# KVM Attention — fast kernels & a scheduler playground

A small, hackable PyTorch implementation of **Key-Value Means (KVM)** attention,
built around two goals:

1. **Fast, differentiable kernels.** Custom Triton kernels (forward *and*
   hand-written backward) for both phases of KVM-family attention — so it is
   practical to *train*, not just run, and scales cleanly past 32k context.
2. **A scheduler playground.** KVM's "which exited token merges into which state
   slot?" is a pluggable *routing decision*. This repo implements KVM's official
   learned, data-dependent routing **and** a family of deterministic,
   data-independent **monotone bucket schedules**, so the two compare
   slot-for-slot on equal footing — and adding another scheduler is easy.

Honest summary up front (see [`bench.md`](bench.md)): the monotone schedules win
on *systems* — data-independence makes the compressed state precomputable, which
is exactly what the fast parallel kernels exploit (**~10× training speedup** at
T=32768). On a content-addressed task, though, KVM's *learned* routing wins on
*quality* — and `kvm-sqrt` is also among the fastest paths in real training. The
open direction: better schedulers, somewhere between a fixed positional rule and
full learned routing.

```
pip install -e .
python scripts/demo_kvm.py         # official KVM: equivalence + recurrence
python scripts/demo_monotone.py    # the monotone-schedule variant
python scripts/test_schedulers.py  # invariant tests for every schedule
python scripts/train_demo.py       # tiny char-LM: KVM vs monotone
python scripts/sweep.py            # plain / kvm / monotone loss-curve comparison
python scripts/bench_flex.py       # recurrent vs FlexAttention prefill: precision + speed
python scripts/viz_buckets.py      # plots: monotone schedules vs KVM's budgets
```

## KVM in brief

KVM (arXiv:2605.09877) is plain softmax attention over
`[ compressed STATE | recent raw tokens (BSWA window) ]`, updated
chunk-recurrently: every `chunk_len` tokens, the chunk leaving the window is
folded into the state — each token **appended** as a new slot or **merged**
(argmax-cosine routed) into an existing one, key/value summed in. State keys
read back as `LN(sum)`, values as the mean — hence "Key-Value *Means*". One
dense softmax pass.

`kvm.py` reproduces this faithfully (minus the training backbone, token-shift,
value-residual, and the KV-cache decode path).

## The routing decision — the pluggable axis

When a token leaves the BSWA window, *something* decides whether it starts a
fresh state slot or merges into an existing one. That decision — and **only**
that decision — is what this repo swaps out:

| | KVM's learned routing | monotone schedules |
|---|---|---|
| who decides the merge | cosine novelty + argmax routing | a deterministic integer schedule |
| data-dependent?       | **yes** (depends on K values)   | **no** (pure arithmetic) |
| slot semantics        | unstructured centroid           | clean contiguous token interval |
| precomputable?        | no — a sequential recurrence    | **yes** — a pure function of position |

Everything else is shared and untouched: the BSWA window, sink, partial RoPE,
qk-norm, and — crucially — the *merge operation itself* (`LN(sum of prepared
keys)` → key, token mean → value, `+log(bucket size)` score bias). Only *which
tokens get grouped* changes. Both decisions are **token-based**: they fire on
individual tokens as they exit the window.

## Monotone bucket schedules

A token far in the past rarely needs token-level resolution; the recent context
does. So a monotone schedule compresses the *exited* history geometrically — the
newest exited token stays a singleton, older tokens fuse into buckets that grow
the further back you look. The `log` schedule, at 63 exited tokens:

```
newest -> oldest:   [1] [2] [4] [8] [16] [32]      (bucket sizes, sum = 63)
```

Each size-`k` bucket is `k` original tokens collapsed into one `LN(sum)` key /
mean value — so a query sees the recent past sharply, the distant past coarsely,
with a smooth gradient between, at `O(log t)` / `O(sqrt t)` slots instead of
`O(t)`. Sizes run newest → oldest and always sum to the token count `t`.

Each step inserts a singleton and fuses **at most one adjacent equal-size pair**,
preserving four invariants: (1) count is monotone non-decreasing, (2) only
equal-size adjacent buckets merge, (3) the newest bucket never merges, (4) sizes
are non-decreasing toward the oldest. `scheduler.py` spans the `O(state)`
continuum:

| name | bucket count (`t` = tokens) |
|---|---|
| `log`             | `O(log t)` — de-amortized binary carry, `bit_length(t)` exactly |
| `logbudget(c)`    | `O(c·log t)` — tunable soft-budget log (adds a `coeff` knob) |
| `power(alpha,c)`  | `O(max(c·t^a, log t))` — the `t^alpha` continuum |
| `sqrt(c)`         | `O(c·sqrt t)` — `power(alpha=1/2)` |
| `linear`          | `O(t)` — never merges (the full-attention reference) |

The equal-size-only rule (invariant 2) is what keeps the structure cleanly
dyadic (`1 1 2 4 8 16 …`, every bucket a power-of-two-aligned token interval) —
and it imposes an **`O(log t)` floor**: a position-only schedule cannot drop
below `~bit_length(t)` buckets without fusing *unequal* buckets, whereas KVM's
data-dependent routing can. That floor — and the data-independence that makes
the schedule precomputable — is the whole tradeoff. Adding a new scheduler is a
~30-line subclass of `BucketScheduler`.

## The fast kernels

Same weights, same math, three execution modes for `KVMAttention` /
`MonotoneKVMAttention`:

- **`forward`** — the reference path. Per-token prepared K/V, a `cumsum`, and one
  softmax per query chunk; no Python token loop (a bucket summary over `[a, b)`
  is just `cumsum[b] - cumsum[a]`). What training / prefill semantically *is*.
- **`forward_flex`** — parallel prefill via `flex_attention` (monotone only): the
  data-independent schedule means every bucket is a fixed contiguous interval,
  so one dyadic token pyramid + a static `BlockMask` does the whole read. The
  cross-check path (~1e-7 vs the recurrence); the token pyramid is memory-heavy,
  so it hits a wall well before Triton does.
- **`forward_triton`** — both PHASE 1 (build the compressed state) and PHASE 2
  (the chunked attention) as **custom Triton kernels** with hand-written
  backward — **fully differentiable end to end**. The fastest training path, and
  it keeps scaling where `flex` runs out of memory. Monotone PHASE 1 is a token
  `cumsum` + gather (data-independent); KVM PHASE 1 is the merge recurrence
  (incl. a state-tiled backward). Monotone runs the full feature set (merge
  gate, head temps); KVM runs the budget schedules with vlens / gate / head-temps
  off (the kernel's restriction).

`TinyLM` **auto-selects** the kernel: `attn_impl="auto"` (the default) routes to
`forward_triton` whenever the input is on CUDA with a chunk-aligned tail, and
falls back to the naive recurrence otherwise. Force a path with `attn_impl` =
`"naive"` / `"triton"` / `"flex"`.

## Layout

```
src/monotone_kvm/
  scheduler.py        bucket schedules + merge primitive + invariant checks + simulate()
  helpers.py          partial RoPE, causal mask
  model.py            TinyLM -- small Transformer LM, any of the three attentions
  attention/
    plain.py          PlainAttention        -- full causal attention, the baseline
    kvm.py            KVMAttention          -- official KVM, faithful
    monotone.py       MonotoneKVMAttention  -- KVM mechanism + a bucket schedule
    monotone_flex.py  forward_flex          -- parallel FlexAttention prefill path
  triton/             Triton kernels (PHASE 1 + PHASE 2), forward + backward
    phase2.py         the shared chunked-attention kernels
    monotone_phase1.py  monotone cumsum/gather kernels (data-independent)
    kvm_phase1.py     KVM merge-recurrence kernels (incl. the state-tiled backward)
    entry.py          forward entry points wiring PHASE 1 + PHASE 2
    common.py         shared helpers
  helion/             Helion kernels -- placeholder (Helion is Linux-only)
scripts/              demos, scheduler tests, training, sweep, benchmark, visualizations
temp/                 gitignored scratch (upstream KVM-paper clone, notes, kernel PoCs)
figures/              gitignored generated plots
```

All three attention modules are drop-in causal-attention layers — the recurrence
and RoPE live inside `forward`, so `TinyLM` treats them as ordinary layers.

## Results

**Speed** (RTX 5060 Ti, bf16, T=32768 — full tables in [`bench.md`](bench.md)):

| vs plain attention @ T=32768 | forward | fwd + bwd (training) |
|---|---|---|
| `monotone triton` (`mono-log`)  | **6.6×** | **10.4×** |
| `monotone triton` (`mono-sqrt`) | **4.5×** | **5.3×** |
| `kvm triton` (`kvm-power`)       | **5.1×** | **4.1×** |

A query attends over only `live + window` positions — `mono-log` at T=32768
attends over ~80 of 32768 (a flat **16** live state slots, **410×** compression).
`live` is measured from the PHASE-2 bias mask the kernel actually reads and
cross-checked against the exact recurrence; the Triton path matches the naive
reference to the TF32 floor in forward *and* backward.

**Quality** — `scripts/sweep.py`, a fixed `TinyLM` backbone, six configs, same
corpus / seed, 4000 steps, seq_len 2048 (final loss = mean of the last 100
steps):

| config | state slots | final loss | sweep throughput |
|---|---|---|---|
| plain | full | 0.074 | 257k tok/s |
| kvm-256 | 256 | 0.093 | 110k tok/s |
| kvm-sqrt | 64 | **0.096** | **371k tok/s** |
| mono-sqrt | 64 | 0.105 | 323k tok/s |
| mono-logbudget-c2 | 32 | 0.105 | 372k tok/s |
| mono-log | 16 | 0.105 | 370k tok/s |

Two honest findings. (1) **monotone is budget-insensitive** — M=16/32/64 all
land at ~0.105, and the monotone loss curves flatten by ~step 1500 while KVM and
plain keep descending. A position-only schedule is a ceiling on accessible
long-range info, because this corpus's long-range signal is content-addressed (a
recurring story subject) and positional coarsening blurs it away. (2) **KVM's
learned routing wins at matched budget** — `kvm-sqrt` (0.096) beats `mono-sqrt`
(0.105) at the same 64 slots, *and* trains at the highest throughput in the
sweep. (The forward-only benchmark shows `kvm-sqrt` at 0.89× at T=32768 — a
worst-case artifact of the M=256 forward merge kernel spilling at extreme T; at
the lengths and budgets real training uses it is among the fastest paths, as the
371k tok/s shows.)

So the split is clean: **monotone's contribution is the fast, precomputable
kernels; KVM's learned routing currently owns quality.** Which makes "explore
better schedulers" the interesting direction. (Caveats: one seed, an easy
templated corpus, differences in the tail.)
