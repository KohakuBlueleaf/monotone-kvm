# Monotone-Bucket KVM

A small, hackable PyTorch implementation of **Key-Value Means (KVM)** attention
and a variant that swaps KVM's data-dependent merge routing for a
**deterministic monotone bucket schedule** — plus a tiny Transformer LM to train
both.

```
pip install -e .
python scripts/demo_kvm.py         # official KVM: equivalence + recurrence
python scripts/demo_monotone.py    # monotone-schedule variant
python scripts/test_schedulers.py  # invariant tests for every schedule
python scripts/train_demo.py       # tiny char-LM: KVM vs monotone
python scripts/sweep.py            # plain / kvm / monotone loss-curve comparison
python scripts/bench_flex.py       # recurrent vs FlexAttention prefill: precision + speed
python scripts/viz_buckets.py      # plots: our bucketing vs KVM's
```

## KVM in brief

KVM (arXiv:2605.09877) is plain softmax attention over
`[ compressed STATE | recent raw tokens (BSWA window) ]`, updated
chunk-recurrently: every `chunk_len` tokens, the chunk leaving the window is
folded into the state — each token **appended** as a new slot or **merged**
(argmax-cosine routed) into an existing one, key/value summed in. State keys
read back as `LN(sum)`, values as the mean — hence "Key-Value *Means*". One
dense softmax pass, no custom kernels.

`kvm.py` reproduces this faithfully (minus the training backbone, token-shift,
value-residual, and the KV-cache decode path).

## The monotone idea

A token far in the past rarely needs token-level resolution; the recent context
does. So compress the *exited* history geometrically — the newest exited token
stays a singleton, older tokens fuse into buckets that grow the further back you
look. The `log` schedule, at 63 exited tokens:

```
newest -> oldest:   [1] [2] [4] [8] [16] [32]      (bucket sizes, sum = 63)
```

One summary token for the most recent exit, a 2-token summary behind it, then 4,
8, 16, a 32-token summary for the oldest stretch — each size-`k` bucket is `k`
original tokens collapsed into one `LN(sum)` key / mean value. A query then sees
the recent past sharply and the distant past coarsely, with a smooth gradient
between, at `O(log t)` or `O(sqrt t)` total slots instead of `O(t)`.

In spirit this is what KVM's learned routing *tries* to do — novel tokens get a
slot, redundant ones get merged. The monotone bet is that you don't need to
*learn* the partition: a fixed schedule obeying "recent = fine, old = coarse"
captures most of it. And because the partition is then a pure function of
position — no data dependence — the whole compressed state is **precomputable**,
which is exactly what lets `forward_flex` / `forward_triton` build it in parallel
(a `cumsum`, not a recurrence). The honest tradeoff: a position-only schedule
can't drop below the `O(log t)` floor without fusing unequal-size buckets (which
breaks the dyadic structure — see below), whereas KVM's data-dependent routing
can — at the cost of that precomputability.

## The monotone variant

We keep KVM's whole *mechanism* — BSWA, sink, partial RoPE, qk-norm, the Means
readout, the chunk recurrence — unchanged, and replace **only the routing
decision**. KVM and monotone are both **token-based**: when a token leaves the
BSWA window, something must decide whether it starts a fresh state slot or
merges into an existing one.

| | official KVM | monotone |
|---|---|---|
| who decides the merge | cosine novelty + argmax routing | a deterministic integer schedule |
| data-dependent?       | **yes** (depends on K values)   | **no** (pure arithmetic) |
| slot semantics        | unstructured centroid           | clean contiguous token interval |

The *merge operation itself* — `LN(sum of prepared keys)` for the key, token
mean for the value, `+log(bucket size)` score bias — is KVM's, untouched. The
schedule decides only *which contiguous run of tokens* forms a bucket: each
exiting token is a singleton bucket, and the schedule fuses adjacent equal-size
buckets by a fixed integer rule. Being data-independent, it is fully
precomputable — the key to the parallel forms below.

## Bucket schedules

Buckets run newest → oldest, sizes in **token units** — a bucket of "size k" is
k original tokens summarized into one slot, and the sizes always sum to the
token count `t`. Each step inserts a singleton and merges **at most one**
adjacent equal-size pair, preserving four invariants: (1) count is monotone
non-decreasing, (2) only equal-size adjacent buckets merge, (3) the newest
bucket never merges, (4) sizes are non-decreasing toward the oldest.

The schedule's timestep `t` is the **token count** — the same unit KVM's budget
uses — so the schedules are directly comparable to KVM slot-for-slot.
`scheduler.py` spans the `O(state)` continuum:

| name | bucket count (`t` = tokens) |
|---|---|
| `log`             | `O(log t)` — de-amortized binary carry, `bit_length(t)` exactly |
| `logbudget(c)`    | `O(c·log t)` — tunable soft-budget log (adds a `coeff` knob) |
| `power(alpha,c)`  | `O(max(c·t^a, log t))` — the `t^alpha` continuum |
| `sqrt(c)`         | `O(c·sqrt t)` — `power(alpha=1/2)` |
| `linear`          | `O(t)` — never merges (the full-attention reference) |

Every schedule fuses **equal-size buckets only** — that is invariant 2, and it
is not optional (the dyadic `1 1 2 4 8 16 …` structure depends on it). It
imposes an **`O(log t)` floor**: you cannot drop below `~bit_length(t)` buckets,
so `log` is the most aggressive schedule in the family. There is deliberately no
`O(1)` schedule — a true constant cap would have to fuse *unequal*-size buckets,
which the invariant forbids. The `coeff` knob (`sqrt` / `power` / `logbudget`)
just scales the budget; no unit conversion is needed, since `t` is already in
tokens.

## Layout

```
src/monotone_kvm/
  scheduler.py        bucket schedules + merge primitive + invariant checks + simulate()
  helpers.py          partial RoPE, causal mask
  model.py            TinyLM -- small Transformer LM, any of the three attentions
  attention/          the attention layers (drop-in causal-attention modules)
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
temp/                 gitignored scratch (upstream KVM-paper clone, notes, kernel PoCs/benchmarks)
figures/              gitignored generated plots
```

All three attention modules are drop-in: the recurrence and RoPE live inside
`forward`, so `TinyLM` treats them as ordinary causal attention layers.

## Training & sweeps

`train_demo.py` trains a `TinyLM` at the character level on a
procedurally-generated tiny-stories corpus (zero data dependencies); `--attn
both` overlays KVM vs monotone. `sweep.py` trains a fixed backbone over a
**state-budget ladder** on one corpus/seed — `mono-log`, `mono-logbudget-c2`,
`mono-sqrt`, `kvm-sqrt`, `kvm-256`, `plain` — and plots their loss curves.
Because both methods are token-based, `mono-sqrt` and `kvm-sqrt` carry the
*same* `~sqrt(T)` state budget — so the sweep isolates exactly one variable: the
routing decision (deterministic schedule vs data-dependent KVM merge).
`viz_buckets.py` writes `bucket_counts.png` (slot count vs context) and
`bucket_structure.png` (the dyadic token-interval structure over time).

## Efficiency — execution modes

Same weights, same math, three ways to run `KVMAttention` / `MonotoneKVMAttention`:

- **`forward`** — the reference path. Per-token prepared K/V, a `cumsum`, and
  one softmax per query chunk: because the schedule is data-independent, a
  bucket summary over a token interval `[a, b)` is just `cumsum[b] - cumsum[a]`,
  so there is no Python token loop — only a short loop over query chunks. This
  is what training / prefill semantically *is*.
- **`forward_flex`** — parallel prefill via `flex_attention` (monotone only).
  Because the monotone schedule is **data-independent**, every bucket is a fixed
  contiguous token interval: all bucket summaries come from one dyadic token
  pyramid, and a static `BlockMask` lets a single `flex_attention` call do the
  read — no Python loop. It is the cross-check path (it agrees with the naive
  recurrence to ~1e-7), but the token dyadic pyramid is `chunk_len×` larger than
  a chunk pyramid, so it hits a memory wall well before the Triton path does.
- **`forward_triton`** — both PHASE 1 (build the compressed state) and PHASE 2
  (the chunked attention) as **custom Triton kernels**, with hand-written
  backward kernels — **fully differentiable end-to-end**. The fastest training
  path, and it keeps scaling where `flex` runs out of memory. Monotone runs the
  full feature set (merge gate, head temps); KVM runs the budget schedules
  (vlens / gate / head-temps off — the kernel's restriction).

`TinyLM` **auto-selects** the kernel: `attn_impl="auto"` (the default) routes to
`forward_triton` whenever the input is on CUDA with a chunk-aligned tail, and
falls back to the naive recurrence otherwise. Set `attn_impl` to
`"naive"` / `"triton"` / `"flex"` to force a path.

**Numbers** (RTX 5060 Ti, bf16 — see [`bench.md`](bench.md) for the full tables):

| vs plain attention @ T=32768 | forward | fwd + bwd (training) |
|---|---|---|
| `monotone triton` (`mono-log`)  | **6.6×** | **10.4×** |
| `monotone triton` (`mono-sqrt`) | **4.5×** | **5.3×** |
| `kvm triton` (`kvm-power`)       | **5.1×** | **4.1×** |

The headline: a query attends over only `live + window` positions — `mono-log`
at T=32768 attends over **~80 of 32768** (a flat **16** live state slots, a
**410×** compression). `live` is *measured* from the PHASE-2 bias mask the
kernel actually reads — not estimated — and cross-checked against the exact
recurrence. The Triton kernels turn that structure into real wall-clock
speedups, all verified against PyTorch references (`temp/verify_*.py`): the
naive recurrence is bit-exact vs an independent reference, and the Triton path
matches it to the TF32 floor in forward *and* backward.
