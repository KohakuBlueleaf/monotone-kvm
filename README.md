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

## The monotone variant

We keep KVM's whole *mechanism* — BSWA, sink, partial RoPE, qk-norm, the Means
readout, the chunk recurrence — and replace only the **merge policy**:

| | official KVM | monotone |
|---|---|---|
| which tokens merge | cosine novelty + argmax routing | a fixed integer schedule |
| data-dependent?    | **yes** (depends on K values)   | **no** (pure arithmetic) |
| slot semantics     | unstructured centroid           | clean contiguous interval |

Each overflow chunk becomes one bucket; the schedule decides whether to fuse one
adjacent equal-size pair. Being data-independent, it is fully precomputable —
the key to the parallel form below.

## Bucket schedules

Buckets run newest → oldest, sizes in chunk units. Each step inserts a singleton
and merges **at most one** adjacent equal-size pair, preserving four invariants:
(1) count is monotone non-decreasing, (2) only equal-size adjacent buckets merge,
(3) the newest bucket never merges, (4) sizes are non-decreasing toward the
oldest.

`scheduler.py` spans the full `O(state)` continuum:

| name | bucket count |
|---|---|
| `fixed(k)`     | `O(1)` — true cap (force-merges; the linear-attention end) |
| `log`          | `O(log t)` — de-amortized binary carry, `bit_length(t)` exactly |
| `power(alpha)` | `O(max(t^a, log t))` — the `t^alpha` continuum |
| `sqrt`         | `O(sqrt t)` — `power(alpha=1/2)` |
| `linear`       | `O(t)` — never merges (the full-attention end) |

Same-size-only merges impose an **`O(log t)` floor**: `log` is the most
aggressive clean-dyadic schedule, and `fixed(k)` drops below the floor only by
force-merging the two oldest buckets.

## Layout

```
src/monotone_kvm/
  scheduler.py  bucket schedules + merge primitive + invariant checks + simulate()
  helpers.py    partial RoPE, causal mask
  kvm.py        KVMAttention          -- official KVM, faithful
  monotone.py   MonotoneKVMAttention  -- KVM mechanism + a bucket schedule
  plain.py      PlainAttention        -- full causal attention, the baseline
  model.py      TinyLM                -- small Transformer LM, any of the three
scripts/        demos, scheduler tests, training, sweep, visualizations
temp/           gitignored scratch (upstream KVM-paper clone, old notes/PoC)
figures/        gitignored generated plots
```

All three attention modules are drop-in: the recurrence and RoPE live inside
`forward`, so `TinyLM` treats them as ordinary causal attention layers.

## Training & sweeps

`train_demo.py` trains a `TinyLM` at the character level on a
procedurally-generated tiny-stories corpus (zero data dependencies); `--attn
both` overlays KVM vs monotone. `sweep.py` trains a fixed backbone with five
configs on one corpus/seed — `plain`, `kvm fixed`, `kvm power_law`,
`monotone log`, `monotone sqrt` — and plots their loss curves, isolating routing
method and state budget. `viz_buckets.py` writes `bucket_counts.png` (slot count
vs context) and `bucket_structure.png` (the dyadic interval structure over time).

## Efficiency

The chunked `forward` is `O(T·(log T + chunk_len))` FLOPs — subquadratic — but a
Python loop over chunks: the right shape for autoregressive decode, launch-bound
for training (official KVM is the same and leans on `torch.compile`).

Because the monotone schedule is **data-independent**, a parallel prefill exists
that KVM structurally cannot have: every bucket is a fixed contiguous interval,
so all bucket summaries come from one `cumsum` (or a dyadic pyramid), and a
static `BlockMask` lets a single `flex_attention` call do the read. Roadmap:
keep the loop as the decode path, add the prefix-sum + FlexAttention prefill
path — same weights, same math, two execution modes.
