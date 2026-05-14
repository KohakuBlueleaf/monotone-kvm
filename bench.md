# Benchmark results

Comprehensive benchmark of every attention variant — speed, VRAM, the effective
KV length, and accuracy. Snapshot on an **RTX 5060 Ti**, bf16, batch 4,
hidden 512, 8 heads, `chunk_len=32`, `n_bswa_chunks=2`. Reproduce with
`python temp/bench_comprehensive.py` (writes the full report + PNG plots).

The two trainable Triton variants compared:
* **monotone-KVM** — KVM's mechanism with *only* the routing decision replaced
  by a data-independent bucket schedule. The Triton path runs the full feature
  set (merge gate, head temps) and is verified bit-exact-class vs the naive
  recurrence.
* **KVM** — official data-dependent merge routing; the Triton path runs the
  budget schedules with vlens / gate / head-temps off (the kernel's restriction).

## Effective KV length — *why* it's fast

How many KV positions a query streams over per softmax pass: `plain` = T;
KVM / monotone = `live + bswa_len`. The `bswa_len` (64) part is the raw sliding
window. The `live` part is the **compressed state slots a query actually
attends over** — each slot is one summary (`LN(sum)` of keys, mean of values)
of a contiguous run of history tokens.

`live` is **measured**, not estimated: it is counted directly from the PHASE-2
`bias` mask the Triton kernel reads — a `-inf` bias entry is a padding slot the
softmax zeroes out, so it does not count. (`pad` below is `M`, the padded
power-of-2 width the kernel *loads*; `live ≤ pad`.) The benchmark also
cross-checks `live` against the exact naive recurrence's recorded bucket trace —
they match at every point.

Both monotone-KVM and KVM are **token-based**, so `live` is in token units for
both and the two are directly comparable slot-for-slot:

* **monotone** — a deterministic schedule partitions the exited tokens into
  contiguous buckets. `mono-log` keeps `bit_length(exited)` buckets — a *flat*
  `live ≈ 12–16` across the whole table; `mono-sqrt` keeps `~sqrt(exited)`.
* **KVM** — data-dependent cosine-novelty / argmax routing fills a token-unit
  budget; `kvm-sqrt` keeps `~sqrt` slots, **budget-matched to `mono-sqrt`**.

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | 2048 (1×) | 4096 (1×) | 8192 (1×) | 16384 (1×) | 32768 (1×) |
| kvm-power | 96 (live=32, 21×) | 108 (live=44, 38×) | 127 (live=63, 65×) | 153 (live=89, 107×) | 190 (live=126, 172×) |
| kvm-sqrt | 108 (live=44, 19×) | 127 (live=63, 32×) | 154 (live=90, 53×) | 191 (live=127, 86×) | 244 (live=180, 134×) |
| kvm-256 | 320 (live=256, 6×) | 320 (live=256, 13×) | 320 (live=256, 26×) | 320 (live=256, 51×) | 320 (live=256, 102×) |
| mono-log | 76 (live=12, 27×) | 77 (live=13, 53×) | 78 (live=14, 105×) | 79 (live=15, 207×) | 80 (live=16, **410×**) |
| mono-sqrt | 110 (live=46, 19×) | 129 (live=65, 32×) | 156 (live=92, 53×) | 193 (live=129, 85×) | 246 (live=182, 133×) |

`mono-log` is the aggressive end — a flat `live ≈ 16` no matter how long the
context. `mono-sqrt` (`live=182`) and `kvm-sqrt` (`live=180`) carry the same
`~sqrt` budget at T=32768 — so that pair isolates exactly the routing decision.

**Cross-check** (`live` from the plan the kernels consume vs the real bucket
count the exact naive recurrence builds — must match):

| config | T=2048 | T=8192 |
|---|---|---|
| mono-log | built 12, plan 12 ✓ | built 14, plan 14 ✓ |
| mono-sqrt | built 46, plan 46 ✓ | built 92, plan 92 ✓ |

## Forward speed — `ms (× vs plain)`

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | 2.0 (1.00×) | 5.6 (1.00×) | 17.4 (1.00×) | 60.4 (1.00×) | 222.9 (1.00×) |
| kvm-power naive | 29.8 (0.07×) | 67.6 (0.08×) | 142.3 (0.12×) | — | — |
| **kvm-power triton** | 1.9 (1.06×) | 4.3 (1.31×) | 9.0 (1.92×) | 21.9 (2.75×) | 43.9 (**5.07×**) |
| kvm-sqrt triton | 2.0 (0.99×) | 4.3 (1.31×) | 11.0 (1.58×) | 22.0 (2.75×) | 251.4 (0.89×) ⚠ |
| kvm-256 triton | 14.5 (0.14×) | 29.8 (0.19×) | 59.8 (0.29×) | 125.2 (0.48×) | 250.7 (0.89×) ⚠ |
| mono-log naive | 27.0 (0.07×) | 58.2 (0.10×) | 122.6 (0.14×) | — | — |
| **mono-log triton** | 1.9 (1.06×) | 4.0 (1.40×) | 8.5 (2.05×) | 16.9 (3.57×) | 33.8 (**6.60×**) |
| mono-log flex | 4.5 (0.44×) | 10.4 (0.54×) | 21.6 (0.80×) | 46.3 (1.31×) | 95.6 (2.33×) 🔴 |
| **mono-sqrt triton** | 2.7 (0.74×) | 4.7 (1.18×) | 10.2 (1.70×) | 23.0 (2.63×) | 49.9 (**4.47×**) |
| mono-sqrt flex | 5.0 (0.40×) | 10.3 (0.54×) | 21.4 (0.81×) | 43.2 (1.40×) | 91.2 (2.45×) 🔴 |

## End-to-end training speed (fwd + bwd) — `ms (× vs plain)`

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | 6.5 (1.00×) | 19.5 (1.00×) | 68.9 (1.00×) | 257.3 (1.00×) | 942.9 (1.00×) |
| **kvm-power triton** | 4.6 (1.41×) | 10.5 (1.86×) | 22.4 (3.08×) | 114.4 (2.25×) | 227.9 (4.14×) |
| kvm-sqrt triton | 5.2 (1.24×) | 10.6 (1.84×) | 56.3 (1.22×) | 112.8 (2.28×) | 347.2 (2.72×) |
| kvm-256 triton | 20.3 (0.32×) | 41.0 (0.47×) | 85.7 (0.80×) | 177.0 (1.45×) | 353.1 (2.67×) |
| **mono-log triton** | 5.1 (1.27×) | 10.3 (1.88×) | 22.0 (3.14×) | 43.7 (5.89×) | 90.6 (**10.41×**) |
| mono-log flex | 12.7 (0.51×) | 26.0 (0.75×) | 55.9 (1.23×) | 118.7 (2.17×) | 249.2 (3.78×) 🔴 |
| **mono-sqrt triton** | 8.1 (0.80×) | 12.8 (1.52×) | 30.9 (2.23×) | 66.2 (3.89×) | 177.0 (**5.33×**) |
| mono-sqrt flex | 12.7 (0.51×) | 25.6 (0.76×) | 55.6 (1.24×) | 112.9 (2.28×) | 238.8 (3.95×) 🔴 |

## Peak VRAM (end-to-end training) — GB

The Triton path scales cleanly. The FlexAttention path hits a **memory wall**:
its dyadic pyramid is built over *tokens* (not chunks), so the reduction is
`chunk_len ×` larger than the old chunk-based pyramid — it blows past the card's
16 GB by T=16384.

| config | T=2048 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|
| plain | 0.16 | 0.50 | 0.95 | 1.84 |
| mono-log triton | 0.48 | 0.97 | 1.75 | **3.45** |
| mono-sqrt triton | 0.29 | 1.24 | 3.49 | 6.94 |
| mono-log / mono-sqrt flex | 0.44 / 0.31 | 3.65 | **14.09** | **55.54** 🔴 |

## Notes

* 🔴 `mono flex` — the FlexAttention path's token dyadic pyramid blows up in
  memory (14 GB at T=16384, 55 GB at T=32768 → far past the card, thrashing).
  flex is the *cross-check* path (it agrees with the naive recurrence to ~1e-7);
  the Triton kernels have no such intermediate and keep scaling. ⚠ `kvm-256` and
  `kvm-sqrt@T=32768` are slow because the KVM *forward* merge kernel carries the
  256-slot state in registers and spills to local memory — the one remaining
  Triton perf gap (the backward is state-tiled).
* **Headline:** `mono-log triton` is the fastest trainable path — **10.4×** vs
  plain for E2E training at T=32768 (a flat `live ≈ 16` state, **410×** KV
  compression), scaling cleanly. `mono-sqrt triton` carries the same `~sqrt`
  budget as `kvm-sqrt` and trains at **5.3×**; `kvm-power triton` is 4.1×. All
  are fully differentiable (custom fwd + bwd Triton kernels).

## Accuracy (forward, T=2048, bf16 vs an fp32 naive reference)

`naive bf16` is the precision floor (same model, same dtype, no kernel) — the
Triton path lands in the same band.

| config | triton bf16 vs fp32 | naive bf16 vs fp32 (floor) |
|---|---|---|
| mono-log triton | abs avg 1.2e-4, rel avg 4.3e-2 | abs avg 6.7e-4, rel avg 3.3e-1 |
| mono-sqrt triton | abs avg 1.3e-4, rel avg 4.5e-2 | abs avg 7.5e-4, rel avg 3.3e-1 |
| mono-log flex | abs avg 1.5e-4, rel avg 5.4e-2 | abs avg 7.1e-4, rel avg 3.2e-1 |
| kvm-power triton | abs avg 2.7e-3, rel avg 5.7e-1 | abs avg 3.2e-3, rel avg 6.5e-1 |
| kvm-256 triton | abs avg 2.7e-3, rel avg 1.1e0 | abs avg 3.1e-3, rel avg 1.2e0 |

Monotone is data-independent → its Triton path is numerically tight
(bit-exact-class vs the naive recurrence; the bf16 error is just input
quantization). KVM routing (cosine novelty + argmax) is *chaotic* in low
precision — a 1-ulp wobble flips a route — so its bf16 error is intrinsically
"loud"; the `naive bf16` column shows the same loudness, i.e. it is the method,
not the kernel. The backward kernels are separately verified against PyTorch
references (`temp/verify_*.py`): monotone's naive recurrence is bit-exact vs an
independent reference, its Triton path lands at the TF32 `tl.dot` floor (~4e-4
relative, fwd + bwd), its flex path at ~1e-7; KVM `our` tracks `pt` at matched
precision (M≤128 SRAM-resident, M=256 state-tiled).
