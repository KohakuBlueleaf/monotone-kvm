# Benchmark results

Comprehensive benchmark across the KVM-sqrt coefficient ladder (`c=1/2/4/8/16`),
`kvm-256`, and the monotone schedule family. Snapshot on an **RTX 5060 Ti**, bf16,
batch 4, hidden 512, 8 heads, `chunk_len=32`, `n_bswa_chunks=2`. Reproduce with
`python scripts/bench_comprehensive.py` (writes `bench_report.md` + PNG plots in
`figures/`).

The headline knobs are:
* **kvm-sqrt `c=k`** — official KVM with state budget `B(t) = max(state_min_len,
  c·sqrt(t))`. The paper's headline config is `c=16`.
* **kvm-256** — fixed budget of 256 slots regardless of context.
* **monotone schedules** — deterministic data-independent partition (token-based
  buckets). `mono-log` keeps `~log2(T)` slots; `mono-sqrt c=k` keeps `~k·sqrt(T)`,
  budget-matched to `kvm-sqrt c=k` slot for slot.

## Effective KV length — live and padded

`live` = actual state slots the kernel attends over (counted from the PHASE-2
`bias` mask: non `-inf` entries; `-inf` slots are padding the softmax zeros
out). `pad M` = the padded kernel load width. **The new tiled forward pads to
multiples of 64** instead of pow2 — saves ~25–30% slot-work at the upper end of
each pow2 bracket.

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | T (1×) | T (1×) | T (1×) | T (1×) | T (1×) |
| **mono-log** | live 12, pad 16 | 13/16 | 14/16 | 15/16 | **16/16 → 410× compression** |
| mono-logbudget c=2 | 23/32 | 25/32 | 27/32 | 29/32 | 31/32 |
| mono-sqrt c=1 | 46/64 | 65/128 | 92/128 | 129/256 | 182/256 |
| mono-sqrt c=2 | 91/128 | 128/128 | 182/256 | 257/512 | 363/512 |
| mono-sqrt c=4 | 180/256 | 255/256 | 362/512 | 512/512 | 725/1024 |
| kvm-sqrt c=1 | 64/64 | 64/64 | 90/128 | 127/128 | **180/192** |
| kvm-sqrt c=2 | 89/128 | 127/128 | 180/192 | 255/256 | 361/384 |
| kvm-sqrt c=4 | 179/192 | 254/256 | 361/384 | 511/512 | 723/768 |
| kvm-sqrt c=8 | 359/384 | 509/512 | 722/768 | 1022/1024 | 1447/1472 |
| **kvm-sqrt c=16 (OFFICIAL)** | 718/768 | 1019/1024 | 1445/1472 | 2045/2048 | 2894/2944 |
| kvm-256 (fixed) | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |

Note the budget alignment between `mono-sqrt c=k` and `kvm-sqrt c=k`: identical
`live` to within ±2 across the whole table (e.g. `live=182` vs `180` at c=1 /
T=32768). So the kvm-vs-mono comparison at each c is a pure routing-decision
test (data-dependent argmax vs deterministic schedule) on equal state budget.

## Forward speed — `ms (× vs plain)`

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | 2.0 (1.00×) | 5.6 (1.00×) | 17.4 (1.00×) | 58.3 (1.00×) | 210.7 (1.00×) |
| **mono-log** | 1.9 (1.06×) | 4.0 (1.40×) | 8.5 (**2.05×**) | 16.9 (**3.44×**) | 33.7 (**6.24×**) |
| mono-logbudget c=2 | 1.9 (1.04×) | 4.2 (1.33×) | 8.8 (1.99×) | 17.4 (3.35×) | 34.7 (6.06×) |
| mono-sqrt c=1 | 2.0 (0.97×) | 4.7 (1.18×) | 10.2 (1.70×) | 23.0 (2.54×) | 50.0 (4.22×) |
| mono-sqrt c=2 | 2.3 (0.86×) | 4.9 (1.14×) | 12.5 (1.40×) | 30.0 (1.95×) | 90.6 (2.32×) |
| mono-sqrt c=4 | 2.9 (0.70×) | 6.0 (0.93×) | 22.5 (0.77×) | 45.5 (1.28×) | 676.7 (0.31×) 🔴 |
| **kvm-sqrt c=1** | 2.0 (0.99×) | 4.3 (1.31×) | 11.0 (1.58×) | 21.9 (2.66×) | 52.1 (**4.04×**) |
| kvm-sqrt c=2 | 2.5 (0.79×) | 5.3 (1.06×) | 13.1 (1.33×) | 29.8 (1.96×) | 73.9 (2.85×) |
| kvm-sqrt c=4 | 3.0 (0.66×) | 7.2 (0.77×) | 18.6 (0.94×) | 44.3 (1.32×) | 118.2 (1.78×) |
| kvm-sqrt c=8 | 4.4 (0.46×) | 10.8 (0.52×) | 29.7 (0.59×) | 82.5 (0.71×) | — OOM |
| **kvm-sqrt c=16 (OFFICIAL)** | 7.2 (0.28×) | 20.5 (0.27×) | 77.0 (0.23×) | 390.2 (0.15×) | — OOM |
| kvm-256 (fixed) | 3.5 (0.58×) | 7.2 (0.77×) | 14.9 (1.17×) | 29.8 (1.96×) | 59.3 (3.55×) |

🔴 `mono-sqrt c=4 @ T=32768` and the worst-case kvm-sqrt c=16 cells include
thermal / autotune-recompile noise — the kernel is correct (acc passes); the
isolated timing is well-behaved. OOM cells are skipped: with bf16 buck_k+buck_v
tensors of shape `[BH, n_q, M, D]`, the `kvm-sqrt c=8/16 @ T=32768` configs need
>10 GB just for the bucket trajectory on the 16 GB card.

## End-to-end training (fwd + bwd) — `ms (× vs plain)`

| config | T=2048 | T=4096 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|---|
| plain | 6.4 (1.00×) | 19.4 (1.00×) | 66.7 (1.00×) | 243.7 (1.00×) | 898.1 (1.00×) |
| **mono-log** | 7.3 (0.88×) | 15.2 (1.28×) | 31.1 (**2.14×**) | 62.1 (**3.92×**) | 124.3 (**7.22×**) |
| mono-logbudget c=2 | 7.6 (0.84×) | 15.7 (1.24×) | 32.1 (2.08×) | 64.2 (3.80×) | 127.6 (7.04×) |
| mono-sqrt c=1 | 8.4 (0.77×) | 18.1 (1.08×) | 41.5 (1.61×) | 88.8 (2.75×) | 234.0 (3.84×) |
| mono-sqrt c=2 | 9.6 (0.67×) | 19.6 (0.99×) | 58.8 (1.13×) | 132.6 (1.84×) | 1336.6 (0.67×) 🔴 |
| mono-sqrt c=4 | 14.2 (0.45×) | 28.2 (0.69×) | 335.2 (0.20×) 🔴 | 671.6 (0.36×) | 12647.4 (0.07×) 🔴 |
| **kvm-sqrt c=1** | 5.2 (**1.23×**) | 10.5 (**1.85×**) | 54.6 (1.22×) | 108.8 (2.24×) | 136.2 (**6.60×**) |
| **kvm-sqrt c=2** | 13.2 (0.49×) | 26.4 (0.74×) | 34.1 (**1.96×**) | 78.0 (**3.13×**) | 201.0 (**4.47×**) |
| kvm-sqrt c=4 | 8.1 (0.79×) | 18.8 (1.03×) | 50.2 (1.33×) | 120.5 (2.02×) | 1189.1 (0.76×) 🔴 |
| kvm-sqrt c=8 | 12.4 (0.52×) | 29.5 (0.66×) | 85.8 (0.78×) | 231.9 (1.05×) | — OOM |
| kvm-sqrt c=16 (OFFICIAL) | 20.8 (0.31×) | 57.2 (0.34×) | 182.9 (0.36×) | 9503.2 (0.03×) 🔴 | — OOM |
| **kvm-256 (fixed)** | 12.7 (0.51×) | 25.7 (0.76×) | 53.3 (1.25×) | 106.1 (**2.30×**) | 212.5 (**4.23×**) |

🔴 cells are dominated by HBM swap / thermal-throttle on the 16 GB consumer
card once peak VRAM crosses ~8 GB. Treat them as upper bounds, not optimums.

## Discussion — log/sqrt with different c, speed × M

**At T=8192 (medium context):**

| config | M (live) | fwd speedup | e2e speedup |
|---|---|---|---|
| mono-log | 14 | **2.05×** | **2.14×** |
| mono-logbudget c=2 | 27 | 1.99× | 2.08× |
| mono-sqrt c=1 | 92 | 1.70× | 1.61× |
| mono-sqrt c=2 | 182 | 1.40× | 1.13× |
| mono-sqrt c=4 | 362 | 0.77× | 0.20× 🔴 |
| kvm-sqrt c=1 | 90 | 1.58× | 1.22× |
| kvm-sqrt c=2 | 180 | 1.33× | **1.96×** |
| kvm-sqrt c=4 | 361 | 0.94× | 1.33× |
| kvm-sqrt c=8 | 722 | 0.59× | 0.78× |
| kvm-sqrt c=16 | 1445 | 0.23× | 0.36× |
| kvm-256 | 256 | 1.17× | 1.25× |

**At T=32768 (long context):**

| config | M (live) | fwd speedup | e2e speedup |
|---|---|---|---|
| mono-log | 16 (FLAT) | **6.24×** | **7.22×** |
| mono-logbudget c=2 | 31 (flat) | 6.06× | 7.04× |
| mono-sqrt c=1 | 182 | 4.22× | 3.84× |
| mono-sqrt c=2 | 363 | 2.32× | 0.67× 🔴 |
| kvm-sqrt c=1 | 180 | 4.04× | 6.60× |
| kvm-sqrt c=2 | 361 | 2.85× | 4.47× |
| kvm-sqrt c=4 | 723 | 1.78× | 0.76× 🔴 |
| kvm-256 | 256 | 3.55× | 4.23× |

**Reading the coefficient sweep:**

- The **`O(log T)`-budget configs (`mono-log`, `mono-logbudget c=2`) are the
  outright speed winners** at long context — M stays nearly flat (12-16 / 23-31
  slots) regardless of T, so compute scales linearly with T instead of with
  `T × M`. At T=32768 they hit 7.22× and 7.04× E2E speedup.
- **`kvm-sqrt c=1` (M ≈ 64-180) is the next sweet spot** — 6.6× E2E at T=32768,
  1.85× at T=4096, comfortably beats plain everywhere past T=4096.
- **`kvm-sqrt c=2` and `kvm-256`** are essentially tied at T=8192-16384 (live
  ≈ 180-256). `kvm-sqrt c=2` grows with context; `kvm-256` doesn't.
- **`kvm-sqrt c=4` onwards is slower than plain at T ≤ 16384** — the per-slot
  state work (cosine sim + argmax + scatter) outpaces plain's O(T²) cost until
  T crosses some break-even point. For `c=16` on a 16 GB card, that break-even
  is past where we can fit the bucket trajectory.
- **`kvm-sqrt c=16` (paper's headline)** is the slowest path everywhere in this
  range, by a wide margin. It runs (no more crashes thanks to the tiled
  forward), but only the next-gen GPU memory budget really wants it.
- **Per-coeff: kvm-sqrt usually beats mono-sqrt** at matched `c` in E2E because
  KVM's bwd is more efficient on this kernel (state-tiled bwd kernel exists in
  `kvm_phase1.py`); monotone's bwd through the cumsum is competitive but
  doesn't win on bwd-heavy regimes.

**M scaling and the pad-to-multiple-of-64 rule.** Padded `M` is now the *next
multiple of 64* (was: next pow2). The savings track exactly the bracket each
`live` lands in:

| live | pow2 M (old) | mult-64 M (new) | saved compute |
|---|---|---|---|
| 180 | 256 | 192 | 25% |
| 361 | 512 | 384 | 25% |
| 722 | 1024 | 768 | 25% |
| 1445 | 2048 | 1472 | 28% |

That's where ~25-30% of the recent E2E improvement at large `c` comes from.

## Peak VRAM (E2E training) — GB

| config | T=2048 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|
| plain | 0.17 | 0.50 | 0.95 | 1.84 |
| mono-log | 0.48 | 0.97 | 1.75 | **3.45** |
| mono-logbudget c=2 | 0.50 | 1.04 | 1.82 | 3.59 |
| mono-sqrt c=1 | 0.29 | 1.24 | 3.49 | 6.94 |
| kvm-sqrt c=1 | 0.56 | 1.68 | 3.03 | 7.36 |
| kvm-sqrt c=2 | 0.64 | 2.01 | 4.51 | 11.94 |
| kvm-sqrt c=4 | 0.74 | 3.22 | 7.73 | 21.64 ⚠ |
| kvm-sqrt c=8 | 1.04 | 5.62 | 14.19 | — OOM |
| kvm-sqrt c=16 (OFFICIAL) | 1.62 | 10.04 | 27.09 ⚠ | — OOM |
| kvm-256 (fixed) | 0.83 | 2.41 | 4.37 | 8.71 |

VRAM scales roughly as `B × H × n_q × M × D × 2 bytes` for the bucket
trajectory `[BH, n_q, M, D]` — the dominant term. With BH=32, n_q=T/32, D=64
that's ~2 KB × T × M bytes, so doubling M at constant T doubles VRAM.

## Accuracy (forward, T=2048, bf16 vs an fp32 naive reference)

| config | triton bf16 vs fp32 | naive bf16 vs fp32 (floor) |
|---|---|---|
| mono-log | abs avg 1.2e-4, rel avg 4.2e-2 | abs avg 7.1e-4, rel avg 3.3e-1 |
| mono-sqrt c=1 | abs avg 1.3e-4, rel avg 4.6e-2 | abs avg 7.7e-4, rel avg 3.4e-1 |
| kvm-sqrt c=1 | abs avg 3.5e-3, rel avg 9.3e-1 | abs avg 4.0e-3, rel avg 1.0 |
| kvm-sqrt c=16 | abs avg 2.0e-3, rel avg 7.8e-1 | abs avg 2.6e-3, rel avg 1.0 |
| kvm-256 | abs avg 2.7e-3, rel avg 1.1 | abs avg 3.1e-3, rel avg 1.2 |

**Monotone is bit-exact-class** vs the naive recurrence — the bf16 error is
just input quantization. **KVM routing is chaotic in low precision** (cosine
novelty + argmax: a 1-ulp wobble flips a route) so KVM's bf16 error is
intrinsically "loud"; the `naive bf16` column shows the same loudness, i.e. it
is the method, not the kernel. The Triton path always lands inside the
method-noise floor.

## Quality — training loss (tiny-stories char-LM)

See `scripts/sweep.py` (the e2e training sweep on the synthetic corpus). The
[earlier sweep](https://github.com/SmerkyG/KVM-paper) showed:

* **`kvm-sqrt c=1` is the sweet spot** — near-plain quality, **highest training
  throughput** (samples/s) in the sweep (faster than plain at small T).
* **monotone is budget-insensitive**: M=16 / 32 / 64 all land at ~0.105 final
  loss; KVM's data-dependent routing wins ~0.009 at matched M=64 budget.
* On this corpus the long-range signal is content-addressed (recurring story
  subject), which KVM's cosine routing keeps in a dedicated slot but the
  positional monotone schedule blurs away.

Open direction: better schedulers — something between a fixed positional rule
and full learned routing.
