"""Precision check + speed benchmark for the FlexAttention prefill path.

Compares `MonotoneKVMAttention.forward` (the chunk recurrence) against
`.forward_flex` (dyadic pyramid + one flex_attention call) -- same weights, same
math, two execution modes. Reports max|diff| and wall-clock (min over iters, the
steady-state number) for forward and forward+backward.

Sweeps a few model shapes (batch / hidden / heads) and sequence lengths. The
first flex run per shape triggers `torch.compile` of the flex_attention op;
`--warmup` iterations absorb it before timing. Speed is ~identical for the `log`
and `sqrt` schedules (they only change the small pyramid), so the sweep uses
`log` by default.

Run:  python scripts/bench_flex.py
      python scripts/bench_flex.py --seqs 2048 8192 --schedules log sqrt
"""

import argparse
import time

import torch

from monotone_kvm import MonotoneKVMAttention, MonotoneKVMConfig

# (batch, hidden, heads)
MODEL_CONFIGS = [
    (4, 512, 8),  # baseline
    (16, 512, 8),  # larger batch
    (4, 1024, 16),  # larger dim / more heads
]


def bench(fn, iters: int, warmup: int, device) -> float:
    """Min seconds/iter after a warmup (min = steady-state, least noise)."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_one(attn, x, iters, warmup, device):
    """Precision + forward + forward/backward timing for one (model, T)."""
    with torch.no_grad():
        ref = attn.forward(x)
        flx = attn.forward_flex(x)
    diff = (ref - flx).abs().max().item()

    with torch.no_grad():
        t_rec = bench(lambda: attn.forward(x), iters, warmup, device)
        t_flx = bench(lambda: attn.forward_flex(x), iters, warmup, device)

    def rec_fb():
        attn.zero_grad(set_to_none=True)
        attn.forward(x).sum().backward()

    def flex_fb():
        attn.zero_grad(set_to_none=True)
        attn.forward_flex(x).sum().backward()

    t_rec_fb = bench(rec_fb, iters, warmup, device)
    t_flx_fb = bench(flex_fb, iters, warmup, device)
    return diff, t_rec, t_flx, t_rec_fb, t_flx_fb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--schedules", nargs="+", default=["log"])
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument("--n-bswa-chunks", type=int, default=2)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    print(
        f"device={device}  chunk_len={args.chunk_len}  "
        f"n_bswa_chunks={args.n_bswa_chunks}  warmup={args.warmup}  iters={args.iters}"
    )
    print("timings are min ms/iter; speedup = recurrent / flex\n")

    for sched in args.schedules:
        for batch, hidden, heads in MODEL_CONFIGS:
            cfg = MonotoneKVMConfig(
                hidden_size=hidden,
                num_heads=heads,
                chunk_len=args.chunk_len,
                n_bswa_chunks=args.n_bswa_chunks,
                sink_len=1,
                schedule=sched,
            )
            attn = MonotoneKVMAttention(cfg).to(device)
            print(
                f"=== schedule={sched}  batch={batch}  hidden={hidden}  heads={heads} ==="
            )
            print(
                f"{'T':>6} | {'max|diff|':>10} | {'rec fwd':>9} {'flex fwd':>9} "
                f"{'speedup':>8} | {'rec f+b':>9} {'flex f+b':>9} {'speedup':>8}"
            )

            for T in args.seqs:
                try:
                    x = torch.randn(batch, T, hidden, device=device)
                    diff, t_rec, t_flx, t_rec_fb, t_flx_fb = run_one(
                        attn, x, args.iters, args.warmup, device
                    )
                    print(
                        f"{T:>6} | {diff:>10.2e} | "
                        f"{t_rec * 1e3:>8.2f}m {t_flx * 1e3:>8.2f}m "
                        f"{t_rec / t_flx:>7.2f}x | "
                        f"{t_rec_fb * 1e3:>8.2f}m {t_flx_fb * 1e3:>8.2f}m "
                        f"{t_rec_fb / t_flx_fb:>7.2f}x"
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{T:>6} | OOM -- skipped")
                    torch.cuda.empty_cache()
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            print()


if __name__ == "__main__":
    main()
