"""Accuracy + speed for the Triton chunked-attention forward (PHASE 2 kernel).

Compares `forward_triton` (PHASE 1 in PyTorch + PHASE 2 Triton kernel) against
the recurrent `forward` and the FlexAttention `forward_flex`. Forward only --
backward comes after the forward design is frozen.

Run:  python scripts/bench_triton.py
      python scripts/bench_triton.py --seqs 2048 8192 --batch 8
"""

import argparse
import time

import torch

from monotone_kvm import (
    KVMAttention,
    KVMConfig,
    MonotoneKVMAttention,
    MonotoneKVMConfig,
)


def report(name, got, ref):
    d = (got - ref).abs()
    rel = d / ref.abs().clamp_min(1e-6)
    print(
        f"  {name:26}  abs: max={d.max().item():.2e} avg={d.mean().item():.2e}  "
        f"|  rel: max={rel.max().item():.2e} avg={rel.mean().item():.2e}"
    )


def bench(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def make(kind, hidden, heads, cl, nb, dev):
    if kind.startswith("monotone"):
        sched = kind.split("-")[1]
        return MonotoneKVMAttention(
            MonotoneKVMConfig(
                hidden_size=hidden,
                num_heads=heads,
                chunk_len=cl,
                n_bswa_chunks=nb,
                sink_len=1,
                schedule=sched,
            )
        ).to(dev)
    mode = kind.split("-")[1]
    kw = (
        dict(state_budget_mode="fixed", state_min_len=64)
        if mode == "fixed"
        else dict(
            state_budget_mode="power_law",
            state_growth_factor=2.0,
            state_growth_exponent=0.5,
            state_min_len=32,
        )
    )
    return KVMAttention(
        KVMConfig(
            hidden_size=hidden,
            num_heads=heads,
            chunk_len=cl,
            n_bswa_chunks=nb,
            sink_len=1,
            **kw,
        )
    ).to(dev)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument("--n-bswa-chunks", type=int, default=2)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    kinds = ["monotone-log", "monotone-sqrt", "kvm-fixed", "kvm-power_law"]

    # ---------------- accuracy ----------------
    print("=== accuracy: forward_triton vs recurrent forward ===")
    for dtype in (torch.float32, torch.float16):
        print(f"-- dtype={dtype} --")
        for kind in kinds:
            attn = make(
                kind, args.hidden, args.heads, args.chunk_len, args.n_bswa_chunks, dev
            ).to(dtype)
            x = torch.randn(2, 1024, args.hidden, device=dev, dtype=dtype)
            with torch.no_grad():
                report(kind, attn.forward_triton(x), attn.forward(x))

    # ---------------- speed (forward only, fp16) ----------------
    print(
        f"\n=== forward speed (fp16, batch={args.batch}, "
        f"hidden={args.hidden}, heads={args.heads}) ==="
    )
    print(
        f"{'kind':16} {'T':>6} | {'recurrent':>10} {'flex':>10} {'triton':>10} "
        f"| {'tri/rec':>8} {'tri/flex':>8}"
    )
    for kind in kinds:
        attn = make(
            kind, args.hidden, args.heads, args.chunk_len, args.n_bswa_chunks, dev
        ).to(torch.float16)
        has_flex = hasattr(attn, "forward_flex")  # only MonotoneKVMAttention
        for T in args.seqs:
            x = torch.randn(args.batch, T, args.hidden, device=dev, dtype=torch.float16)
            with torch.no_grad():
                t_rec = bench(lambda: attn.forward(x), args.iters, args.warmup, dev)
                t_flx = (
                    bench(lambda: attn.forward_flex(x), args.iters, args.warmup, dev)
                    if has_flex
                    else float("nan")
                )
                t_tri = bench(
                    lambda: attn.forward_triton(x), args.iters, args.warmup, dev
                )
            flx_s = f"{t_flx:>9.2f}m" if has_flex else f"{'n/a':>10}"
            ratio_flx = f"{t_flx / t_tri:>7.2f}x" if has_flex else f"{'n/a':>8}"
            print(
                f"{kind:16} {T:>6} | {t_rec:>9.2f}m {flx_s} {t_tri:>9.2f}m "
                f"| {t_rec / t_tri:>7.2f}x {ratio_flx}"
            )
        print()


if __name__ == "__main__":
    main()
