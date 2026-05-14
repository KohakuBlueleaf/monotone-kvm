"""Training sweep: plain vs KVM routing vs monotone routing.

Trains a fixed `TinyLM` backbone with five attention configurations on the same
synthetic tiny-stories corpus, with the same seed (so every run sees identical
minibatches), and overlays their loss curves.

  plain          full causal attention            -- the ceiling baseline
  kvm fixed      KVM routing, fixed O(1) budget
  kvm power_law  KVM routing, ~O(sqrt) budget
  monotone log   monotone routing, O(log t) buckets
  monotone sqrt  monotone routing, O(sqrt t) buckets

This isolates two axes: routing method (data-dependent KVM vs data-independent
monotone) and state budget. compile is intentionally off.

Run:  python scripts/sweep.py                 # 1000 steps each, CUDA if available
      python scripts/sweep.py --steps 2000 --seq-len 1024
"""

import argparse
from pathlib import Path

import torch

from monotone_kvm import TinyLMConfig
from train_demo import CharTokenizer, make_corpus, plot_curves, run_training

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# (label, TinyLMConfig overrides) -- backbone dims are shared, set in main()
SWEEP = [
    ("plain", dict(attn="plain")),
    ("kvm fixed", dict(attn="kvm", state_budget_mode="fixed", state_min_len=64)),
    (
        "kvm power_law",
        dict(
            attn="kvm",
            state_budget_mode="power_law",
            state_growth_factor=2.0,
            state_growth_exponent=0.5,
            state_min_len=32,
        ),
    ),
    ("monotone log", dict(attn="monotone", schedule="log")),
    ("monotone sqrt", dict(attn="monotone", schedule="sqrt")),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--chunk-len", type=int, default=32)
    p.add_argument("--n-bswa-chunks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="LR warmup steps (cosine decay to zero afterwards)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--plot-skip",
        type=int,
        default=None,
        help="drop the first N steps from the plot (default: = warmup)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    plot_skip = args.warmup if args.plot_skip is None else args.plot_skip

    device = torch.device(args.device)
    text = make_corpus()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(
        f"corpus: {len(text)} chars, vocab {tok.vocab_size}  |  "
        f"sweep: {args.steps} steps x {len(SWEEP)} configs on {device}"
    )

    base = dict(
        vocab_size=tok.vocab_size,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        chunk_len=args.chunk_len,
        n_bswa_chunks=args.n_bswa_chunks,
        sink_len=1,
    )

    curves: dict[str, list[float]] = {}
    for label, overrides in SWEEP:
        cfg = TinyLMConfig(**base, **overrides)
        curves[label] = run_training(
            cfg,
            data,
            device,
            label=label,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_every=max(1, args.steps // 5),
            warmup=args.warmup,
        )

    plot_curves(
        curves,
        FIG_DIR / "sweep_loss.png",
        f"Routing & budget sweep -- tiny-stories char LM "
        f"({args.steps} steps, from step {plot_skip})",
        skip=plot_skip,
    )

    print("\n=== final loss (mean of last 50 steps) ===")
    for label, losses in curves.items():
        tail = losses[-50:]
        print(f"  {label:16s}  {sum(tail) / len(tail):.4f}")


if __name__ == "__main__":
    main()
