"""Training sweep: a state-budget ladder, monotone routing vs KVM routing.

Trains a fixed `TinyLM` backbone with six attention configurations on the same
synthetic tiny-stories corpus, same seed (identical minibatches), and overlays
their loss curves. The configs form a clean state-budget ladder:

  mono-log           dyadic LogScheduler          -- the smallest, aggressive
  mono-logbudget-c2  soft-budget log, coeff 2     -- between log and sqrt
  mono-sqrt          sqrt schedule, coeff 1       -- ~sqrt(T), budget-matched
                                                    to kvm-sqrt
  kvm-sqrt           KVM routing, ~O(sqrt) budget
  kvm-256            KVM routing, ~256-slot budget
  plain              full causal attention       -- the ceiling baseline

Both monotone-KVM and KVM are **token-based**: the merge decision fires when a
token exits the BSWA window. The monotone schedule replaces *only* KVM's routing
decision with a deterministic integer rule. So `mono-sqrt` and `kvm-sqrt` carry
the *same* state budget (~sqrt(T) slots) and differ in exactly one thing -- the
routing decision -- which is the clean comparison this sweep is built around.
The `coeff` knob (PowerScheduler / LogBudgetScheduler) just scales the budget;
no chunk/token unit conversion is needed.

`TinyLM` auto-selects the Triton kernels (attn_impl="auto"), so the sweep runs
on the fast path. compile is intentionally off.

Run:  python scripts/sweep.py                       # default longer sweep
      python scripts/sweep.py --steps 6000 --seq-len 4096
"""

import argparse
from pathlib import Path

import torch

from monotone_kvm import TinyLMConfig, build_attention
from monotone_kvm.triton import _kvm_budget_plan, _monotone_plan
from train_demo import CharTokenizer, make_corpus, plot_curves, run_training

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# KVM features off -> the Triton KVM kernel path is usable (its restriction).
KVM_OFF = dict(use_vlens=False, use_merge_gate=False, use_head_temps=False)

# (label, TinyLMConfig overrides) -- backbone dims are shared, set in main().
# Ordered as a budget ladder: smallest monotone state -> sqrt -> KVM -> plain.
SWEEP = [
    ("mono-log", dict(attn="monotone", schedule="log")),
    (
        "mono-logbudget-c2",
        dict(attn="monotone", schedule="logbudget", schedule_kwargs={"coeff": 2.0}),
    ),
    (
        "mono-sqrt",
        dict(attn="monotone", schedule="sqrt", schedule_kwargs={"coeff": 1.0}),
    ),
    (
        "kvm-sqrt",
        dict(
            attn="kvm",
            state_budget_mode="power_law",
            state_growth_factor=1.0,
            state_growth_exponent=0.5,
            state_min_len=32,
            **KVM_OFF,
        ),
    ),
    (
        "kvm-256",
        dict(
            attn="kvm",
            state_budget_mode="power_law",
            state_growth_factor=1.0,
            state_growth_exponent=0.5,
            state_min_len=256,
            **KVM_OFF,
        ),
    ),
    ("plain", dict(attn="plain")),
]


def effective_M(cfg: TinyLMConfig, seq_len: int) -> int | None:
    """The padded state-slot count M this config feeds to PHASE 2 (None = plain).
    This is the effective compressed-state width -- the headline knob the sweep
    varies. plain attends over the full sequence, so it has no M."""
    attn = build_attention(cfg)
    if cfg.attn == "monotone":
        return _monotone_plan(attn, seq_len, "cpu")["M"]
    if cfg.attn == "kvm":
        return _kvm_budget_plan(attn, seq_len, "cpu")["M"]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--chunk-len", type=int, default=32)
    p.add_argument("--n-bswa-chunks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--warmup",
        type=int,
        default=300,
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
        f"sweep: {args.steps} steps x {len(SWEEP)} configs, "
        f"seq_len={args.seq_len} on {device}"
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

    # show the effective state width M each config will use at this seq_len
    cfgs = {label: TinyLMConfig(**base, **ov) for label, ov in SWEEP}
    print(
        "\neffective state width M (slots fed to PHASE 2) at "
        f"seq_len={args.seq_len}:"
    )
    for label, cfg in cfgs.items():
        M = effective_M(cfg, args.seq_len)
        print(f"  {label:18}  M = {M if M is not None else f'{args.seq_len} (full)'}")

    curves: dict[str, list[float]] = {}
    for label, cfg in cfgs.items():
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
        f"State-budget ladder -- tiny-stories char LM "
        f"({args.steps} steps, seq_len {args.seq_len}, from step {plot_skip})",
        skip=plot_skip,
    )

    print("\n=== final loss (mean of last 100 steps)  |  effective M ===")
    for label, losses in curves.items():
        tail = losses[-100:]
        M = effective_M(cfgs[label], args.seq_len)
        mstr = str(M) if M is not None else "full"
        print(f"  {label:18}  {sum(tail) / len(tail):.4f}   (M={mstr})")


if __name__ == "__main__":
    main()
