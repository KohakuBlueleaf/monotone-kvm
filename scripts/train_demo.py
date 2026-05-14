"""Tiny end-to-end training demo: a small LM on synthetic "tiny stories".

Trains a `TinyLM` (plain / KVM / monotone attention) at the character level on a
procedurally-generated templated-story corpus -- zero data dependencies, runs on
CPU or GPU. The point is to show the attention layers train end-to-end and the
loss actually drops; with `--attn both` it trains KVM and monotone side by side
and saves a loss-curve comparison to `figures/`.

The reusable pieces (`make_corpus`, `CharTokenizer`, `run_training`) are imported
by `scripts/sweep.py`.

Examples:
  python scripts/train_demo.py                          # both, default small model
  python scripts/train_demo.py --attn monotone --schedule sqrt
  python scripts/train_demo.py --attn plain --steps 600
  python scripts/train_demo.py --hidden 512 --layers 8  # ~few dozen M (wants a GPU)
  python scripts/train_demo.py --data path/to/text.txt  # train on your own text
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch

from monotone_kvm import TinyLM, TinyLMConfig

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# --------------------------------------------------------------------------
# synthetic "tiny stories" corpus -- a small templated grammar
# --------------------------------------------------------------------------
_SUBJECTS = [
    "the cat",
    "a dog",
    "the little girl",
    "a small boy",
    "the old man",
    "a tiny bird",
    "the brown mouse",
    "a white rabbit",
    "the big bear",
    "a red fox",
    "the young prince",
    "a kind queen",
]
_ADJ = [
    "happy",
    "sad",
    "tired",
    "hungry",
    "curious",
    "brave",
    "sleepy",
    "kind",
    "lonely",
    "cheerful",
    "scared",
    "proud",
]
_VERBS = [
    "ran",
    "jumped",
    "slept",
    "ate the apple",
    "played",
    "sang a song",
    "walked slowly",
    "looked around",
    "smiled",
    "danced",
    "found a key",
    "told a story",
]
_PLACES = [
    "in the park",
    "by the river",
    "at home",
    "under a tall tree",
    "on the green hill",
    "near the blue lake",
    "in the garden",
    "by the wide sea",
    "inside the warm house",
    "across the bridge",
]


def make_corpus(n_stories: int = 6000, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_stories):
        s1, s2 = rng.sample(_SUBJECTS, 2)
        story = (
            f"once upon a time, {s1} was very {rng.choice(_ADJ)}. "
            f"{s1} {rng.choice(_VERBS)} {rng.choice(_PLACES)}. "
            f"then {s2} {rng.choice(_VERBS)} {rng.choice(_PLACES)}. "
            f"{s1} and {s2} were {rng.choice(_ADJ)}. the end.\n"
        )
        out.append(story)
    return "".join(out)


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device, rng):
    ix = [rng.randint(0, len(data) - seq_len - 1) for _ in range(batch_size)]
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).to(device)
    return x, y


def run_training(
    cfg: TinyLMConfig,
    data: torch.Tensor,
    device,
    *,
    label: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int = 0,
    log_every: int = 100,
    warmup: int | None = None,
    compile: bool = False,
    tok: "CharTokenizer | None" = None,
) -> list[float]:
    """Train one `TinyLM` config and return the per-step loss list.

    Shared by `train_demo.py` and `sweep.py`. The same `seed` makes every run
    see the same minibatches, so loss curves are directly comparable. The LR
    follows a linear warmup + cosine decay to zero -- without it a high constant
    LR makes the loss bounce around its basin (visible bumps + early plateau).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    raw_model = TinyLM(cfg).to(device)
    opt = torch.optim.AdamW(
        raw_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    model = torch.compile(raw_model) if compile else raw_model

    warmup = warmup if warmup is not None else max(steps // 20, 10)

    def _lr_lambda(e: int) -> float:
        step = e + 1
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine: 1 -> 0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    print(
        f"\n=== {label}  ({raw_model.num_params() / 1e6:.2f}M params, "
        f"device={device}" + (", compiled" if compile else "") + ") ==="
    )

    losses: list[float] = []
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(data, batch_size, seq_len, device, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            toks = step * batch_size * seq_len
            print(
                f"  step {step:5d}/{steps}  loss {loss.item():.4f}  "
                f"lr {opt.param_groups[0]['lr']:.1e}  "
                f"({toks / (time.time() - t0):.0f} tok/s)"
            )
        sched.step()

    if tok is not None:
        prompt = "once upon a time, "
        idx = torch.tensor([tok.encode(prompt)], device=device)
        out = raw_model.generate(idx, max_new_tokens=160, temperature=0.8, top_k=20)
        print(f"  sample: {tok.decode(out[0].tolist())!r}")
    return losses


def _cfg_from_args(attn: str, args, vocab_size: int) -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        attn=attn,
        chunk_len=args.chunk_len,
        n_bswa_chunks=args.n_bswa_chunks,
        sink_len=1,
        schedule=args.schedule,
        schedule_kwargs={"alpha": args.alpha} if args.schedule == "power" else {},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--attn", choices=["plain", "kvm", "monotone", "both"], default="both"
    )
    p.add_argument(
        "--schedule",
        default="log",
        help="monotone schedule: log/logbudget/sqrt/power/linear",
    )
    p.add_argument("--alpha", type=float, default=1 / 3, help="power-schedule exponent")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--chunk-len", type=int, default=32)
    p.add_argument("--n-bswa-chunks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="LR warmup steps (default: 5%% of --steps)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--compile", action="store_true", help="wrap the model in torch.compile"
    )
    p.add_argument(
        "--data", default=None, help="optional text file to train on instead"
    )
    args = p.parse_args()

    device = torch.device(args.device)
    if args.data:
        text = Path(args.data).read_text(encoding="utf-8")
        print(f"loaded {len(text)} chars from {args.data}")
    else:
        text = make_corpus()
        print(f"generated synthetic tiny-stories corpus: {len(text)} chars")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"vocab size = {tok.vocab_size}, tokens = {len(data)}")

    attns = ["kvm", "monotone"] if args.attn == "both" else [args.attn]
    curves = {}
    for attn in attns:
        label = attn + (f" ({args.schedule})" if attn == "monotone" else "")
        curves[label] = run_training(
            _cfg_from_args(attn, args, tok.vocab_size),
            data,
            device,
            label=label,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_every=args.log_every,
            warmup=args.warmup,
            compile=args.compile,
            tok=tok,
        )

    if len(curves) > 1:
        plot_curves(
            curves,
            FIG_DIR / "train_loss.png",
            "KVM vs monotone-KVM -- tiny-stories char LM",
        )


def plot_curves(curves: dict, out_path: Path, title: str, skip: int = 0):
    """Plot smoothed loss curves for a set of named runs.

    `skip` drops the first N steps from the *plot* (not the smoothing window),
    so early transients don't compress the interesting late-training range.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(xs, w=25):
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo : i + 1]) / (i - lo + 1))
        return out

    out_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for name, losses in curves.items():
        sm = smooth(losses)
        start = min(skip, len(sm) - 1)
        plt.plot(range(start, len(sm)), sm[start:], label=name, linewidth=1.8)
    plt.xlabel("step")
    plt.ylabel("train loss (smoothed)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"\nsaved loss curve to {out_path}")


if __name__ == "__main__":
    main()
