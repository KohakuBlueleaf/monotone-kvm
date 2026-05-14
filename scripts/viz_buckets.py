"""Visualize our monotone bucketing vs KVM's budget schedules.

Saves two figures to `figures/`:

  bucket_counts.png    -- state-slot count vs context length: the monotone
                          schedules (log / sqrt / power / fixed / linear) and
                          KVM's budget modes (fixed / power_law / saturation),
                          against log / sqrt / linear reference curves.

  bucket_structure.png -- the *interval structure* of the monotone schedules:
                          for each timestep, which dyadic bucket every past
                          chunk currently belongs to. KVM has no static
                          equivalent -- its slots are data-dependent centroids.

Run:  python scripts/viz_buckets.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monotone_kvm import KVMAttention, KVMConfig, get_scheduler, simulate

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
CHUNK_LEN = 64  # 1 chunk == this many tokens, for the x-axis in tokens


def kvm_budget_curve(mode: str, n_chunks: int, **kw) -> list[int]:
    """KVM's state-slot count vs chunk index, straight from `_desired_state_len`."""
    cfg = KVMConfig(
        hidden_size=64, num_heads=1, chunk_len=CHUNK_LEN, state_budget_mode=mode, **kw
    )
    attn = KVMAttention(cfg)
    counts, cur = [], CHUNK_LEN  # KVM seeds state with the first full chunk
    for t in range(1, n_chunks + 1):
        ctx = t * CHUNK_LEN
        cur = attn._desired_state_len(ctx_len=ctx, avail=ctx, cur=cur)
        counts.append(cur)
    return counts


def plot_counts(n_chunks: int = 512):
    xs = [t * CHUNK_LEN for t in range(1, n_chunks + 1)]  # context length in tokens

    monotone = {
        "monotone log": get_scheduler("log"),
        "monotone sqrt": get_scheduler("sqrt"),
        "monotone power(1/3)": get_scheduler("power", alpha=1 / 3),
        "monotone fixed(k=16)": get_scheduler("fixed", k=16),
    }
    plt.figure(figsize=(10, 6))
    for label, sched in monotone.items():
        counts = [len(s) for s in simulate(sched, n_chunks)]
        plt.plot(xs, counts, label=label, linewidth=2)

    for mode, kw in [
        ("fixed", dict(state_min_len=256, n_max_d_chunks=1)),
        (
            "power_law",
            dict(state_growth_factor=4.0, state_growth_exponent=0.5, state_min_len=64),
        ),
        ("saturation", dict(state_saturation_n=4096, state_min_len=64)),
    ]:
        counts = kvm_budget_curve(mode, n_chunks, **kw)
        plt.plot(xs, counts, "--", label=f"KVM {mode}", linewidth=1.5)

    plt.plot(
        xs,
        [math.log2(t + 1) for t in range(1, n_chunks + 1)],
        ":",
        color="gray",
        label="ref: log2(chunks)",
    )
    plt.plot(
        xs,
        [math.sqrt(t) for t in range(1, n_chunks + 1)],
        ":",
        color="black",
        label="ref: sqrt(chunks)",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"context length (tokens, chunk_len={CHUNK_LEN})")
    plt.ylabel("active state slots")
    plt.title(
        "State-slot count: monotone schedules vs KVM budgets\n"
        "(note: a monotone slot is a whole bucket; a KVM slot is ~one token)"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "bucket_counts.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


def structure_grid(sched, n_chunks: int) -> np.ndarray:
    """grid[t, c] = log2(size of the bucket containing chunk c at timestep t)."""
    grid = np.full((n_chunks, n_chunks), np.nan)
    for t, sizes in enumerate(simulate(sched, n_chunks)):  # sizes: newest-first
        pos = sum(sizes)
        for sz in sizes:
            for c in range(pos - sz, pos):
                grid[t, c] = math.log2(sz)
            pos -= sz
    return grid


def plot_structure(n_chunks: int = 64):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name in zip(axes, ("log", "sqrt")):
        grid = structure_grid(get_scheduler(name), n_chunks)
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
        )
        ax.set_title(f"monotone '{name}' schedule")
        ax.set_xlabel("chunk position (oldest -> newest)")
        ax.set_ylabel("timestep t")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log2(bucket size)")
    fig.suptitle(
        "Monotone bucketing: which dyadic bucket each chunk belongs to over time\n"
        "(every bucket is a clean contiguous interval -- KVM has no static equivalent)"
    )
    fig.tight_layout()
    out = FIG_DIR / "bucket_structure.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    plot_counts()
    plot_structure()


if __name__ == "__main__":
    main()
