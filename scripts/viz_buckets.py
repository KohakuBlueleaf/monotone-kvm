"""Visualize the monotone bucket schedules vs KVM's budget schedules.

Both monotone-KVM and KVM are **token-based**: the merge decision is made when a
token exits the BSWA window. The monotone schedule replaces *only* KVM's routing
decision with a deterministic integer rule -- so the two are directly
comparable, slot for slot, on a token x-axis.

Saves two figures to `figures/`:

  bucket_counts.png    -- state-slot count vs context length (tokens): the
                          monotone schedules (log / sqrt / power / logbudget)
                          and KVM's budget modes (fixed / power_law /
                          saturation), against log / sqrt reference curves.
                          Both methods keep the first `bswa_len` tokens raw in
                          the sliding window -- the compressed state holds only
                          tokens that have *exited* it -- so every curve is a
                          function of `context - bswa_len` and starts there.

  bucket_structure.png -- the *interval structure* of the monotone schedules:
                          for each timestep, which dyadic bucket every past
                          token currently belongs to. KVM has no static
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
KVM_CHUNK_LEN = 64  # KVM updates its state every chunk_len tokens (its recurrence)
N_BSWA_CHUNKS = 2  # bswa window = N_BSWA_CHUNKS * KVM_CHUNK_LEN raw recent tokens
BSWA_LEN = N_BSWA_CHUNKS * KVM_CHUNK_LEN  # NO compressed state exists below this:
# the newest BSWA_LEN tokens are the raw sliding window, for BOTH methods. The
# schedule (monotone) / routing (KVM) only ever acts on tokens that have *exited*
# the window -- so at context length T the state summarizes T - BSWA_LEN tokens.


def kvm_budget_curve(mode: str, n_tokens: int, **kw) -> tuple[list[int], list[int]]:
    """KVM's state-slot count vs context length, from `_desired_state_len`. The
    compressed state holds only tokens that have EXITED the bswa window; KVM
    seeds it with the first chunk (one slot per token) and updates per chunk.
    Sampled at chunk boundaries past the window. Returns (context, slot_count)."""
    cfg = KVMConfig(
        hidden_size=64,
        num_heads=1,
        chunk_len=KVM_CHUNK_LEN,
        n_bswa_chunks=N_BSWA_CHUNKS,
        state_budget_mode=mode,
        **kw,
    )
    attn = KVMAttention(cfg)
    cur = KVM_CHUNK_LEN  # KVM seeds the state with the first chunk (1 slot/token)
    # at ctx == BSWA_LEN the window is exactly full: the state is the untouched
    # seed (KVM_CHUNK_LEN one-token slots). Anchor the curve there so it
    # originates at the bswa line instead of floating in from the first chunk
    # boundary past it.
    xs, counts = [BSWA_LEN], [cur]
    for ctx in range(BSWA_LEN + KVM_CHUNK_LEN, n_tokens + 1, KVM_CHUNK_LEN):
        avail = ctx - BSWA_LEN  # only exited tokens can be in the state
        cur = attn._desired_state_len(ctx_len=ctx, avail=avail, cur=cur)
        xs.append(ctx)
        counts.append(cur)
    return xs, counts


def plot_counts(n_tokens: int = 16384):
    # the compressed state only ever holds tokens that have EXITED the bswa
    # window -- T - BSWA_LEN of them at context length T, for BOTH methods. So
    # every curve is a function of that exited count, drawn on a context-length
    # x-axis: nothing compressed exists until context passes BSWA_LEN.
    n_exit = n_tokens - BSWA_LEN
    exit_xs = list(range(1, n_exit + 1))  # exited-token count
    ctx_xs = [m + BSWA_LEN for m in exit_xs]  # ... mapped onto context length

    monotone = {
        "monotone log": get_scheduler("log"),
        "monotone logbudget(c=2)": get_scheduler("logbudget", coeff=2.0),
        "monotone sqrt": get_scheduler("sqrt"),
        "monotone power(1/3)": get_scheduler("power", alpha=1 / 3),
    }
    plt.figure(figsize=(10, 6))
    for label, sched in monotone.items():
        counts = [len(s) for s in simulate(sched, n_exit)]
        plt.plot(ctx_xs, counts, label=label, linewidth=2)

    for mode, kw in [
        ("fixed", dict(state_min_len=256, n_max_d_chunks=1)),
        (
            "power_law",
            dict(state_growth_factor=1.0, state_growth_exponent=0.5, state_min_len=64),
        ),
        ("saturation", dict(state_saturation_n=4096, state_min_len=64)),
    ]:
        kx, counts = kvm_budget_curve(mode, n_tokens, **kw)
        plt.plot(kx, counts, "--", label=f"KVM {mode}", linewidth=1.5)

    # references: the schedule shapes, as a function of the exited-token count
    plt.plot(
        ctx_xs,
        [math.log2(m + 1) for m in exit_xs],
        ":",
        color="gray",
        label="ref: log2(exited)",
    )
    plt.plot(
        ctx_xs,
        [math.sqrt(m) for m in exit_xs],
        ":",
        color="black",
        label="ref: sqrt(exited)",
    )
    plt.axvline(
        BSWA_LEN,
        color="crimson",
        lw=1,
        ls=":",
        alpha=0.7,
        label=f"bswa window = {BSWA_LEN} (no compressed state below this)",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("context length (tokens)")
    plt.ylabel("active compressed-state slots")
    plt.title(
        "State-slot count: monotone schedules vs KVM budgets\n"
        "(both token-based; the state holds only tokens that have exited the "
        f"{BSWA_LEN}-token bswa window)"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "bucket_counts.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


def structure_grid(sched, n_tokens: int) -> np.ndarray:
    """grid[t, c] = log2(size of the bucket containing token c at timestep t)."""
    grid = np.full((n_tokens, n_tokens), np.nan)
    for t, sizes in enumerate(simulate(sched, n_tokens)):  # sizes: newest-first
        pos = sum(sizes)
        for sz in sizes:
            for c in range(pos - sz, pos):
                grid[t, c] = math.log2(sz)
            pos -= sz
    return grid


def plot_structure(n_tokens: int = 64):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name in zip(axes, ("log", "sqrt")):
        grid = structure_grid(get_scheduler(name), n_tokens)
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
        )
        ax.set_title(f"monotone '{name}' schedule")
        ax.set_xlabel("token position (oldest -> newest)")
        ax.set_ylabel("timestep t (tokens seen)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log2(bucket size in tokens)")
    fig.suptitle(
        "Monotone bucketing: which dyadic bucket each token belongs to over time\n"
        "(every bucket is a clean contiguous token interval -- KVM has no static "
        "equivalent)"
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
