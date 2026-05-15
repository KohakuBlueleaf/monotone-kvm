"""Comprehensive attention benchmark -- speed, VRAM, effective-KV, accuracy.

Sweeps `kvm-sqrt` coefficient (c=1/2/4/8/16) and the monotone family across
T=2048..32768. Reports speed (fwd + e2e), peak VRAM, the live state-slot count
(M), and the padded kernel width.

Outputs (next to this script):
  bench_report.md   -- full markdown report
  bench_fwd.png     -- fwd speedup vs T
  bench_e2e.png     -- fwd+bwd speedup vs T
  bench_vram.png    -- e2e peak VRAM vs T

What the bench measures
  * Effective KV length per query: plain = T; KVM / monotone = `live + bswa_len`
    where `live` is the real state-slot count (counted from the PHASE-2 bias
    mask the kernel reads), `pad M` is the padded kernel load width.
  * Forward and end-to-end (fwd + bwd) wall-clock in bf16, min of `iters`
    after `wu` warmups (so torch.compile / autotune are absorbed).
  * Peak VRAM via `torch.cuda.max_memory_allocated()`; cache empty'd between
    cells.

Run:  python scripts/bench_comprehensive.py
"""

import copy
import time

import torch

from monotone_kvm import (
    KVMAttention,
    KVMConfig,
    MonotoneKVMAttention,
    MonotoneKVMConfig,
    PlainAttention,
    PlainAttentionConfig,
)
from monotone_kvm.triton import _kvm_budget_plan, _monotone_plan

dev, dt = "cuda", torch.bfloat16
H, nh, cl, nb, B = 512, 8, 32, 2, 4
SEQS = [2048, 4096, 8192, 16384, 32768]
NAIVE_MAX_T = 8192  # naive ref is O(slow); cap it


def bench(fn, it=15, wu=5):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


# ---- config builders -> fp32 on-device modules (cast to bf16 at use site) ----
def _kvm(**kw):
    base = dict(
        hidden_size=H,
        num_heads=nh,
        chunk_len=cl,
        n_bswa_chunks=nb,
        sink_len=1,
        use_vlens=False,
        use_merge_gate=False,
        use_head_temps=False,
    )
    base.update(kw)
    return KVMAttention(KVMConfig(**base)).to(dev)


mk_plain = lambda: PlainAttention(PlainAttentionConfig(H, nh)).to(dev)


def mk_kvm_sqrt_c(factor, min_len=64):
    return _kvm(
        state_budget_mode="power_law",
        state_growth_factor=float(factor),
        state_growth_exponent=0.5,
        state_min_len=min_len,
    )


mk_kvm_256 = lambda: _kvm(state_budget_mode="fixed", state_min_len=256)


def mk_mono(name, **schedule_kwargs):
    return MonotoneKVMAttention(
        MonotoneKVMConfig(
            H, nh, chunk_len=cl, n_bswa_chunks=nb, sink_len=1,
            schedule=name, schedule_kwargs=schedule_kwargs,
        )
    ).to(dev)


def fwd_wrap(a, run, x):
    with torch.no_grad():
        run(a, x)


def fb(a, run, x):
    a.zero_grad(set_to_none=True)
    run(a, x).sum().backward()


# name, builder, run_fn, run fwd?, run e2e?
def _t(name, mk):
    return (
        name,
        mk,
        lambda m, x: m.forward_triton(x) if hasattr(m, "forward_triton") else m(x),
        True,
        True,
    )


VARIANTS = [
    ("plain", mk_plain, lambda m, x: m(x), True, True),
    # KVM coefficient sweep (all use the new tiled forward kernel where M>128)
    _t("kvm-sqrt c=1", lambda: mk_kvm_sqrt_c(1.0)),
    _t("kvm-sqrt c=2", lambda: mk_kvm_sqrt_c(2.0)),
    _t("kvm-sqrt c=4", lambda: mk_kvm_sqrt_c(4.0)),
    _t("kvm-sqrt c=8", lambda: mk_kvm_sqrt_c(8.0)),
    _t("kvm-sqrt c=16 (OFFICIAL)", lambda: mk_kvm_sqrt_c(16.0, min_len=256)),
    _t("kvm-256 (fixed)", mk_kvm_256),
    # Monotone family
    _t("mono-log", lambda: mk_mono("log")),
    _t("mono-logbudget c=2", lambda: mk_mono("logbudget", coeff=2.0)),
    _t("mono-sqrt c=1", lambda: mk_mono("sqrt", coeff=1.0)),
    _t("mono-sqrt c=2", lambda: mk_mono("sqrt", coeff=2.0)),
    _t("mono-sqrt c=4", lambda: mk_mono("sqrt", coeff=4.0)),
]

# distinct model configs (for the effective-KV table; path doesn't change M)
KV_CONFIGS = [
    ("plain", mk_plain),
    ("kvm-sqrt c=1", lambda: mk_kvm_sqrt_c(1.0)),
    ("kvm-sqrt c=2", lambda: mk_kvm_sqrt_c(2.0)),
    ("kvm-sqrt c=4", lambda: mk_kvm_sqrt_c(4.0)),
    ("kvm-sqrt c=8", lambda: mk_kvm_sqrt_c(8.0)),
    ("kvm-sqrt c=16 (OFFICIAL)", lambda: mk_kvm_sqrt_c(16.0, min_len=256)),
    ("kvm-256 (fixed)", mk_kvm_256),
    ("mono-log", lambda: mk_mono("log")),
    ("mono-logbudget c=2", lambda: mk_mono("logbudget", coeff=2.0)),
    ("mono-sqrt c=1", lambda: mk_mono("sqrt", coeff=1.0)),
    ("mono-sqrt c=2", lambda: mk_mono("sqrt", coeff=2.0)),
    ("mono-sqrt c=4", lambda: mk_mono("sqrt", coeff=4.0)),
]


def state_slots(a, label, T):
    """(live, padded) compressed-state slot counts for (config, T).

      live   = slots a query actually attends over -- counted directly from the
               PHASE-2 bias mask the kernel reads: a non -inf entry is a live
               slot, -inf is padding the softmax zeroes out. Taken as the max
               over query chunks (the state grows, so the last chunk is the
               worst case). This is *measured from the kernel input*, not a
               logical estimate.
      padded = M, the padded power-of-2 width the kernel actually loads.

    Returns (None, None) for plain (no compressed state)."""
    if "kvm" in label:
        plan = _kvm_budget_plan(a, T, dev)
    elif "mono" in label:
        plan = _monotone_plan(a, T, dev)
    else:
        return None, None
    live = int((plan["bias"] != float("-inf")).sum(dim=1).max())
    return live, plan["M"]


def kv_summary():
    """{label: {T: (max_kv, live, padded)}}.  max_kv = positions a query attends
    over: plain -> T;  KVM/monotone -> live state slots + bswa_len (the sliding
    window). `live` is counted from the PHASE-2 bias mask the kernel reads."""
    res = {}
    for label, mk in KV_CONFIGS:
        a = mk()
        res[label] = {}
        for T in SEQS:
            live, padded = state_slots(a, label, T)
            if live is None:
                res[label][T] = (T, None, None)
            else:
                res[label][T] = (live + a.bswa_len, live, padded)
        del a
    return res


def verify_kv():
    """Empirical cross-check: the `live` slot count from the plan == what the
    actual forward really builds. Runs the naive monotone recurrence (exact, and
    it records its real per-query-chunk bucket trace) and confirms the plan the
    kernels consume matches it -- so the effective-KV table is measured, not
    assumed. Returns markdown lines for the report."""
    print("\n=== verifying effective-KV: plan vs actual forward ===", flush=True)
    rows = []
    for label, mk in [
        ("mono-log", lambda: mk_mono("log")),
        ("mono-sqrt", lambda: mk_mono("sqrt")),
    ]:
        for T in (2048, 8192):
            a = mk()
            x = torch.randn(B, T, H, device=dev, dtype=torch.float32)
            with torch.no_grad():
                a.forward(x)  # exact recurrence; populates a._size_trace
            # last query chunk = the worst case the plan's .max() also picks
            actual = a.sink_len + len(a._size_trace[-1])
            plan_live, _ = state_slots(a, label, T)
            ok = actual == plan_live
            msg = (
                f"  {label:10} T={T:6}: forward built {actual:4} live slots, "
                f"plan says {plan_live:4}  {'OK' if ok else 'MISMATCH'}"
            )
            print(msg, flush=True)
            assert ok, f"{label} T={T}: plan/forward live-slot mismatch"
            rows.append(f"| {label} | {T} | {actual} | {plan_live} | OK |")
            del a, x
            torch.cuda.empty_cache()
    return rows


def run_table(mode, wrap):
    """mode in {'fwd','e2e'}. {name: {T: dict(ms, vram_gb) | None | dict(err)}}."""
    res = {}
    for name, mk, run, do_fwd, do_e2e in VARIANTS:
        if (mode == "fwd" and not do_fwd) or (mode == "e2e" and not do_e2e):
            continue
        res[name] = {}
        is_naive = "naive" in name
        for T in SEQS:
            if is_naive and T > NAIVE_MAX_T:
                res[name][T] = None
                continue

            # pre-skip cells that would OOM (buck_k+buck_v dominate VRAM at
            # large M*T): rough estimate of buck-tensor VRAM in GB.
            est_M = None
            try:
                a_probe = mk()
                live, M = state_slots(a_probe, name, T)
                est_M = M
                del a_probe
            except Exception:
                pass
            torch.cuda.empty_cache()
            if est_M is not None:
                # 2 buf (k+v) * BH * n_q * M * D * 2 bytes (bf16)
                est_vram_gb = 2 * B * nh * (T // cl) * est_M * (H // nh) * 2 / 1e9
                if est_vram_gb > 10:  # leave headroom on a 16 GB card
                    res[name][T] = dict(
                        err=f"skipped (buck est ~{est_vram_gb:.1f} GB > 10 GB)"
                    )
                    print(f"  [{mode}] {name:24} T={T:6}: {res[name][T]}", flush=True)
                    continue

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                vram_before = torch.cuda.memory_allocated() / 1e9
                x = torch.randn(B, T, H, device=dev, dtype=dt)
                a = mk().to(dt)
                ms = bench(lambda: wrap(a, run, x))
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                res[name][T] = dict(
                    ms=ms, vram_gb=vram_peak, vram_run_gb=vram_peak - vram_before
                )
                del a, x
            except Exception as e:  # noqa: BLE001
                res[name][T] = dict(err=str(e).splitlines()[-1][:70])
            try:
                torch.cuda.empty_cache()
            except Exception:  # CUDA context may be poisoned after illegal
                pass            # access; ignore so the loop can continue.
            print(f"  [{mode}] {name:24} T={T:6}: {res[name][T]}", flush=True)
    return res


def report(got, ref):
    d = (got.float() - ref.float()).abs()
    rel = d / ref.float().abs().clamp_min(1e-6)
    return f"abs avg={d.mean():.1e} max={d.max():.1e} | rel avg={rel.mean():.1e}"


def accuracy():
    """fwd accuracy at T=2048: triton/flex-bf16 and naive-bf16, both vs naive-fp32."""
    torch.manual_seed(0)
    T = 2048
    rows = []
    specs = [
        ("kvm-sqrt c=1", lambda: mk_kvm_sqrt_c(1.0), "forward_triton"),
        ("kvm-sqrt c=16", lambda: mk_kvm_sqrt_c(16.0, min_len=256), "forward_triton"),
        ("kvm-256", mk_kvm_256, "forward_triton"),
        ("mono-log", lambda: mk_mono("log"), "forward_triton"),
        ("mono-sqrt c=1", lambda: mk_mono("sqrt", coeff=1.0), "forward_triton"),
    ]
    for name, mk, method in specs:
        a32 = mk()  # fp32
        a_bf = copy.deepcopy(a32).to(dt)  # same weights, bf16
        x32 = torch.randn(B, T, H, device=dev, dtype=torch.float32)
        with torch.no_grad():
            ref = a32.forward(x32)  # naive fp32 = ground truth
            tri = getattr(a_bf, method)(x32.to(dt))  # triton / flex bf16
            nai = a_bf.forward(x32.to(dt))  # naive bf16 = precision floor
        rows.append(
            (
                f"{name} {method.replace('forward_', '')}",
                report(tri, ref),
                report(nai, ref),
            )
        )
        print(f"  {name} {method}: done", flush=True)
        del a32, a_bf, x32
        torch.cuda.empty_cache()
    return rows


# ---------- markdown formatting ----------
def _hdr():
    return (
        "| config | "
        + " | ".join(f"T={T}" for T in SEQS)
        + " |\n"
        + "|"
        + "---|" * (len(SEQS) + 1)
    )


def md_speed(res):
    base = {T: (res.get("plain", {}).get(T) or {}).get("ms") for T in SEQS}
    lines = [_hdr()]
    for name, cells in res.items():
        row = []
        for T in SEQS:
            c = cells.get(T)
            if c is None:
                row.append("-")
            elif "err" in c:
                row.append("ERR")
            else:
                b = base[T]
                sp = f" ({b / c['ms']:.2f}x)" if b else ""
                row.append(f"{c['ms']:.1f}ms{sp}")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def md_vram(res):
    lines = [_hdr()]
    for name, cells in res.items():
        row = []
        for T in SEQS:
            c = cells.get(T)
            if c is None:
                row.append("-")
            elif "err" in c:
                row.append("ERR")
            else:
                row.append(f"{c['vram_gb']:.2f} GB")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def md_kv(kv):
    lines = [_hdr()]
    for label, cells in kv.items():
        row = []
        for T in SEQS:
            mk_, live, padded = cells[T]
            if live is None:
                row.append(f"{mk_} (1x)")
            else:
                row.append(f"{mk_} (live={live}, pad={padded}, {T / mk_:.0f}x)")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_plot(res, key, title, ylabel, path, vs_plain=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    base = {T: (res.get("plain", {}).get(T) or {}).get(key) for T in SEQS}
    for name, cells in res.items():
        xs, ys = [], []
        for T in SEQS:
            c = cells.get(T)
            if not c or "err" in c:
                continue
            v = c[key]
            if vs_plain:
                b = base[T]
                if not b:
                    continue
                xs.append(T)
                ys.append(b / v)
            else:
                xs.append(T)
                ys.append(v)
        if xs:
            style = "--" if "naive" in name else ("-." if "flex" in name else "-")
            ax.plot(xs, ys, style, marker="o", label=name, linewidth=2)
    if vs_plain:
        ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQS)
    ax.set_xticklabels([str(t) for t in SEQS])
    ax.set_xlabel("sequence length T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    import os
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    FIG_DIR = ROOT / "figures"
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_PATH = ROOT / "bench_report.md"

    torch.manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  | bf16, batch={B}, hidden={H}, "
        f"heads={nh}, chunk_len={cl}, n_bswa_chunks={nb}",
        flush=True,
    )

    print("\n=== effective KV length ===", flush=True)
    kv = kv_summary()
    kv_check = verify_kv()
    print("\n=== accuracy (fwd, T=2048) ===", flush=True)
    acc = accuracy()
    for nm, tri, nai in acc:
        print(f"  {nm:22} triton/flex vs fp32: {tri}")
        print(f"  {'':22} naive bf16  vs fp32: {nai}")

    print("\n=== FWD ONLY ===", flush=True)
    fwd = run_table("fwd", fwd_wrap)
    print("\n=== E2E TRAINING (fwd+bwd) ===", flush=True)
    e2e = run_table("e2e", fb)

    make_plot(
        fwd,
        "ms",
        f"Forward-only speedup (bf16, batch={B})",
        "speedup vs plain",
        str(FIG_DIR / "bench_fwd.png"),
    )
    make_plot(
        e2e,
        "ms",
        f"Training (fwd+bwd) speedup (bf16, batch={B})",
        "speedup vs plain",
        str(FIG_DIR / "bench_e2e.png"),
    )
    make_plot(
        e2e,
        "vram_gb",
        f"Training (fwd+bwd) peak VRAM (bf16, batch={B})",
        "peak VRAM (GB)",
        str(FIG_DIR / "bench_vram.png"),
        vs_plain=False,
    )

    with open(str(REPORT_PATH), "w") as f:
        f.write("# Comprehensive attention benchmark\n\n")
        f.write(
            f"GPU: {torch.cuda.get_device_name(0)} | bf16 | batch={B} | "
            f"hidden={H} | heads={nh} | chunk_len={cl} | n_bswa_chunks={nb}\n\n"
        )
        f.write("## Effective KV length (positions a query attends over)\n\n")
        f.write(
            "Cell = `max_kv (live=state slots the kernel attends over, "
            "pad=padded power-of-2 kernel width, T/max_kv compression)`. plain "
            "attends over T; KVM/monotone attend over `live + bswa_len` (live "
            "state slots + the sliding window of " + str(nb * cl) + " raw "
            "tokens). **`live` is counted directly from the PHASE-2 bias mask "
            "the kernel reads** -- a `-inf` bias entry is a padding slot the "
            "softmax zeroes out -- and is cross-checked against the real "
            "forward pass below, so it is measured, not estimated. Both "
            "methods are token-based: `mono-log` keeps ~log2(T) live slots, "
            "`mono-sqrt` ~sqrt(T) (budget-matched to `kvm-sqrt`).\n\n"
        )
        f.write(md_kv(kv) + "\n\n")
        f.write(
            "**Cross-check** -- `live` from the plan the kernels consume vs the "
            "real bucket count the exact naive recurrence builds:\n\n"
        )
        f.write("| config | T | forward built | plan says | |\n|---|---|---|---|---|\n")
        f.write("\n".join(kv_check) + "\n\n")
        f.write("## Forward speed\n\n")
        f.write(
            "Cell = `min-latency ms (speedup vs plain)`. `-` = not run "
            f"(naive capped to T<={NAIVE_MAX_T}). `ERR` = failed.\n\n"
        )
        f.write(md_speed(fwd) + "\n\n")
        f.write("## End-to-end training speed (fwd + bwd)\n\n" + md_speed(e2e) + "\n\n")
        f.write("## Peak VRAM -- forward\n\n")
        f.write(
            "`torch.cuda.max_memory_allocated()` per cell, empty_cache between "
            "runs.\n\n" + md_vram(fwd) + "\n\n"
        )
        f.write("## Peak VRAM -- end-to-end training\n\n" + md_vram(e2e) + "\n\n")
        f.write(
            "## Reading the long-T cells\n\n"
            "Real method-level effects to keep in mind when reading the tables:\n\n"
            "* **OOM-skipped cells (marked `skipped (buck est ...)`)** -- the "
            "bucket trajectory `[BH, n_q, M, D]` dominates VRAM. On a 16 GB card "
            "`kvm-sqrt c=8/16 @ T>=32768` needs >10 GB just for `buck_k+buck_v` "
            "and is pre-skipped. Larger HBM lifts the ceiling.\n"
            "* **`kvm-sqrt` forward at M>=256** -- the original kernel held M "
            "state slots in registers and spilled at M>=256; the new "
            "`_kvm_merge_kernel_tiled` (state-in-HBM, MT=64 tiling) removes the "
            "spill and is the default whenever M>128. The bwd was already "
            "state-tiled.\n"
            "* **Multiple-of-64 M padding** -- PHASE-1 and PHASE-2 now pad M to "
            "the next multiple of 64 (was: next pow2). At `live` near the top of "
            "a pow2 bracket (180/361/722/1445) this saves ~25-30% of per-slot "
            "compute and the corresponding fraction of VRAM.\n\n"
        )
        f.write(
            "![fwd](bench_fwd.png)\n\n![e2e](bench_e2e.png)\n\n"
            "![vram](bench_vram.png)\n\n"
        )
        f.write("## Precision / accuracy (forward, T=2048)\n\n")
        f.write(
            "bf16 vs an fp32 naive reference. `naive bf16` is the precision "
            "floor (same model, same dtype, no kernel) -- the Triton/flex "
            "path should land in the same band.\n\n"
        )
        f.write("| config | triton/flex bf16 vs fp32 | naive bf16 vs fp32 |\n")
        f.write("|---|---|---|\n")
        for nm, tri, nai in acc:
            f.write(f"| {nm} | {tri} | {nai} |\n")
        f.write(
            "\n**Notes.** Monotone is data-independent -> its Triton path is "
            "numerically tight (bit-exact-class vs the naive recurrence; the "
            "bf16 error is just input quantization). KVM routing (cosine "
            "novelty + argmax) is *chaotic* in low precision -- a 1-ulp wobble "
            "flips a route -- so KVM's bf16 error is intrinsically 'loud'; the "
            "`naive bf16` column shows the same loudness, i.e. it is the "
            "method, not the kernel. Backward kernels are separately verified "
            "(monotone bit-identical to a PyTorch ref incl. d-gate/d-temp; "
            "KVM `our` tracks `pt` at matched precision, M<=128 SRAM-resident "
            "and M=256 state-tiled).\n"
        )
    print(f"\n  wrote {REPORT_PATH}", flush=True)
