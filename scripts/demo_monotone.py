"""Demo: KVM attention with a monotone deterministic bucket schedule.

  (1) the schedules themselves -- pure integer arithmetic, data-independent;
  (2) the full attention module -- run the recurrence, confirm the slot count
      stays monotone and the bucket-size invariants hold.

Run:  python scripts/demo_monotone.py
"""

import torch

from monotone_kvm import (
    MonotoneKVMAttention,
    MonotoneKVMConfig,
    check_invariants,
    get_scheduler,
    simulate,
)


def main():
    torch.manual_seed(0)

    # (1) the schedules: bucket sizes (chunk units), newest-first.
    for name, kwargs in [
        ("log", {}),
        ("sqrt", {}),
        ("power", {"alpha": 1 / 3}),
        ("fixed", {"k": 5}),
    ]:
        sched = get_scheduler(name, **kwargs)
        hist = simulate(sched, 17)
        print(f"\n[{sched.name}]")
        for t, sizes in enumerate(hist, start=1):
            check_invariants(sizes)
            note = f"  (== bit_length({t}))" if name == "log" else ""
            assert name != "log" or len(sizes) == t.bit_length()
            print(f"  t={t:2d}: count={len(sizes):2d}  {sizes}{note}")

    # (2) the full attention module.
    B, T, H, d = 2, 2048, 4, 32
    hidden = H * d
    x = torch.randn(B, T, hidden)
    print()
    for schedule, kwargs in [("log", {}), ("sqrt", {}), ("power", {"alpha": 1 / 3})]:
        cfg = MonotoneKVMConfig(
            hidden_size=hidden,
            num_heads=H,
            chunk_len=64,
            n_bswa_chunks=2,
            sink_len=1,
            schedule=schedule,
            schedule_kwargs=kwargs,
        )
        model = MonotoneKVMAttention(cfg)
        y = model(x)
        y.square().mean().backward()
        gnorm = (
            sum(
                (p.grad.detach() ** 2).sum()
                for p in model.parameters()
                if p.grad is not None
            )
            .sqrt()
            .item()
        )
        trace = model._trace
        assert all(b >= a for a, b in zip(trace, trace[1:])), "slot count not monotone"
        for sizes in model._size_trace:
            check_invariants(sizes)
        label = model.scheduler.name
        print(f"[{label:13s}] y={tuple(y.shape)}  slots/chunk = {trace}")
        print(
            f"{'':16s}final buckets (chunk units) = {model._size_trace[-1]}  "
            f"grad_norm={gnorm:.3f}"
        )


if __name__ == "__main__":
    main()
