"""Invariant tests for every bucket schedule.

Checks, for each scheduler over a long run:
  * the four monotone-bucket invariants hold at every step;
  * the bucket count is monotone non-decreasing;
  * each schedule matches its expected asymptotic count;
  * total tokens are conserved (sum of bucket sizes == t).

Run:  python scripts/test_schedulers.py
"""

from monotone_kvm import check_invariants, get_scheduler, simulate


def test_scheduler(name, n=4096, **kwargs):
    sched = get_scheduler(name, **kwargs)
    hist = simulate(sched, n)

    prev_count = 0
    for t, sizes in enumerate(hist, start=1):
        check_invariants(sizes)  # invariants 1-4
        assert len(sizes) >= prev_count, f"{sched.name}: bucket count dropped at t={t}"
        prev_count = len(sizes)
        assert (
            sum(sizes) == t
        ), f"{sched.name}: tokens not conserved at t={t} ({sum(sizes)} != {t})"

    counts = [len(s) for s in hist]
    final = counts[-1]

    # asymptotic sanity
    if name == "log":
        assert all(
            len(s) == t.bit_length() for t, s in enumerate(hist, 1)
        ), "log schedule must hold exactly bit_length(t) buckets"
    elif name == "linear":
        assert final == n, "linear schedule must never merge"
    elif name == "fixed":
        k = kwargs.get("k", 8)
        assert max(counts) <= k, f"fixed(k={k}) exceeded its hard cap: {max(counts)}"
        assert final == min(n, k), f"fixed(k={k}) should settle at min(n,k)"
    else:  # power / sqrt -- soft budget, floored at the O(log t) dyadic floor
        floor = n.bit_length()
        ref = max(sched.expected_count(n), floor)
        assert (
            final <= ref + 3
        ), f"{sched.name}: final count {final} exceeds max(budget, log-floor)={ref}"

    print(
        f"  PASS  {sched.name:14s}  final count = {final:5d}  "
        f"(ref ~ {sched.expected_count(n):.0f})"
    )
    return counts


def main():
    print("running scheduler invariant tests (n=4096)")
    test_scheduler("log")
    test_scheduler("sqrt")
    test_scheduler("power", alpha=1 / 3)
    test_scheduler("power", alpha=1 / 4)
    test_scheduler("fixed", k=8)
    test_scheduler("fixed", k=32)
    test_scheduler("linear")
    print("all scheduler tests passed.")


if __name__ == "__main__":
    main()
