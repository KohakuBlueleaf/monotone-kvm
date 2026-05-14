"""Monotone bucket schedules.

A *bucket schedule* decides, deterministically and without looking at any tensor
data, how a growing sequence of fixed-content "buckets" is compressed over time.
Buckets are ordered newest -> oldest; sizes are in *chunk units*.

Every schedule preserves four invariants on each step:

  1. bucket count is monotone non-decreasing -- it never drops;
  2. only adjacent equal-size buckets merge -- keeps the hierarchy dyadic;
  3. the newest bucket (index 0) is never merged -- stays directly readable;
  4. sizes are non-decreasing newest -> oldest -- memory grows smoothly.

`step(sizes, t)` takes the bucket sizes *before* chunk `t` (newest-first) and
returns `(new_sizes, merge_pair)` where `new_sizes` already has the new
singleton prepended at index 0 and `merge_pair` is `None` or `(i, i+1)` -- the
adjacent pair fused (indices into `new_sizes`' pre-merge layout).

Every schedule in the family obeys invariant 2 *strictly* -- it only ever fuses
adjacent **equal-size** buckets. The family spans the O(state) continuum::

    log             O(log t)         -- de-amortized binary carry (no coeff)
    logbudget(c)    O(c log t)       -- tunable soft-budget log
    power(alpha,c)  O(max(c t^a,log t))-- the t**alpha continuum (1/3, 1/2, ...)
    sqrt(c)         O(c sqrt t)      -- power(alpha=1/2)
    linear          O(t)             -- never merges, full-attention end

Note the **O(log t) floor**: with same-size-only adjacent merges you cannot drop
below ~bit_length(t) buckets, so `log` is the most aggressive schedule in the
family. `power(alpha)` for small alpha is therefore really O(max(t**alpha,
log t)). There is no O(1) schedule here: a true constant cap would have to fuse
*unequal*-size buckets, which invariant 2 forbids -- so the floor stands.
"""

import math
from abc import ABC, abstractmethod


# --------------------------------------------------------------------------
# integer helpers
# --------------------------------------------------------------------------
def lowbit(x: int) -> int:
    """Lowest set bit of x (x & -x)."""
    return x & -x


def ceil_sqrt(x: int) -> int:
    r = math.isqrt(x)
    return r if r * r == x else r + 1


def merge_one(
    sizes: list[int], *, required_size: int | None = None, order: str = "oldest"
):
    """Merge exactly one adjacent equal-size pair, never touching index 0.

    `sizes` is newest-first and already includes the new singleton at index 0.
    `order="oldest"` prefers the oldest valid pair (compress old memory first);
    `order="newest"` prefers the newest. A merge is only allowed if it keeps
    sizes non-decreasing toward the oldest end.

    Returns `(new_sizes, (i, i+1))`, or `(sizes, None)` if no valid pair exists.
    """
    n = len(sizes)
    scan = range(n - 2, 0, -1) if order == "oldest" else range(1, n - 1)
    for i in scan:
        a, b = sizes[i], sizes[i + 1]
        if a != b:
            continue
        if required_size is not None and a != required_size:
            continue
        if i + 2 < n and a + b > sizes[i + 2]:  # invariant 4
            continue
        return sizes[:i] + [a + b] + sizes[i + 2 :], (i, i + 1)
    return sizes, None


def check_invariants(sizes: list[int]) -> None:
    """Assert the four monotone-bucket invariants on a single state."""
    if not sizes:
        return
    assert sizes[0] == 1, f"newest bucket must be a singleton: {sizes}"
    for i in range(len(sizes) - 1):
        assert (
            sizes[i] <= sizes[i + 1]
        ), f"sizes must be non-decreasing newest->oldest: {sizes}"


# --------------------------------------------------------------------------
# scheduler interface
# --------------------------------------------------------------------------
class BucketScheduler(ABC):
    """Base class for monotone bucket schedules."""

    name: str = "base"

    @abstractmethod
    def step(self, sizes: list[int], t: int):
        """Advance one chunk. See module docstring for the contract."""

    def expected_count(self, t: int) -> float:
        """Reference bucket-count target at chunk `t` (for plots / sanity)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LogScheduler(BucketScheduler):
    """De-amortized binary carry: exactly ``bit_length(t)`` buckets = O(log t).

    Inside each power-of-two phase the merge size cycles 1, 2, 1, 4, 1, 2, 1, 8,
    ..., so memory is transferred toward the old end gradually instead of
    collapsing all at once (1 1 2 4 ... -> 1 1 1 1 ... 64).
    """

    name = "log"

    def step(self, sizes: list[int], t: int):
        sizes = [1] + list(sizes)
        base = 1 << (t.bit_length() - 1)
        phase = t - base
        if phase == 0:  # power-of-two boundary: grow
            return sizes, None
        new_sizes, pair = merge_one(sizes, required_size=lowbit(phase), order="oldest")
        if pair is None:
            raise RuntimeError(f"log schedule: no valid merge at t={t}, sizes={sizes}")
        return new_sizes, pair

    def expected_count(self, t: int) -> float:
        return float(t.bit_length())


class SoftBudgetScheduler(BucketScheduler):
    """Merge a newest-side pair whenever the bucket count exceeds ``budget(t)``.

    Because the newest bucket can't merge and only equal-size pairs may fuse,
    the budget is *soft* in two ways: the count can exceed it by a small slack,
    and it can never drop below the O(log t) dyadic floor (see module docstring)
    -- so the realised count is roughly ``max(budget(t), bit_length(t))``.
    """

    @abstractmethod
    def budget(self, t: int) -> float: ...

    def step(self, sizes: list[int], t: int):
        sizes = [1] + list(sizes)
        if len(sizes) > self.budget(t):
            return merge_one(sizes, order="newest")
        return sizes, None

    def expected_count(self, t: int) -> float:
        return float(self.budget(t))


class PowerScheduler(SoftBudgetScheduler):
    """O(t**alpha) buckets. ``alpha=1/2`` is sqrt, ``1/3`` cube-root, etc."""

    def __init__(self, alpha: float = 0.5, coeff: float = 1.0):
        self.alpha = float(alpha)
        self.coeff = float(coeff)
        self.name = f"power(a={self.alpha:g})"

    def budget(self, t: int) -> float:
        return math.ceil(self.coeff * (t**self.alpha))

    def __repr__(self) -> str:
        return f"PowerScheduler(alpha={self.alpha:g}, coeff={self.coeff:g})"


class SqrtScheduler(PowerScheduler):
    """O(sqrt t) buckets -- ``PowerScheduler(alpha=1/2)``."""

    def __init__(self, coeff: float = 1.0):
        super().__init__(alpha=0.5, coeff=coeff)
        self.name = "sqrt"

    def __repr__(self) -> str:
        return f"SqrtScheduler(coeff={self.coeff:g})"


class LogBudgetScheduler(SoftBudgetScheduler):
    """O(coeff * log2 t) buckets -- a *tunable* soft-budget log.

    `LogScheduler` is the de-amortized binary carry: structurally exactly
    ``bit_length(t)`` buckets, with the gradual slow-transfer property -- and so
    it has no `coeff` lever (the dyadic carry *is* the schedule). This is the
    tunable alternative: a soft-budget schedule (like `PowerScheduler`) whose
    budget is ``coeff * log2(t)``. It gains a `coeff` knob uniform with
    power/sqrt, but gives up the dyadic slow-transfer (it merges a newest-side
    pair on overflow). ``coeff=1`` sits near the O(log t) floor.
    """

    def __init__(self, coeff: float = 1.0):
        self.coeff = float(coeff)
        self.name = f"logbudget(c={self.coeff:g})"

    def budget(self, t: int) -> float:
        return math.ceil(self.coeff * max(t.bit_length(), 1))

    def __repr__(self) -> str:
        return f"LogBudgetScheduler(coeff={self.coeff:g})"


class LinearScheduler(BucketScheduler):
    """O(t) buckets: never merges -- the full-attention end of the family.

    Trivially honours invariant 2 (it performs zero merges); it is the
    no-compression reference, not a real compression schedule.
    """

    name = "linear"

    def step(self, sizes: list[int], t: int):
        return [1] + list(sizes), None

    def expected_count(self, t: int) -> float:
        return float(t)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
_FACTORIES = {
    "log": LogScheduler,
    "logbudget": LogBudgetScheduler,
    "sqrt": SqrtScheduler,
    "power": PowerScheduler,
    "linear": LinearScheduler,
}


def get_scheduler(name: str, **kwargs) -> BucketScheduler:
    """Build a scheduler by name. Extra kwargs go to the constructor
    (e.g. ``get_scheduler("power", alpha=1/3)``, ``get_scheduler("sqrt", coeff=2)``).
    """
    key = name.lower()
    if key not in _FACTORIES:
        raise ValueError(f"unknown scheduler {name!r}; choices: {sorted(_FACTORIES)}")
    return _FACTORIES[key](**kwargs)


def simulate(scheduler: BucketScheduler, n_tokens: int) -> list[list[int]]:
    """Run `scheduler` for `n_tokens` steps, returning the size list per step.

    The schedule is **token-based**: step `t` folds in one more token, so
    `history[t-1]` is the bucket-size partition of the first `t` tokens.
    """
    sizes: list[int] = []
    history: list[list[int]] = []
    for t in range(1, n_tokens + 1):
        sizes, _ = scheduler.step(sizes, t)
        check_invariants(sizes)
        history.append(list(sizes))
    return history


def intervals(sizes_newest_first: list[int]) -> list[tuple[int, int]]:
    """Bucket sizes (newest-first token counts) -> contiguous [a, b) token
    intervals, oldest-first. The intervals tile [0, sum(sizes)) exactly."""
    out, pos = [], sum(sizes_newest_first)
    for sz in sizes_newest_first:
        out.append((pos - sz, pos))
        pos -= sz
    return list(reversed(out))


if __name__ == "__main__":
    demos = [
        get_scheduler("log"),
        get_scheduler("sqrt"),
        get_scheduler("power", alpha=1 / 3),
        get_scheduler("logbudget", coeff=2.0),
        get_scheduler("linear"),
    ]
    for sched in demos:
        print(f"\n[{sched.name}]  {sched!r}")
        hist = simulate(sched, 20)
        prev = 0
        for t, sizes in enumerate(hist, start=1):
            assert len(sizes) >= prev, "bucket count must be monotone"
            prev = len(sizes)
            print(f"  t={t:2d}: count={len(sizes):2d}  {sizes}")
