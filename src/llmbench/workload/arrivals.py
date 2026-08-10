"""Arrival-process generation.

The whole benchmark rests on this module being right, so it is pure, seeded and
has no I/O: given a seed it returns the same schedule on every machine and every
run, which is what makes cross-configuration comparison meaningful.

**Open-loop.** A schedule is a list of wall-clock offsets fixed *before* the run
begins. Requests are dispatched at those instants whether or not earlier
requests have completed. A closed-loop generator instead waits for a completion
before issuing the next request, so when the server slows the generator slows
with it and the slow period is under-sampled — *coordinated omission*. The
resulting tail latency is systematically optimistic, and the tail is the number
people actually care about.

``closed_loop_schedule`` exists only so the benchmark can *demonstrate* that
error against its own open-loop numbers on identical hardware. It must never
produce a headline result.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ArrivalSchedule",
    "num_requests_for_duration",
    "poisson_schedule",
    "trace_schedule",
]


@dataclass(frozen=True, slots=True)
class ArrivalSchedule:
    """Immutable list of dispatch offsets, in seconds from run start.

    Offsets are non-decreasing. The schedule is the *intent*; how closely the
    client actually hit it is measured separately as dispatch lag, because a
    client that cannot keep up silently turns an open-loop benchmark back into a
    closed-loop one.
    """

    offsets_s: tuple[float, ...]
    #: The nominal rate this schedule was generated for. ``None`` for trace
    #: replay, where arrivals come from recorded production timestamps.
    nominal_rate_rps: float | None

    def __post_init__(self) -> None:
        if any(b < a for a, b in zip(self.offsets_s, self.offsets_s[1:], strict=False)):
            msg = "arrival offsets must be non-decreasing"
            raise ValueError(msg)
        if self.offsets_s and self.offsets_s[0] < 0.0:
            msg = f"arrival offsets must be non-negative, got {self.offsets_s[0]}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.offsets_s)

    @property
    def span_s(self) -> float:
        """Wall-clock time from first to last dispatch."""
        if len(self.offsets_s) < 2:
            return 0.0
        return self.offsets_s[-1] - self.offsets_s[0]

    @property
    def realized_rate_rps(self) -> float:
        """Rate the schedule actually encodes.

        For a finite Poisson draw this differs from the nominal rate by sampling
        noise; reporting the realized value keeps the offered load honest.
        """
        if self.span_s <= 0.0:
            return 0.0
        return (len(self.offsets_s) - 1) / self.span_s

    def inter_arrivals_s(self) -> tuple[float, ...]:
        """Gaps between consecutive arrivals — the quantity that is Exponential."""
        return tuple(b - a for a, b in zip(self.offsets_s, self.offsets_s[1:], strict=False))


def num_requests_for_duration(rate_rps: float, duration_s: float) -> int:
    """Requests needed to cover ``duration_s`` at ``rate_rps``.

    Rounded up, with a floor of 1, so a short smoke run at a low rate still
    issues something rather than silently measuring nothing.
    """
    if rate_rps <= 0.0:
        msg = f"rate_rps must be positive, got {rate_rps}"
        raise ValueError(msg)
    if duration_s < 0.0:
        msg = f"duration_s must be non-negative, got {duration_s}"
        raise ValueError(msg)
    return max(1, math.ceil(rate_rps * duration_s))


def poisson_schedule(rate_rps: float, num_requests: int, seed: int) -> ArrivalSchedule:
    """Generate a Poisson arrival schedule.

    A Poisson process has Exponentially distributed inter-arrival times with
    mean ``1 / rate_rps``; the schedule is their cumulative sum. This models
    independent clients arriving without coordination, which is the standard
    assumption for serving benchmarks — and, being an assumption, is checked
    against a real production trace elsewhere in the sweep.

    The first request is dispatched at offset 0 so that a run's measurement
    window starts immediately rather than after a random initial gap.

    Args:
        rate_rps: Offered load, requests per second. Must be positive.
        num_requests: Number of arrivals to generate. Must be positive.
        seed: Seeds a PCG64 generator. Identical seeds give identical schedules
            across machines and numpy versions.

    Raises:
        ValueError: On a non-positive rate or request count.
    """
    if rate_rps <= 0.0:
        msg = f"rate_rps must be positive, got {rate_rps}"
        raise ValueError(msg)
    if num_requests <= 0:
        msg = f"num_requests must be positive, got {num_requests}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    # n-1 gaps, because the first arrival is pinned at t=0.
    gaps = rng.exponential(scale=1.0 / rate_rps, size=num_requests - 1)
    offsets = np.concatenate(([0.0], np.cumsum(gaps)))
    return ArrivalSchedule(offsets_s=tuple(float(x) for x in offsets), nominal_rate_rps=rate_rps)


def trace_schedule(timestamps_s: Sequence[float]) -> ArrivalSchedule:
    """Build a schedule from recorded production arrival timestamps.

    Timestamps are rebased so the first arrival sits at offset 0. No rate is
    attached: the point of a trace replay is that its burstiness is *not*
    described by a single rate parameter, and pretending otherwise would erase
    the very property the run exists to expose.

    Raises:
        ValueError: If the trace is empty or not sorted.
    """
    if not timestamps_s:
        msg = "trace schedule requires at least one timestamp"
        raise ValueError(msg)
    if any(b < a for a, b in itertools.pairwise(timestamps_s)):
        msg = "trace timestamps must be non-decreasing"
        raise ValueError(msg)

    origin = timestamps_s[0]
    return ArrivalSchedule(
        offsets_s=tuple(float(t - origin) for t in timestamps_s),
        nominal_rate_rps=None,
    )
