"""Percentile computation and distribution summaries.

The percentile routine here is written out by hand rather than delegated to
``numpy.percentile``. That is deliberate: percentiles are the headline output of
this benchmark, and ``tests/test_percentiles.py`` checks this implementation
against numpy as an independent oracle across sizes, duplicate-heavy inputs and
degenerate cases. Calling numpy here would make that test tautological.

Interpolation matches numpy's default ``linear`` method, so the two agree
exactly rather than approximately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmbench.schema import Stats

__all__ = ["percentile", "summarize"]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile (0-100) of an already-sorted sequence.

    Uses linear interpolation between the two nearest ranks, which is numpy's
    default and the convention most serving benchmarks implicitly assume.

    Args:
        sorted_values: Values in non-decreasing order. Not re-sorted — sorting
            once in the caller keeps the cost of summarising many percentiles
            linear rather than repeated.
        q: Percentile in [0, 100].

    Raises:
        ValueError: If the sequence is empty or ``q`` is out of range.
    """
    if not sorted_values:
        msg = "percentile of an empty sequence is undefined"
        raise ValueError(msg)
    if not 0.0 <= q <= 100.0:
        msg = f"q must be in [0, 100], got {q}"
        raise ValueError(msg)

    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])

    # Virtual index into the sorted sample.
    rank = (q / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return float(sorted_values[int(rank)])

    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _stdev(values: Sequence[float], mean: float) -> float:
    """Population standard deviation.

    Population rather than sample: these are complete observations of a run, not
    a sample drawn from it. Matches ``numpy.std`` default (ddof=0).
    """
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def summarize(values: Sequence[float]) -> Stats:
    """Summarise a sample into the schema's :class:`Stats` record.

    An empty sample yields an all-zero record with ``count=0`` rather than
    raising. A run that completed no requests still has to produce a valid,
    storable result — the fact that it completed nothing is the finding, and
    discarding the record would hide it.
    """
    if not values:
        return Stats(
            count=0, mean=0.0, std=0.0, min=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0.0
        )

    ordered = sorted(float(v) for v in values)
    mean = sum(ordered) / len(ordered)

    return Stats(
        count=len(ordered),
        mean=mean,
        std=_stdev(ordered, mean),
        min=ordered[0],
        p50=percentile(ordered, 50.0),
        p90=percentile(ordered, 90.0),
        p95=percentile(ordered, 95.0),
        p99=percentile(ordered, 99.0),
        max=ordered[-1],
    )
