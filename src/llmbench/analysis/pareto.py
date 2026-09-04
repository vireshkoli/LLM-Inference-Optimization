"""Pareto frontier over quality, latency and cost.

The headline chart. A configuration is *dominated* when another is at least as
good on every axis and strictly better on one; dominated configurations are kept
and drawn, not filtered out, because seeing what lost is most of the argument
for what won.

Axis directions are declared explicitly rather than inferred. Getting a sign
wrong silently inverts the frontier and produces a confident, wrong
recommendation — the kind of error that looks like a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ParetoPoint", "dominates", "pareto_frontier"]


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One configuration on the frontier.

    Attributes are stored in their natural units and compared with explicit
    directions: quality higher-is-better, latency and cost lower-is-better.
    """

    config_id: str
    quality: float
    latency_s: float
    cost_per_1m_usd: float
    #: Carried through for labelling; not part of the domination test.
    throughput_tokens_s: float = 0.0


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better somewhere.

    Quality is maximised; latency and cost are minimised.
    """
    at_least_as_good = (
        a.quality >= b.quality
        and a.latency_s <= b.latency_s
        and a.cost_per_1m_usd <= b.cost_per_1m_usd
    )
    strictly_better = (
        a.quality > b.quality or a.latency_s < b.latency_s or a.cost_per_1m_usd < b.cost_per_1m_usd
    )
    return at_least_as_good and strictly_better


def pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[list[ParetoPoint], list[ParetoPoint]]:
    """Split points into (frontier, dominated).

    Returns:
        ``(frontier, dominated)``. Both are returned because the dominated set
        is evidence: a frontier drawn without the points it beat is an
        assertion rather than a demonstration.
    """
    frontier = [p for p in points if not any(dominates(q, p) for q in points if q is not p)]
    dominated = [p for p in points if p not in frontier]
    return frontier, dominated
