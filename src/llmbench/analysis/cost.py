"""Cost model and SLA-conditioned configuration choice.

    cost_per_1M_tokens = (gpu_hourly_usd / output_tokens_per_hour) * 1e6

**On price sensitivity.** Every configuration here runs on the same GPU, so the
hourly rate enters as a linear scalar and the *ranking* of configurations is
invariant to it. There is no crossover price at which a different configuration
wins, and publishing a price-sensitivity table would imply a result that does
not exist.

What *does* reorder the frontier is the latency SLA. A p95 TTFT budget
determines which offered rates a configuration can serve, which caps its
sustainable throughput, which sets its cost per token. Two configurations can
swap places as the budget tightens, so that is the sensitivity worth reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llmbench.metrics.percentiles import summarize
from llmbench.schema import RunResult

__all__ = [
    "ConfigOperatingPoint",
    "CostAssumption",
    "best_under_sla",
    "cost_per_million_tokens",
    "operating_points",
]


@dataclass(frozen=True, slots=True)
class CostAssumption:
    gpu_hourly_usd: float
    source_vendor: str
    source_url: str
    accessed_date: str


def cost_per_million_tokens(tokens_per_second: float, gpu_hourly_usd: float) -> float:
    """Cost of a million output tokens at a sustained token rate.

    Raises:
        ValueError: On a non-positive token rate — a configuration that
            produces nothing has no cost per token, and returning infinity
            would quietly rank it somewhere on a chart.
    """
    if tokens_per_second <= 0:
        msg = f"tokens_per_second must be positive, got {tokens_per_second}"
        raise ValueError(msg)
    return (gpu_hourly_usd / (tokens_per_second * 3600.0)) * 1e6


@dataclass(frozen=True, slots=True)
class ConfigOperatingPoint:
    """One configuration's best sustainable point under a latency budget."""

    config_id: str
    rate_rps: float
    throughput_tokens_s: float
    throughput_std: float
    ttft_p95_s: float
    tpot_p95_s: float
    repeats: int
    cost_per_1m_usd: float

    @property
    def meets(self) -> bool:
        return True


def operating_points(
    runs: Sequence[RunResult], gpu_hourly_usd: float
) -> dict[str, list[ConfigOperatingPoint]]:
    """Aggregate reportable runs into per-configuration operating points.

    Only ``VALID`` runs are used. Oversubscribed runs are excluded on purpose:
    past capacity the latency is a function of run duration rather than of
    offered load, so treating one as an operating point would put a number on
    the frontier that no amount of provisioning could reproduce.
    """
    grouped: dict[tuple[str, float], list[RunResult]] = {}
    for run in runs:
        if not run.is_reportable or run.workload.request_rate_rps is None:
            continue
        grouped.setdefault((run.config_id, run.workload.request_rate_rps), []).append(run)

    points: dict[str, list[ConfigOperatingPoint]] = {}
    for (config_id, rate), group in sorted(grouped.items()):
        thr = summarize([r.output_token_throughput for r in group])
        ttft = summarize([r.ttft_s.p95 for r in group])
        tpot = summarize([r.tpot_s.p95 for r in group])
        points.setdefault(config_id, []).append(
            ConfigOperatingPoint(
                config_id=config_id,
                rate_rps=rate,
                throughput_tokens_s=thr.mean,
                throughput_std=thr.std,
                ttft_p95_s=ttft.mean,
                tpot_p95_s=tpot.mean,
                repeats=len(group),
                cost_per_1m_usd=cost_per_million_tokens(thr.mean, gpu_hourly_usd),
            )
        )
    return points


def best_under_sla(
    points: dict[str, list[ConfigOperatingPoint]],
    *,
    max_ttft_p95_s: float,
    max_tpot_p95_s: float | None = None,
) -> dict[str, ConfigOperatingPoint]:
    """Highest-throughput admissible point per configuration.

    "Admissible" means the measured p95s sit inside the budget. Throughput is
    maximised rather than latency minimised because, for a fixed SLA, the
    cheapest configuration is the one that serves the most tokens per GPU-hour
    while still meeting it.
    """
    best: dict[str, ConfigOperatingPoint] = {}
    for config_id, candidates in points.items():
        admissible = [
            p
            for p in candidates
            if p.ttft_p95_s <= max_ttft_p95_s
            and (max_tpot_p95_s is None or p.tpot_p95_s <= max_tpot_p95_s)
        ]
        if admissible:
            best[config_id] = max(admissible, key=lambda p: p.throughput_tokens_s)
    return best
