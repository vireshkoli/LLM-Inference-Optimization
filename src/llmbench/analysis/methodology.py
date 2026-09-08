"""Analysis for the runs that test the methodology itself.

Three comparisons, each answering a question the headline sweep cannot:

* **Coordinated omission** — how much tail latency a closed-loop generator hides
  on this hardware, against an open-loop run of matched achieved throughput.
* **Burstiness** — what the Poisson assumption costs, against a replay of real
  production arrivals at a matched mean rate.
* **Drift** — whether the environment moved across the sweep, by re-running the
  first configuration last and comparing.

Each returns a number with its own comparison baked in, because each is only
meaningful *relative* to something. A closed-loop p99 quoted on its own is just
a number; quoted against the open-loop p99 it is the size of an error.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from llmbench.metrics.percentiles import summarize
from llmbench.schema import ArrivalProcess, RunResult

__all__ = [
    "DriftComparison",
    "ProcessComparison",
    "drift_comparison",
    "process_comparison",
]


@dataclass(frozen=True, slots=True)
class ProcessComparison:
    """One arrival process measured against the open-loop Poisson baseline."""

    label: str
    baseline_label: str
    #: Achieved output-token throughput, both sides. Reported because the
    #: comparison is only interpretable when the server did comparable work;
    #: a large gap here invalidates the latency comparison rather than
    #: explaining it.
    throughput: float
    baseline_throughput: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    baseline_ttft_p50_ms: float
    baseline_ttft_p95_ms: float
    baseline_ttft_p99_ms: float
    runs: int
    #: Repeat-to-repeat spread on each side. Without these the comparison is a
    #: difference of two means with no way to tell it from noise — and a tail
    #: percentile is a high-variance statistic, so that distinction is not
    #: academic. Measured here: the trace replay's p99 sat 56 ms above the
    #: Poisson baseline against a pooled standard error of 57 ms, while the
    #: baseline's own two measurements of the identical configuration differed
    #: by 118 ms.
    ttft_p99_std_ms: float = 0.0
    baseline_ttft_p99_std_ms: float = 0.0
    baseline_runs: int = 0

    @property
    def throughput_ratio(self) -> float:
        return self.throughput / self.baseline_throughput if self.baseline_throughput else 0.0

    @property
    def comparable(self) -> bool:
        """Whether the two sides did similar enough work to compare tails.

        Ten per cent is generous, and deliberately so: the point is to catch a
        comparison that is meaningless, not to insist on a precision neither run
        was designed to deliver.
        """
        return abs(self.throughput_ratio - 1.0) <= 0.10

    @property
    def p99_pooled_stderr_ms(self) -> float:
        """Standard error of the difference between the two p99 means."""
        n, m = max(self.runs, 1), max(self.baseline_runs, 1)
        variance = (self.ttft_p99_std_ms**2 / n) + (self.baseline_ttft_p99_std_ms**2 / m)
        return float(variance**0.5)

    @property
    def p99_significant(self) -> bool:
        """Whether the p99 difference exceeds two standard errors.

        A comparison that fails this has found nothing, and reporting its ratio
        as an effect would be publishing noise — which is precisely the error
        class this repository exists to avoid.
        """
        err = self.p99_pooled_stderr_ms
        if err <= 0:
            return abs(self.ttft_p99_ms - self.baseline_ttft_p99_ms) > 1e-9
        return abs(self.ttft_p99_ms - self.baseline_ttft_p99_ms) > 2.0 * err

    def understatement_ratio(self, percentile: str = "p99") -> float:
        """How many times *smaller* this process reports the tail.

        Above 1.0 means the process understates tail latency relative to
        open-loop Poisson — which for a closed-loop generator is precisely
        coordinated omission, expressed as a factor rather than a claim.
        """
        mine = getattr(self, f"ttft_{percentile}_ms")
        base = getattr(self, f"baseline_ttft_{percentile}_ms")
        return base / mine if mine else 0.0


def _aggregate_ttft(runs: Sequence[RunResult]) -> tuple[float, float, float, float, float]:
    """Mean TTFT p50/p95/p99, p99 spread, and throughput across repeats."""
    p99 = summarize([r.ttft_s.p99 for r in runs])
    return (
        summarize([r.ttft_s.p50 for r in runs]).mean * 1e3,
        summarize([r.ttft_s.p95 for r in runs]).mean * 1e3,
        p99.mean * 1e3,
        p99.std * 1e3,
        summarize([r.output_token_throughput for r in runs]).mean,
    )


def process_comparison(
    runs: Sequence[RunResult],
    *,
    process: ArrivalProcess,
    baseline_config_id: str,
    baseline_rate_rps: float,
    label: str | None = None,
) -> ProcessComparison | None:
    """Compare one arrival process against the matched open-loop Poisson run.

    Returns ``None`` when either side is missing, rather than raising: the
    report is generated from whatever has been measured, and a methodology run
    that has not been executed should leave a gap in the document instead of
    breaking its generation.
    """
    subject = [r for r in runs if r.workload.arrival_process is process]
    baseline = [
        r
        for r in runs
        if r.config_id == baseline_config_id
        and r.workload.arrival_process is ArrivalProcess.POISSON
        and r.workload.request_rate_rps == baseline_rate_rps
        and r.is_reportable
    ]
    if not subject or not baseline:
        return None

    p50, p95, p99, p99_std, throughput = _aggregate_ttft(subject)
    b50, b95, b99, b99_std, b_throughput = _aggregate_ttft(baseline)

    return ProcessComparison(
        label=label or process.value,
        baseline_label=f"{baseline_config_id} @ {baseline_rate_rps:g} rps (Poisson)",
        throughput=throughput,
        baseline_throughput=b_throughput,
        ttft_p50_ms=p50,
        ttft_p95_ms=p95,
        ttft_p99_ms=p99,
        baseline_ttft_p50_ms=b50,
        baseline_ttft_p95_ms=b95,
        baseline_ttft_p99_ms=b99,
        runs=len(subject),
        ttft_p99_std_ms=p99_std,
        baseline_ttft_p99_std_ms=b99_std,
        baseline_runs=len(baseline),
    )


@dataclass(frozen=True, slots=True)
class DriftComparison:
    """The same configuration measured twice, at the start and end of a sweep."""

    config_id: str
    rate_rps: float
    first_ttft_p95_ms: float
    later_ttft_p95_ms: float
    first_throughput: float
    later_throughput: float
    #: Pooled standard deviation of the original repeats, in ms. The drift is
    #: judged against the sweep's own repeat-to-repeat noise rather than against
    #: an arbitrary percentage, because "within noise" has to mean *this* run's
    #: noise to mean anything.
    first_ttft_std_ms: float

    @property
    def ttft_drift_ms(self) -> float:
        return self.later_ttft_p95_ms - self.first_ttft_p95_ms

    @property
    def throughput_drift_pct(self) -> float:
        if not self.first_throughput:
            return 0.0
        return (self.later_throughput / self.first_throughput - 1.0) * 100.0

    @property
    def within_noise(self) -> bool:
        """True when the re-run sits inside two standard deviations of the first.

        If this holds, environmental drift across the sweep is bounded by the
        same variation the error bars already show, and results measured hours
        apart are comparable. If it fails, every cross-configuration claim in
        the report is weakened — which is why the canary is worth its GPU time.
        """
        if self.first_ttft_std_ms <= 0:
            return abs(self.ttft_drift_ms) < 1e-9
        return abs(self.ttft_drift_ms) <= 2.0 * self.first_ttft_std_ms


def drift_comparison(
    runs: Sequence[RunResult],
    *,
    config_id: str,
    canary_label: str,
    rate_rps: float,
    min_separation_s: float = 3600.0,
    min_repeats_per_side: int = 2,
) -> DriftComparison | None:
    """Compare a canary re-run against the original measurement.

    A canary is the same configuration measured again *hours later*, so the two
    groups are separated by finding the largest gap in start time. Two guards
    stop that from inventing a canary out of ordinary repeats:

    ``min_separation_s`` — consecutive repeats of one rate point run minutes
    apart, so the largest gap between them is small. Without a floor, splitting
    on it produces two arbitrary groups and reports the difference between them
    as environmental drift. That happened: three repeats four minutes apart were
    split 1-and-2 and reported +77.6 ms of drift against a standard deviation of
    zero, because a single-run group has no variance.

    ``min_repeats_per_side`` — the verdict is "within the sweep's own noise", and
    a group of one measurement carries no noise estimate to compare against.

    Returns ``None`` when no genuine canary is present, so the report says the
    canary has not been run rather than showing a fabricated one.
    """
    same = [
        r
        for r in runs
        if r.config_id == config_id
        and r.workload.request_rate_rps == rate_rps
        and r.workload.arrival_process is ArrivalProcess.POISSON
        and r.is_reportable
    ]
    if len(same) < 2:
        return None

    same = sorted(same, key=lambda r: r.started_at)
    # Split on the largest gap in start time: the canary runs at the end of the
    # sweep, hours after the original, so the seam is unambiguous.
    gaps = [
        (b.started_at - a.started_at, i + 1) for i, (a, b) in enumerate(itertools.pairwise(same))
    ]
    if not gaps:
        return None
    largest, split = max(gaps, key=lambda g: g[0])
    if largest.total_seconds() < min_separation_s:
        return None
    first, later = same[:split], same[split:]
    if len(first) < min_repeats_per_side or len(later) < min_repeats_per_side:
        return None

    first_ttft = summarize([r.ttft_s.p95 * 1e3 for r in first])
    later_ttft = summarize([r.ttft_s.p95 * 1e3 for r in later])

    return DriftComparison(
        config_id=canary_label,
        rate_rps=rate_rps,
        first_ttft_p95_ms=first_ttft.mean,
        later_ttft_p95_ms=later_ttft.mean,
        first_throughput=summarize([r.output_token_throughput for r in first]).mean,
        later_throughput=summarize([r.output_token_throughput for r in later]).mean,
        first_ttft_std_ms=first_ttft.std,
    )
