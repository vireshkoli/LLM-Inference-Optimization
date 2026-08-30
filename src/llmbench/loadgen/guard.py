"""Run-validity assessment — deciding whether a measurement may be reported.

The failure mode this guards against is the one that quietly ruins open-loop
benchmarks. At high offered load the *client* can run out of headroom: the event
loop falls behind its own schedule, requests go out late, and the offered load
silently collapses back toward closed-loop. The server then looks faster than it
is, precisely in the high-load region where the interesting numbers live.

Dispatch lag — actual minus scheduled dispatch time — makes this visible. A run
whose lag has grown is measuring the load generator, not the server, and is
marked invalid rather than reported.

Invalid runs are **kept, not discarded**: the rate at which a harness or a
configuration runs out of headroom is itself a result, and deleting the record
would hide it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llmbench.metrics.percentiles import percentile
from llmbench.schema import RunValidity

__all__ = ["SaturationPolicy", "ValidityAssessment", "assess_validity"]


@dataclass(frozen=True, slots=True)
class SaturationPolicy:
    """Thresholds separating a trustworthy run from a compromised one.

    Dispatch lag is bounded two ways because a single threshold cannot serve
    every rate. The absolute ceiling catches gross stalls at low rates, where a
    generous relative budget would wave through a 400 ms hiccup. The relative
    bound catches drift at high rates, where lag smaller than the absolute
    ceiling can still exceed a whole inter-arrival gap and reorder the offered
    load. A run must satisfy both.
    """

    #: Hard ceiling on p99 dispatch lag, seconds.
    max_p99_dispatch_lag_s: float = 0.050
    #: p99 dispatch lag must also stay under this fraction of the mean
    #: inter-arrival time, so the bound tightens automatically as rate rises.
    max_p99_lag_as_interarrival_fraction: float = 0.5
    #: Above this share of failed requests the server, not the client, has
    #: broken and the latency distribution is not meaningful.
    max_failure_rate: float = 0.01
    #: Above this share of telemetry samples reporting an active throttle
    #: reason, the GPU was not in thermal steady state.
    max_throttled_fraction: float = 0.05
    #: A run that completed fewer than this share of its scheduled requests did
    #: not run long enough to characterise anything.
    min_completion_rate: float = 0.95
    #: Below this ratio of achieved to offered request rate, the server could
    #: not keep up: the queue grows without bound and latency becomes a function
    #: of run duration rather than a steady-state property.
    min_achieved_rate_ratio: float = 0.80
    #: Dispatch lag only contaminates a measurement in proportion to what is
    #: being measured. Lag below this fraction of observed TTFT is immaterial,
    #: however it compares to the absolute or inter-arrival bounds. Measured:
    #: at 24 rps beyond saturation, lag p99 was 185 ms against a TTFT p95 of
    #: 456 s — 0.04%, which is plainly not the client's doing.
    max_lag_fraction_of_latency: float = 0.05


@dataclass(frozen=True, slots=True)
class ValidityAssessment:
    validity: RunValidity
    notes: tuple[str, ...]

    @property
    def is_reportable(self) -> bool:
        return self.validity is RunValidity.VALID


def assess_validity(
    *,
    dispatch_lags_s: Sequence[float],
    requests_scheduled: int,
    requests_completed: int,
    requests_failed: int,
    mean_interarrival_s: float | None = None,
    offered_rate_rps: float | None = None,
    achieved_rate_rps: float | None = None,
    observed_latency_s: float | None = None,
    throttled_fraction: float = 0.0,
    policy: SaturationPolicy | None = None,
) -> ValidityAssessment:
    """Classify a completed run.

    Every violated condition is recorded in ``notes``, but a single validity
    label is returned, chosen by how fundamentally the condition undermines the
    measurement:

    1. ``ENGINE_ERROR``     — the server failed; latencies describe nothing.
    2. ``OVERSUBSCRIBED``   — offered load exceeded capacity; latency is
       duration-dependent rather than steady-state.
    3. ``CLIENT_SATURATED`` — the harness failed; latencies describe the client.
    4. ``INCOMPLETE``       — too little data to characterise the configuration.
    5. ``THERMAL_THROTTLED``— real data, but not at steady-state clocks.

    Engine errors outrank the rest because a failing server produces latencies
    for requests that never really ran.

    **Oversubscription outranks client saturation**, because past capacity the
    two are not independent: a server that cannot keep up leaves thousands of
    requests in flight, and that alone induces dispatch lag in any client. The
    lag is then a symptom, not the cause, and reporting it as client saturation
    would blame the harness for the server's limit.

    Args:
        dispatch_lags_s: Per-request actual-minus-scheduled dispatch time.
        mean_interarrival_s: Mean gap of the *intended* schedule. When omitted
            only the absolute lag ceiling applies.
        offered_rate_rps / achieved_rate_rps: Used to detect oversubscription.
        observed_latency_s: Typically TTFT p95. Dispatch lag is judged against
            it, because lag only contaminates a measurement in proportion to
            what is being measured.
    """
    policy = policy or SaturationPolicy()
    notes: list[str] = []

    failure_rate = requests_failed / requests_scheduled if requests_scheduled else 0.0
    completion_rate = requests_completed / requests_scheduled if requests_scheduled else 0.0

    engine_failed = failure_rate > policy.max_failure_rate
    if engine_failed:
        notes.append(
            f"failure rate {failure_rate:.3%} exceeds {policy.max_failure_rate:.3%} "
            f"({requests_failed}/{requests_scheduled} requests failed)"
        )

    oversubscribed = False
    if offered_rate_rps and achieved_rate_rps is not None:
        ratio = achieved_rate_rps / offered_rate_rps
        if ratio < policy.min_achieved_rate_ratio:
            oversubscribed = True
            notes.append(
                f"achieved only {achieved_rate_rps:.2f} of {offered_rate_rps:.2f} offered req/s "
                f"({ratio:.0%}) — the server did not keep up, so the queue grew without bound "
                f"and latency reflects run duration rather than steady state"
            )

    saturated = False
    if dispatch_lags_s:
        p99_lag = percentile(sorted(dispatch_lags_s), 99.0)

        # Lag that is negligible against the latency being measured cannot have
        # contaminated it, whatever the absolute thresholds say.
        immaterial = (
            observed_latency_s is not None
            and observed_latency_s > 0
            and p99_lag / observed_latency_s < policy.max_lag_fraction_of_latency
        )

        if immaterial:
            pass
        elif p99_lag > policy.max_p99_dispatch_lag_s:
            saturated = True
            notes.append(
                f"p99 dispatch lag {p99_lag * 1e3:.1f} ms exceeds absolute ceiling "
                f"{policy.max_p99_dispatch_lag_s * 1e3:.1f} ms — the client fell behind its "
                f"own schedule, so offered load was not the intended open-loop process"
            )

        if not immaterial and mean_interarrival_s is not None and mean_interarrival_s > 0.0:
            relative_ceiling = policy.max_p99_lag_as_interarrival_fraction * mean_interarrival_s
            if p99_lag > relative_ceiling:
                saturated = True
                notes.append(
                    f"p99 dispatch lag {p99_lag * 1e3:.1f} ms exceeds "
                    f"{policy.max_p99_lag_as_interarrival_fraction:.0%} of the "
                    f"{mean_interarrival_s * 1e3:.1f} ms mean inter-arrival gap — lag of this "
                    f"size relative to the gap reorders the offered load"
                )

    incomplete = completion_rate < policy.min_completion_rate
    if requests_scheduled == 0:
        # Distinct from "most requests failed": nothing was ever offered, so
        # there is no distribution to report. Called out separately because
        # "completion rate 0.0% (0/0)" reads like a division bug.
        notes.append("run scheduled zero requests — nothing was measured")
    elif incomplete:
        notes.append(
            f"completion rate {completion_rate:.1%} below {policy.min_completion_rate:.1%} "
            f"({requests_completed}/{requests_scheduled} completed)"
        )

    throttled = throttled_fraction > policy.max_throttled_fraction
    if throttled:
        notes.append(
            f"GPU reported an active throttle reason in {throttled_fraction:.1%} of samples, "
            f"above {policy.max_throttled_fraction:.1%} — not thermal steady state"
        )

    # Precedence, most fundamental failure first. Every violated condition is
    # already in `notes`, so ranking chooses the label without hiding anything.
    ranked: tuple[tuple[bool, RunValidity], ...] = (
        (engine_failed, RunValidity.ENGINE_ERROR),
        (oversubscribed, RunValidity.OVERSUBSCRIBED),
        (saturated, RunValidity.CLIENT_SATURATED),
        (incomplete, RunValidity.INCOMPLETE),
        (throttled, RunValidity.THERMAL_THROTTLED),
    )
    validity = next(
        (verdict for triggered, verdict in ranked if triggered),
        RunValidity.VALID,
    )

    return ValidityAssessment(validity=validity, notes=tuple(notes))
