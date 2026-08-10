"""Run-validity assessment.

These tests pin the behaviour that keeps a compromised measurement out of the
headline results — specifically, detecting when the load generator rather than
the server became the bottleneck.
"""

from __future__ import annotations

from llmbench.loadgen.guard import SaturationPolicy, assess_validity
from llmbench.schema import RunValidity


def healthy(**overrides: object) -> dict[str, object]:
    """A clean run; individual tests perturb exactly one dimension."""
    base: dict[str, object] = {
        "dispatch_lags_s": [0.0005] * 1000,
        "requests_scheduled": 1000,
        "requests_completed": 1000,
        "requests_failed": 0,
        "mean_interarrival_s": 0.125,  # 8 rps
        "throttled_fraction": 0.0,
    }
    base.update(overrides)
    return base


class TestHealthyRuns:
    def test_clean_run_is_valid(self) -> None:
        assessment = assess_validity(**healthy())  # type: ignore[arg-type]
        assert assessment.validity is RunValidity.VALID
        assert assessment.notes == ()
        assert assessment.is_reportable is True

    def test_tolerates_a_single_failure_within_budget(self) -> None:
        assessment = assess_validity(
            **healthy(requests_completed=999, requests_failed=1)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.VALID


class TestClientSaturation:
    def test_absolute_lag_ceiling_catches_gross_stalls(self) -> None:
        assessment = assess_validity(
            **healthy(dispatch_lags_s=[0.2] * 1000, mean_interarrival_s=None)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.CLIENT_SATURATED
        assert any("absolute ceiling" in n for n in assessment.notes)

    def test_relative_bound_catches_drift_at_high_rate(self) -> None:
        """40 ms lag is under the absolute ceiling but is most of a 41 ms gap.

        At 24 rps that reorders the offered load, so the run cannot be reported
        even though no single threshold in isolation looks alarming.
        """
        assessment = assess_validity(
            **healthy(  # type: ignore[arg-type]
                dispatch_lags_s=[0.040] * 1000,
                mean_interarrival_s=1.0 / 24.0,
            )
        )
        assert assessment.validity is RunValidity.CLIENT_SATURATED
        assert any("mean inter-arrival gap" in n for n in assessment.notes)

    def test_same_lag_is_acceptable_at_low_rate(self) -> None:
        """40 ms of lag against a 1 s gap is immaterial.

        A single fixed threshold would either flag this run or miss the one
        above; that is why the policy carries both bounds.
        """
        assessment = assess_validity(
            **healthy(dispatch_lags_s=[0.040] * 1000, mean_interarrival_s=1.0)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.VALID

    def test_uses_p99_not_mean(self) -> None:
        """A few very late dispatches invalidate a run whose mean looks fine."""
        lags = [0.0001] * 985 + [0.5] * 15
        assessment = assess_validity(**healthy(dispatch_lags_s=lags))  # type: ignore[arg-type]
        assert assessment.validity is RunValidity.CLIENT_SATURATED

    def test_empty_run_is_incomplete_rather_than_a_crash(self) -> None:
        """A run that offered no load has no distribution to report.

        It must not divide by zero, and it must not be reportable either —
        VALID here would let an empty run reach the Pareto frontier.
        """
        assessment = assess_validity(
            **healthy(dispatch_lags_s=[], requests_scheduled=0, requests_completed=0)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.INCOMPLETE
        assert any("zero requests" in n for n in assessment.notes)


class TestEngineFailure:
    def test_high_failure_rate_is_an_engine_error(self) -> None:
        assessment = assess_validity(
            **healthy(requests_completed=900, requests_failed=100)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.ENGINE_ERROR

    def test_engine_error_outranks_client_saturation(self) -> None:
        """A saturated client still yields a coherent distribution for some
        load; a failing server yields latencies for requests that never ran."""
        assessment = assess_validity(
            **healthy(  # type: ignore[arg-type]
                dispatch_lags_s=[0.5] * 1000,
                requests_completed=500,
                requests_failed=500,
            )
        )
        assert assessment.validity is RunValidity.ENGINE_ERROR
        # Both problems are still recorded, so nothing is hidden by the ranking.
        assert len(assessment.notes) >= 2


class TestThrottlingAndCompleteness:
    def test_throttling_is_flagged(self) -> None:
        assessment = assess_validity(**healthy(throttled_fraction=0.25))  # type: ignore[arg-type]
        assert assessment.validity is RunValidity.THERMAL_THROTTLED
        assert any("throttle reason" in n for n in assessment.notes)

    def test_brief_throttling_is_tolerated(self) -> None:
        assessment = assess_validity(**healthy(throttled_fraction=0.01))  # type: ignore[arg-type]
        assert assessment.validity is RunValidity.VALID

    def test_low_completion_is_incomplete(self) -> None:
        assessment = assess_validity(
            **healthy(requests_completed=500, requests_failed=0)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.INCOMPLETE

    def test_saturation_outranks_incompleteness(self) -> None:
        assessment = assess_validity(
            **healthy(dispatch_lags_s=[0.5] * 1000, requests_completed=500)  # type: ignore[arg-type]
        )
        assert assessment.validity is RunValidity.CLIENT_SATURATED


class TestPolicyConfiguration:
    def test_thresholds_are_tunable(self) -> None:
        strict = SaturationPolicy(max_p99_dispatch_lag_s=0.0001)
        assessment = assess_validity(**healthy(), policy=strict)  # type: ignore[arg-type]
        assert assessment.validity is RunValidity.CLIENT_SATURATED

    def test_notes_quantify_the_violation(self) -> None:
        """A note that says only 'invalid' is useless when triaging a sweep."""
        assessment = assess_validity(
            **healthy(dispatch_lags_s=[0.25] * 1000, mean_interarrival_s=None)  # type: ignore[arg-type]
        )
        assert any("250.0 ms" in n for n in assessment.notes)
