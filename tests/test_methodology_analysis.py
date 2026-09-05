"""Comparisons for the methodology runs.

Each of these turns an exhibit into a number. The tests pin the directions,
because a sign error here would report coordinated omission backwards — that a
closed-loop generator *overstates* the tail — which is a confident, wrong,
publishable-looking result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llmbench.analysis.methodology import drift_comparison, process_comparison
from llmbench.schema import ArrivalProcess, RunResult

from .conftest import make_stats


def run_at(
    base: RunResult,
    *,
    config_id: str | None = None,
    process: ArrivalProcess = ArrivalProcess.POISSON,
    rate: float | None = 4.0,
    concurrency: int | None = None,
    ttft_ms: tuple[float, float, float] = (100.0, 200.0, 300.0),
    throughput: float = 1000.0,
    started: datetime | None = None,
) -> RunResult:
    """Clone the fixture with the fields these comparisons actually read."""
    p50, p95, p99 = (v / 1e3 for v in ttft_ms)
    payload = base.model_dump()
    payload["config_id"] = config_id or base.config_id
    payload["workload"] = {
        **payload["workload"],
        "arrival_process": process,
        "request_rate_rps": rate,
        "concurrency": concurrency,
    }
    payload["ttft_s"] = make_stats(
        mean=p50, minimum=p50 / 2, p50=p50, p90=p95, p95=p95, p99=p99, maximum=p99
    ).model_dump()
    payload["output_token_throughput"] = throughput
    if started is not None:
        payload["started_at"] = started
        payload["finished_at"] = started + timedelta(minutes=3)
    return RunResult.model_validate(payload)


class TestCoordinatedOmission:
    def test_closed_loop_understating_the_tail_reports_above_one(
        self, run_result: RunResult
    ) -> None:
        """The direction that matters. A closed-loop generator under-samples the
        slow periods, so its reported p99 is *smaller* than the truth."""
        runs = [
            run_at(
                run_result,
                ttft_ms=(90.0, 150.0, 200.0),
                process=ArrivalProcess.CLOSED_LOOP,
                rate=None,
                concurrency=64,
            ),
            run_at(run_result, ttft_ms=(100.0, 400.0, 900.0)),
        ]
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.CLOSED_LOOP,
            baseline_config_id=run_result.config_id,
            baseline_rate_rps=4.0,
        )
        assert cmp is not None
        assert cmp.understatement_ratio("p99") == pytest.approx(4.5)
        assert cmp.understatement_ratio("p95") == pytest.approx(400.0 / 150.0)

    def test_comparable_when_throughput_matches(self, run_result: RunResult) -> None:
        runs = [
            run_at(
                run_result,
                process=ArrivalProcess.CLOSED_LOOP,
                rate=None,
                concurrency=64,
                throughput=1020.0,
            ),
            run_at(run_result, throughput=1000.0),
        ]
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.CLOSED_LOOP,
            baseline_config_id=run_result.config_id,
            baseline_rate_rps=4.0,
        )
        assert cmp is not None
        assert cmp.comparable is True

    def test_not_comparable_when_the_server_did_different_work(self, run_result: RunResult) -> None:
        """A large throughput gap invalidates the latency comparison rather than
        explaining it — the exhibit must say so instead of quoting a ratio."""
        runs = [
            run_at(
                run_result,
                process=ArrivalProcess.CLOSED_LOOP,
                rate=None,
                concurrency=64,
                throughput=400.0,
            ),
            run_at(run_result, throughput=1000.0),
        ]
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.CLOSED_LOOP,
            baseline_config_id=run_result.config_id,
            baseline_rate_rps=4.0,
        )
        assert cmp is not None
        assert cmp.comparable is False

    def test_missing_exhibit_yields_none_not_an_error(self, run_result: RunResult) -> None:
        """An unrun methodology run should leave a gap in the report, not break
        its generation."""
        assert (
            process_comparison(
                [run_at(run_result)],
                process=ArrivalProcess.CLOSED_LOOP,
                baseline_config_id=run_result.config_id,
                baseline_rate_rps=4.0,
            )
            is None
        )

    def test_missing_baseline_yields_none(self, run_result: RunResult) -> None:
        runs = [run_at(run_result, process=ArrivalProcess.CLOSED_LOOP, rate=None, concurrency=64)]
        assert (
            process_comparison(
                runs,
                process=ArrivalProcess.CLOSED_LOOP,
                baseline_config_id=run_result.config_id,
                baseline_rate_rps=4.0,
            )
            is None
        )


class TestTraceReplay:
    def test_bursty_trace_showing_a_worse_tail_reports_below_one(
        self, run_result: RunResult
    ) -> None:
        """Real arrivals are burstier than Poisson, so the replay's tail should
        be *worse*. Understatement below 1.0 is the correct direction here."""
        runs = [
            run_at(
                run_result,
                process=ArrivalProcess.TRACE_REPLAY,
                rate=None,
                ttft_ms=(110.0, 600.0, 1400.0),
            ),
            run_at(run_result, ttft_ms=(100.0, 400.0, 900.0)),
        ]
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.TRACE_REPLAY,
            baseline_config_id=run_result.config_id,
            baseline_rate_rps=4.0,
        )
        assert cmp is not None
        assert cmp.understatement_ratio("p99") < 1.0


class TestDriftCanary:
    def test_matching_rerun_is_within_noise(self, run_result: RunResult) -> None:
        t0 = datetime(2026, 8, 10, 10, 0, tzinfo=UTC)
        runs = [
            run_at(run_result, ttft_ms=(100.0, 200.0, 300.0), started=t0),
            run_at(run_result, ttft_ms=(100.0, 210.0, 300.0), started=t0 + timedelta(minutes=5)),
            run_at(run_result, ttft_ms=(100.0, 205.0, 300.0), started=t0 + timedelta(hours=9)),
            run_at(
                run_result,
                ttft_ms=(100.0, 208.0, 300.0),
                started=t0 + timedelta(hours=9, minutes=4),
            ),
        ]
        drift = drift_comparison(
            runs, config_id=run_result.config_id, canary_label="drift-canary", rate_rps=4.0
        )
        assert drift is not None
        assert drift.within_noise is True

    def test_large_drift_is_flagged(self, run_result: RunResult) -> None:
        """If this fails in a real sweep, every cross-configuration claim in the
        report is weakened — which is the whole reason the canary is run."""
        t0 = datetime(2026, 8, 10, 10, 0, tzinfo=UTC)
        runs = [
            run_at(run_result, ttft_ms=(100.0, 200.0, 300.0), started=t0),
            run_at(run_result, ttft_ms=(100.0, 201.0, 300.0), started=t0 + timedelta(minutes=5)),
            run_at(run_result, ttft_ms=(100.0, 900.0, 1200.0), started=t0 + timedelta(hours=9)),
            run_at(
                run_result,
                ttft_ms=(100.0, 890.0, 1200.0),
                started=t0 + timedelta(hours=9, minutes=4),
            ),
        ]
        drift = drift_comparison(
            runs, config_id=run_result.config_id, canary_label="drift-canary", rate_rps=4.0
        )
        assert drift is not None
        assert drift.within_noise is False
        assert drift.ttft_drift_ms > 0

    def test_ordinary_repeats_are_not_mistaken_for_a_canary(self, run_result: RunResult) -> None:
        """Regression. Three repeats minutes apart were split on their largest
        gap and reported +77.6 ms of drift against a standard deviation of zero,
        because a single-run group has no variance to compare against."""
        t0 = datetime(2026, 8, 19, 11, 4, tzinfo=UTC)
        runs = [
            run_at(run_result, ttft_ms=(100.0, 200.8, 300.0), started=t0),
            run_at(run_result, ttft_ms=(100.0, 274.8, 300.0), started=t0 + timedelta(minutes=4)),
            run_at(run_result, ttft_ms=(100.0, 282.1, 300.0), started=t0 + timedelta(minutes=8)),
        ]
        assert (
            drift_comparison(
                runs, config_id=run_result.config_id, canary_label="drift-canary", rate_rps=4.0
            )
            is None
        )

    def test_single_measurement_yields_none(self, run_result: RunResult) -> None:
        assert (
            drift_comparison(
                [run_at(run_result)],
                config_id=run_result.config_id,
                canary_label="drift-canary",
                rate_rps=4.0,
            )
            is None
        )
