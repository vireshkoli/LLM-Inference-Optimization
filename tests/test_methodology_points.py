"""How a declared methodology run becomes the points to measure.

The precedence rule here is not cosmetic. Getting it backwards cost a full
sweep: six requested concurrencies all ran at the matrix's declared 64 and
overwrote a single set of result files, and nothing in the output said so
because every run was internally consistent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llmbench.config import MethodologyRun
from llmbench.runner import RatePoint, SweepRunner
from llmbench.schema import ArrivalProcess
from llmbench.workload.arrivals import ArrivalSchedule


def points(runner: SweepRunner, entry: MethodologyRun, override: int | None) -> list[RatePoint]:
    stub = RatePoint(rate_rps=4.0, schedule=ArrivalSchedule((0.0,), 4.0), specs=())
    with patch.object(SweepRunner, "rate_point", return_value=stub):
        return runner._methodology_points(
            entry,
            trace_path=__import__("pathlib").Path("unused.csv"),
            matched_rate_rps=4.0,
            closed_loop_concurrency=override,
        )


@pytest.fixture
def closed_loop() -> MethodologyRun:
    return MethodologyRun(
        id="vllm-bf16-closed-loop",
        config="vllm-bf16",
        arrival_process=ArrivalProcess.CLOSED_LOOP,
        concurrency=64,
    )


class TestConcurrencyPrecedence:
    def test_explicit_override_beats_the_matrix(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        """Regression. `entry.concurrency or override` ignored the caller
        whenever the matrix declared a value, so a sweep over concurrencies
        silently ran every point at the matrix's own number."""
        assert points(sweep_runner, closed_loop, 40)[0].concurrency == 40

    def test_matrix_value_used_when_no_override(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        assert points(sweep_runner, closed_loop, None)[0].concurrency == 64

    def test_missing_everywhere_fails_loudly(self, sweep_runner: SweepRunner) -> None:
        """Concurrency *is* the offered load for this generator, so there is no
        default that would be meaningful."""
        entry = MethodologyRun(
            id="no-concurrency",
            config="vllm-bf16",
            arrival_process=ArrivalProcess.CLOSED_LOOP,
        )
        with pytest.raises(ValueError, match="neither the matrix nor the caller"):
            points(sweep_runner, entry, None)


class TestArrivalProcessRouting:
    def test_closed_loop_point_carries_no_rate(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        """A closed loop has no offered rate; carrying one would put it on the
        frontier, which keys on exactly that field."""
        point = points(sweep_runner, closed_loop, 40)[0]
        assert point.rate_rps is None
        assert point.arrival_process is ArrivalProcess.CLOSED_LOOP

    def test_drift_canary_sweeps_the_whole_ladder(self, sweep_runner: SweepRunner) -> None:
        """Its value lies entirely in being identical to the original run."""
        entry = MethodologyRun(id="drift-canary", repeat_of="vllm-bf16")
        got = points(sweep_runner, entry, None)
        assert len(got) == len(sweep_runner.config.workload.request_rates_rps)
        assert all(p.arrival_process is ArrivalProcess.POISSON for p in got)
