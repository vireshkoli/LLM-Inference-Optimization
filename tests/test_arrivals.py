"""Arrival-process correctness.

If the generated schedule is not actually Poisson, every latency number in the
repository describes a workload other than the one claimed. These tests check
the distribution itself rather than merely that some numbers came out — a
benchmark's statistical core is exactly the code that most needs testing and
most often has none.

All tests are seeded, so results are deterministic and cannot flake.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as scipy_stats

from llmbench.workload.arrivals import (
    ArrivalSchedule,
    num_requests_for_duration,
    poisson_schedule,
    trace_schedule,
)


class TestPoissonDistribution:
    @pytest.mark.parametrize("rate", [0.5, 1.0, 8.0, 24.0])
    def test_inter_arrivals_are_exponential(self, rate: float) -> None:
        """Kolmogorov-Smirnov against Exponential(1/rate).

        A Poisson arrival process is *defined* by exponential gaps. This is the
        assertion that the load generator offers the load it claims to.
        """
        schedule = poisson_schedule(rate_rps=rate, num_requests=20_000, seed=42)
        gaps = np.asarray(schedule.inter_arrivals_s())

        result = scipy_stats.kstest(gaps, "expon", args=(0.0, 1.0 / rate))
        assert result.pvalue > 0.01, (
            f"inter-arrival times are not Exponential(1/{rate}): KS p={result.pvalue:.4g}"
        )

    @pytest.mark.parametrize("rate", [1.0, 8.0, 24.0])
    def test_mean_gap_matches_the_nominal_rate(self, rate: float) -> None:
        schedule = poisson_schedule(rate_rps=rate, num_requests=50_000, seed=7)
        mean_gap = float(np.mean(schedule.inter_arrivals_s()))
        assert mean_gap == pytest.approx(1.0 / rate, rel=0.02)

    @pytest.mark.parametrize("rate", [1.0, 8.0])
    def test_realized_rate_matches_nominal(self, rate: float) -> None:
        schedule = poisson_schedule(rate_rps=rate, num_requests=50_000, seed=11)
        assert schedule.realized_rate_rps == pytest.approx(rate, rel=0.02)

    def test_variance_matches_exponential(self) -> None:
        """Exponential has std == mean. A uniform-gap bug would pass a mean
        check while producing a completely different queueing regime."""
        schedule = poisson_schedule(rate_rps=10.0, num_requests=50_000, seed=3)
        gaps = np.asarray(schedule.inter_arrivals_s())
        assert float(gaps.std()) == pytest.approx(float(gaps.mean()), rel=0.05)

    def test_is_not_uniformly_spaced(self) -> None:
        """Guards against silently degenerating into a constant-rate generator."""
        schedule = poisson_schedule(rate_rps=10.0, num_requests=5_000, seed=5)
        gaps = np.asarray(schedule.inter_arrivals_s())
        assert float(gaps.std()) > 0.5 * float(gaps.mean())


class TestDeterminism:
    def test_same_seed_gives_an_identical_schedule(self) -> None:
        """The property that makes cross-configuration comparison valid.

        Every configuration under test must face a byte-identical offered load;
        otherwise differences between engines are confounded with differences in
        the workload they happened to be given.
        """
        a = poisson_schedule(rate_rps=8.0, num_requests=1_000, seed=20260810)
        b = poisson_schedule(rate_rps=8.0, num_requests=1_000, seed=20260810)
        assert a.offsets_s == b.offsets_s

    def test_different_seeds_give_different_schedules(self) -> None:
        a = poisson_schedule(rate_rps=8.0, num_requests=1_000, seed=1)
        b = poisson_schedule(rate_rps=8.0, num_requests=1_000, seed=2)
        assert a.offsets_s != b.offsets_s

    def test_schedule_is_stable_across_processes(self) -> None:
        """Pinned values from PCG64, which numpy guarantees is reproducible.

        If these drift, results generated before and after the change are not
        comparable and the schema version must be bumped.
        """
        schedule = poisson_schedule(rate_rps=2.0, num_requests=4, seed=0)
        expected = np.concatenate(
            ([0.0], np.cumsum(np.random.default_rng(0).exponential(scale=0.5, size=3)))
        )
        assert schedule.offsets_s == pytest.approx(tuple(expected))


class TestScheduleInvariants:
    def test_first_arrival_is_at_zero(self) -> None:
        """The measurement window starts immediately, not after a random gap."""
        assert poisson_schedule(rate_rps=1.0, num_requests=10, seed=1).offsets_s[0] == 0.0

    def test_offsets_are_monotonic(self) -> None:
        offsets = poisson_schedule(rate_rps=5.0, num_requests=1_000, seed=1).offsets_s
        assert all(b >= a for a, b in itertools.pairwise(offsets))

    def test_length_matches_request_count(self) -> None:
        assert len(poisson_schedule(rate_rps=5.0, num_requests=137, seed=1)) == 137

    def test_rejects_decreasing_offsets(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            ArrivalSchedule(offsets_s=(0.0, 2.0, 1.0), nominal_rate_rps=1.0)

    def test_rejects_negative_offsets(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ArrivalSchedule(offsets_s=(-1.0, 2.0), nominal_rate_rps=1.0)

    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_rejects_non_positive_rate(self, rate: float) -> None:
        with pytest.raises(ValueError, match="rate_rps must be positive"):
            poisson_schedule(rate_rps=rate, num_requests=10, seed=1)

    def test_rejects_zero_requests(self) -> None:
        with pytest.raises(ValueError, match="num_requests must be positive"):
            poisson_schedule(rate_rps=1.0, num_requests=0, seed=1)


class TestDurationSizing:
    @pytest.mark.parametrize(
        ("rate", "duration", "expected"),
        [(8.0, 180.0, 1440), (1.0, 30.0, 30), (24.0, 180.0, 4320), (0.5, 10.0, 5)],
    )
    def test_request_count_for_a_window(self, rate: float, duration: float, expected: int) -> None:
        assert num_requests_for_duration(rate, duration) == expected

    def test_rounds_up_so_the_window_is_covered(self) -> None:
        assert num_requests_for_duration(3.0, 10.0) == 30
        assert num_requests_for_duration(3.3, 10.0) == 33

    def test_never_returns_zero(self) -> None:
        """A short low-rate smoke run must still issue something."""
        assert num_requests_for_duration(0.1, 1.0) == 1


class TestTraceReplay:
    def test_rebases_to_zero(self) -> None:
        schedule = trace_schedule([100.0, 100.5, 102.0])
        assert schedule.offsets_s == (0.0, 0.5, 2.0)

    def test_carries_no_nominal_rate(self) -> None:
        """A bursty trace is not described by a single rate parameter.

        Attaching one would erase the burstiness the replay exists to expose.
        """
        assert trace_schedule([0.0, 1.0]).nominal_rate_rps is None

    def test_preserves_burstiness(self) -> None:
        schedule = trace_schedule([0.0, 0.01, 0.02, 5.0, 5.01])
        gaps = schedule.inter_arrivals_s()
        assert max(gaps) / min(gaps) > 100

    def test_rejects_unsorted_timestamps(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            trace_schedule([0.0, 5.0, 1.0])

    def test_rejects_empty_trace(self) -> None:
        with pytest.raises(ValueError, match="at least one timestamp"):
            trace_schedule([])
