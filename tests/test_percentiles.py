"""Percentile computation checked against numpy as an independent oracle.

``llmbench.metrics.percentiles`` deliberately does not call numpy, so this is a
genuine cross-implementation check rather than a tautology. Percentiles are the
headline output of the benchmark; an off-by-one in the interpolation would shift
every reported p95 and p99 by a small, plausible, entirely wrong amount.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmbench.metrics.percentiles import percentile, summarize

QUANTILES = [0.0, 25.0, 50.0, 90.0, 95.0, 99.0, 99.9, 100.0]


class TestAgainstNumpyOracle:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 99, 100, 101, 1000])
    @pytest.mark.parametrize("q", QUANTILES)
    def test_matches_numpy_on_random_samples(self, n: int, q: float) -> None:
        rng = np.random.default_rng(seed=n * 7919)
        values = np.sort(rng.lognormal(0.0, 1.0, size=n))
        expected = float(np.percentile(values, q, method="linear"))
        assert percentile(values.tolist(), q) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize("q", QUANTILES)
    def test_matches_numpy_with_heavy_duplicates(self, q: float) -> None:
        """Latency samples cluster hard; ties must not shift the interpolation."""
        values = sorted([1.0] * 50 + [2.0] * 30 + [3.0] * 20)
        expected = float(np.percentile(values, q, method="linear"))
        assert percentile(values, q) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("q", QUANTILES)
    def test_matches_numpy_on_a_long_tail(self, q: float) -> None:
        """The shape that matters: mostly fast, a few very slow requests."""
        values = sorted([0.01] * 990 + [5.0] * 10)
        expected = float(np.percentile(values, q, method="linear"))
        assert percentile(values, q) == pytest.approx(expected, rel=1e-12)


class TestPercentileEdgeCases:
    def test_single_value(self) -> None:
        assert percentile([42.0], 99.0) == 42.0

    def test_two_values_interpolates(self) -> None:
        assert percentile([0.0, 10.0], 50.0) == pytest.approx(5.0)

    def test_bounds_are_min_and_max(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 100.0) == 4.0

    def test_empty_is_an_error_not_a_zero(self) -> None:
        """Silently returning 0.0 would look like an impossibly fast run."""
        with pytest.raises(ValueError, match="empty sequence"):
            percentile([], 50.0)

    @pytest.mark.parametrize("q", [-1.0, 100.1, 1000.0])
    def test_rejects_out_of_range_quantiles(self, q: float) -> None:
        with pytest.raises(ValueError, match=r"q must be in \[0, 100\]"):
            percentile([1.0, 2.0], q)


class TestSummarize:
    def test_agrees_with_numpy_across_the_board(self) -> None:
        rng = np.random.default_rng(1234)
        values = rng.lognormal(-2.0, 0.8, size=5000).tolist()
        stats = summarize(values)
        arr = np.asarray(values)

        assert stats.count == 5000
        assert stats.mean == pytest.approx(float(arr.mean()), rel=1e-12)
        assert stats.std == pytest.approx(float(arr.std()), rel=1e-12)
        assert stats.min == pytest.approx(float(arr.min()), rel=1e-12)
        assert stats.max == pytest.approx(float(arr.max()), rel=1e-12)
        for name, q in [("p50", 50.0), ("p90", 90.0), ("p95", 95.0), ("p99", 99.0)]:
            assert getattr(stats, name) == pytest.approx(
                float(np.percentile(arr, q, method="linear")), rel=1e-12
            )

    def test_uses_population_std_not_sample_std(self) -> None:
        """These are complete observations of a run, not a draw from it."""
        values = [1.0, 2.0, 3.0, 4.0]
        assert summarize(values).std == pytest.approx(float(np.std(values)))

    def test_empty_sample_yields_a_valid_zero_record(self) -> None:
        """A run that completed nothing still needs a storable result.

        Raising here would mean the fact that a configuration completed zero
        requests could not be recorded — and that fact is a finding.
        """
        stats = summarize([])
        assert stats.count == 0
        assert stats.p99 == 0.0

    def test_single_observation_has_zero_spread(self) -> None:
        stats = summarize([0.25])
        assert stats.count == 1
        assert stats.std == 0.0
        assert stats.min == stats.max == stats.p99 == 0.25

    def test_output_validates_against_the_schema(self) -> None:
        """summarize must never emit a Stats the schema would reject."""
        rng = np.random.default_rng(99)
        stats = summarize(rng.exponential(0.1, size=250).tolist())
        assert stats.min <= stats.p50 <= stats.p90 <= stats.p95 <= stats.p99 <= stats.max
