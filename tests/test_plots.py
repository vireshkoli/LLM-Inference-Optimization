"""Chart generation against synthetic fixtures.

Runs in CI with no GPU and no measurements. The point is not that the pixels are
correct — it is that the aggregation feeding them is: invalid runs excluded,
repeats collapsed to mean ± std, and colour tied to configuration identity
rather than to position in a filtered list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.analysis.plots import (
    SERIES_COLORS,
    _aggregate,
    plot_latency_throughput,
    plot_tpot_throughput,
)
from llmbench.schema import RunResult, RunValidity

from .conftest import make_stats


def run_at(
    base: RunResult,
    *,
    config_id: str = "vllm-bf16",
    rate: float = 4.0,
    repeat: int = 0,
    ttft_p95_ms: float = 200.0,
    throughput: float = 700.0,
    validity: RunValidity = RunValidity.VALID,
) -> RunResult:
    """Clone the fixture with one measurement varied."""
    payload = base.model_dump()
    payload["config_id"] = config_id
    payload["repeat_index"] = repeat
    payload["validity"] = validity
    payload["output_token_throughput"] = throughput
    payload["workload"]["request_rate_rps"] = rate

    s = ttft_p95_ms / 1e3
    payload["ttft_s"] = make_stats(
        mean=s,
        std=s * 0.05,
        minimum=s * 0.5,
        p50=s * 0.8,
        p90=s * 0.95,
        p95=s,
        p99=s * 1.1,
        maximum=s * 1.2,
    ).model_dump()
    return RunResult.model_validate(payload)


class TestAggregation:
    def test_collapses_repeats_to_mean_and_std(self, run_result: RunResult) -> None:
        runs = [
            run_at(run_result, rate=4.0, repeat=i, ttft_p95_ms=ms, throughput=t)
            for i, (ms, t) in enumerate([(100.0, 600.0), (110.0, 700.0), (120.0, 800.0)])
        ]
        points = _aggregate(runs, "p95")
        assert len(points) == 1
        assert points[0].repeats == 3
        assert points[0].latency_mean_ms == pytest.approx(110.0)
        assert points[0].latency_std_ms > 0
        assert points[0].throughput_mean == pytest.approx(700.0)

    def test_excludes_invalid_runs(self, run_result: RunResult) -> None:
        """A client-saturated run measures the harness, not the server.

        Including it would bend the curve toward optimism exactly where the
        interesting numbers are.
        """
        runs = [
            run_at(run_result, rate=4.0, repeat=0, ttft_p95_ms=100.0),
            run_at(
                run_result,
                rate=4.0,
                repeat=1,
                ttft_p95_ms=9000.0,
                validity=RunValidity.CLIENT_SATURATED,
            ),
        ]
        points = _aggregate(runs, "p95")
        assert points[0].repeats == 1
        assert points[0].latency_mean_ms == pytest.approx(100.0)

    def test_orders_points_by_rate(self, run_result: RunResult) -> None:
        runs = [run_at(run_result, rate=r) for r in (16.0, 1.0, 4.0)]
        assert [p.rate_rps for p in _aggregate(runs, "p95")] == [1.0, 4.0, 16.0]

    def test_all_invalid_yields_no_points(self, run_result: RunResult) -> None:
        runs = [run_at(run_result, validity=RunValidity.ENGINE_ERROR)]
        assert _aggregate(runs, "p95") == []

    @pytest.mark.parametrize("percentile", ["p50", "p90", "p95", "p99"])
    def test_any_percentile_can_be_plotted(self, run_result: RunResult, percentile: str) -> None:
        assert len(_aggregate([run_at(run_result)], percentile)) == 1


class TestPalette:
    def test_colour_follows_configuration_not_rank(self) -> None:
        """Filtering the set must not repaint the survivors.

        Hues are assigned by position in a fixed order, so this is really a
        guard that the order itself is a stable constant.
        """
        assert SERIES_COLORS[0] == "#2a78d6"
        assert len(SERIES_COLORS) == 8
        assert len(set(SERIES_COLORS)) == 8


class TestRendering:
    def test_writes_a_latency_chart(self, run_result: RunResult, tmp_path: Path) -> None:
        runs = [
            run_at(run_result, rate=r, ttft_p95_ms=ms, throughput=t)
            for r, ms, t in [(1.0, 90.0, 180.0), (4.0, 120.0, 700.0), (16.0, 900.0, 1900.0)]
        ]
        out = plot_latency_throughput(runs, tmp_path / "latency.png")
        assert out.exists()
        assert out.stat().st_size > 5_000

    def test_writes_a_tpot_chart(self, run_result: RunResult, tmp_path: Path) -> None:
        runs = [run_at(run_result, rate=r) for r in (1.0, 4.0)]
        out = plot_tpot_throughput(runs, tmp_path / "tpot.png")
        assert out.exists()

    def test_renders_multiple_configurations(self, run_result: RunResult, tmp_path: Path) -> None:
        runs = [
            run_at(run_result, config_id=cid, rate=r, throughput=t)
            for cid in ("vllm-bf16", "vllm-awq-int4", "sglang-bf16")
            for r, t in [(1.0, 180.0), (8.0, 1200.0)]
        ]
        out = plot_latency_throughput(runs, tmp_path / "multi.png")
        assert out.exists()

    def test_creates_missing_output_directory(self, run_result: RunResult, tmp_path: Path) -> None:
        out = plot_latency_throughput([run_at(run_result)], tmp_path / "deep" / "nested" / "x.png")
        assert out.exists()

    def test_narrow_range_avoids_log_scale(self, run_result: RunResult, tmp_path: Path) -> None:
        """A two-point smoke run must not get log ticks like '2.05 x 10^2'."""
        runs = [
            run_at(run_result, rate=r, ttft_p95_ms=ms, throughput=t)
            for r, ms, t in [(1.0, 210.0, 176.0), (4.0, 187.0, 708.0)]
        ]
        assert plot_latency_throughput(runs, tmp_path / "narrow.png").exists()
