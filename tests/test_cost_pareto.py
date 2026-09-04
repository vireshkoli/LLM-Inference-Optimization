"""Cost model and Pareto frontier.

Pure arithmetic over committed results, so it runs in CI with no GPU. Sign
errors here are the dangerous kind: they invert the frontier and produce a
confident, wrong recommendation that looks exactly like a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.analysis.cost import (
    ConfigOperatingPoint,
    best_under_sla,
    cost_per_million_tokens,
)
from llmbench.analysis.pareto import ParetoPoint, dominates, pareto_frontier
from llmbench.report.render import MissingBlockError, pareto_markers, render_into


def point(cfg: str, q: float, lat: float, cost: float) -> ParetoPoint:
    return ParetoPoint(config_id=cfg, quality=q, latency_s=lat, cost_per_1m_usd=cost)


def op(cfg: str, rate: float, thr: float, ttft: float, tpot: float = 0.05) -> ConfigOperatingPoint:
    return ConfigOperatingPoint(
        config_id=cfg,
        rate_rps=rate,
        throughput_tokens_s=thr,
        throughput_std=0.0,
        ttft_p95_s=ttft,
        tpot_p95_s=tpot,
        repeats=3,
        cost_per_1m_usd=cost_per_million_tokens(thr, 0.40),
    )


class TestCostArithmetic:
    def test_known_value(self) -> None:
        """1000 tok/s at $0.40/h -> 3.6M tokens/hour -> $0.1111 per 1M."""
        assert cost_per_million_tokens(1000.0, 0.40) == pytest.approx(0.40 / 3.6e6 * 1e6)

    def test_doubling_throughput_halves_cost(self) -> None:
        a = cost_per_million_tokens(1000.0, 0.40)
        b = cost_per_million_tokens(2000.0, 0.40)
        assert b == pytest.approx(a / 2)

    def test_price_scales_cost_linearly(self) -> None:
        """The basis of the claim that ranking is price-invariant."""
        a = cost_per_million_tokens(1000.0, 0.40)
        b = cost_per_million_tokens(1000.0, 0.80)
        assert b == pytest.approx(2 * a)

    def test_ranking_is_invariant_to_gpu_price(self) -> None:
        """Every configuration runs on the same GPU, so price is a scalar.

        There is no crossover rate at which a different configuration wins;
        a price-sensitivity table would imply a result that does not exist.
        """
        thr = [265.0, 596.0, 1112.0]
        cheap = [cost_per_million_tokens(t, 0.40) for t in thr]
        dear = [cost_per_million_tokens(t, 1.28) for t in thr]
        assert sorted(range(3), key=lambda i: cheap[i]) == sorted(range(3), key=lambda i: dear[i])

    def test_zero_throughput_is_an_error_not_infinity(self) -> None:
        """Infinity would quietly place a dead configuration on a chart."""
        with pytest.raises(ValueError, match="must be positive"):
            cost_per_million_tokens(0.0, 0.40)


class TestSlaSelection:
    def points(self) -> dict[str, list[ConfigOperatingPoint]]:
        return {
            "bf16": [op("bf16", 1, 265, 0.142), op("bf16", 4, 1028, 0.253)],
            "awq-int4": [
                op("awq-int4", 1, 286, 0.104),
                op("awq-int4", 4, 1112, 0.194),
                op("awq-int4", 8, 2035, 3.288),
            ],
        }

    def test_picks_highest_throughput_within_budget(self) -> None:
        best = best_under_sla(self.points(), max_ttft_p95_s=0.30)
        assert best["awq-int4"].rate_rps == 4
        assert best["bf16"].rate_rps == 4

    def test_a_tighter_budget_excludes_points(self) -> None:
        """The SLA, not the GPU price, is what reorders the frontier."""
        best = best_under_sla(self.points(), max_ttft_p95_s=0.15)
        assert best["awq-int4"].rate_rps == 1
        assert best["bf16"].rate_rps == 1

    def test_a_generous_budget_admits_the_saturating_point(self) -> None:
        best = best_under_sla(self.points(), max_ttft_p95_s=5.0)
        assert best["awq-int4"].rate_rps == 8

    def test_a_configuration_with_no_admissible_point_is_omitted(self) -> None:
        """Silently reporting its slowest point would imply it met the SLA."""
        best = best_under_sla(self.points(), max_ttft_p95_s=0.05)
        assert best == {}

    def test_tpot_budget_is_applied_too(self) -> None:
        pts = {"x": [op("x", 1, 500, 0.10, tpot=0.20)]}
        assert best_under_sla(pts, max_ttft_p95_s=1.0) != {}
        assert best_under_sla(pts, max_ttft_p95_s=1.0, max_tpot_p95_s=0.05) == {}


class TestParetoDomination:
    def test_strictly_better_dominates(self) -> None:
        assert dominates(point("a", 0.75, 0.10, 0.10), point("b", 0.70, 0.20, 0.20))

    def test_equal_points_do_not_dominate_each_other(self) -> None:
        a, b = point("a", 0.75, 0.10, 0.10), point("b", 0.75, 0.10, 0.10)
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_a_tradeoff_is_not_domination(self) -> None:
        """Better quality but worse cost is a choice, not a winner."""
        a, b = point("a", 0.80, 0.10, 0.30), point("b", 0.70, 0.10, 0.10)
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_quality_is_maximised_not_minimised(self) -> None:
        """A sign error here silently inverts the whole frontier."""
        better, worse = point("a", 0.90, 0.10, 0.10), point("b", 0.10, 0.10, 0.10)
        assert dominates(better, worse)
        assert not dominates(worse, better)

    def test_latency_and_cost_are_minimised(self) -> None:
        fast, slow = point("a", 0.75, 0.05, 0.10), point("b", 0.75, 0.50, 0.10)
        assert dominates(fast, slow)
        cheap, dear = point("a", 0.75, 0.10, 0.05), point("b", 0.75, 0.10, 0.50)
        assert dominates(cheap, dear)


class TestFrontier:
    def test_splits_frontier_from_dominated(self) -> None:
        pts = [
            point("int4", 0.745, 0.104, 0.10),  # best everywhere
            point("bf16", 0.745, 0.142, 0.42),  # dominated: slower and dearer
            point("int8", 0.758, 0.097, 0.12),  # better quality and latency
        ]
        frontier, dominated = pareto_frontier(pts)
        ids = {p.config_id for p in frontier}
        assert "int8" in ids
        assert {p.config_id for p in dominated} == {"bf16"}

    def test_dominated_points_are_returned_not_discarded(self) -> None:
        """Seeing what lost is most of the argument for what won."""
        pts = [point("good", 0.9, 0.1, 0.1), point("bad", 0.5, 0.9, 0.9)]
        frontier, dominated = pareto_frontier(pts)
        assert len(frontier) + len(dominated) == len(pts)

    def test_a_single_point_is_its_own_frontier(self) -> None:
        frontier, dominated = pareto_frontier([point("only", 0.7, 0.1, 0.1)])
        assert len(frontier) == 1
        assert dominated == []

    def test_empty_input(self) -> None:
        assert pareto_frontier([]) == ([], [])


class TestReportBlocks:
    """`make report` must be idempotent, or "no number here was typed by hand"
    is a claim rather than a checkable property."""

    @staticmethod
    def _doc(tmp_path: Path, body: str) -> Path:
        p = tmp_path / "DOC.md"
        p.write_text(body)
        return p

    def test_fills_an_empty_placeholder(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "intro\n\n<!-- BEGIN:tbl -->\n<!-- END:tbl -->\n\nouttro\n")
        render_into(doc, {"tbl": "| a |\n|---|"})
        assert "| a |" in doc.read_text()
        assert doc.read_text().startswith("intro")
        assert doc.read_text().endswith("outtro\n")

    def test_second_render_is_byte_identical(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "<!-- BEGIN:tbl -->\n<!-- END:tbl -->\n")
        render_into(doc, {"tbl": "x"})
        once = doc.read_text()
        render_into(doc, {"tbl": "x"})
        assert doc.read_text() == once

    def test_replaces_stale_content_rather_than_appending(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "<!-- BEGIN:tbl -->\nOLD NUMBER\n<!-- END:tbl -->\n")
        render_into(doc, {"tbl": "NEW NUMBER"})
        assert "OLD NUMBER" not in doc.read_text()
        assert "NEW NUMBER" in doc.read_text()

    def test_prose_outside_markers_is_untouched(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "hand written\n<!-- BEGIN:tbl -->\n<!-- END:tbl -->\nalso hand\n")
        render_into(doc, {"tbl": "gen"})
        text = doc.read_text()
        assert "hand written" in text
        assert "also hand" in text

    def test_missing_marker_raises_rather_than_silently_dropping(self, tmp_path: Path) -> None:
        """A dropped block would leave a stale hand-written table looking generated."""
        doc = self._doc(tmp_path, "no markers here\n")
        with pytest.raises(MissingBlockError, match="tbl"):
            render_into(doc, {"tbl": "x"})

    def test_unrelated_blocks_are_left_alone(self, tmp_path: Path) -> None:
        doc = self._doc(
            tmp_path,
            "<!-- BEGIN:a -->\nkeep\n<!-- END:a -->\n<!-- BEGIN:b -->\n<!-- END:b -->\n",
        )
        render_into(doc, {"b": "new"})
        assert "keep" in doc.read_text()


class TestParetoMarkers:
    def test_configs_without_quality_are_omitted_not_guessed(self) -> None:
        """A guessed quality score would decide the frontier — the one value a
        chart like this must never invent."""
        markers = pareto_markers([], [], gpu_hourly_usd=0.4, max_ttft_p95_s=0.5)
        assert markers == []
