"""Generated blocks in README and REPORT.

Several of these guard a number that was once typed into prose and then went
stale while the tables beside it regenerated -- the reason they are generated now.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from llmbench.cli import _DOCUMENT_BLOCKS
from llmbench.report.render import (
    _signed,
    cost_note,
    crossvalidation_table,
    findings_summary,
    load_quality,
    load_runs,
    methodology_summary,
)

ROOT = Path(__file__).resolve().parent.parent


def metric(
    name: str, ours: float, upstream: float, *, sensitive: bool = False
) -> dict[str, object]:
    delta = (ours / upstream - 1) * 100
    return {
        "metric": name,
        "ours": ours,
        "upstream": upstream,
        "unit": "ms",
        "delta_pct": delta,
        "agrees": abs(delta) <= 5.0,
        "length_sensitive": sensitive,
    }


def payload(*, matched: bool, throughput: tuple[float, float]) -> dict[str, object]:
    tokens = 256.0 if matched else 300.0
    upstream_tokens = 256.0 if matched else 190.0
    return {
        "config_id": "vllm-bf16",
        "rate_rps": 4.0,
        "matched_lengths": matched,
        "fixed_input_tokens": 256,
        "fixed_output_tokens": 256,
        "output_tokens_per_request": {
            "ours": tokens,
            "upstream": upstream_tokens,
        },
        "metrics": [
            metric("TTFT mean", 170.0, 168.0),
            metric("TPOT mean", 40.0, 41.0),
            metric(
                "E2E mean",
                170.0 + 40.0 * (tokens - 1),
                168.0 + 41.0 * (upstream_tokens - 1),
                sensitive=True,
            ),
            metric("Output throughput", *throughput, sensitive=True),
        ],
    }


class TestCrossValidationTable:
    def test_a_miss_with_matched_lengths_is_a_real_miss(self, tmp_path: Path) -> None:
        """Regression. Matched-length throughput was flagged "no*" (explained
        by a length difference) although the lengths were identical."""
        (tmp_path / "xv_matched.json").write_text(
            json.dumps(payload(matched=True, throughput=(927.6, 977.3)))
        )
        table = crossvalidation_table(tmp_path / "xv.json")
        assert "| Output throughput |" in table
        assert "**NO** |" in table
        assert "no* |" not in table
        assert "Outside ±5 % with matched lengths: Output throughput" in table

    def test_a_length_sensitive_miss_with_different_lengths_is_starred(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "xv.json").write_text(
            json.dumps(payload(matched=False, throughput=(1028.0, 685.0)))
        )
        table = crossvalidation_table(tmp_path / "xv.json")
        assert "no* |" in table

    def test_the_consistency_check_is_computed_not_typed(self, tmp_path: Path) -> None:
        """Regression. The footnote stated "within 0.05 %" for a check that
        came out at 0.11 % on this harness's side."""
        data = payload(matched=False, throughput=(1028.0, 685.0))
        (tmp_path / "xv.json").write_text(json.dumps(data))
        table = crossvalidation_table(tmp_path / "xv.json")
        assert "to within 0.00 % here and 0.00 % upstream" in table
        assert "1.494" not in table


class TestSmallFormatting:
    def test_no_negative_zero(self) -> None:
        assert _signed(-0.004) == "+0.0"
        assert _signed(-0.06) == "-0.1"
        assert _signed(3.21) == "+3.2"

    def test_cost_note_states_the_alternates(self) -> None:
        note = cost_note(0.40, "RunPod", "https://example", "2026-08-10", [("Secure", 0.79)])
        assert "Secure at $0.79 (x1.98)" in note


class TestDocumentMarkers:
    """Every block a document is meant to carry has markers, and no marker is
    left without a generator. `llmbench report` passes each document its full
    list, so a deleted marker raises instead of leaving a stale copy."""

    @pytest.mark.parametrize("doc", sorted(_DOCUMENT_BLOCKS))
    def test_markers_and_generators_agree(self, doc: str) -> None:
        text = (ROOT / doc).read_text()
        begins = set(re.findall(r"<!--\s*BEGIN:([a-z0-9-]+)\s*-->", text))
        ends = set(re.findall(r"<!--\s*END:([a-z0-9-]+)\s*-->", text))
        assert begins == ends
        assert begins == set(_DOCUMENT_BLOCKS[doc])


@pytest.fixture(scope="module")
def runs() -> list[object]:
    return list(load_runs(ROOT / "results" / "runs"))


class TestSummariesOnTheCommittedData:
    """Structure only: the values move whenever results are added."""

    def test_findings_name_every_format(self, runs: list[object]) -> None:
        text = findings_summary(runs, load_quality(ROOT / "results" / "quality"))  # type: ignore[arg-type]
        assert "INT8-W8A8 shrinks the declared weights" in text
        assert "x GPTQ" in text
        assert "x AWQ" in text
        assert "the highest rate all three quantized formats are still valid" in text

    def test_methodology_summary_has_a_row_per_exhibit(self, runs: list[object]) -> None:
        text = methodology_summary(
            runs,  # type: ignore[arg-type]
            load_quality(ROOT / "results" / "quality"),
            crossvalidation_path=ROOT / "results" / "crossvalidation.json",
            baseline_config_id="vllm-bf16",
            rate_rps=4.0,
        )
        for row in (
            "Cross-validation vs `vllm bench serve`",
            "Closed-loop exhibit",
            "Azure trace replay",
            "Drift canary",
            "Same checkpoint, two engines",
            "INT8 ceiling",
        ):
            assert row in text
        assert "count dispersion" not in text
