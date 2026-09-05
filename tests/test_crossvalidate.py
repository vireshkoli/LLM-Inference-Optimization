"""Cross-validation against vLLM's own harness.

The comparison is only meaningful if the two harnesses are given the same
workload, so most of these tests pin the arguments rather than the arithmetic.
A silently different arrival process or dataset would turn "the harnesses
disagree" into a statement about the workload instead of about the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmbench.analysis.crossvalidate import (
    MetricAgreement,
    compare_to_upstream,
    load_upstream_result,
    upstream_args,
)
from llmbench.schema import ArrivalProcess, RunResult

from .conftest import make_stats

ARGS = upstream_args(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="/data/sharegpt.json",
    num_prompts=720,
    request_rate=4.0,
    seed=20260810,
    port=8000,
    result_dir="/out",
    result_filename="upstream.json",
)


def flag(name: str) -> str:
    return ARGS[ARGS.index(name) + 1]


class TestWorkloadIsMatched:
    def test_burstiness_is_explicitly_poisson(self) -> None:
        """The single easiest way to compare two different workloads and call
        the disagreement a bug is to leave this to a default."""
        assert flag("--burstiness") == "1.0"

    def test_rate_and_seed_are_passed_through(self) -> None:
        assert flag("--request-rate") == "4"
        assert flag("--seed") == "20260810"

    def test_ignore_eos_is_set(self) -> None:
        """Ours enforces output length this way; without it the two harnesses
        would generate different numbers of tokens."""
        assert "--ignore-eos" in ARGS

    def test_completions_endpoint_matches_ours(self) -> None:
        """No chat template on either side, so the prompt sent is the prompt
        measured for both."""
        assert flag("--endpoint") == "/v1/completions"

    def test_percentiles_cover_what_we_report(self) -> None:
        assert set(flag("--metric-percentiles").split(",")) >= {"50", "95", "99"}

    def test_results_are_saved(self) -> None:
        assert "--save-result" in ARGS
        assert flag("--result-filename") == "upstream.json"


class TestUpstreamResultLoading:
    def test_reads_numeric_metrics(self, tmp_path: Path) -> None:
        p = tmp_path / "upstream.json"
        p.write_text(
            json.dumps(
                {
                    "mean_ttft_ms": 120.5,
                    "p99_ttft_ms": 400.0,
                    "output_throughput": 1000.0,
                    "date": "2026-09-05",
                }
            )
        )
        loaded = load_upstream_result(p)
        assert loaded["mean_ttft_ms"] == 120.5
        assert "date" not in loaded

    def test_missing_keys_raise_rather_than_compare_nothing(self, tmp_path: Path) -> None:
        """If upstream's schema moves, the comparison must fail loudly instead
        of quietly comparing an empty set of metrics and reporting agreement."""
        p = tmp_path / "upstream.json"
        p.write_text(json.dumps({"something_else": 1.0}))
        with pytest.raises(ValueError, match="upstream result schema has changed"):
            load_upstream_result(p)


class TestAgreement:
    def test_within_five_percent_agrees(self) -> None:
        assert MetricAgreement("TTFT mean", ours=103.0, upstream=100.0).agrees is True

    def test_beyond_five_percent_is_a_finding(self) -> None:
        a = MetricAgreement("TTFT p99", ours=140.0, upstream=100.0)
        assert a.agrees is False
        assert a.delta_pct == pytest.approx(40.0)

    def test_delta_is_signed(self) -> None:
        assert MetricAgreement("x", ours=90.0, upstream=100.0).delta_pct == pytest.approx(-10.0)

    def test_zero_upstream_does_not_divide(self) -> None:
        assert MetricAgreement("x", ours=5.0, upstream=0.0).delta_pct == 0.0


class TestComparison:
    def _run(self, base: RunResult, *, ttft_mean_ms: float, throughput: float) -> RunResult:
        payload = base.model_dump()
        payload["workload"] = {
            **payload["workload"],
            "arrival_process": ArrivalProcess.POISSON,
            "request_rate_rps": 4.0,
        }
        s = ttft_mean_ms / 1e3
        payload["ttft_s"] = make_stats(
            mean=s, minimum=s / 2, p50=s, p90=s * 2, p95=s * 2, p99=s * 3, maximum=s * 3
        ).model_dump()
        payload["output_token_throughput"] = throughput
        return RunResult.model_validate(payload)

    def test_pairs_our_metrics_with_theirs(self, run_result: RunResult) -> None:
        runs = [self._run(run_result, ttft_mean_ms=100.0, throughput=1000.0)]
        got = compare_to_upstream(
            runs,
            {"mean_ttft_ms": 98.0, "p99_ttft_ms": 300.0, "output_throughput": 1010.0},
            config_id=run_result.config_id,
            rate_rps=4.0,
        )
        names = {a.metric for a in got}
        assert {"TTFT mean", "TTFT p99", "Output throughput"} <= names
        assert all(a.agrees for a in got)

    def test_metrics_upstream_did_not_report_are_omitted(self, run_result: RunResult) -> None:
        """Absent is not zero. A missing upstream metric must not appear as a
        100% disagreement."""
        runs = [self._run(run_result, ttft_mean_ms=100.0, throughput=1000.0)]
        got = compare_to_upstream(
            runs,
            {"mean_ttft_ms": 98.0, "p99_ttft_ms": 300.0, "output_throughput": 1010.0},
            config_id=run_result.config_id,
            rate_rps=4.0,
        )
        assert "TPOT mean" not in {a.metric for a in got}

    def test_no_matching_runs_yields_empty_not_an_error(self) -> None:
        assert (
            compare_to_upstream(
                [],
                {"mean_ttft_ms": 1.0, "p99_ttft_ms": 1.0, "output_throughput": 1.0},
                config_id="vllm-bf16",
                rate_rps=4.0,
            )
            == []
        )
