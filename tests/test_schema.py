"""Guards on the results schema.

The schema is frozen in Phase 1. These tests exist so that a change to it is a
deliberate act with a visible diff, rather than something that happens quietly
between two sweeps and silently makes earlier results incomparable.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llmbench.schema import (
    SCHEMA_VERSION,
    ArrivalProcess,
    LengthSource,
    RunResult,
    RunValidity,
    Stats,
    WorkloadConfig,
)

from .conftest import make_stats


class TestSchemaVersion:
    def test_is_semver(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+", SCHEMA_VERSION)

    def test_pinned(self) -> None:
        """Bumping this is allowed; doing so accidentally is not.

        If this fails, confirm the change is intentional, update the constant
        here, and note the migration in METHODOLOGY.md.
        """
        assert SCHEMA_VERSION == "1.0.0"

    def test_stamped_on_records(self, run_result: RunResult) -> None:
        assert run_result.schema_version == SCHEMA_VERSION


class TestRoundTrip:
    def test_json_round_trip_is_lossless(self, run_result: RunResult) -> None:
        restored = RunResult.model_validate_json(run_result.model_dump_json())
        assert restored == run_result

    def test_round_trips_through_plain_python_objects(self, run_result: RunResult) -> None:
        """`json.loads` then validate — the path the analysis code actually uses."""
        payload = json.loads(run_result.model_dump_json())
        assert RunResult.model_validate(payload) == run_result

    def test_enums_serialise_as_strings(self, run_result: RunResult) -> None:
        payload = json.loads(run_result.model_dump_json())
        assert payload["validity"] == "valid"
        assert payload["engine"]["name"] == "vllm"
        assert payload["model"]["quantization"] == "gptq-int4"


class TestStrictness:
    def test_unknown_field_is_rejected(self, run_result: RunResult) -> None:
        """A typo'd key must fail loudly rather than be silently dropped."""
        payload = json.loads(run_result.model_dump_json())
        payload["output_token_througput"] = 1234.0  # deliberate typo
        with pytest.raises(ValidationError, match="output_token_througput"):
            RunResult.model_validate(payload)

    def test_records_are_immutable(self, run_result: RunResult) -> None:
        with pytest.raises(ValidationError):
            run_result.output_token_throughput = 9999.0  # type: ignore[misc]


class TestStatsOrdering:
    def test_rejects_out_of_order_percentiles(self) -> None:
        with pytest.raises(ValidationError, match="non-decreasing"):
            make_stats(p95=1.6, p99=0.4)

    def test_accepts_a_degenerate_single_value(self) -> None:
        s = make_stats(
            count=1, mean=0.5, std=0.0, minimum=0.5, p50=0.5, p90=0.5, p95=0.5, p99=0.5, maximum=0.5
        )
        assert s.p99 == 0.5

    def test_empty_sample_skips_ordering_check(self) -> None:
        """A run that produced no completed requests still needs a record."""
        s = Stats(count=0, mean=0.0, std=0.0, min=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0.0)
        assert s.count == 0

    def test_negative_std_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_stats(std=-0.1)


class TestRunAccounting:
    def test_completed_plus_failed_cannot_exceed_sent(self, run_result: RunResult) -> None:
        payload = json.loads(run_result.model_dump_json())
        payload["requests_completed"] = 1440
        payload["requests_failed"] = 1
        with pytest.raises(ValidationError, match="exceeds sent"):
            RunResult.model_validate(payload)

    def test_finish_cannot_precede_start(self, run_result: RunResult) -> None:
        payload = json.loads(run_result.model_dump_json())
        payload["finished_at"] = datetime(2026, 8, 10, 11, 0, 0, tzinfo=UTC).isoformat()
        with pytest.raises(ValidationError, match="precedes"):
            RunResult.model_validate(payload)

    @pytest.mark.parametrize(
        "validity",
        [
            RunValidity.CLIENT_SATURATED,
            RunValidity.ENGINE_ERROR,
            RunValidity.THERMAL_THROTTLED,
            RunValidity.INCOMPLETE,
        ],
    )
    def test_only_valid_runs_are_reportable(
        self, run_result: RunResult, validity: RunValidity
    ) -> None:
        """Invalid runs are kept, never reported.

        Why a configuration could not be measured is itself a finding, so the
        record survives; the Pareto frontier must not see it.
        """
        payload = json.loads(run_result.model_dump_json())
        payload["validity"] = validity.value
        assert RunResult.model_validate(payload).is_reportable is False

    def test_valid_run_is_reportable(self, run_result: RunResult) -> None:
        assert run_result.is_reportable is True


class TestWorkloadConsistency:
    def test_poisson_requires_a_request_rate(self, stats: Stats) -> None:
        with pytest.raises(ValidationError, match="requires an explicit request_rate_rps"):
            WorkloadConfig(
                arrival_process=ArrivalProcess.POISSON,
                request_rate_rps=None,
                length_source=LengthSource.SHAREGPT,
                seed=1,
                num_requests=10,
                warmup_requests=0,
                measurement_duration_s=10.0,
                ignore_eos=True,
                input_len_tokens=stats,
                output_len_tokens=stats,
            )

    def test_trace_replay_rejects_a_request_rate(self, stats: Stats) -> None:
        """Trace replay derives arrivals from the trace; an offered rate would
        silently override the very thing the run exists to demonstrate."""
        with pytest.raises(ValidationError, match="derives arrivals from the trace"):
            WorkloadConfig(
                arrival_process=ArrivalProcess.TRACE_REPLAY,
                request_rate_rps=8.0,
                length_source=LengthSource.AZURE_TRACE,
                seed=1,
                num_requests=10,
                warmup_requests=0,
                measurement_duration_s=10.0,
                ignore_eos=True,
                input_len_tokens=stats,
                output_len_tokens=stats,
            )

    def test_trace_replay_without_a_rate_is_valid(self, stats: Stats) -> None:
        wl = WorkloadConfig(
            arrival_process=ArrivalProcess.TRACE_REPLAY,
            request_rate_rps=None,
            length_source=LengthSource.AZURE_TRACE,
            seed=1,
            num_requests=10,
            warmup_requests=0,
            measurement_duration_s=10.0,
            ignore_eos=True,
            input_len_tokens=stats,
            output_len_tokens=stats,
        )
        assert wl.request_rate_rps is None


class TestConfoundsTravelWithResults:
    """A result that loses its asterisk is worse than no result."""

    def test_neighbor_gpu_busy_is_recorded(self, run_result: RunResult) -> None:
        payload = json.loads(run_result.model_dump_json())
        assert "neighbor_gpu_busy" in payload["hardware"]

    def test_clock_policy_is_recorded_even_when_unlocked(self, run_result: RunResult) -> None:
        payload = json.loads(run_result.model_dump_json())
        assert payload["hardware"]["clocks"]["locked"] is True

    def test_selected_kernel_is_recorded(self, run_result: RunResult) -> None:
        """Marlin selection is asserted from engine logs, never assumed."""
        assert run_result.engine.selected_kernel == "gptq_marlin"

    def test_image_digest_is_pinned_not_a_tag(self, run_result: RunResult) -> None:
        assert run_result.engine.image_digest.startswith("sha256:")

    def test_model_revision_is_a_commit_sha(self, run_result: RunResult) -> None:
        assert re.fullmatch(r"[0-9a-f]{40}", run_result.model.revision)
