"""GPU telemetry parsing and aggregation.

All parsing is exercised against captured ``nvidia-smi`` output, so it runs in
CI with no GPU. The rows below are real readings taken from the target machine —
an A40 pair where GPU 0 was running a training job and GPU 1 was idle.
"""

from __future__ import annotations

import pytest

from llmbench.telemetry.gpu import (
    INVALIDATING_REASONS,
    GpuSample,
    decode_event_reasons,
    parse_sample_line,
    summarize_samples,
)

# Captured verbatim from the target machine.
IDLE_ROW = "GPU-88098c5c-ddc0-9c3c-b403-ea19597482ba, 24, 210, 405, 21.52, 0, 0, 0x0000000000000001"
LOADED_ROW = (
    "GPU-88098c5c-ddc0-9c3c-b403-ea19597482ba, 68, 1605, 7251, 284.31, 40391, 96, "
    "0x0000000000000004"
)
THERMAL_ROW = (
    "GPU-88098c5c-ddc0-9c3c-b403-ea19597482ba, 91, 1200, 7251, 299.50, 40391, 99, "
    "0x0000000000000044"
)


class TestBitmaskDecoding:
    def test_idle_flag(self) -> None:
        assert decode_event_reasons(0x1) == ("gpu_idle",)

    def test_sw_power_cap(self) -> None:
        assert decode_event_reasons(0x4) == ("sw_power_cap",)

    def test_combined_flags(self) -> None:
        assert decode_event_reasons(0x44) == ("sw_power_cap", "hw_thermal_slowdown")

    def test_no_flags(self) -> None:
        assert decode_event_reasons(0x0) == ()


class TestThrottleClassification:
    """The distinction that decides whether any run is reportable at all."""

    def test_sw_power_cap_is_not_throttling(self) -> None:
        """Measured: an A40 at 96% util reports sw_power_cap continuously.

        It is the card holding its 300 W budget — normal operation. Counting it
        as throttling would mark every loaded run THERMAL_THROTTLED and leave
        the entire sweep unreportable.
        """
        assert parse_sample_line(LOADED_ROW).is_throttled is False

    def test_gpu_idle_is_not_throttling(self) -> None:
        assert parse_sample_line(IDLE_ROW).is_throttled is False

    def test_thermal_slowdown_is_throttling(self) -> None:
        assert parse_sample_line(THERMAL_ROW).is_throttled is True

    def test_invalidating_set_excludes_power_cap(self) -> None:
        assert "sw_power_cap" not in INVALIDATING_REASONS
        assert "gpu_idle" not in INVALIDATING_REASONS

    def test_invalidating_set_covers_thermal_and_brake(self) -> None:
        assert {
            "hw_slowdown",
            "sw_thermal_slowdown",
            "hw_thermal_slowdown",
            "hw_power_brake_slowdown",
        } == INVALIDATING_REASONS


class TestSampleParsing:
    def test_parses_a_loaded_reading(self) -> None:
        s = parse_sample_line(LOADED_ROW)
        assert s.temperature_c == 68.0
        assert s.sm_clock_mhz == 1605.0
        assert s.power_w == pytest.approx(284.31)
        assert s.memory_used_mib == 40391
        assert s.utilization_pct == 96.0
        assert s.event_reasons == ("sw_power_cap",)

    def test_carries_the_uuid(self) -> None:
        """Identity is pinned by UUID because container GPU indices are
        renumbered: `--gpus device=1` appears as index 0 inside the container."""
        assert parse_sample_line(IDLE_ROW).uuid.startswith("GPU-88098c5c")

    def test_handles_unsupported_fields(self) -> None:
        """Some fields report [N/A]; a sweep must not die over one of them."""
        row = "GPU-abc, [N/A], 210, 405, [Not Supported], 0, 0, 0x0"
        s = parse_sample_line(row)
        assert s.temperature_c == 0.0
        assert s.power_w == 0.0

    def test_rejects_a_malformed_row(self) -> None:
        with pytest.raises(ValueError, match="expected 8 fields"):
            parse_sample_line("GPU-abc, 24, 210")


class TestAggregation:
    def test_summarizes_into_schema_record(self) -> None:
        samples = [parse_sample_line(LOADED_ROW) for _ in range(10)]
        telemetry = summarize_samples(samples)

        assert telemetry.sample_count == 10
        assert telemetry.temperature_c.mean == 68.0
        assert telemetry.memory_used_mib_max == 40391
        assert telemetry.throttled_fraction == 0.0

    def test_reports_informational_reasons_without_invalidating(self) -> None:
        """sw_power_cap is recorded so a reader sees it, but does not condemn
        the run. Hiding it would be as dishonest as counting it."""
        telemetry = summarize_samples([parse_sample_line(LOADED_ROW)])
        assert "sw_power_cap" in telemetry.throttle_reasons_observed
        assert telemetry.throttled_fraction == 0.0

    def test_throttled_fraction_counts_only_invalidating_samples(self) -> None:
        samples = [parse_sample_line(LOADED_ROW)] * 8 + [parse_sample_line(THERMAL_ROW)] * 2
        telemetry = summarize_samples(samples)
        assert telemetry.throttled_fraction == pytest.approx(0.2)
        assert "hw_thermal_slowdown" in telemetry.throttle_reasons_observed

    def test_empty_sample_set_is_valid(self) -> None:
        telemetry = summarize_samples([])
        assert telemetry.sample_count == 0
        assert telemetry.throttled_fraction == 0.0

    def test_tracks_peak_memory_not_last(self) -> None:
        samples = [
            GpuSample("GPU-x", 60.0, 1600.0, 7251.0, 250.0, mem, 90.0, ())
            for mem in (1000, 40000, 2000)
        ]
        assert summarize_samples(samples).memory_used_mib_max == 40000
