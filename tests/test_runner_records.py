"""What the runner records beside each result.

Three notes and one file, each added because its absence left a question about
the committed results that the data could not answer: whether the clock lock
held, whether a "valid" run was a steady state, which stretch of a trace was
replayed, and what the individual requests of a run looked like.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from llmbench.config import load_sweep_config
from llmbench.engines.preflight import PreflightReport
from llmbench.loadgen.client import RequestRecord
from llmbench.runner import (
    SweepRunner,
    _clock_check_note,
    _queue_growth_note,
    _trace_window_note,
)
from llmbench.schema import RunResult
from llmbench.telemetry.gpu import GpuSample
from llmbench.workload.trace import TraceRequest, TraceWindow


def record(index: int, ttft_s: float) -> RequestRecord:
    return RequestRecord(
        index=index,
        is_warmup=False,
        scheduled_offset_s=index * 0.25,
        dispatch_lag_s=0.001,
        input_tokens=100,
        requested_max_tokens=50,
        ttft_s=ttft_s,
        itl_s=(0.02,) * 49,
        e2e_s=ttft_s + 0.98,
        output_tokens=50,
        finish_reason="length",
        status_code=200,
        error=None,
        dispatch_offset_s=index * 0.25 + 0.001,
    )


def sample(sm_mhz: float, *, util: float = 95.0, capped: bool = False) -> GpuSample:
    return GpuSample(
        uuid="GPU-test",
        temperature_c=70.0,
        sm_clock_mhz=sm_mhz,
        mem_clock_mhz=7251.0,
        power_w=290.0,
        memory_used_mib=40000,
        utilization_pct=util,
        event_reasons=("sw_power_cap",) if capped else (),
    )


def preflight(*, locked: bool = True) -> PreflightReport:
    return PreflightReport(
        gpu_index=1,
        gpu_uuid="GPU-test",
        neighbor_gpu_busy=False,
        neighbor_details=(),
        free_disk_gib=100.0,
        free_vram_gib=45.0,
        clocks_locked=locked,
        sm_clock_mhz=1740 if locked else None,
        warnings=(),
    )


class TestQueueGrowthNote:
    def test_a_growing_queue_is_noted(self) -> None:
        """TTFT climbing through the run is the signature of an arrival rate
        above capacity -- which the 80 % oversubscription rule can miss."""
        ok = [record(i, 0.2 + 0.02 * i) for i in range(300)]
        note = _queue_growth_note(ok)
        assert note is not None
        assert "not a steady state" in note

    def test_a_stable_queue_is_not(self) -> None:
        ok = [record(i, 0.2 + 0.05 * ((i * 7) % 5)) for i in range(300)]
        assert _queue_growth_note(ok) is None

    def test_a_start_up_burst_is_not_mistaken_for_growth(self) -> None:
        """A closed loop's burst makes the *first* third slow, not the last."""
        ok = [record(i, 3.0 if i < 60 else 0.2) for i in range(300)]
        assert _queue_growth_note(ok) is None

    def test_too_few_requests_say_nothing(self) -> None:
        assert _queue_growth_note([record(i, 0.1 * i) for i in range(20)]) is None


class TestClockCheckNote:
    def test_unlocked_runs_get_no_note(self) -> None:
        assert _clock_check_note([sample(1740.0)] * 10, preflight(locked=False)) is None

    def test_a_held_lock_is_confirmed(self) -> None:
        note = _clock_check_note([sample(1740.0)] * 10, preflight())
        assert note is not None
        assert "100% within 30 MHz" in note
        assert note.endswith("lock held")

    def test_dips_under_the_power_cap_are_attributed_to_it(self) -> None:
        samples = [sample(1740.0)] * 5 + [sample(1560.0, capped=True)] * 5
        note = _clock_check_note(samples, preflight())
        assert note is not None
        assert "only while the driver was power-capping" in note

    def test_unexplained_dips_suggest_a_lapsed_lock(self) -> None:
        samples = [sample(1740.0)] * 5 + [sample(1400.0)] * 5
        note = _clock_check_note(samples, preflight())
        assert note is not None
        assert "may have lapsed" in note

    def test_idle_samples_do_not_count(self) -> None:
        """An idle A40 drops to ~210 MHz even when locked."""
        samples = [sample(210.0, util=0.0)] * 20 + [sample(1740.0)] * 5
        note = _clock_check_note(samples, preflight())
        assert note is not None
        assert "5 loaded samples" in note
        assert note.endswith("lock held")

    def test_no_loaded_sample_is_stated(self) -> None:
        note = _clock_check_note([sample(210.0, util=0.0)] * 5, preflight())
        assert note is not None
        assert "not re-checked" in note


class TestTraceWindowNote:
    def test_names_the_window_and_its_scaling(self) -> None:
        requests = tuple(
            TraceRequest(arrival_s=i * 0.25, input_tokens=100, output_tokens=50) for i in range(88)
        )
        window = TraceWindow(requests=requests, duration_s=20.0, start_s=2804.3)
        note = _trace_window_note(window.scaled_to_rate(4.0))
        assert note.startswith("trace window 2804.3-2824.3 s of the trace")
        assert "time-scaled x1.100 to 4.00 rps" in note
        assert "CV^2" in note


class TestPerRequestRecords:
    def test_records_are_written_beside_the_summary(
        self, tmp_path: Path, run_result: RunResult
    ) -> None:
        """Only summaries used to be kept, so no percentile could be recomputed
        and the closed-loop start-up burst could not be looked for."""
        runner = SweepRunner(
            load_sweep_config(Path("configs/sweep.yaml")),
            gpu_index=0,
            results_dir=tmp_path / "runs",
        )
        runner.results_dir.mkdir()
        runner._raw_records[run_result.run_id] = tuple(record(i, 0.2) for i in range(3))

        summary = runner._write(run_result, label="probe")

        raw = tmp_path / "raw" / summary.name.replace(".json", ".jsonl.gz")
        with gzip.open(raw, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert [r["index"] for r in rows] == [0, 1, 2]
        assert rows[0]["dispatch_offset_s"] == pytest.approx(0.001)
        assert run_result.run_id not in runner._raw_records

    def test_a_run_without_records_writes_only_its_summary(
        self, tmp_path: Path, run_result: RunResult
    ) -> None:
        runner = SweepRunner(
            load_sweep_config(Path("configs/sweep.yaml")),
            gpu_index=0,
            results_dir=tmp_path / "runs",
        )
        runner.results_dir.mkdir()
        runner._write(run_result)
        assert not (tmp_path / "raw").exists()
