"""One measurement end to end, with no GPU: dispatch, assembly and writing.

``SweepRunner.measure`` is the path every committed number came through, yet
it could only run against a live engine, so it went untested. Here the engine is
the in-process mock server, and the two things that need hardware -- the GPU
sampler and the environment probe -- are replaced by fixed stand-ins.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmbench.config import load_sweep_config
from llmbench.engines.base import EngineHandle, EngineLaunchSpec
from llmbench.engines.preflight import PreflightReport
from llmbench.loadgen import client as client_module
from llmbench.loadgen import closed_loop as closed_loop_module
from llmbench.runner import RatePoint, SweepRunner
from llmbench.schema import ArrivalProcess, GPUTelemetry, HardwareInfo, RunValidity
from llmbench.telemetry.gpu import GpuSample
from llmbench.workload.arrivals import poisson_schedule

from .mock_server import MockLLMServer
from .test_loadgen import client_for, specs


class _Sampler:
    """Stands in for the nvidia-smi sampler: the GPU held its lock throughout."""

    def __init__(self, telemetry: GPUTelemetry, **_: object) -> None:
        self._telemetry = telemetry

    def start(self) -> None:
        pass

    def stop(self) -> GPUTelemetry:
        return self._telemetry

    @property
    def samples(self) -> tuple[GpuSample, ...]:
        return tuple(
            GpuSample(
                uuid="GPU-test",
                temperature_c=70.0,
                sm_clock_mhz=1740.0,
                mem_clock_mhz=7251.0,
                power_w=280.0,
                memory_used_mib=40000,
                utilization_pct=95.0,
                event_reasons=(),
            )
            for _ in range(5)
        )


@pytest.fixture
def runner(tmp_path: Path) -> SweepRunner:
    r = SweepRunner(
        load_sweep_config(Path("configs/sweep.yaml")),
        gpu_index=1,
        results_dir=tmp_path / "runs",
    )
    r.results_dir.mkdir()
    return r


@pytest.fixture
def handle() -> EngineHandle:
    spec = EngineLaunchSpec(
        config_id="vllm-bf16",
        image="vllm/vllm-openai",
        tag="v0.26.0",
        image_digest="sha256:" + "ff" * 32,
        model_hf_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        gpu_index=1,
        port=8000,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
        hf_cache_dir=Path("/tmp/hf"),
        expected_kernel=None,
        kernel_log_patterns={},
        extra_args={"--seed": "0"},
    )
    return EngineHandle(
        spec=spec,
        container_id="c0ffee",
        engine_version="0.26.0",
        selected_kernel=None,
        startup_log="",
        startup_duration_s=1.0,
    )


@pytest.fixture
def preflight() -> PreflightReport:
    return PreflightReport(
        gpu_index=1,
        gpu_uuid="GPU-test",
        neighbor_gpu_busy=False,
        neighbor_details=(),
        free_disk_gib=100.0,
        free_vram_gib=45.0,
        clocks_locked=True,
        sm_clock_mhz=1740,
        warnings=(),
    )


@pytest.fixture
def offline(gpu_telemetry: GPUTelemetry, hardware: HardwareInfo) -> Iterator[None]:
    """Route every request to the mock server and stub out the hardware."""
    server = MockLLMServer(ttft_s=0.005, itl_s=0.001)
    open_loop, closed_loop = client_module.run_open_loop, closed_loop_module.run_closed_loop

    def via_mock_open(*args: Any, **kwargs: Any) -> Any:
        return open_loop(*args, client=client_for(server), **kwargs)

    def via_mock_closed(*args: Any, **kwargs: Any) -> Any:
        return closed_loop(*args, client=client_for(server), **kwargs)

    with (
        patch("llmbench.runner.run_open_loop", via_mock_open),
        patch("llmbench.runner.run_closed_loop", via_mock_closed),
        patch("llmbench.runner.GpuSampler", lambda **kw: _Sampler(gpu_telemetry, **kw)),
        patch("llmbench.runner.environment_info", return_value=hardware),
        patch("llmbench.runner.time.sleep"),
    ):
        yield


@pytest.mark.usefixtures("offline")
class TestOneMeasurement:
    def test_open_loop_run_is_assembled_and_written(
        self, runner: SweepRunner, handle: EngineHandle, preflight: PreflightReport
    ) -> None:
        point = RatePoint(
            rate_rps=200.0,
            schedule=poisson_schedule(rate_rps=200.0, num_requests=60, seed=1),
            specs=specs(60),
        )
        run = runner.measure(handle, point, 0, preflight)

        assert run.validity is RunValidity.VALID
        assert run.workload.warmup_requests == 50
        assert run.requests_sent == 10
        assert run.engine.extra_args == {"--seed": "0"}
        assert any("clock lock re-checked from 5 loaded samples" in n for n in run.validity_notes)

        summary = runner._write(run, label="probe")
        raw = summary.parent.parent / "raw" / summary.name.replace(".json", ".jsonl.gz")
        with gzip.open(raw, "rt", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 60
        assert sum(r["is_warmup"] for r in rows) == 50

    def test_closed_loop_warms_up_through_its_burst(
        self, runner: SweepRunner, handle: EngineHandle, preflight: PreflightReport
    ) -> None:
        point = RatePoint(
            rate_rps=None,
            schedule=poisson_schedule(rate_rps=200.0, num_requests=80, seed=1),
            specs=specs(80),
            arrival_process=ArrivalProcess.CLOSED_LOOP,
            concurrency=64,
            warmup_requests=64,
        )
        run = runner.measure(handle, point, 0, preflight)

        assert run.workload.warmup_requests == 64
        assert run.requests_sent == 16
        assert run.workload.concurrency == 64

    def test_trace_point_note_is_recorded(
        self, runner: SweepRunner, handle: EngineHandle, preflight: PreflightReport
    ) -> None:
        point = RatePoint(
            rate_rps=None,
            schedule=poisson_schedule(rate_rps=200.0, num_requests=60, seed=1),
            specs=specs(60),
            arrival_process=ArrivalProcess.TRACE_REPLAY,
            note="trace window 10.0-190.0 s of the trace: (test)",
        )
        run = runner.measure(handle, point, 0, preflight)
        assert "trace window 10.0-190.0 s of the trace: (test)" in run.validity_notes
