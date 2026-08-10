"""Shared fixtures.

Builders produce *valid* records so that each test can mutate exactly one field
and assert on that field alone. Every value here is synthetic — nothing in the
test suite requires a GPU, an engine, or a network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmbench.schema import (
    ArrivalProcess,
    ClockPolicy,
    EngineConfig,
    EngineName,
    GPUInfo,
    GPUTelemetry,
    HardwareInfo,
    HostInfo,
    LengthSource,
    ModelConfig,
    Quantization,
    RunResult,
    RunValidity,
    Stats,
    WorkloadConfig,
)


def make_stats(
    *,
    count: int = 100,
    mean: float = 1.0,
    std: float = 0.1,
    minimum: float = 0.5,
    p50: float = 0.9,
    p90: float = 1.4,
    p95: float = 1.6,
    p99: float = 1.9,
    maximum: float = 2.0,
) -> Stats:
    return Stats(
        count=count,
        mean=mean,
        std=std,
        min=minimum,
        p50=p50,
        p90=p90,
        p95=p95,
        p99=p99,
        max=maximum,
    )


@pytest.fixture
def stats() -> Stats:
    return make_stats()


@pytest.fixture
def gpu_info() -> GPUInfo:
    """The actual measurement device: A40, sm_86, ECC on."""
    return GPUInfo(
        index=1,
        name="NVIDIA A40",
        compute_capability="8.6",
        vram_total_mib=46068,
        driver_version="570.133.07",
        cuda_version="12.8",
        ecc_enabled=True,
    )


@pytest.fixture
def hardware(gpu_info: GPUInfo) -> HardwareInfo:
    return HardwareInfo(
        gpu=gpu_info,
        host=HostInfo(
            hostname="lab-node",
            cpu_model="Intel Xeon Processor (Icelake)",
            cpu_count=16,
            ram_total_gib=31.0,
            platform="Linux-5.4.0",
            python_version="3.12.13",
        ),
        clocks=ClockPolicy(locked=True, sm_clock_mhz=1740, mem_clock_mhz=7251),
        neighbor_gpu_busy=False,
    )


@pytest.fixture
def engine_config() -> EngineConfig:
    return EngineConfig(
        name=EngineName.VLLM,
        version="0.11.0",
        image="vllm/vllm-openai:v0.11.0",
        image_digest="sha256:" + "ab" * 32,
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
        selected_kernel="gptq_marlin",
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        hf_id="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        revision="6a426ef8adc0b4b96408001a9628d71f01c9ceca",
        quantization=Quantization.GPTQ_INT4,
        dtype="auto",
        max_model_len=4096,
        weights_gib=5.7,
    )


@pytest.fixture
def workload_config(stats: Stats) -> WorkloadConfig:
    return WorkloadConfig(
        arrival_process=ArrivalProcess.POISSON,
        request_rate_rps=8.0,
        length_source=LengthSource.SHAREGPT,
        seed=20260810,
        num_requests=1440,
        warmup_requests=50,
        measurement_duration_s=180.0,
        ignore_eos=True,
        input_len_tokens=stats,
        output_len_tokens=stats,
    )


@pytest.fixture
def gpu_telemetry(stats: Stats) -> GPUTelemetry:
    return GPUTelemetry(
        sample_count=180,
        temperature_c=stats,
        sm_clock_mhz=stats,
        power_w=stats,
        memory_used_mib_max=42000,
        throttle_reasons_observed=[],
        throttled_fraction=0.0,
    )


@pytest.fixture
def run_result(
    engine_config: EngineConfig,
    model_config: ModelConfig,
    workload_config: WorkloadConfig,
    hardware: HardwareInfo,
    gpu_telemetry: GPUTelemetry,
    stats: Stats,
) -> RunResult:
    return RunResult(
        run_id="01JD00000000000000000000",
        config_id="vllm-gptq-int4",
        repeat_index=0,
        started_at=datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 10, 12, 4, 0, tzinfo=UTC),
        engine=engine_config,
        model=model_config,
        workload=workload_config,
        hardware=hardware,
        requests_sent=1440,
        requests_completed=1440,
        requests_failed=0,
        ttft_s=stats,
        tpot_s=stats,
        e2e_latency_s=stats,
        output_tokens=stats,
        output_token_throughput=2100.0,
        request_throughput=8.0,
        # A healthy open-loop client: sub-millisecond scheduling error, so the
        # offered load actually matched the intended Poisson process.
        dispatch_lag_s=make_stats(
            mean=0.001,
            std=0.0008,
            minimum=0.0,
            p50=0.001,
            p90=0.002,
            p95=0.003,
            p99=0.004,
            maximum=0.006,
        ),
        gpu_telemetry=gpu_telemetry,
        validity=RunValidity.VALID,
    )
