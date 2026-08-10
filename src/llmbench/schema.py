"""Versioned result schema for every artifact this benchmark produces.

This module is deliberately *pure data*: no measurement logic, no I/O, no numpy
beyond typing. It is defined once and frozen in Phase 1. Downstream phases may
**extend** it (new optional fields, bumping the minor version) but must never
repurpose or remove a field — a benchmark whose output format drifts mid-project
cannot be compared against its own earlier runs, which makes the whole exercise
worthless.

Every record is self-describing: given a single ``RunResult`` JSON you can
reconstruct exactly what hardware, engine build, model revision and offered load
produced it, and whether the measurement was valid.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bumped only on a breaking change. tests/test_schema.py guards this.
SCHEMA_VERSION = "1.0.0"

NonNegFloat = Annotated[float, Field(ge=0.0)]
PosInt = Annotated[int, Field(gt=0)]


class _Base(BaseModel):
    """Strict base: unknown keys are an error, not a silent shrug.

    A typo in a config key that silently produces a differently-shaped result is
    exactly the class of bug that invalidates a benchmark weeks later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Quantization(StrEnum):
    BF16 = "bf16"
    GPTQ_INT4 = "gptq-int4"
    AWQ_INT4 = "awq-int4"
    INT8_W8A8 = "int8-w8a8"
    NF4 = "nf4"


class EngineName(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class ArrivalProcess(StrEnum):
    """How request arrival times are generated.

    POISSON is the primary open-loop process. TRACE_REPLAY replays real
    production inter-arrival times to show what Poisson's own assumption costs.
    CLOSED_LOOP exists only so the repo can *demonstrate* coordinated omission,
    never to report a headline number.
    """

    POISSON = "poisson"
    TRACE_REPLAY = "trace-replay"
    CLOSED_LOOP = "closed-loop"


class LengthSource(StrEnum):
    SHAREGPT = "sharegpt"
    AZURE_TRACE = "azure-trace"
    FIXED = "fixed"


class RunValidity(StrEnum):
    """Whether a run may be reported as a measurement.

    Anything other than VALID must be excluded from headline results. The point
    of recording rather than discarding invalid runs is that the *reason* a
    configuration could not be measured is itself a finding.
    """

    VALID = "valid"
    CLIENT_SATURATED = "client-saturated"
    ENGINE_ERROR = "engine-error"
    THERMAL_THROTTLED = "thermal-throttled"
    INCOMPLETE = "incomplete"


class QualityTask(StrEnum):
    WIKITEXT2_PPL = "wikitext2-ppl"
    GSM8K = "gsm8k"
    IFEVAL = "ifeval"


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------
class Stats(_Base):
    """Summary of a sample distribution.

    Built by ``llmbench.metrics.percentiles.summarize``; never constructed by
    hand in measurement code. Percentiles are carried explicitly because a mean
    alone hides precisely the tail behaviour this benchmark exists to expose.
    """

    count: int = Field(ge=0)
    mean: float
    std: float = Field(ge=0.0)
    min: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Stats:
        ordered = [self.min, self.p50, self.p90, self.p95, self.p99, self.max]
        if self.count > 0 and any(b < a for a, b in pairwise(ordered)):
            msg = f"percentiles must be non-decreasing, got {ordered}"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class GPUInfo(_Base):
    index: int = Field(ge=0)
    name: str
    compute_capability: str
    vram_total_mib: PosInt
    driver_version: str
    cuda_version: str
    ecc_enabled: bool


class HostInfo(_Base):
    hostname: str
    cpu_model: str
    cpu_count: PosInt
    ram_total_gib: float = Field(gt=0)
    platform: str
    python_version: str


class ClockPolicy(_Base):
    """Whether GPU clocks were pinned, and to what.

    Reported either way. An unlocked run is not invalid, but it must be
    labelled so a reader can weigh late-run drift for themselves.
    """

    locked: bool
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    persistence_mode: bool = False


class GPUTelemetry(_Base):
    """Sampled throughout the measurement window, not just at the edges.

    ``throttled_fraction`` is the share of samples reporting any active
    throttle reason. Non-zero does not automatically invalidate a run; it
    quantifies a confound that would otherwise be invisible.
    """

    sample_count: int = Field(ge=0)
    temperature_c: Stats
    sm_clock_mhz: Stats
    power_w: Stats
    memory_used_mib_max: int = Field(ge=0)
    throttle_reasons_observed: list[str] = Field(default_factory=list)
    throttled_fraction: float = Field(ge=0.0, le=1.0)


class HardwareInfo(_Base):
    gpu: GPUInfo
    host: HostInfo
    clocks: ClockPolicy
    #: True when a *different* GPU in the same chassis was under load during this
    #: run. A40s are passively cooled and share chassis airflow, so a busy
    #: neighbour is a real thermal confound that must travel with the result.
    neighbor_gpu_busy: bool = False


# ---------------------------------------------------------------------------
# Configuration under test
# ---------------------------------------------------------------------------
class ModelConfig(_Base):
    hf_id: str
    #: Resolved commit SHA, not a branch name. "main" is not reproducible.
    revision: str
    quantization: Quantization
    dtype: str
    max_model_len: PosInt
    weights_gib: float = Field(gt=0)


class EngineConfig(_Base):
    name: EngineName
    version: str
    image: str
    #: Pinned digest (sha256:...). Never ``:latest`` — an image that silently
    #: moves under you makes every earlier result unreproducible.
    image_digest: str
    tensor_parallel_size: PosInt = 1
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_num_seqs: PosInt
    #: The kernel the engine actually selected, parsed from its startup log.
    #: Asserted, never assumed: a GPTQ checkpoint with ``desc_act=True`` can
    #: silently fall off the fast Marlin path and quietly halve throughput.
    selected_kernel: str | None = None
    extra_args: dict[str, str] = Field(default_factory=dict)


class WorkloadConfig(_Base):
    arrival_process: ArrivalProcess
    #: Offered load. None for TRACE_REPLAY, where arrivals come from the trace.
    request_rate_rps: float | None = Field(default=None, gt=0)
    length_source: LengthSource
    seed: int
    num_requests: PosInt
    warmup_requests: int = Field(ge=0)
    measurement_duration_s: NonNegFloat
    #: Output length is enforced via max_tokens with ignore_eos so that every
    #: configuration performs *identical* work. Without it, quantization changes
    #: where the model stops and you are silently comparing different workloads.
    ignore_eos: bool
    input_len_tokens: Stats
    output_len_tokens: Stats

    @model_validator(mode="after")
    def _rate_matches_process(self) -> WorkloadConfig:
        if self.arrival_process is ArrivalProcess.TRACE_REPLAY:
            if self.request_rate_rps is not None:
                msg = "trace-replay derives arrivals from the trace; request_rate_rps must be None"
                raise ValueError(msg)
        elif self.request_rate_rps is None:
            msg = f"{self.arrival_process} requires an explicit request_rate_rps"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
class RunResult(_Base):
    """One (configuration, request-rate, repeat) measurement."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    config_id: str
    repeat_index: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime

    engine: EngineConfig
    model: ModelConfig
    workload: WorkloadConfig
    hardware: HardwareInfo

    requests_sent: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    requests_failed: int = Field(ge=0)

    # The metrics, kept separate on purpose: reporting only end-to-end latency
    # hides whether a configuration is prefill- or decode-bound.
    ttft_s: Stats
    tpot_s: Stats
    e2e_latency_s: Stats
    output_tokens: Stats

    output_token_throughput: NonNegFloat
    request_throughput: NonNegFloat

    #: actual minus scheduled dispatch time, per request. If this grows, the
    #: *client* has become the bottleneck and the run understates tail latency.
    dispatch_lag_s: Stats
    gpu_telemetry: GPUTelemetry

    validity: RunValidity
    validity_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _accounting_balances(self) -> RunResult:
        if self.requests_completed + self.requests_failed > self.requests_sent:
            msg = (
                f"completed+failed ({self.requests_completed}+{self.requests_failed}) "
                f"exceeds sent ({self.requests_sent})"
            )
            raise ValueError(msg)
        if self.finished_at < self.started_at:
            msg = "finished_at precedes started_at"
            raise ValueError(msg)
        return self

    @property
    def is_reportable(self) -> bool:
        return self.validity is RunValidity.VALID


class QualityScore(_Base):
    task: QualityTask
    metric: str
    value: float
    #: Standard error where the harness reports one; perplexity has none.
    stderr: float | None = Field(default=None, ge=0.0)
    num_samples: PosInt


class QualityResult(_Base):
    """Accuracy for one model configuration.

    Measured *through the running engine* rather than a separate transformers
    path, so it exercises the same kernels the latency numbers came from and can
    catch a quantized-kernel bug that perplexity-on-checkpoint would miss.
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str
    config_id: str
    started_at: datetime
    finished_at: datetime

    engine: EngineConfig
    model: ModelConfig
    hardware: HardwareInfo

    #: Greedy decoding, fixed seed, identical prompts across configurations.
    temperature: float = Field(ge=0.0)
    seed: int
    #: vLLM greedy output is not bitwise-deterministic across batch sizes
    #: (reduction order varies), so quality runs pin a low concurrency.
    max_concurrency: PosInt

    scores: list[QualityScore]


class CostAssumptions(_Base):
    """Stated prominently because the GPU is a lab machine with no invoice."""

    gpu_hourly_usd: float = Field(gt=0)
    source_vendor: str
    source_url: str
    accessed_date: str
    notes: str = ""


class SweepManifest(_Base):
    """Provenance for a whole sweep — what was run, from which commit."""

    schema_version: str = SCHEMA_VERSION
    sweep_id: str
    created_at: datetime
    git_commit: str
    git_dirty: bool
    config_path: str
    cost: CostAssumptions
    run_ids: list[str] = Field(default_factory=list)
    quality_run_ids: list[str] = Field(default_factory=list)
