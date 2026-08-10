"""Sweep orchestration.

Ties together preflight, engine lifecycle, workload generation, load
dispatching, telemetry and validity assessment into a schema-validated
:class:`~llmbench.schema.RunResult` per (configuration, rate, repeat) point.

Two structural decisions carry most of the value:

**The engine is started once per configuration**, then every rate and repeat is
swept against it. vLLM takes ~220 s to reach health — weight load, compilation
and CUDA-graph capture — so restarting per rate point would multiply the sweep's
cost by roughly the number of rates and add nothing.

**The workload is generated once per rate**, seeded, and reused across every
configuration. Each engine and quantization level therefore faces a
byte-identical offered load, which is what makes differences between them
attributable to the configuration rather than to the sample they happened to be
given.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llmbench.config import EngineProfile, SweepConfig, load_engine_profile
from llmbench.engines.base import EngineHandle, EngineLaunchSpec, EngineProcess, resolve_digest
from llmbench.engines.preflight import PreflightReport, run_preflight
from llmbench.engines.sglang import SglangEngine
from llmbench.engines.vllm import VllmEngine
from llmbench.loadgen.client import LoadGenConfig, LoadGenResult, RequestRecord, run_open_loop
from llmbench.loadgen.guard import assess_validity
from llmbench.metrics.percentiles import summarize
from llmbench.schema import (
    ArrivalProcess,
    ClockPolicy,
    EngineConfig,
    EngineName,
    GPUInfo,
    GPUTelemetry,
    HardwareInfo,
    HostInfo,
    ModelConfig,
    Quantization,
    RunResult,
    WorkloadConfig,
)
from llmbench.telemetry.gpu import GpuSampler, nvidia_smi_query
from llmbench.workload.arrivals import ArrivalSchedule, num_requests_for_duration, poisson_schedule
from llmbench.workload.corpus import ShareGptCorpus, load_sharegpt
from llmbench.workload.lengths import EmpiricalLengthSampler, clamp_to_context
from llmbench.workload.prompts import RequestSpec, build_requests
from llmbench.workload.tokenizer import DEFAULT_HF_CACHE, load_tokenizer

__all__ = ["SweepRunner", "environment_info"]

_ENGINES: dict[EngineName, type[EngineProcess]] = {
    EngineName.VLLM: VllmEngine,
    EngineName.SGLANG: SglangEngine,
}


@dataclass(frozen=True, slots=True)
class RatePoint:
    """One offered-load level: the workload every configuration will face."""

    rate_rps: float
    schedule: ArrivalSchedule
    specs: tuple[RequestSpec, ...]

    @property
    def mean_interarrival_s(self) -> float:
        gaps = self.schedule.inter_arrivals_s()
        return sum(gaps) / len(gaps) if gaps else 0.0


def _cuda_version() -> str:
    """Driver's supported CUDA version.

    Only printed in the nvidia-smi header — there is no ``--query-gpu`` field
    for it — so it is scraped rather than queried.
    """
    out = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    ).stdout
    match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    return match.group(1) if match else "unknown"


def _total_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / (1024**2)
    return 0.0


def environment_info(gpu_index: int, preflight: PreflightReport) -> HardwareInfo:
    """Capture the environment exactly as it was for this run."""
    fields = "name,compute_cap,memory.total,driver_version,ecc.mode.current"
    row = nvidia_smi_query(
        [f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(gpu_index)]
    ).splitlines()[0]
    name, compute_cap, vram, driver, ecc = (p.strip() for p in row.split(","))

    return HardwareInfo(
        gpu=GPUInfo(
            index=gpu_index,
            name=name,
            compute_capability=compute_cap,
            vram_total_mib=int(float(vram)),
            driver_version=driver,
            cuda_version=_cuda_version(),
            ecc_enabled=ecc.lower().startswith("enabled"),
        ),
        host=HostInfo(
            hostname=socket.gethostname(),
            cpu_model=platform.processor() or "unknown",
            # Affinity, not total core count: the benchmark's real CPU budget.
            cpu_count=len(os.sched_getaffinity(0)),
            ram_total_gib=_total_ram_gib(),
            platform=platform.platform(),
            python_version=sys.version.split()[0],
        ),
        clocks=ClockPolicy(
            locked=preflight.clocks_locked,
            sm_clock_mhz=preflight.sm_clock_mhz,
            persistence_mode=True,
        ),
        neighbor_gpu_busy=preflight.neighbor_gpu_busy,
    )


class SweepRunner:
    """Executes a sweep matrix, writing one validated JSON per measurement."""

    def __init__(
        self,
        config: SweepConfig,
        *,
        gpu_index: int,
        results_dir: Path,
        configs_dir: Path = Path("configs"),
        dataset_path: Path = Path("data/sharegpt_v3.json"),
        hf_cache_dir: Path = DEFAULT_HF_CACHE,
        require_locked_clocks: bool = False,
    ) -> None:
        self.config = config
        self.gpu_index = gpu_index
        self.results_dir = results_dir
        self.configs_dir = configs_dir
        self.dataset_path = dataset_path
        self.hf_cache_dir = hf_cache_dir
        self.require_locked_clocks = require_locked_clocks

        self._corpus: ShareGptCorpus | None = None
        self._rate_points: dict[float, RatePoint] = {}

    # ------------------------------------------------------------------
    # Workload
    # ------------------------------------------------------------------
    def _load_corpus(self) -> ShareGptCorpus:
        if self._corpus is None:
            tokenizer = load_tokenizer(
                self.config.model.hf_id, self.config.model.revision, self.hf_cache_dir
            )
            self._corpus = load_sharegpt(self.dataset_path, tokenizer)
        return self._corpus

    def rate_point(self, rate_rps: float) -> RatePoint:
        """Build (once) the workload every configuration faces at this rate.

        Cached, so all configurations share the identical seeded schedule and
        prompt list — the property that makes cross-configuration comparison
        meaningful.
        """
        if rate_rps in self._rate_points:
            return self._rate_points[rate_rps]

        corpus = self._load_corpus()
        tokenizer = load_tokenizer(
            self.config.model.hf_id, self.config.model.revision, self.hf_cache_dir
        )
        defaults = self.config.defaults
        seed = self.config.workload.seed

        n_measured = num_requests_for_duration(rate_rps, defaults.measurement_duration_s)
        total = n_measured + defaults.warmup_requests

        sampler = EmpiricalLengthSampler(pairs=corpus.pairs)
        pairs = clamp_to_context(sampler.sample(total, seed=seed), self.config.model.max_model_len)
        specs = build_requests(pairs, corpus.corpus_token_ids, tokenizer, seed=seed)
        schedule = poisson_schedule(rate_rps=rate_rps, num_requests=total, seed=seed)

        point = RatePoint(rate_rps=rate_rps, schedule=schedule, specs=specs)
        self._rate_points[rate_rps] = point
        return point

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------
    def _launch_spec(self, config_id: str, profile: EngineProfile) -> EngineLaunchSpec:
        quant = self.config.quantization_for(config_id)
        digest = profile.image_digest or resolve_digest(profile.image, profile.tag)

        return EngineLaunchSpec(
            config_id=config_id,
            image=profile.image,
            tag=profile.tag,
            image_digest=digest,
            model_hf_id=quant.hf_id,
            model_revision=quant.revision,
            gpu_index=self.gpu_index,
            port=profile.port,
            max_model_len=self.config.model.max_model_len,
            gpu_memory_utilization=self.config.defaults.gpu_memory_utilization,
            max_num_seqs=self.config.defaults.max_num_seqs,
            hf_cache_dir=self.hf_cache_dir,
            expected_kernel=quant.expected_kernel,
            kernel_log_patterns=profile.kernel_log_patterns,
            startup_timeout_s=profile.startup_timeout_s,
            health_path=profile.health_path,
            metrics_path=profile.metrics_path,
        )

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def measure(
        self,
        handle: EngineHandle,
        point: RatePoint,
        repeat_index: int,
        preflight: PreflightReport,
    ) -> RunResult:
        """Run one (rate, repeat) measurement against an already-live engine."""
        quant = self.config.quantization_for(handle.spec.config_id)
        defaults = self.config.defaults
        started = datetime.now(UTC)

        loadgen_config = LoadGenConfig(
            base_url=handle.base_url,
            model=quant.hf_id,
            ignore_eos=defaults.ignore_eos,
            temperature=0.0,
        )

        sampler = GpuSampler(gpu_index=self.gpu_index, interval_s=1.0)
        sampler.start()
        result = asyncio.run(
            run_open_loop(
                point.schedule,
                point.specs,
                loadgen_config,
                warmup_requests=defaults.warmup_requests,
            )
        )
        telemetry = sampler.stop()
        finished = datetime.now(UTC)

        # Settle between rate points so queued work from this run cannot leak
        # into the next one's measurement window.
        time.sleep(defaults.settle_s)

        return self._assemble(
            handle=handle,
            point=point,
            repeat_index=repeat_index,
            preflight=preflight,
            loadgen=result,
            telemetry=telemetry,
            started=started,
            finished=finished,
        )

    def _assemble(
        self,
        *,
        handle: EngineHandle,
        point: RatePoint,
        repeat_index: int,
        preflight: PreflightReport,
        loadgen: LoadGenResult,
        telemetry: GPUTelemetry,
        started: datetime,
        finished: datetime,
    ) -> RunResult:
        spec = handle.spec
        quant = self.config.quantization_for(spec.config_id)
        defaults = self.config.defaults

        measured: Sequence[RequestRecord] = loadgen.measured
        ok = [r for r in measured if r.succeeded]
        failed = len(measured) - len(ok)

        window = loadgen.measurement_window_s or 1e-9
        assessment = assess_validity(
            dispatch_lags_s=[r.dispatch_lag_s for r in measured],
            requests_scheduled=len(measured),
            requests_completed=len(ok),
            requests_failed=failed,
            mean_interarrival_s=point.mean_interarrival_s,
            throttled_fraction=telemetry.throttled_fraction,
        )

        notes = list(assessment.notes)
        notes.extend(preflight.warnings)

        return RunResult(
            run_id=uuid.uuid4().hex,
            config_id=spec.config_id,
            repeat_index=repeat_index,
            started_at=started,
            finished_at=finished,
            engine=EngineConfig(
                # From the config entry, not parsed out of the id string: an id
                # like "vllm-gptq-int4" only happens to split correctly, and a
                # rename would silently mislabel every result.
                name=self.config.configuration(spec.config_id).engine,
                version=handle.engine_version,
                image=f"{spec.image}:{spec.tag}",
                image_digest=spec.image_digest,
                gpu_memory_utilization=spec.gpu_memory_utilization,
                max_num_seqs=spec.max_num_seqs,
                selected_kernel=handle.selected_kernel,
            ),
            model=ModelConfig(
                hf_id=quant.hf_id,
                revision=quant.revision,
                quantization=Quantization(self.config.configuration(spec.config_id).quantization),
                dtype=quant.dtype,
                max_model_len=spec.max_model_len,
                weights_gib=quant.expected_weights_gib,
            ),
            workload=WorkloadConfig(
                arrival_process=ArrivalProcess.POISSON,
                request_rate_rps=point.rate_rps,
                length_source=self.config.workload.length_source,
                seed=self.config.workload.seed,
                num_requests=len(measured),
                warmup_requests=defaults.warmup_requests,
                measurement_duration_s=loadgen.measurement_window_s,
                ignore_eos=defaults.ignore_eos,
                input_len_tokens=summarize([float(r.input_tokens) for r in measured]),
                output_len_tokens=summarize([float(r.output_tokens) for r in ok]),
            ),
            hardware=environment_info(self.gpu_index, preflight),
            requests_sent=len(measured),
            requests_completed=len(ok),
            requests_failed=failed,
            ttft_s=summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
            tpot_s=summarize([r.tpot_s for r in ok if r.tpot_s is not None]),
            e2e_latency_s=summarize([r.e2e_s for r in ok if r.e2e_s is not None]),
            output_tokens=summarize([float(r.output_tokens) for r in ok]),
            output_token_throughput=sum(r.output_tokens for r in ok) / window,
            request_throughput=len(ok) / window,
            dispatch_lag_s=summarize([r.dispatch_lag_s for r in measured]),
            gpu_telemetry=telemetry,
            validity=assessment.validity,
            validity_notes=notes,
        )

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------
    def run(self) -> list[Path]:
        """Execute the whole matrix. Returns the paths written."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir.parent / "raw").mkdir(parents=True, exist_ok=True)

        preflight = run_preflight(
            self.gpu_index,
            results_path=str(self.results_dir),
            require_locked_clocks=self.require_locked_clocks,
        )
        for warning in preflight.warnings:
            print(f"  [preflight] {warning}")

        written: list[Path] = []

        for entry in self.config.configurations:
            profile = load_engine_profile(entry.engine, self.configs_dir)
            engine = _ENGINES[entry.engine]()
            spec = self._launch_spec(entry.id, profile)

            print(f"\n=== {entry.id} ({entry.engine.value}, {entry.quantization}) ===")
            print("  starting engine...")
            handle = engine.start(spec)
            print(
                f"  ready in {handle.startup_duration_s:.0f}s "
                f"(v{handle.engine_version}, kernel={handle.selected_kernel})"
            )
            self._persist_startup_log(entry.id, handle)

            try:
                for rate in self.config.workload.request_rates_rps:
                    point = self.rate_point(rate)
                    for repeat in range(self.config.defaults.repeats):
                        run = self.measure(handle, point, repeat, preflight)
                        path = self._write(run)
                        written.append(path)
                        print(
                            f"  {rate:>5.1f} rps rep{repeat}: "
                            f"TTFT p95 {run.ttft_s.p95 * 1e3:7.1f} ms | "
                            f"TPOT p95 {run.tpot_s.p95 * 1e3:6.1f} ms | "
                            f"{run.output_token_throughput:7.1f} tok/s | "
                            f"{run.validity.value}"
                        )
            finally:
                engine.stop(spec)
                print("  engine stopped")

        return written

    def _persist_startup_log(self, config_id: str, handle: EngineHandle) -> None:
        """Keep the kernel-selection evidence alongside the results."""
        log_dir = self.results_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{config_id}.startup.log").write_text(handle.startup_log)

    def _write(self, run: RunResult) -> Path:
        name = f"{run.config_id}__{run.workload.request_rate_rps:g}rps__rep{run.repeat_index}.json"
        path = self.results_dir / name
        path.write_text(json.dumps(json.loads(run.model_dump_json()), indent=2) + "\n")
        return path
