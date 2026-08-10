"""Typed loading of the sweep matrix.

The sweep is config-driven so that adding or removing a configuration never
requires editing benchmark code. That only holds if the config is validated as
strictly as the results are: a typo in a key that silently falls back to a
default would produce a run that looks fine and measures something other than
what was asked for.

Everything here is therefore ``extra="forbid"``, and cross-references (a
configuration naming a quantization level that does not exist) are checked at
load time rather than at launch time, hours into a lab session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmbench.schema import ArrivalProcess, EngineName, LengthSource

__all__ = ["EngineProfile", "SweepConfig", "load_engine_profile", "load_sweep_config"]

PosInt = Annotated[int, Field(gt=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Strict):
    hf_id: str
    #: Resolved commit SHA. A branch name is not reproducible.
    revision: str
    max_model_len: PosInt


class Defaults(_Strict):
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_num_seqs: PosInt
    repeats: PosInt
    warmup_requests: int = Field(ge=0)
    measurement_duration_s: float = Field(gt=0)
    settle_s: float = Field(ge=0)
    ignore_eos: bool


class WorkloadSpec(_Strict):
    arrival_process: ArrivalProcess
    length_source: LengthSource
    seed: int
    request_rates_rps: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _rates_positive_and_sorted(self) -> WorkloadSpec:
        if any(r <= 0 for r in self.request_rates_rps):
            msg = f"request rates must be positive, got {self.request_rates_rps}"
            raise ValueError(msg)
        if self.request_rates_rps != sorted(self.request_rates_rps):
            # Ascending order matters: the runner ramps upward and stops at the
            # first sustained saturation, which only works if rates increase.
            msg = f"request rates must be ascending, got {self.request_rates_rps}"
            raise ValueError(msg)
        return self


class QuantizationSpec(_Strict):
    hf_id: str
    revision: str
    dtype: str
    expected_weights_gib: float = Field(gt=0)
    #: ``None`` for BF16, which has no quantized kernel to select. Any other
    #: value is asserted against the engine's startup log before measuring.
    expected_kernel: str | None = None
    notes: str = ""


class ConfigurationEntry(_Strict):
    id: str
    engine: EngineName
    quantization: str


class MethodologyRun(_Strict):
    """A run that tests the methodology rather than ranking configurations."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    config: str | None = None
    arrival_process: ArrivalProcess | None = None
    length_source: LengthSource | None = None
    concurrency: int | None = None
    repeat_of: str | None = None


class SweepConfig(_Strict):
    schema_version: str
    model: ModelSpec
    defaults: Defaults
    workload: WorkloadSpec
    quantizations: dict[str, QuantizationSpec]
    configurations: list[ConfigurationEntry] = Field(min_length=1)
    methodology_runs: list[MethodologyRun] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_resolve(self) -> SweepConfig:
        """Catch dangling references at load time, not at launch time.

        A configuration naming a quantization level that does not exist would
        otherwise surface after the engine image has been pulled and the model
        loaded — minutes of GPU time to learn about a typo.
        """
        unknown = [c.id for c in self.configurations if c.quantization not in self.quantizations]
        if unknown:
            available = ", ".join(sorted(self.quantizations))
            msg = f"configurations {unknown} reference unknown quantizations; have: {available}"
            raise ValueError(msg)

        ids = [c.id for c in self.configurations]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            msg = f"duplicate configuration ids: {sorted(duplicates)}"
            raise ValueError(msg)

        known = set(ids)
        dangling = [m.id for m in self.methodology_runs if m.config and m.config not in known]
        if dangling:
            msg = f"methodology runs {dangling} reference unknown configurations"
            raise ValueError(msg)
        return self

    def configuration(self, config_id: str) -> ConfigurationEntry:
        for entry in self.configurations:
            if entry.id == config_id:
                return entry
        msg = f"no configuration with id {config_id!r}"
        raise KeyError(msg)

    def quantization_for(self, config_id: str) -> QuantizationSpec:
        return self.quantizations[self.configuration(config_id).quantization]


class EngineProfile(_Strict):
    """Per-engine launch profile from ``configs/engines/*.yaml``."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: EngineName
    image: str
    tag: str
    image_digest: str = ""
    port: int = 8000
    health_path: str = "/health"
    metrics_path: str = "/metrics"
    openai_base_path: str = "/v1"
    args: dict[str, str] = Field(default_factory=dict)
    kernel_log_patterns: dict[str, str] = Field(default_factory=dict)
    startup_timeout_s: int = 900
    metrics: dict[str, str] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"config not found: {path}"
        raise FileNotFoundError(msg)
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        msg = f"{path} must contain a YAML mapping, got {type(loaded).__name__}"
        raise ValueError(msg)
    return loaded


def load_sweep_config(path: Path | str) -> SweepConfig:
    return SweepConfig.model_validate(_read_yaml(Path(path)))


def load_engine_profile(engine: EngineName, configs_dir: Path | str = "configs") -> EngineProfile:
    return EngineProfile.model_validate(
        _read_yaml(Path(configs_dir) / "engines" / f"{engine.value}.yaml")
    )
