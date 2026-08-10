"""Sweep-config validation.

A config error that only surfaces at launch costs an engine pull and a model
load to discover. Every check here is designed to fail in milliseconds instead,
and the committed configs are validated as part of the suite so a hand edit
cannot quietly break the matrix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llmbench.config import SweepConfig, load_engine_profile, load_sweep_config
from llmbench.schema import EngineName

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def minimal() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "model": {
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "a" * 40,
            "max_model_len": 4096,
        },
        "defaults": {
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 256,
            "repeats": 3,
            "warmup_requests": 50,
            "measurement_duration_s": 180.0,
            "settle_s": 20.0,
            "ignore_eos": True,
        },
        "workload": {
            "arrival_process": "poisson",
            "length_source": "sharegpt",
            "seed": 1,
            "request_rates_rps": [1.0, 4.0],
        },
        "quantizations": {
            "bf16": {
                "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
                "revision": "a" * 40,
                "dtype": "bfloat16",
                "expected_weights_gib": 15.0,
                "expected_kernel": None,
            }
        },
        "configurations": [{"id": "vllm-bf16", "engine": "vllm", "quantization": "bf16"}],
        "methodology_runs": [],
    }


class TestCommittedConfigs:
    """The real configs must stay valid; they are the sweep matrix."""

    @pytest.mark.parametrize("name", ["sweep.yaml", "smoke.yaml"])
    def test_loads(self, name: str) -> None:
        config = load_sweep_config(CONFIGS / name)
        assert config.configurations
        assert config.workload.request_rates_rps

    def test_sweep_covers_the_intended_quantization_axis(self) -> None:
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        assert set(config.quantizations) == {"bf16", "int8-w8a8", "gptq-int4", "awq-int4"}

    def test_fp8_is_absent(self) -> None:
        """The A40 is sm_86 and has no FP8 tensor cores.

        vLLM will still *load* an FP8 checkpoint by dequantizing to FP16,
        producing a plausible-looking and meaningless number. Its absence from
        the matrix is a deliberate result, so it is asserted.
        """
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        assert not any("fp8" in key.lower() for key in config.quantizations)

    def test_every_quantized_config_asserts_a_kernel(self) -> None:
        """Only BF16 may skip the kernel assertion."""
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        for name, quant in config.quantizations.items():
            if name == "bf16":
                assert quant.expected_kernel is None
            else:
                assert quant.expected_kernel, f"{name} must declare an expected kernel"

    def test_revisions_are_commit_shas(self) -> None:
        """A branch name is not reproducible."""
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        for quant in config.quantizations.values():
            assert len(quant.revision) == 40, f"{quant.hf_id} revision is not a commit SHA"

    @pytest.mark.parametrize("engine", [EngineName.VLLM, EngineName.SGLANG])
    def test_engine_profiles_load(self, engine: EngineName) -> None:
        profile = load_engine_profile(engine, CONFIGS)
        assert profile.name is engine
        assert profile.tag

    def test_engine_tags_are_pinned_not_moving(self) -> None:
        """`latest` and `nightly` move underneath you and make earlier results
        unreproducible."""
        for engine in (EngineName.VLLM, EngineName.SGLANG):
            tag = load_engine_profile(engine, CONFIGS).tag
            assert tag not in {"latest", "nightly"}
            assert not tag.startswith("nightly")

    def test_sglang_uses_a_cuda_12_image(self) -> None:
        """CUDA 13 needs driver >= 580; this host runs 570.

        A cu130 image fails at container start with a CUDA init error that never
        mentions the driver, so the constraint is pinned here.
        """
        assert "cu129" in load_engine_profile(EngineName.SGLANG, CONFIGS).tag


class TestCrossReferences:
    def test_unknown_quantization_is_rejected(self) -> None:
        raw = minimal()
        raw["configurations"] = [{"id": "x", "engine": "vllm", "quantization": "fp8"}]
        with pytest.raises(ValidationError, match="unknown quantizations"):
            SweepConfig.model_validate(raw)

    def test_duplicate_config_ids_are_rejected(self) -> None:
        raw = minimal()
        raw["configurations"] = [
            {"id": "dup", "engine": "vllm", "quantization": "bf16"},
            {"id": "dup", "engine": "sglang", "quantization": "bf16"},
        ]
        with pytest.raises(ValidationError, match="duplicate configuration ids"):
            SweepConfig.model_validate(raw)

    def test_methodology_run_referencing_a_missing_config_is_rejected(self) -> None:
        raw = minimal()
        raw["methodology_runs"] = [{"id": "canary", "config": "does-not-exist"}]
        with pytest.raises(ValidationError, match="unknown configurations"):
            SweepConfig.model_validate(raw)


class TestStrictness:
    def test_typo_in_a_key_is_an_error(self) -> None:
        """Silently falling back to a default would run a different experiment
        than the one written down."""
        raw = minimal()
        raw["defaults"]["gpu_memory_utilisation"] = 0.8  # type: ignore[index]
        with pytest.raises(ValidationError, match="gpu_memory_utilisation"):
            SweepConfig.model_validate(raw)

    def test_descending_rates_are_rejected(self) -> None:
        """The runner ramps upward and stops at first sustained saturation,
        which only works if rates ascend."""
        raw = minimal()
        raw["workload"]["request_rates_rps"] = [8.0, 2.0]  # type: ignore[index]
        with pytest.raises(ValidationError, match="ascending"):
            SweepConfig.model_validate(raw)

    def test_non_positive_rate_is_rejected(self) -> None:
        raw = minimal()
        raw["workload"]["request_rates_rps"] = [0.0, 4.0]  # type: ignore[index]
        with pytest.raises(ValidationError, match="positive"):
            SweepConfig.model_validate(raw)

    def test_memory_utilization_above_one_is_rejected(self) -> None:
        raw = minimal()
        raw["defaults"]["gpu_memory_utilization"] = 1.5  # type: ignore[index]
        with pytest.raises(ValidationError):
            SweepConfig.model_validate(raw)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="config not found"):
            load_sweep_config(CONFIGS / "nope.yaml")


class TestLookups:
    def test_quantization_for_config(self) -> None:
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        assert config.quantization_for("vllm-gptq-int4").expected_kernel == "gptq_marlin"
        assert config.quantization_for("vllm-awq-int4").expected_kernel == "awq_marlin"

    def test_unknown_config_id_raises(self) -> None:
        config = load_sweep_config(CONFIGS / "sweep.yaml")
        with pytest.raises(KeyError, match="no configuration with id"):
            config.configuration("nope")


class TestYamlHygiene:
    @pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.name)
    def test_every_committed_yaml_parses(self, path: Path) -> None:
        assert isinstance(yaml.safe_load(path.read_text()), dict)
