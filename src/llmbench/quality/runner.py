"""Quality-evaluation orchestration.

Mirrors the latency sweep's structure — preflight, launch the engine once per
configuration, measure, tear down — but the measurement is accuracy rather than
time, and the constraints differ in one important way.

**Concurrency is pinned low.** vLLM's greedy decoding is not bitwise
deterministic across batch sizes: reduction order inside fused kernels depends
on batch shape, so the same prompt can produce different tokens at different
concurrency. Latency runs deliberately vary concurrency; quality runs must not,
or a configuration's score would move with the load it happened to be measured
under. The pinned value is recorded in every ``QualityResult``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from llmbench.config import SweepConfig, load_engine_profile
from llmbench.engines.base import EngineHandle, EngineLaunchSpec, EngineProcess, resolve_digest
from llmbench.engines.preflight import run_preflight
from llmbench.engines.sglang import SglangEngine
from llmbench.engines.vllm import VllmEngine
from llmbench.quality.datasets import load_wikitext_tokens
from llmbench.quality.harness import HarnessConfig, HarnessError, run_lm_eval
from llmbench.quality.perplexity import compute_perplexity
from llmbench.runner import environment_info
from llmbench.schema import (
    EngineConfig,
    EngineName,
    ModelConfig,
    QualityResult,
    QualityScore,
    QualityTask,
    Quantization,
)
from llmbench.workload.tokenizer import DEFAULT_HF_CACHE, load_tokenizer

__all__ = ["QualityRunner"]

_ENGINES: dict[EngineName, type[EngineProcess]] = {
    EngineName.VLLM: VllmEngine,
    EngineName.SGLANG: SglangEngine,
}

#: Greedy decoding at fixed low concurrency; see the module docstring.
QUALITY_CONCURRENCY = 4


class QualityRunner:
    def __init__(
        self,
        config: SweepConfig,
        *,
        gpu_index: int,
        results_dir: Path,
        wikitext_path: Path,
        configs_dir: Path = Path("configs"),
        hf_cache_dir: Path = DEFAULT_HF_CACHE,
        ppl_tokens: int = 100_000,
        task_limit: int | None = None,
        skip_tasks: bool = False,
    ) -> None:
        self.config = config
        self.gpu_index = gpu_index
        self.results_dir = results_dir
        self.wikitext_path = wikitext_path
        self.configs_dir = configs_dir
        self.hf_cache_dir = hf_cache_dir
        self.ppl_tokens = ppl_tokens
        self.task_limit = task_limit
        self.skip_tasks = skip_tasks

    def _launch_spec(self, config_id: str) -> tuple[EngineLaunchSpec, EngineProcess]:
        entry = self.config.configuration(config_id)
        profile = load_engine_profile(entry.engine, self.configs_dir)
        quant = self.config.quantization_for(config_id)
        digest = profile.image_digest or resolve_digest(profile.image, profile.tag)

        spec = EngineLaunchSpec(
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
        return spec, _ENGINES[entry.engine]()

    def measure(self, handle: EngineHandle) -> QualityResult:
        spec = handle.spec
        quant = self.config.quantization_for(spec.config_id)
        started = datetime.now(UTC)
        scores: list[QualityScore] = []

        tokenizer = load_tokenizer(quant.hf_id, quant.revision, self.hf_cache_dir)
        tokens = load_wikitext_tokens(self.wikitext_path, tokenizer, max_tokens=self.ppl_tokens)
        ppl = compute_perplexity(
            handle.base_url,
            quant.hf_id,
            tokens,
            context_len=spec.max_model_len,
            # Half the context: every token is scored with at least
            # max_model_len/2 tokens of left-context.
            stride=spec.max_model_len // 2,
        )
        print(
            f"  wikitext2 ppl {ppl.perplexity:.4f} "
            f"({ppl.tokens_scored:,} tokens, {ppl.windows} windows)"
        )
        scores.append(
            QualityScore(
                task=QualityTask.WIKITEXT2_PPL,
                metric="perplexity",
                value=ppl.perplexity,
                stderr=None,
                num_samples=ppl.tokens_scored,
            )
        )

        if not self.skip_tasks:
            harness = HarnessConfig(
                base_url=handle.base_url,
                model=quant.hf_id,
                max_concurrent=QUALITY_CONCURRENCY,
                limit=self.task_limit,
            )
            try:
                task_scores = run_lm_eval(
                    harness,
                    ["gsm8k", "ifeval"],
                    self.results_dir / "lm-eval" / spec.config_id,
                )
                scores.extend(task_scores)
                for score in task_scores:
                    print(f"  {score.task.value:<14} {score.metric:<34} {score.value:.4f}")
            except HarnessError as exc:
                # Perplexity already succeeded; losing the task scores should
                # not discard a measurement that cost an engine launch.
                print(f"  [warn] task benchmarks failed: {exc}")

        preflight = run_preflight(
            self.gpu_index,
            results_path=str(self.results_dir),
            gpu_memory_utilization=spec.gpu_memory_utilization,
        )

        return QualityResult(
            run_id=uuid.uuid4().hex,
            config_id=spec.config_id,
            started_at=started,
            finished_at=datetime.now(UTC),
            engine=EngineConfig(
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
            hardware=environment_info(self.gpu_index, preflight),
            temperature=0.0,
            seed=self.config.workload.seed,
            max_concurrency=QUALITY_CONCURRENCY,
            scores=scores,
        )

    def run(self, config_ids: list[str] | None = None) -> list[Path]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        targets = config_ids or [
            c.id for c in self.config.configurations if c.engine is EngineName.VLLM
        ]
        written: list[Path] = []

        for config_id in targets:
            spec, engine = self._launch_spec(config_id)
            print(f"\n=== {config_id} (quality) ===")
            handle = engine.start(spec)
            print(f"  engine ready (v{handle.engine_version}, kernel={handle.selected_kernel})")
            try:
                result = self.measure(handle)
                path = self.results_dir / f"{config_id}__quality.json"
                path.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2) + "\n")
                written.append(path)
            finally:
                engine.stop(spec)
                print("  engine stopped")

        return written
