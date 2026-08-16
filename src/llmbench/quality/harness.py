"""Task benchmarks via lm-evaluation-harness against the live engine.

Scoring is delegated to lm-evaluation-harness rather than hand-rolled. Exact-match
normalisation for GSM8K and the verifiable constraint checks in IFEval are both
fiddly enough that a bespoke implementation would be a source of error rather
than a source of confidence, and using the standard harness keeps the numbers
comparable to published results.

The harness runs in its **own virtualenv**, invoked as a subprocess. It depends
on torch, and the load generator must stay lightweight — a heavy client is the
thing this project measures rather than ships. Keeping torch out of the main
dependency set also keeps CI fast and GPU-free.

Tasks, and why these two:

* **GSM8K** (8-shot, exact match) — generative and genuinely sensitive to
  quantization damage, unlike loglikelihood-ranking benchmarks.
* **IFEval** — programmatically verifiable instruction following, ~540 prompts.
  Cheap, and it catches a failure mode arithmetic accuracy does not.

MMLU is excluded: 14k mostly-knowledge questions, expensive, and comparatively
insensitive to quantization.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llmbench.schema import QualityScore, QualityTask

__all__ = ["TASK_METRICS", "HarnessError", "run_lm_eval"]


class HarnessError(RuntimeError):
    """lm-evaluation-harness failed to run or produced no parsable scores."""


#: Metric to extract per task, and the schema task it maps to. lm-eval reports
#: several metrics per task; picking explicitly avoids silently reporting
#: whichever happens to come first.
TASK_METRICS: dict[str, tuple[QualityTask, tuple[str, ...]]] = {
    "gsm8k": (QualityTask.GSM8K, ("exact_match,strict-match", "exact_match,flexible-extract")),
    "ifeval": (
        QualityTask.IFEVAL,
        ("prompt_level_strict_acc,none", "inst_level_strict_acc,none"),
    ),
}


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """How to reach the engine and where the harness virtualenv lives."""

    base_url: str
    model: str
    venv_python: Path = Path(".venv-lmeval/bin/python")
    #: Fixed low concurrency. vLLM greedy decoding is not bitwise-deterministic
    #: across batch sizes — reduction order in fused kernels varies with batch
    #: shape — so quality runs pin concurrency to keep configurations
    #: comparable to each other.
    max_concurrent: int = 4
    num_fewshot: int | None = None
    limit: int | None = None


def _build_command(config: HarnessConfig, tasks: Sequence[str], output_dir: Path) -> list[str]:
    model_args = ",".join(
        [
            f"model={config.model}",
            f"base_url={config.base_url.rstrip('/')}/v1/completions",
            f"num_concurrent={config.max_concurrent}",
            "tokenized_requests=False",
            "max_retries=3",
        ]
    )
    command = [
        str(config.venv_python),
        "-m",
        "lm_eval",
        # Talks to the running server over HTTP; no second copy of the model.
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "1",
        "--output_path",
        str(output_dir),
        # Greedy, seeded, identical prompts across configurations.
        "--seed",
        "0",
    ]
    if config.num_fewshot is not None:
        command += ["--num_fewshot", str(config.num_fewshot)]
    if config.limit is not None:
        command += ["--limit", str(config.limit)]
    return command


def parse_lm_eval_results(payload: dict[str, object]) -> list[QualityScore]:
    """Extract the metrics of interest from an lm-eval results document."""
    results = payload.get("results")
    if not isinstance(results, dict):
        msg = "lm-eval output contained no 'results' object"
        raise HarnessError(msg)

    counts = payload.get("n-samples")
    scores: list[QualityScore] = []

    for task_name, task_result in results.items():
        mapping = TASK_METRICS.get(task_name)
        if mapping is None or not isinstance(task_result, dict):
            continue
        task, metric_keys = mapping

        n_samples = 1
        if isinstance(counts, dict) and isinstance(counts.get(task_name), dict):
            effective = counts[task_name].get("effective")
            if isinstance(effective, int) and effective > 0:
                n_samples = effective

        for metric_key in metric_keys:
            value = task_result.get(metric_key)
            if not isinstance(value, int | float):
                continue
            stderr_key = metric_key.replace(",", "_stderr,", 1)
            stderr = task_result.get(stderr_key)
            scores.append(
                QualityScore(
                    task=task,
                    metric=metric_key,
                    value=float(value),
                    stderr=float(stderr) if isinstance(stderr, int | float) else None,
                    num_samples=n_samples,
                )
            )

    if not scores:
        msg = f"no recognised metrics in lm-eval output; saw tasks {sorted(results)}"
        raise HarnessError(msg)
    return scores


def _newest_results_file(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        msg = f"lm-eval wrote no results file under {output_dir}"
        raise HarnessError(msg)
    return candidates[-1]


def run_lm_eval(
    config: HarnessConfig, tasks: Sequence[str], output_dir: Path, *, timeout_s: float = 7200.0
) -> list[QualityScore]:
    """Run the harness and return parsed scores.

    Raises:
        HarnessError: If the virtualenv is missing, the harness exits non-zero,
            or its output contains none of the expected metrics.
    """
    if not config.venv_python.exists():
        msg = (
            f"lm-eval virtualenv not found at {config.venv_python}. "
            f"Create it with `make lmeval-env` — it is kept separate because "
            f"lm-eval depends on torch and the load generator must stay light."
        )
        raise HarnessError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(config, tasks, output_dir)

    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_s)
    if result.returncode != 0:
        msg = f"lm-eval exited {result.returncode}:\n{result.stderr[-2000:]}"
        raise HarnessError(msg)

    payload = json.loads(_newest_results_file(output_dir).read_text())
    return parse_lm_eval_results(payload)
