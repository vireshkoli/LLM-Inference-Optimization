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
from dataclasses import dataclass, replace
from pathlib import Path

from llmbench.schema import QualityScore, QualityTask

__all__ = [
    "TASK_BACKEND",
    "TASK_CHAT_TEMPLATE",
    "TASK_FEWSHOT",
    "TASK_METRICS",
    "HarnessError",
    "run_lm_eval",
]


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

#: Few-shot count per task, stated explicitly rather than left to lm-eval's
#: per-task defaults.
#:
#: This forces **one invocation per task**: ``--num_fewshot`` applies to every
#: task in a run, so evaluating both together would either leave GSM8K at
#: lm-eval's 5-shot default (contradicting the documented 8-shot) or force
#: few-shot prompting onto IFEval, which is 0-shot by design and whose
#: verifiable constraints assume it.
TASK_FEWSHOT: dict[str, int] = {
    "gsm8k": 8,
    "ifeval": 0,
}

#: Which lm-eval backend and endpoint each task is served through.
#:
#: Not interchangeable. IFEval is a 0-shot *instruction-following* benchmark and
#: must go through /v1/chat/completions so the server applies the model's chat
#: template; through raw /v1/completions an instruct model merely continues text
#: and the score collapses. Measured on Llama-3.1-8B-Instruct BF16:
#:
#:     prompt_level_strict_acc   0.4603 raw  ->  0.7000 templated
#:     inst_level_strict_acc     0.6019 raw  ->  0.7937 templated
#:
#: GSM8K stays on raw completions: its 8-shot exemplars establish the pattern
#: without a template, which is the conventional few-shot setup and reproduces
#: published numbers here (0.7286 strict / 0.7832 flexible).
TASK_BACKEND: dict[str, tuple[str, str]] = {
    "gsm8k": ("local-completions", "/v1/completions"),
    "ifeval": ("local-chat-completions", "/v1/chat/completions"),
}

#: Whether a task needs the model's chat template applied.
#:
#: This is not cosmetic. IFEval is a 0-shot *instruction-following* benchmark:
#: served through raw /v1/completions with no template, an instruct model simply
#: continues the text instead of behaving as an assistant, and the score
#: collapses. Measured on Llama-3.1-8B-Instruct BF16, prompt-level strict
#: accuracy was 0.4603 without the template against ~0.78-0.80 published — a
#: ~32-point gap that is an artifact of prompting, not of the model.
#:
#: GSM8K deliberately stays off: its 8-shot exemplars establish the pattern
#: without a template, which is the conventional few-shot completion setup and
#: reproduces published numbers (0.7286 strict / 0.7832 flexible here).
TASK_CHAT_TEMPLATE: dict[str, bool] = {
    "gsm8k": False,
    "ifeval": True,
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
    #: Apply the model's chat template. Required for instruction-following
    #: tasks; see TASK_CHAT_TEMPLATE.
    apply_chat_template: bool = False


def _build_command(config: HarnessConfig, tasks: Sequence[str], output_dir: Path) -> list[str]:
    backend, endpoint = TASK_BACKEND.get(tasks[0], ("local-completions", "/v1/completions"))
    model_args = ",".join(
        [
            f"model={config.model}",
            f"base_url={config.base_url.rstrip('/')}{endpoint}",
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
        backend,
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
    if config.apply_chat_template:
        command.append("--apply_chat_template")
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


def _run_one_task(
    config: HarnessConfig, task: str, output_dir: Path, timeout_s: float
) -> list[QualityScore]:
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    per_task = replace(
        config,
        num_fewshot=TASK_FEWSHOT.get(task, config.num_fewshot),
        apply_chat_template=TASK_CHAT_TEMPLATE.get(task, config.apply_chat_template),
    )
    command = _build_command(per_task, [task], task_dir)

    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_s)
    if result.returncode != 0:
        msg = f"lm-eval exited {result.returncode} on task {task!r}:\n{result.stderr[-2000:]}"
        raise HarnessError(msg)

    return parse_lm_eval_results(json.loads(_newest_results_file(task_dir).read_text()))


def run_lm_eval(
    config: HarnessConfig, tasks: Sequence[str], output_dir: Path, *, timeout_s: float = 7200.0
) -> list[QualityScore]:
    """Run each task in its own invocation and return the combined scores.

    One invocation per task is required, not merely tidier: ``--num_fewshot`` is
    global to a run, and GSM8K (8-shot) and IFEval (0-shot) need different
    values. Batching them would silently mis-prompt one of the two.

    Raises:
        HarnessError: If the virtualenv is missing, a task exits non-zero, or
            its output contains none of the expected metrics.
    """
    if not config.venv_python.exists():
        msg = (
            f"lm-eval virtualenv not found at {config.venv_python}. "
            f"Create it with `make lmeval-env` — it is kept separate because "
            f"lm-eval depends on torch and the load generator must stay light."
        )
        raise HarnessError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[QualityScore] = []
    for task in tasks:
        scores.extend(_run_one_task(config, task, output_dir, timeout_s))
    return scores
