"""Render result tables from committed JSON.

Every figure in the documents traces to a file in ``results/``. The generator
refuses to invent values: a configuration with no reportable measurement is
shown as absent rather than filled with its nearest neighbour, because a table
that silently interpolates is worse than one with a gap.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llmbench.analysis.cost import operating_points
from llmbench.schema import QualityResult, RunResult, RunValidity

__all__ = ["latency_table", "load_quality", "load_runs", "quality_table", "validity_summary"]


def load_runs(results_dir: Path) -> list[RunResult]:
    """Load and re-validate every latency record."""
    return [
        RunResult.model_validate_json(p.read_text()) for p in sorted(results_dir.glob("*.json"))
    ]


def load_quality(results_dir: Path) -> list[QualityResult]:
    return [
        QualityResult.model_validate_json(p.read_text())
        for p in sorted(results_dir.glob("*__quality.json"))
    ]


def _fmt(value: float | None, spec: str, dash: str = "—") -> str:
    return dash if value is None else format(value, spec)


def quality_table(quality: Sequence[QualityResult]) -> str:
    """Quality across quantization levels, deltas relative to BF16."""
    rows: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    for q in quality:
        scores = {f"{s.task.value}|{s.metric}": s.value for s in q.scores}
        rows[q.config_id] = scores
        weights[q.config_id] = q.model.weights_gib

    base = rows.get("vllm-bf16", {})
    ppl_k = "wikitext2-ppl|perplexity"
    gsm_k = "gsm8k|exact_match,strict-match"
    ife_k = "ifeval|prompt_level_strict_acc,none"

    out = [
        "| Config | Weights | WikiText-2 PPL | ΔPPL | GSM8K | Δ | IFEval | Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in ("vllm-bf16", "vllm-int8-w8a8", "vllm-awq-int4", "vllm-gptq-int4"):
        r = rows.get(cfg)
        if r is None:
            continue

        def delta(key: str, row: dict[str, float] = r) -> float | None:
            return row[key] - base[key] if key in row and key in base else None

        out.append(
            f"| `{cfg}` | {weights[cfg]:.1f} GiB "
            f"| {_fmt(r.get(ppl_k), '.4f')} | {_fmt(delta(ppl_k), '+.4f')} "
            f"| {_fmt(r.get(gsm_k), '.4f')} | {_fmt(delta(gsm_k), '+.4f')} "
            f"| {_fmt(r.get(ife_k), '.4f')} | {_fmt(delta(ife_k), '+.4f')} |"
        )
    return "\n".join(out)


def latency_table(runs: Sequence[RunResult], gpu_hourly_usd: float) -> str:
    """Reportable operating points with cost per million output tokens."""
    points = operating_points(runs, gpu_hourly_usd)
    out = [
        "| Config | Rate | Throughput | TTFT p95 | TPOT p95 | $/1M tokens | Repeats |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in sorted(points):
        for p in points[cfg]:
            out.append(
                f"| `{cfg}` | {p.rate_rps:g} rps "
                f"| {p.throughput_tokens_s:.0f} ± {p.throughput_std:.0f} tok/s "
                f"| {p.ttft_p95_s * 1e3:.0f} ms | {p.tpot_p95_s * 1e3:.1f} ms "
                f"| ${p.cost_per_1m_usd:.4f} | {p.repeats} |"
            )
    return "\n".join(out)


def validity_summary(runs: Sequence[RunResult]) -> str:
    """How many measurements were reportable, and why the rest were not.

    Published prominently rather than buried: a benchmark that shows only its
    usable runs has hidden its own error bars.
    """
    counts: dict[RunValidity, int] = {}
    for r in runs:
        counts[r.validity] = counts.get(r.validity, 0) + 1

    lines = ["| Validity | Runs | Meaning |", "|---|---|---|"]
    meaning = {
        RunValidity.VALID: "reportable",
        RunValidity.OVERSUBSCRIBED: "offered load beyond capacity; latency reflects run duration",
        RunValidity.CLIENT_SATURATED: "load generator became the bottleneck",
        RunValidity.ENGINE_ERROR: "server failed",
        RunValidity.THERMAL_THROTTLED: "GPU not at steady-state clocks",
        RunValidity.INCOMPLETE: "too few requests completed",
    }
    for validity, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{validity.value}` | {n} | {meaning.get(validity, '')} |")
    return "\n".join(lines)


def write_summary_json(runs: Sequence[RunResult], out_path: Path, gpu_hourly_usd: float) -> Path:
    """Flat records for the static results explorer."""
    points = operating_points(runs, gpu_hourly_usd)
    payload = [
        {
            "config_id": p.config_id,
            "rate_rps": p.rate_rps,
            "throughput_tokens_s": round(p.throughput_tokens_s, 2),
            "throughput_std": round(p.throughput_std, 2),
            "ttft_p95_ms": round(p.ttft_p95_s * 1e3, 2),
            "tpot_p95_ms": round(p.tpot_p95_s * 1e3, 3),
            "cost_per_1m_usd": round(p.cost_per_1m_usd, 5),
            "repeats": p.repeats,
        }
        for cfg in sorted(points)
        for p in points[cfg]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def cost_note(gpu_hourly_usd: float, vendor: str, url: str, accessed: str) -> str:
    return (
        f"Cost assumes **${gpu_hourly_usd:.2f}/GPU-hour** ({vendor}, {url}, accessed {accessed}). "
        f"The benchmark GPU is a lab machine with no invoice, so this is an assumption. "
        f"Because every configuration runs on the same GPU, the hourly rate is a linear scalar "
        f"and the *ranking* is invariant to it — there is no crossover price."
    )
