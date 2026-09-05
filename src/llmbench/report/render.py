"""Render result tables from committed JSON.

Every figure in the documents traces to a file in ``results/``. The generator
refuses to invent values: a configuration with no reportable measurement is
shown as absent rather than filled with its nearest neighbour, because a table
that silently interpolates is worse than one with a gap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from llmbench.analysis.cost import best_under_sla, operating_points
from llmbench.analysis.methodology import drift_comparison, process_comparison
from llmbench.analysis.pareto import ParetoPoint, pareto_frontier
from llmbench.analysis.plots import ParetoMarker
from llmbench.schema import ArrivalProcess, QualityResult, RunResult, RunValidity

__all__ = [
    "MissingBlockError",
    "crossvalidation_table",
    "drift_table",
    "latency_table",
    "load_quality",
    "load_runs",
    "methodology_table",
    "pareto_markers",
    "quality_scores",
    "quality_table",
    "render_into",
    "sla_table",
    "validity_summary",
]


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


#: Quality axis for the headline chart. GSM8K strict-match is generative and
#: genuinely sensitive to quantization damage, unlike a multiple-choice task.
_HEADLINE_TASK = "gsm8k"
_HEADLINE_METRIC = "exact_match,strict-match"


def quality_scores(
    quality: Sequence[QualityResult],
    task: str = _HEADLINE_TASK,
    metric: str = _HEADLINE_METRIC,
) -> dict[str, tuple[float, float]]:
    """Map config id to ``(value, stderr)`` for one task metric.

    A missing stderr becomes 0.0 rather than being dropped: perplexity has no
    sampling error to report, and a configuration without an interval should
    still appear on the chart with no error bar.
    """
    out: dict[str, tuple[float, float]] = {}
    for q in quality:
        for s in q.scores:
            if s.task.value == task and s.metric == metric:
                out[q.config_id] = (s.value, s.stderr or 0.0)
    return out


def pareto_markers(
    runs: Sequence[RunResult],
    quality: Sequence[QualityResult],
    *,
    gpu_hourly_usd: float,
    max_ttft_p95_s: float,
) -> list[ParetoMarker]:
    """Place each configuration on the quality/cost plane under one SLA.

    Configurations without a quality measurement are omitted rather than given
    a placeholder score. A guessed quality value would decide the frontier,
    which is the one thing a chart like this must never invent.
    """
    scores = quality_scores(quality)
    best = best_under_sla(operating_points(runs, gpu_hourly_usd), max_ttft_p95_s=max_ttft_p95_s)

    points = [
        ParetoPoint(
            config_id=cid,
            quality=scores[cid][0],
            latency_s=op.ttft_p95_s,
            cost_per_1m_usd=op.cost_per_1m_usd,
            throughput_tokens_s=op.throughput_tokens_s,
        )
        for cid, op in sorted(best.items())
        if cid in scores
    ]
    frontier, _ = pareto_frontier(points)
    on_frontier = {p.config_id for p in frontier}

    return [
        ParetoMarker(
            config_id=p.config_id,
            cost_per_1m_usd=p.cost_per_1m_usd,
            quality=p.quality,
            quality_ci=1.96 * scores[p.config_id][1],
            throughput_tokens_s=p.throughput_tokens_s,
            on_frontier=p.config_id in on_frontier,
        )
        for p in points
    ]


def sla_table(
    runs: Sequence[RunResult],
    *,
    gpu_hourly_usd: float,
    ttft_budgets_ms: Sequence[float],
) -> str:
    """Cost-optimal configuration as a function of the p95 TTFT budget.

    This is the sensitivity analysis that has a real answer. Because every
    configuration runs on the same GPU, the hourly price is a linear scalar and
    cannot reorder anything — but the latency budget decides which offered rates
    are admissible, which caps sustainable throughput, which sets cost per
    token. Configurations therefore *can* swap places as the budget tightens.
    """
    points = operating_points(runs, gpu_hourly_usd)
    lines = [
        "| p95 TTFT budget | Cheapest config | Rate | Throughput | $/1M tokens | Admissible |",
        "|---|---|---|---|---|---|",
    ]
    for budget_ms in ttft_budgets_ms:
        best = best_under_sla(points, max_ttft_p95_s=budget_ms / 1e3)
        if not best:
            lines.append(f"| {budget_ms:.0f} ms | — | — | — | — | 0 |")
            continue
        winner = min(best.values(), key=lambda p: p.cost_per_1m_usd)
        lines.append(
            f"| {budget_ms:.0f} ms | `{winner.config_id}` | {winner.rate_rps:g} rps "
            f"| {winner.throughput_tokens_s:.0f} tok/s | ${winner.cost_per_1m_usd:.4f} "
            f"| {len(best)} of {len(points)} |"
        )
    return "\n".join(lines)


_BLOCK = re.compile(
    r"<!--\s*BEGIN:(?P<key>[a-z0-9-]+)\s*-->.*?<!--\s*END:(?P=key)\s*-->",
    re.DOTALL,
)


class MissingBlockError(RuntimeError):
    """A generated block was produced but the document has nowhere to put it."""


def render_into(path: Path, blocks: Mapping[str, str]) -> set[str]:
    """Replace marked regions of a Markdown document in place.

    Prose stays hand-written; every number lives inside a
    ``<!-- BEGIN:key -->`` / ``<!-- END:key -->`` pair and is replaced wholesale
    from the result JSON. That split is what lets ``make report`` be idempotent:
    rebuilding on unchanged results must leave ``git diff`` empty, which is the
    only real proof that no figure in the documents was typed by hand.

    Returns:
        The keys actually substituted.

    Raises:
        MissingBlockError: If a supplied block has no marker in the document.
            Silently dropping it would let a stale hand-written table survive a
            rebuild and look generated.
    """
    text = path.read_text()
    seen: set[str] = set()

    def substitute(match: re.Match[str]) -> str:
        key = match.group("key")
        if key not in blocks:
            return match.group(0)
        seen.add(key)
        # Emitted in a normal form rather than preserving whatever whitespace
        # the document had, so a second run produces a byte-identical file. An
        # empty placeholder block and a filled one round-trip the same way.
        return f"<!-- BEGIN:{key} -->\n{blocks[key]}\n<!-- END:{key} -->"

    updated = _BLOCK.sub(substitute, text)
    missing = set(blocks) - seen
    if missing:
        msg = f"{path}: no <!-- BEGIN:... --> marker for {sorted(missing)}"
        raise MissingBlockError(msg)

    if updated != text:
        path.write_text(updated)
    return seen


def methodology_table(
    runs: Sequence[RunResult], *, baseline_config_id: str, rate_rps: float
) -> str:
    """Coordinated omission and burstiness, each against its matched baseline.

    Rows appear only for exhibits that have actually been run. A table that
    invented a row for an unexecuted run would be claiming a measurement.
    """
    lines = [
        "| Arrival process | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | vs Poisson (p99) |",
        "|---|---|---|---|---|---|",
    ]
    baseline_shown = False
    rows = 0

    for process, label in (
        (ArrivalProcess.CLOSED_LOOP, "Closed loop (exhibit)"),
        (ArrivalProcess.TRACE_REPLAY, "Azure trace replay"),
    ):
        cmp = process_comparison(
            runs,
            process=process,
            baseline_config_id=baseline_config_id,
            baseline_rate_rps=rate_rps,
            label=label,
        )
        if cmp is None:
            continue
        if not baseline_shown:
            lines.append(
                f"| **Open-loop Poisson** (baseline) | {cmp.baseline_ttft_p50_ms:.0f} ms "
                f"| {cmp.baseline_ttft_p95_ms:.0f} ms | {cmp.baseline_ttft_p99_ms:.0f} ms "
                f"| {cmp.baseline_throughput:.0f} tok/s | — |"
            )
            baseline_shown = True
        ratio = cmp.understatement_ratio("p99")
        verdict = f"{ratio:.2f}x {'understated' if ratio > 1 else 'worse'}"
        if not cmp.comparable:
            verdict += f" (throughput {cmp.throughput_ratio:.0%} of baseline — not comparable)"
        lines.append(
            f"| {cmp.label} | {cmp.ttft_p50_ms:.0f} ms | {cmp.ttft_p95_ms:.0f} ms "
            f"| {cmp.ttft_p99_ms:.0f} ms | {cmp.throughput:.0f} tok/s | {verdict} |"
        )
        rows += 1

    if rows == 0:
        return "_No methodology exhibits have been run yet._"
    return "\n".join(lines)


def drift_table(runs: Sequence[RunResult], *, config_id: str, rate_rps: float) -> str:
    """Whether the environment moved across the sweep."""
    drift = drift_comparison(runs, config_id=config_id, canary_label=config_id, rate_rps=rate_rps)
    if drift is None:
        return "_The drift canary has not been run yet._"

    verdict = (
        "within noise — environmental drift across the sweep is bounded"
        if drift.within_noise
        else "**outside noise** — results measured hours apart are not directly comparable"
    )
    return "\n".join(
        [
            "| | First measurement | Re-run at end of sweep | Δ |",
            "|---|---|---|---|",
            f"| TTFT p95 | {drift.first_ttft_p95_ms:.1f} ms | {drift.later_ttft_p95_ms:.1f} ms "
            f"| {drift.ttft_drift_ms:+.1f} ms |",
            f"| Throughput | {drift.first_throughput:.0f} tok/s "
            f"| {drift.later_throughput:.0f} tok/s | {drift.throughput_drift_pct:+.1f} % |",
            "",
            f"Repeat-to-repeat std of the original: ±{drift.first_ttft_std_ms:.1f} ms. "
            f"The re-run is {verdict}.",
        ]
    )


def crossvalidation_table(path: Path) -> str:
    """Agreement with vLLM's own harness, read from a completed comparison."""
    if not path.exists():
        return "_Cross-validation against `vllm bench serve` has not been run yet._"

    payload = json.loads(path.read_text())
    lines = [
        f"One configuration (`{payload['config_id']}` at {payload['rate_rps']:g} rps) measured "
        f"independently by vLLM's own `vllm bench serve`, against the same live server:",
        "",
        "| Metric | This harness | `vllm bench serve` | Δ | Agrees (±5 %) |",
        "|---|---|---|---|---|",
    ]
    for m in payload["metrics"]:
        lines.append(
            f"| {m['metric']} | {m['ours']:.2f} {m['unit']} | {m['upstream']:.2f} {m['unit']} "
            f"| {m['delta_pct']:+.1f} % | {'yes' if m['agrees'] else '**NO**'} |"
        )
    return "\n".join(lines)
