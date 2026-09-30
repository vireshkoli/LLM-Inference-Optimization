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
from typing import Any

from llmbench.analysis.cost import ConfigOperatingPoint, best_under_sla, operating_points
from llmbench.analysis.methodology import (
    ProcessComparison,
    _latest_session,
    drift_comparison,
    process_comparison,
)
from llmbench.analysis.pareto import ParetoPoint, pareto_frontier
from llmbench.analysis.plots import ParetoMarker
from llmbench.schema import ArrivalProcess, QualityResult, RunResult, RunValidity

__all__ = [
    "MissingBlockError",
    "bandwidth_table",
    "crossvalidation_table",
    "drift_table",
    "engine_axis_table",
    "findings_summary",
    "headline_facts",
    "latency_table",
    "load_quality",
    "load_runs",
    "methodology_summary",
    "methodology_table",
    "pareto_markers",
    "quality_scores",
    "quality_table",
    "regime_table",
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


def cost_note(
    gpu_hourly_usd: float,
    vendor: str,
    url: str,
    accessed: str,
    alternates: Sequence[tuple[str, float]] = (),
) -> str:
    note = (
        f"Cost assumes **${gpu_hourly_usd:.2f}/GPU-hour** ({vendor}, {url}, accessed {accessed}). "
        f"The benchmark GPU is a lab machine with no invoice, so this is an assumption. "
        f"Because every configuration runs on the same GPU, the hourly rate is a linear scalar "
        f"and the *ranking* is invariant to it — there is no crossover price."
    )
    if alternates:
        listed = ", ".join(
            f"{label} at ${price:.2f} (x{price / gpu_hourly_usd:.2f})"
            for label, price in alternates
        )
        note += (
            f" At the alternate prices in the cost file — {listed} — every cost scales by "
            f"that factor."
        )
    return note


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
    """Burstiness against Poisson at one rate; closed loop at every matched rate.

    The closed-loop exhibit is shown at each open-loop rate it was matched to,
    because its result *changes sign* with load: worse than open loop at light
    load, indistinguishable in the middle, better near capacity. One row would
    have to pick a load, and whichever it picked would misrepresent the others.

    Rows appear only for exhibits that have actually been run. A table that
    invented a row for an unexecuted run would be claiming a measurement.
    """
    lines = [
        "| Arrival process | Matched to | TTFT p50 | TTFT p95 | TTFT p99 | Throughput "
        "| Tail vs open-loop Poisson |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = 0

    def verdict(cmp: ProcessComparison) -> str:
        if not cmp.comparable:
            return (
                f"not comparable — throughput {cmp.throughput_ratio:.0%} of baseline, "
                f"so this measures the load, not the generator"
            )
        gap = cmp.ttft_p99_ms - cmp.baseline_ttft_p99_ms
        if not cmp.p99_significant:
            return (
                f"indistinguishable ({gap:+.0f} ms against ±{2 * cmp.p99_pooled_stderr_ms:.0f} ms)"
            )
        ratio = cmp.understatement_ratio("p99")
        if ratio > 1:
            return f"**{_times(ratio)} understated** (p99 {gap:+.0f} ms)"
        return f"**{_times(1 / ratio)} worse** (p99 {gap:+.0f} ms)"

    trace = process_comparison(
        runs,
        process=ArrivalProcess.TRACE_REPLAY,
        baseline_config_id=baseline_config_id,
        baseline_rate_rps=rate_rps,
        label="Azure trace replay",
    )
    if trace is not None:
        lines.append(
            f"| **Open-loop Poisson** (baseline) | {rate_rps:g} rps "
            f"| {trace.baseline_ttft_p50_ms:.0f} ms | {trace.baseline_ttft_p95_ms:.0f} ms "
            f"| {trace.baseline_ttft_p99_ms:.0f} ms | {trace.baseline_throughput:.0f} tok/s | — |"
        )
        lines.append(
            f"| {trace.label} | {rate_rps:g} rps | {trace.ttft_p50_ms:.0f} ms "
            f"| {trace.ttft_p95_ms:.0f} ms | {trace.ttft_p99_ms:.0f} ms "
            f"| {trace.throughput:.0f} tok/s | {verdict(trace)} |"
        )
        rows += 1

    # Every open-loop rate that has a throughput-matched closed-loop partner.
    rates = sorted(
        {
            r.workload.request_rate_rps
            for r in runs
            if r.config_id == baseline_config_id
            and r.workload.request_rate_rps is not None
            and r.is_reportable
        }
    )
    for rate in rates:
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.CLOSED_LOOP,
            baseline_config_id=baseline_config_id,
            baseline_rate_rps=rate,
            label="Closed loop",
        )
        if cmp is None or not cmp.comparable:
            continue
        lines.append(
            f"| {cmp.label} | {rate:g} rps ({cmp.baseline_ttft_p99_ms:.0f} ms p99) "
            f"| {cmp.ttft_p50_ms:.0f} ms | {cmp.ttft_p95_ms:.0f} ms | {cmp.ttft_p99_ms:.0f} ms "
            f"| {cmp.throughput:.0f} tok/s | {verdict(cmp)} |"
        )
        rows += 1

    if rows == 0:
        return "_No methodology exhibits have been run yet._"
    return "\n".join(lines)


def _times(ratio: float) -> str:
    """A ratio as ``1.6x``, or ``1.04x`` when one decimal would round it to 1.0."""
    return f"{ratio:.2f}x" if ratio < 1.5 else f"{ratio:.1f}x"


def _signed(value: float, digits: int = 1) -> str:
    """``+1.2``/``-1.2`` without the ``-0.0`` that rounding a tiny negative gives."""
    rounded = round(value, digits)
    return f"{(rounded or 0.0):+.{digits}f}"


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
            f"| {drift.later_throughput:.0f} tok/s | {_signed(drift.throughput_drift_pct)} % |",
            "",
            f"Repeat-to-repeat std of the original: ±{drift.first_ttft_std_ms:.1f} ms. "
            f"The re-run is {verdict}.",
        ]
    )


#: The note the runner records for every trace replay (runner._trace_window_note).
_TRACE_NOTE = re.compile(
    r"trace window (?P<start>[\d.]+)-[\d.]+ s of the trace: .*? time-scaled "
    r"x(?P<scale>[\d.]+) to [\d.]+ rps; CV\^2 (?P<cv2>[\d.]+)"
)


def _closed_loop_summary(runs: Sequence[RunResult], *, baseline_config_id: str) -> str | None:
    rates = sorted(
        {
            r.workload.request_rate_rps
            for r in runs
            if r.config_id == baseline_config_id
            and r.workload.request_rate_rps is not None
            and r.is_reportable
        }
    )
    parts: list[str] = []
    better = 0
    for rate in rates:
        cmp = process_comparison(
            runs,
            process=ArrivalProcess.CLOSED_LOOP,
            baseline_config_id=baseline_config_id,
            baseline_rate_rps=rate,
            label="closed loop",
        )
        if cmp is None or not cmp.comparable:
            continue
        n = re.search(r"N=(\d+)", cmp.label)
        where = f"{rate:g} rps (N={n.group(1)})" if n else f"{rate:g} rps"
        if not cmp.p99_significant:
            parts.append(f"{where} indistinguishable")
            continue
        ratio = cmp.understatement_ratio("p99")
        if ratio > 1:
            better += 1
            parts.append(f"{where} {_times(ratio)} better")
        else:
            parts.append(f"{where} {_times(1 / ratio)} worse")
    if not parts:
        return None
    verdict = (
        "never reported a better tail than the open loop"
        if better == 0
        else f"reported a better tail at {better} of {len(parts)} matched rates"
    )
    return (
        f"p99 TTFT at matched throughput, closed against open loop: {'; '.join(parts)}. "
        f"Coordinated omission predicts a closed loop flatters the tail; here it {verdict}."
    )


def _trace_summary(
    runs: Sequence[RunResult], *, baseline_config_id: str, rate_rps: float
) -> str | None:
    cmp = process_comparison(
        runs,
        process=ArrivalProcess.TRACE_REPLAY,
        baseline_config_id=baseline_config_id,
        baseline_rate_rps=rate_rps,
        label="trace",
    )
    if cmp is None:
        return None
    latest = _latest_session(
        [r for r in runs if r.workload.arrival_process is ArrivalProcess.TRACE_REPLAY]
    )
    found = [m for r in latest for n in r.validity_notes if (m := _TRACE_NOTE.search(n))]
    starts = {m["start"] for m in found}
    if found:
        scales = [float(m["scale"]) for m in found]
        cv2 = [float(m["cv2"]) for m in found]
        described = (
            f"{len(starts)} distinct non-overlapping windows (time-scaled x{min(scales):.2f}-"
            f"x{max(scales):.2f} to {rate_rps:g} rps; CV² {min(cv2):.2f}-{max(cv2):.2f})"
        )
    else:
        described = f"{cmp.runs} replays of the trace"
    gap = cmp.ttft_p99_ms - cmp.baseline_ttft_p99_ms
    band = 2 * cmp.p99_pooled_stderr_ms
    verdict = (
        "indistinguishable: no measurable tail cost from real arrival timing at this rate"
        if not cmp.p99_significant
        else ("worse than Poisson" if gap > 0 else "better than Poisson")
    )
    return (
        f"{described} against {cmp.baseline_runs} Poisson draws: p99 TTFT {gap:+.0f} ms "
        f"against ±{band:.0f} ms — {verdict}."
    )


def methodology_summary(
    runs: Sequence[RunResult],
    quality: Sequence[QualityResult],
    *,
    crossvalidation_path: Path,
    baseline_config_id: str,
    rate_rps: float,
) -> str:
    """README's one-line verdict for each run that tests the method itself.

    Generated, like the tables it summarises: this table was typed once, and by
    the time of the audit it described five independent trace windows that
    were one window replayed five times.
    """
    rows: list[tuple[str, str]] = []

    matched = crossvalidation_path.with_name(
        f"{crossvalidation_path.stem}_matched{crossvalidation_path.suffix}"
    )
    if matched.exists():
        m = json.loads(matched.read_text())
        agree = sum(1 for x in m["metrics"] if x["agrees"])
        deltas = ", ".join(f"{x['metric']} {x['delta_pct']:+.1f} %" for x in m["metrics"])
        lengths = f"{m['fixed_input_tokens']}/{m['fixed_output_tokens']}"
        text = (
            f"Same live server, constant {lengths}-token requests at {m['rate_rps']:g} rps: "
            f"{agree} of {len(m['metrics'])} metrics within ±5 % ({deltas})."
        )
        if crossvalidation_path.exists():
            u = json.loads(crossvalidation_path.read_text())
            worst = max(u["metrics"], key=lambda x: abs(x["delta_pct"]))
            tokens = u.get("output_tokens_per_request", {})
            text += (
                f" Sampling lengths independently, the gap reached {worst['delta_pct']:+.0f} % "
                f"({worst['metric']}) — the harnesses drew {tokens.get('ours', 0):.0f} and "
                f"{tokens.get('upstream', 0):.0f} output tokens per request."
            )
        rows.append(("**Cross-validation vs `vllm bench serve`**", text))

    closed = _closed_loop_summary(runs, baseline_config_id=baseline_config_id)
    if closed:
        rows.append(("**Closed-loop exhibit**", closed))

    trace = _trace_summary(runs, baseline_config_id=baseline_config_id, rate_rps=rate_rps)
    if trace:
        rows.append(("**Azure trace replay**", trace))

    drift = drift_comparison(
        runs, config_id=baseline_config_id, canary_label=baseline_config_id, rate_rps=rate_rps
    )
    if drift:
        rows.append(
            (
                "**Drift canary**",
                f"`{baseline_config_id}` re-measured at the end of the sweep: p95 TTFT "
                f"{drift.first_ttft_p95_ms:.0f} → {drift.later_ttft_p95_ms:.0f} ms "
                f"({drift.ttft_drift_ms:+.0f} ms; twice the original's std is "
                f"{2 * drift.first_ttft_std_ms:.0f} ms), throughput "
                f"{_signed(drift.throughput_drift_pct)} % — "
                f"{'within' if drift.within_noise else 'outside'} the sweep's own noise.",
            )
        )

    scores = quality_scores(quality)
    pairs = [
        (label, scores[v][0], scores[s][0])
        for label, v, s in (
            ("BF16", "vllm-bf16", "sglang-bf16"),
            ("AWQ", "vllm-awq-int4", "sglang-awq-int4"),
        )
        if v in scores and s in scores
    ]
    if pairs:
        listed = "; ".join(f"{label} {v:.4f} vs {s:.4f}" for label, v, s in pairs)
        rows.append(
            ("**Same checkpoint, two engines**", f"GSM8K strict, vLLM vs SGLang: {listed}.")
        )

    pts = operating_points(runs, 1.0)
    int8 = sorted(pts.get("vllm-int8-w8a8", []), key=lambda p: p.rate_rps)
    over = sorted(
        {
            r.workload.request_rate_rps
            for r in runs
            if r.config_id == "vllm-int8-w8a8"
            and r.validity is RunValidity.OVERSUBSCRIBED
            and r.workload.request_rate_rps is not None
        }
    )
    others = [
        max(p.rate_rps for p in pts[c])
        for c in pts
        if c.startswith("vllm-") and c != "vllm-int8-w8a8"
    ]
    if len(int8) >= 2 and over and others:
        top, below = int8[-1], int8[-2]
        rows.append(
            (
                "**INT8 ceiling**",
                f"Valid up to {top.rate_rps:g} rps ({top.throughput_tokens_s:.0f} tok/s; p95 TTFT "
                f"{top.ttft_p95_s * 1e3:.0f} ms, against {below.ttft_p95_s * 1e3:.0f} ms at "
                f"{below.rate_rps:g} rps), oversubscribed from {over[0]:g} rps. The other vLLM "
                f"configurations are valid up to {min(others):g}-{max(others):g} rps.",
            )
        )

    if not rows:
        return "_No methodology runs have been measured yet._"
    return "\n".join(
        ["| Test | Result |", "|---|---|", *(f"| {test} | {result} |" for test, result in rows)]
    )


def _e2e_identity_error_pct(payload: Mapping[str, Any], side: str) -> float:
    """How far mean E2E is from mean TTFT + mean TPOT x (mean output - 1), in %.

    Per request, E2E = TTFT + TPOT x (n - 1) by definition; for means it holds
    only approximately, since output length varies. A harness whose timing were
    broken would miss it by far more than that approximation.
    """
    by_metric = {m["metric"]: m for m in payload["metrics"]}
    tokens = float(payload.get("output_tokens_per_request", {}).get(side, 0.0))
    e2e = float(by_metric["E2E mean"][side])
    if not tokens or not e2e:
        return float("nan")
    rebuilt = float(by_metric["TTFT mean"][side]) + float(by_metric["TPOT mean"][side]) * (
        tokens - 1
    )
    return abs(rebuilt / e2e - 1.0) * 100.0


def crossvalidation_table(path: Path) -> str:
    """Agreement with vLLM's own harness, matched and unmatched.

    Two files: ``<stem>_matched.json`` from a run with both harnesses on
    constant lengths, and ``<stem>.json`` from each sampling ShareGPT its own
    way. The matched run is the check; the unmatched one is shown because its
    disagreement is the reason matching was necessary, and a reader who sees
    only the clean result would not know that.
    """
    matched = path.with_name(f"{path.stem}_matched{path.suffix}")
    if not matched.exists() and not path.exists():
        return "_Cross-validation against `vllm bench serve` has not been run yet._"

    def table(payload: dict[str, object]) -> list[str]:
        rows = [
            "| Metric | This harness | `vllm bench serve` | Δ | Agrees (±5 %) |",
            "|---|---|---|---|---|",
        ]
        metrics = payload["metrics"]
        assert isinstance(metrics, list)
        # A length-sensitive metric may differ because the workloads differ --
        # but only when they do. With matched lengths a miss is a real one.
        lengths_differ = not payload.get("matched_lengths", False)
        for m in metrics:
            if m["agrees"]:
                flag = "yes"
            elif m.get("length_sensitive") and lengths_differ:
                flag = "no*"
            else:
                flag = "**NO**"
            rows.append(
                f"| {m['metric']} | {m['ours']:.2f} {m['unit']} | {m['upstream']:.2f} {m['unit']} "
                f"| {m['delta_pct']:+.1f} % | {flag} |"
            )
        return rows

    out: list[str] = []
    if matched.exists():
        m = json.loads(matched.read_text())
        out += [
            f"**Matched workload** — `{m['config_id']}` at {m['rate_rps']:g} rps, both harnesses "
            f"on constant {m['fixed_input_tokens']}/{m['fixed_output_tokens']} input/output tokens "
            f"against the same live server:",
            "",
            *table(m),
            "",
        ]
        missed = [x["metric"] for x in m["metrics"] if not x["agrees"]]
        if missed:
            out += [
                f"Outside ±5 % with matched lengths: {', '.join(missed)}. Output throughput "
                f"divides tokens by a measurement window, and the two harnesses define that "
                f"window independently (this one: first measured dispatch to last completion).",
                "",
            ]
    if path.exists():
        u = json.loads(path.read_text())
        ratio = u.get("output_tokens_per_request", {})
        out += [
            f"**Unmatched workload** — the same comparison with each harness sampling ShareGPT "
            f"its own way. Ours drew {ratio.get('ours', 0):.0f} output tokens per request, "
            f"upstream {ratio.get('upstream', 0):.0f} (ratio {ratio.get('ratio', 0):.2f}):",
            "",
            *table(u),
            "",
            r"\* length-sensitive: with different output lengths these metrics differ by "
            f"construction. Each harness is internally consistent: its mean E2E equals its "
            f"mean TTFT + mean TPOT x (mean output - 1) to within "
            f"{_e2e_identity_error_pct(u, 'ours'):.2f} % here and "
            f"{_e2e_identity_error_pct(u, 'upstream'):.2f} % upstream. The disagreement is the "
            "workload, not the code — which is why the matched run above exists.",
        ]
    return "\n".join(out).rstrip()


def headline_facts(
    runs: Sequence[RunResult],
    quality: Sequence[QualityResult],
    *,
    gpu_hourly_usd: float,
    max_ttft_p95_s: float,
    baseline_config_id: str = "vllm-bf16",
) -> str:
    """The sentences in the README that used to be typed.

    They were true when written and went stale twice — once when the INT8
    ladder was extended, once when SGLang gained task scores and the frontier
    grew from four configurations to six. A sentence that carries a number is a
    result, and results are generated here or not stated.
    """
    configs = sorted(
        {r.config_id for r in runs if r.workload.arrival_process is ArrivalProcess.POISSON}
    )
    markers = pareto_markers(
        runs, quality, gpu_hourly_usd=gpu_hourly_usd, max_ttft_p95_s=max_ttft_p95_s
    )
    if not markers:
        return "_No results to summarise yet._"

    cheapest = min(markers, key=lambda m: m.cost_per_1m_usd)
    base = next((m for m in markers if m.config_id == baseline_config_id), None)
    scores = quality_scores(quality)
    values = [v for v, _ in scores.values()]
    spread = max(values) - min(values)
    half_width = 1.96 * max(e for _, e in scores.values())
    inside = base is not None and all(
        abs(m.quality - base.quality) <= base.quality_ci for m in markers
    )

    lines = [
        f"**{len(runs)} measured runs** across {len(configs)} configurations, "
        f"{len(quality)} quality evaluations, open-loop, at least three repeats per point.",
        "",
        f"Under a **p95 TTFT budget of {max_ttft_p95_s * 1e3:.0f} ms**, `{cheapest.config_id}` is "
        f"cost-optimal: **{cheapest.throughput_tokens_s:.0f} output tokens/sec at "
        f"${cheapest.cost_per_1m_usd:.4f} per million tokens**",
    ]
    if base is not None and base.config_id != cheapest.config_id:
        saving = (1 - cheapest.cost_per_1m_usd / base.cost_per_1m_usd) * 100
        dominators = [
            m
            for m in markers
            if m.config_id != base.config_id
            and m.cost_per_1m_usd <= base.cost_per_1m_usd
            and m.quality >= base.quality
        ]
        lines[-1] += (
            f" — {saving:.0f} % cheaper than `{base.config_id}`. `{base.config_id}` is dominated: "
            f"{len(dominators)} configurations are cheaper at equal-or-better measured quality."
        )
    else:
        lines[-1] += "."

    lines += [
        "",
        f"The grey band on the chart is `{baseline_config_id}`'s 95 % confidence interval on "
        f"GSM8K{', and every configuration falls inside it' if inside else ''}. "
        f"The quality spread across all {len(markers)} configurations is {spread:.4f}, "
        f"{'smaller' if spread < half_width else 'larger'} than a single configuration's 95 % "
        f"half-width of {half_width:.4f}"
        + (
            " — **quality does not separate these configurations at this sample size**, so the "
            "decision is made on cost and latency."
            if spread < half_width
            else "."
        ),
    ]
    return "\n".join(lines)


def bandwidth_table(runs: Sequence[RunResult], quality: Sequence[QualityResult]) -> str:
    """Finding 1: decode speedup at 1 rps against the weight-byte ratio."""
    pts = operating_points(runs, 1.0)
    weights = {x.config_id: x.model.weights_gib for x in quality}
    base = next((p for p in pts.get("vllm-bf16", []) if p.rate_rps == 1.0), None)
    if base is None or "vllm-bf16" not in weights:
        return "_Finding 1 needs vllm-bf16 at 1 rps with a quality record._"
    lines = [
        "| Config | Declared weights | Predicted speedup (byte ratio) | Measured TPOT p95 "
        "| Measured speedup |",
        "|---|---|---|---|---|",
    ]
    for cfg in ("vllm-bf16", "vllm-int8-w8a8", "vllm-gptq-int4", "vllm-awq-int4"):
        p = next((p for p in pts.get(cfg, []) if p.rate_rps == 1.0), None)
        if p is None or cfg not in weights:
            continue
        byte_ratio = weights["vllm-bf16"] / weights[cfg]
        speed = base.tpot_p95_s / p.tpot_p95_s
        bold = "**" if cfg == "vllm-int8-w8a8" else ""
        lines.append(
            f"| `{cfg}` | {weights[cfg]:.2f} GiB | {byte_ratio:.2f}x | {p.tpot_p95_s * 1e3:.2f} ms "
            f"| {bold}{speed:.2f}x{bold} |"
        )
    g = next((p for p in pts.get("vllm-gptq-int4", []) if p.rate_rps == 1.0), None)
    a = next((p for p in pts.get("vllm-awq-int4", []) if p.rate_rps == 1.0), None)
    if g and a:
        lines += [
            "",
            f"Act-order overhead, isolated: `gptq_marlin` {g.tpot_p95_s * 1e3:.2f} ms vs "
            f"`awq_marlin` {a.tpot_p95_s * 1e3:.2f} ms — "
            f"**a {(g.tpot_p95_s / a.tpot_p95_s - 1) * 100:.1f} % decode penalty** "
            f"for `desc_act=true` at identical bit width.",
        ]
    return "\n".join(lines)


def regime_table(runs: Sequence[RunResult]) -> str:
    """Finding 2: which configuration wins TPOT and TTFT at each offered rate."""
    pts = operating_points(runs, 1.0)
    vllm = [c for c in pts if c.startswith("vllm-")]
    rates = sorted({p.rate_rps for c in vllm for p in pts[c]})
    lines = [
        "| Offered rate | TPOT p95 winner | TTFT p95 winner | Configs still valid |",
        "|---|---|---|---|",
    ]
    for rate in rates:
        at = {c: p for c in vllm for p in pts[c] if p.rate_rps == rate}
        if not at:
            continue
        tp = min(at.items(), key=lambda kv: kv[1].tpot_p95_s)
        tt = min(at.items(), key=lambda kv: kv[1].ttft_p95_s)
        lines.append(
            f"| {rate:g} rps | `{tp[0]}` ({tp[1].tpot_p95_s * 1e3:.1f} ms) "
            f"| `{tt[0]}` ({tt[1].ttft_p95_s * 1e3:.0f} ms) | {len(at)} of {len(vllm)} |"
        )
    return "\n".join(lines)


_INT4 = ("vllm-gptq-int4", "vllm-awq-int4")


def _contiguous_through(rates: Sequence[float], won: set[float]) -> float | None:
    """Highest rate r such that every rate up to r is in ``won``."""
    last = None
    for rate in rates:
        if rate not in won:
            break
        last = rate
    return last


def findings_summary(runs: Sequence[RunResult], quality: Sequence[QualityResult]) -> str:
    """README's two findings, with every number taken from the data.

    These paragraphs used to be typed, and three of their numbers went stale
    when later runs moved the averages while the tables beside them regenerated.
    """
    pts = operating_points(runs, 1.0)
    weights = {x.config_id: x.model.weights_gib for x in quality}

    def at(cfg: str, rate: float) -> ConfigOperatingPoint | None:
        return next((p for p in pts.get(cfg, []) if p.rate_rps == rate), None)

    base, int8 = at("vllm-bf16", 1.0), at("vllm-int8-w8a8", 1.0)
    gptq, awq = at("vllm-gptq-int4", 1.0), at("vllm-awq-int4", 1.0)
    needed = {"vllm-bf16", "vllm-int8-w8a8", *_INT4}
    if base is None or int8 is None or gptq is None or awq is None or not needed <= weights.keys():
        return "_The findings need all four vLLM configurations at 1 rps with quality records._"

    def byte_ratio(cfg: str) -> float:
        return weights["vllm-bf16"] / weights[cfg]

    def speedup(p: ConfigOperatingPoint) -> float:
        return base.tpot_p95_s / p.tpot_p95_s

    int8_off = abs(speedup(int8) / byte_ratio("vllm-int8-w8a8") - 1.0) * 100.0
    int4_pred = sorted({f"{byte_ratio(c):.2f}x" for c in _INT4})
    finding1 = (
        f"**1. Decode time tracks weight bytes, not FLOPs — quantitatively.** At 1 rps, "
        f"INT8-W8A8 shrinks the declared weights {byte_ratio('vllm-int8-w8a8'):.2f}x and p95 "
        f"decode latency by **{speedup(int8):.2f}x**, within {int8_off:.1f} % of the bandwidth "
        f"prediction. The INT4 configurations *exceed* their predicted {' and '.join(int4_pred)} "
        f"({speedup(gptq):.2f}x GPTQ, {speedup(awq):.2f}x AWQ), which a bytes-read model cannot "
        f"explain; the likely cause is that Marlin is tuned for low batch while vLLM's general "
        f"BF16 path is not. That is reported as unexplained rather than claimed as confirmation."
    )

    vllm = [c for c in pts if c.startswith("vllm-")]
    rates = sorted({p.rate_rps for c in vllm for p in pts[c]})
    winner: dict[float, str] = {}
    for rate in rates:
        here = [(c, p) for c in vllm for p in pts[c] if p.rate_rps == rate]
        if here:
            winner[rate] = min(here, key=lambda cp: cp[1].tpot_p95_s)[0]
    int4_through = _contiguous_through(rates, {r for r, c in winner.items() if c in _INT4})
    int8_from = min((r for r, c in winner.items() if c == "vllm-int8-w8a8"), default=None)
    shared = [r for r in rates if all(at(c, r) for c in ("vllm-int8-w8a8", *_INT4))]
    if int4_through is None or int8_from is None or not shared:
        return finding1
    top = shared[-1]

    def ttft_ms(cfg: str) -> float:
        point = at(cfg, top)
        return point.ttft_p95_s * 1e3 if point else float("nan")

    finding2 = (
        f"**2. The best quantization inverts between load regimes.** INT4 has the lowest p95 "
        f"decode latency at every rate up to {int4_through:g} rps; from {int8_from:g} rps INT8 "
        f"does. At {top:g} rps, the highest rate all three quantized formats are still valid, "
        f"p95 TTFT is **{ttft_ms('vllm-awq-int4'):.0f} ms (AWQ) and "
        f"{ttft_ms('vllm-gptq-int4'):.0f} ms (GPTQ) against {ttft_ms('vllm-int8-w8a8'):.0f} ms "
        f"for INT8**."
    )
    return f"{finding1}\n\n{finding2}"


def _tok(p: ConfigOperatingPoint | None) -> str:
    return "—" if p is None else f"{p.throughput_tokens_s:.0f}"


def _ttft(p: ConfigOperatingPoint | None) -> str:
    return "—" if p is None else f"{p.ttft_p95_s * 1e3:.0f} ms"


def engine_axis_table(runs: Sequence[RunResult]) -> str:
    """vLLM against SGLang on the same checkpoints, at matched offered rates."""
    pts = operating_points(runs, 1.0)
    pairs = (("vllm-bf16", "sglang-bf16"), ("vllm-awq-int4", "sglang-awq-int4"))
    lines = [
        "| Rate | Checkpoint | vLLM tok/s | SGLang tok/s | vLLM TTFT p95 | SGLang TTFT p95 |",
        "|---|---|---|---|---|---|",
    ]
    for rate in (1.0, 4.0, 8.0):
        for v, sg in pairs:
            pv = next((p for p in pts.get(v, []) if p.rate_rps == rate), None)
            ps = next((p for p in pts.get(sg, []) if p.rate_rps == rate), None)
            lines.append(
                f"| {rate:g} rps | {v.removeprefix('vllm-')} | {_tok(pv)} | {_tok(ps)} "
                f"| {_ttft(pv)} | {_ttft(ps)} |"
            )
    return "\n".join(lines)
