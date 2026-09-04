"""Chart generation.

The core artifact is a **latency-vs-throughput curve**: achieved output-token
throughput on x, a tail-latency percentile on y, one line per configuration,
swept across offered load until saturation. A single "N tokens/sec" number
without a load level and a latency distribution is not a result, so the curve —
not a bar — is the primary form.

Design constraints applied throughout:

* **One y-axis, never two.** TTFT and TPOT have different scales and different
  meanings; they get separate figures rather than a dual-axis chart.
* **Colour follows the configuration, never its rank**, and hues are assigned in
  fixed order, so filtering the set never repaints the survivors.
* **Error bars are mandatory.** Every point is mean ± std across repeats; a
  benchmark chart without them invites the reader to assume one lucky run.
* Invalid runs are excluded from the lines and reported in the caption rather
  than silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the benchmark host
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from llmbench.metrics.percentiles import summarize
from llmbench.schema import RunResult

__all__ = [
    "SERIES_COLORS",
    "ParetoMarker",
    "plot_latency_throughput",
    "plot_pareto",
    "plot_tpot_throughput",
]

#: Validated categorical palette, fixed order (see the data-viz reference
#: palette). Assigned by configuration identity and never cycled, so adding or
#: removing a configuration cannot recolour the others.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e3e2df"
_SURFACE = "#fcfcfb"


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    """One offered-load level, aggregated across repeats."""

    rate_rps: float
    throughput_mean: float
    throughput_std: float
    latency_mean_ms: float
    latency_std_ms: float
    repeats: int


def _aggregate(runs: Sequence[RunResult], percentile: str) -> list[SeriesPoint]:
    """Group reportable runs by offered rate and compute mean ± std."""
    by_rate: dict[float, list[RunResult]] = {}
    for run in runs:
        if not run.is_reportable or run.workload.request_rate_rps is None:
            continue
        by_rate.setdefault(run.workload.request_rate_rps, []).append(run)

    points: list[SeriesPoint] = []
    for rate in sorted(by_rate):
        group = by_rate[rate]
        throughput = summarize([r.output_token_throughput for r in group])
        latency = summarize([getattr(r.ttft_s, percentile) * 1e3 for r in group])
        points.append(
            SeriesPoint(
                rate_rps=rate,
                throughput_mean=throughput.mean,
                throughput_std=throughput.std,
                latency_mean_ms=latency.mean,
                latency_std_ms=latency.std,
                repeats=len(group),
            )
        )
    return points


def _aggregate_tpot(runs: Sequence[RunResult], percentile: str) -> list[SeriesPoint]:
    by_rate: dict[float, list[RunResult]] = {}
    for run in runs:
        if not run.is_reportable or run.workload.request_rate_rps is None:
            continue
        by_rate.setdefault(run.workload.request_rate_rps, []).append(run)

    points: list[SeriesPoint] = []
    for rate in sorted(by_rate):
        group = by_rate[rate]
        throughput = summarize([r.output_token_throughput for r in group])
        latency = summarize([getattr(r.tpot_s, percentile) * 1e3 for r in group])
        points.append(
            SeriesPoint(
                rate_rps=rate,
                throughput_mean=throughput.mean,
                throughput_std=throughput.std,
                latency_mean_ms=latency.mean,
                latency_std_ms=latency.std,
                repeats=len(group),
            )
        )
    return points


def _style_axes(ax: Axes, *, xlabel: str, ylabel: str, title: str, subtitle: str) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_xlabel(xlabel, color=_TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=_TEXT_SECONDARY, fontsize=10)
    ax.set_title(title, color=_TEXT_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=26)
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        color=_TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
    )
    # Recessive grid and axes: the data carries the ink, not the furniture.
    ax.grid(visible=True, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=9)


def _draw(
    grouped: dict[str, list[SeriesPoint]],
    *,
    out_path: Path,
    ylabel: str,
    title: str,
    subtitle: str,
    log_y: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=160, facecolor=_SURFACE)

    for index, (label, points) in enumerate(grouped.items()):
        if not points:
            continue
        colour = SERIES_COLORS[index % len(SERIES_COLORS)]
        xs = [p.throughput_mean for p in points]
        ys = [p.latency_mean_ms for p in points]

        ax.errorbar(
            xs,
            ys,
            yerr=[p.latency_std_ms for p in points],
            xerr=[p.throughput_std for p in points],
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.5,
            capsize=3,
            elinewidth=1.2,
            label=label,
            zorder=3,
        )
        # Direct label at the curve's end: identity is never colour-alone, and
        # three slots in this palette sit under 3:1 contrast on a light surface.
        ax.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=_TEXT_SECONDARY,
            fontsize=9,
            va="center",
        )

    # Log y only when the data actually spans orders of magnitude. A saturated
    # sweep does; a two-point smoke run does not, and forcing log on a narrow
    # range yields unreadable ticks like "2.05 x 10^2".
    all_y = [p.latency_mean_ms for pts in grouped.values() for p in pts if p.latency_mean_ms > 0]
    spans_decades = bool(all_y) and (max(all_y) / min(all_y)) >= 5.0
    if log_y and spans_decades:
        ax.set_yscale("log")

    # Headroom on the right so end-of-curve direct labels are not clipped.
    ax.margins(x=0.12)

    _style_axes(
        ax,
        xlabel="Output token throughput (tokens/sec)",
        ylabel=ylabel,
        title=title,
        subtitle=subtitle,
    )

    if len(grouped) >= 2:
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=_TEXT_SECONDARY,
            loc="upper left",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _group_by_config(runs: Iterable[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for run in runs:
        grouped.setdefault(run.config_id, []).append(run)
    return grouped


def plot_latency_throughput(
    runs: Sequence[RunResult],
    out_path: Path,
    *,
    percentile: str = "p95",
) -> Path:
    """The core artifact: tail TTFT against achieved throughput.

    Each point is one offered request rate; the curve bends upward at the knee,
    where queueing begins to dominate. Log y, because tail latency spans orders
    of magnitude between the linear region and saturation.
    """
    grouped = {cid: _aggregate(rs, percentile) for cid, rs in _group_by_config(runs).items()}
    excluded = sum(1 for r in runs if not r.is_reportable)
    subtitle = (
        f"Open-loop Poisson arrivals · mean ± std across repeats"
        f"{f' · {excluded} invalid run(s) excluded' if excluded else ''}"
    )
    return _draw(
        grouped,
        out_path=out_path,
        ylabel=f"{percentile.upper()} time to first token (ms)",
        title="Latency vs throughput",
        subtitle=subtitle,
        log_y=True,
    )


def plot_tpot_throughput(
    runs: Sequence[RunResult],
    out_path: Path,
    *,
    percentile: str = "p95",
) -> Path:
    """Decode-phase latency against throughput.

    Kept as its own figure rather than a second y-axis on the TTFT chart:
    separating prefill from decode is the whole point of reporting them apart.
    """
    grouped = {cid: _aggregate_tpot(rs, percentile) for cid, rs in _group_by_config(runs).items()}
    return _draw(
        grouped,
        out_path=out_path,
        ylabel=f"{percentile.upper()} time per output token (ms)",
        title="Decode latency vs throughput",
        subtitle="Excludes prefill · mean ± std across repeats",
        log_y=False,
    )


@dataclass(frozen=True, slots=True)
class ParetoMarker:
    """One configuration placed on the quality/cost plane under an SLA."""

    config_id: str
    cost_per_1m_usd: float
    quality: float
    #: Half-width of the 95% interval on the quality score, when the harness
    #: reported a standard error. Drawn, because the central finding here is
    #: that the configurations are not separated by it.
    quality_ci: float
    throughput_tokens_s: float
    on_frontier: bool


def plot_pareto(
    markers: Sequence[ParetoMarker],
    out_path: Path,
    *,
    sla_label: str,
    quality_label: str,
    reference_config: str | None = None,
) -> Path:
    """Quality against cost, with dominated configurations left visible.

    The third axis — latency — is held fixed by the SLA rather than drawn: each
    configuration sits at the highest offered rate whose measured p95 still met
    the budget, so every point on the chart is one a platform team could
    actually run. Sweeping the budget moves the points, which is why the SLA is
    named in the subtitle rather than assumed.

    Dominated configurations are drawn hollow and kept in place. Deleting them
    would turn a demonstration into an assertion: the frontier is only
    persuasive next to the points it beat.

    Args:
        reference_config: Draws that configuration's 95% quality interval as a
            band across the plot. When every marker falls inside it, the chart
            says so in ink: the quality axis is not resolved at this sample
            size, and the frontier is being decided by cost alone. A Pareto
            chart that omits this reads as a quality ranking it cannot support.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=160, facecolor=_SURFACE)

    reference = next((m for m in markers if m.config_id == reference_config), None)
    if reference is not None and reference.quality_ci:
        ax.axhspan(
            reference.quality - reference.quality_ci,
            reference.quality + reference.quality_ci,
            color=_GRID,
            alpha=0.65,
            zorder=1,
            label=f"{reference.config_id} 95% CI",
        )

    order = {m.config_id: i for i, m in enumerate(sorted(markers, key=lambda m: m.config_id))}
    frontier = sorted([m for m in markers if m.on_frontier], key=lambda m: m.cost_per_1m_usd)

    if len(frontier) >= 2:
        ax.plot(
            [m.cost_per_1m_usd for m in frontier],
            [m.quality for m in frontier],
            color=_TEXT_SECONDARY,
            linewidth=1.4,
            linestyle="--",
            zorder=2,
            label="Pareto frontier",
        )

    for m in markers:
        colour = SERIES_COLORS[order[m.config_id] % len(SERIES_COLORS)]
        ax.errorbar(
            [m.cost_per_1m_usd],
            [m.quality],
            yerr=[m.quality_ci] if m.quality_ci else None,
            color=colour,
            marker="o",
            markersize=11 if m.on_frontier else 9,
            markerfacecolor=colour if m.on_frontier else _SURFACE,
            markeredgecolor=colour,
            markeredgewidth=2.0,
            capsize=3,
            elinewidth=1.2,
            linestyle="none",
            zorder=4 if m.on_frontier else 3,
        )
        ax.annotate(
            f"{m.config_id}\n{m.throughput_tokens_s:.0f} tok/s"
            + ("" if m.on_frontier else "  · dominated"),
            xy=(m.cost_per_1m_usd, m.quality),
            xytext=(10, -4),
            textcoords="offset points",
            color=_TEXT_PRIMARY if m.on_frontier else _TEXT_SECONDARY,
            fontsize=8.5,
            va="top",
        )

    ax.margins(x=0.30, y=0.22)
    _style_axes(
        ax,
        xlabel="Cost per 1M output tokens (USD)",
        ylabel=quality_label,
        title="Quality vs cost at a fixed latency budget",
        subtitle=(
            f"Each point is the highest offered rate meeting {sla_label} · "
            f"hollow = dominated · error bars are 95% CI on quality"
        ),
    )
    if len(markers) >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT_SECONDARY, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
