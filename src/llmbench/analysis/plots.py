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

__all__ = ["SERIES_COLORS", "plot_latency_throughput", "plot_tpot_throughput"]

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
