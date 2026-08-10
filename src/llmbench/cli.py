"""Command-line entry point.

``llmbench sweep``  — run a matrix, writing one validated JSON per measurement
``llmbench charts`` — regenerate figures from committed results
``llmbench show``   — summarise results already on disk

Everything is driven from YAML. Adding a configuration is a config edit, never a
code edit, and every number that reaches the README is regenerated from the JSON
rather than typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llmbench.analysis.plots import plot_latency_throughput, plot_tpot_throughput
from llmbench.config import load_sweep_config
from llmbench.runner import SweepRunner
from llmbench.schema import RunResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Rigorous open-loop benchmarking of quantized LLM serving.",
)
console = Console()


def _load_runs(results_dir: Path) -> list[RunResult]:
    """Load and re-validate every committed result.

    Re-validating on read is deliberate: it is how a schema change that would
    silently orphan earlier results gets caught, rather than surfacing as a
    confusing chart much later.
    """
    runs: list[RunResult] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(RunResult.model_validate_json(path.read_text()))
        except Exception as exc:
            console.print(f"[red]invalid result[/red] {path.name}: {exc}")
    return runs


@app.command()
def sweep(
    config: Annotated[Path, typer.Option(help="Sweep matrix YAML")] = Path("configs/sweep.yaml"),
    gpu: Annotated[int, typer.Option(help="Host GPU index to measure on")] = 1,
    results: Annotated[Path, typer.Option(help="Where to write result JSON")] = Path(
        "results/runs"
    ),
    dataset: Annotated[Path, typer.Option(help="ShareGPT dataset")] = Path("data/sharegpt_v3.json"),
    require_locked_clocks: Annotated[
        bool, typer.Option(help="Refuse to run unless GPU clocks are pinned")
    ] = False,
    charts: Annotated[bool, typer.Option(help="Regenerate figures afterwards")] = True,
) -> None:
    """Execute a sweep matrix."""
    cfg = load_sweep_config(config)
    console.print(
        f"[bold]{config.name}[/bold]: {len(cfg.configurations)} configuration(s) x "
        f"{len(cfg.workload.request_rates_rps)} rate(s) x {cfg.defaults.repeats} repeat(s) "
        f"= {len(cfg.configurations) * len(cfg.workload.request_rates_rps) * cfg.defaults.repeats}"
        f" measurements"
    )

    runner = SweepRunner(
        cfg,
        gpu_index=gpu,
        results_dir=results,
        dataset_path=dataset,
        require_locked_clocks=require_locked_clocks,
    )
    written = runner.run()
    console.print(f"\n[green]wrote {len(written)} result file(s)[/green] to {results}")

    if charts:
        make_charts(results=results, out=Path("results/figures"))


@app.command("charts")
def make_charts(
    results: Annotated[Path, typer.Option(help="Directory of result JSON")] = Path("results/runs"),
    out: Annotated[Path, typer.Option(help="Where to write figures")] = Path("results/figures"),
) -> None:
    """Regenerate every figure from committed results."""
    runs = _load_runs(results)
    if not runs:
        console.print(f"[yellow]no results found in {results}[/yellow]")
        raise typer.Exit(code=1)

    paths = [
        plot_latency_throughput(runs, out / "latency_vs_throughput.png"),
        plot_tpot_throughput(runs, out / "tpot_vs_throughput.png"),
    ]
    for path in paths:
        console.print(f"  wrote {path}")


@app.command()
def show(
    results: Annotated[Path, typer.Option(help="Directory of result JSON")] = Path("results/runs"),
) -> None:
    """Summarise results on disk."""
    runs = _load_runs(results)
    if not runs:
        console.print(f"[yellow]no results found in {results}[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"{len(runs)} run(s) in {results}", header_style="bold")
    for column in (
        "config",
        "rps",
        "rep",
        "TTFT p95",
        "TPOT p95",
        "tok/s",
        "req/s",
        "lag p99",
        "validity",
    ):
        table.add_column(column, justify="right" if column != "config" else "left")

    for run in sorted(
        runs, key=lambda r: (r.config_id, r.workload.request_rate_rps or 0, r.repeat_index)
    ):
        ok = run.is_reportable
        table.add_row(
            run.config_id,
            f"{run.workload.request_rate_rps:g}",
            str(run.repeat_index),
            f"{run.ttft_s.p95 * 1e3:.1f} ms",
            f"{run.tpot_s.p95 * 1e3:.1f} ms",
            f"{run.output_token_throughput:.1f}",
            f"{run.request_throughput:.2f}",
            f"{run.dispatch_lag_s.p99 * 1e3:.2f} ms",
            f"[green]{run.validity.value}[/green]" if ok else f"[red]{run.validity.value}[/red]",
        )
    console.print(table)

    invalid = [r for r in runs if not r.is_reportable]
    if invalid:
        console.print(f"\n[yellow]{len(invalid)} run(s) not reportable:[/yellow]")
        for run in invalid:
            for note in run.validity_notes:
                console.print(f"  {run.config_id} @ {run.workload.request_rate_rps:g} rps: {note}")


@app.command()
def validate(
    config: Annotated[Path, typer.Option(help="Sweep matrix YAML")] = Path("configs/sweep.yaml"),
) -> None:
    """Validate a sweep config without running anything."""
    cfg = load_sweep_config(config)
    total = len(cfg.configurations) * len(cfg.workload.request_rates_rps) * cfg.defaults.repeats
    console.print(f"[green]{config} is valid[/green]")
    console.print(f"  model         {cfg.model.hf_id} @ {cfg.model.revision[:12]}")
    console.print(f"  max_model_len {cfg.model.max_model_len}")
    console.print(f"  rates         {cfg.workload.request_rates_rps}")
    console.print(f"  measurements  {total}")
    for entry in cfg.configurations:
        quant = cfg.quantizations[entry.quantization]
        console.print(
            f"    {entry.id:<20} {entry.engine.value:<8} {entry.quantization:<12} "
            f"kernel={quant.expected_kernel or '—'}"
        )


@app.command("schema")
def dump_schema(
    out: Annotated[Path, typer.Option(help="Where to write the JSON Schema")] = Path(
        "results/schema.json"
    ),
) -> None:
    """Emit the results JSON Schema, so consumers can validate independently."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RunResult.model_json_schema(), indent=2) + "\n")
    console.print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
