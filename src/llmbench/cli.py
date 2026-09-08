"""Command-line entry point.

``llmbench sweep``  — run a matrix, writing one validated JSON per measurement
``llmbench charts`` — regenerate figures from committed results
``llmbench show``   — summarise results already on disk
``llmbench report`` — regenerate every chart and table in README/REPORT

Everything is driven from YAML. Adding a configuration is a config edit, never a
code edit, and every number that reaches the README is regenerated from the JSON
rather than typed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from llmbench.analysis.crossvalidate import (
    MetricAgreement,
    compare_to_upstream,
    load_upstream_result,
    upstream_args,
)
from llmbench.analysis.plots import plot_latency_throughput, plot_pareto, plot_tpot_throughput
from llmbench.config import load_engine_profile, load_sweep_config
from llmbench.engines.preflight import run_preflight
from llmbench.engines.sglang import SglangEngine
from llmbench.engines.vllm import VllmEngine
from llmbench.report.render import (
    cost_note,
    crossvalidation_table,
    drift_table,
    latency_table,
    load_quality,
    methodology_table,
    pareto_markers,
    quality_table,
    render_into,
    sla_table,
    validity_summary,
    write_summary_json,
)
from llmbench.runner import SweepRunner
from llmbench.schema import EngineName as _EngineName
from llmbench.schema import RunResult

_ENGINE_TYPES = {_EngineName.VLLM: VllmEngine, _EngineName.SGLANG: SglangEngine}

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
    configs_only: Annotated[
        str, typer.Option(help="Comma-separated config ids; default is the whole matrix")
    ] = "",
    charts: Annotated[bool, typer.Option(help="Regenerate figures afterwards")] = True,
) -> None:
    """Execute a sweep matrix.

    ``--configs-only`` exists for interruptibility. The matrix takes ~18
    GPU-hours, and on a shared machine the card can be reclaimed part-way
    through; each (config, rate, repeat) is written as its own validated JSON,
    so a reclaimed GPU costs one measurement and the remaining configurations
    can be resumed by name rather than restarting the sweep.
    """
    cfg = load_sweep_config(config)
    selected = [c.strip() for c in configs_only.split(",") if c.strip()]
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
    written = runner.run(config_ids=selected or None)
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
def quality(
    config: Annotated[Path, typer.Option(help="Sweep matrix YAML")] = Path("configs/sweep.yaml"),
    gpu: Annotated[int, typer.Option(help="Host GPU index to measure on")] = 1,
    results: Annotated[Path, typer.Option(help="Where to write quality JSON")] = Path(
        "results/quality"
    ),
    wikitext: Annotated[Path, typer.Option(help="WikiText-2 test parquet")] = Path(
        "data/wikitext2_test.parquet"
    ),
    configs_only: Annotated[
        str, typer.Option(help="Comma-separated config ids; default is every vLLM config")
    ] = "",
    ppl_tokens: Annotated[int, typer.Option(help="Tokens of WikiText-2 to score")] = 100_000,
    limit: Annotated[int, typer.Option(help="Cap task samples (0 = full set)")] = 0,
    skip_tasks: Annotated[bool, typer.Option(help="Perplexity only; skip GSM8K/IFEval")] = False,
) -> None:
    """Measure perplexity, GSM8K and IFEval through the live engine.

    Runs against the serving engine rather than the raw checkpoint, so the
    quantized kernels actually under test are the ones being scored.
    """
    from llmbench.quality.runner import QualityRunner  # noqa: PLC0415  (torch-free import path)

    cfg = load_sweep_config(config)
    selected = [c.strip() for c in configs_only.split(",") if c.strip()]

    runner = QualityRunner(
        cfg,
        gpu_index=gpu,
        results_dir=results,
        wikitext_path=wikitext,
        ppl_tokens=ppl_tokens,
        task_limit=limit or None,
        skip_tasks=skip_tasks,
    )
    written = runner.run(config_ids=selected or None)
    console.print(f"\n[green]wrote {len(written)} quality record(s)[/green] to {results}")


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


@app.command()
def report(
    results: Annotated[Path, typer.Option(help="Directory of result JSON")] = Path("results/runs"),
    quality_dir: Annotated[Path, typer.Option(help="Directory of quality JSON")] = Path(
        "results/quality"
    ),
    cost_config: Annotated[Path, typer.Option(help="Cost assumptions")] = Path("configs/cost.yaml"),
    out: Annotated[Path, typer.Option(help="Repository root to write into")] = Path("."),
    sla_ttft_ms: Annotated[
        float, typer.Option(help="p95 TTFT budget for the headline chart")
    ] = 500.0,
    sla_methodology_rate: Annotated[
        float, typer.Option(help="Poisson rate the methodology exhibits are compared against")
    ] = 4.0,
) -> None:
    """Regenerate every chart and table in README/REPORT from results JSON.

    Idempotent by construction: running it twice on unchanged results leaves
    ``git diff`` empty. That property is the point — it is what makes "no number
    in these documents was typed by hand" checkable rather than merely claimed.
    """
    runs = _load_runs(results)
    if not runs:
        console.print(f"[yellow]no results found in {results}[/yellow]")
        raise typer.Exit(code=1)

    quality = load_quality(quality_dir) if quality_dir.exists() else []
    cost = yaml.safe_load(cost_config.read_text())
    ref = cost["reference"]
    price = float(ref["gpu_hourly_usd"])
    budgets = [float(x) for x in cost["sla_targets"]["p95_ttft_ms"]]

    figures = out / "results" / "figures"
    paths = [
        plot_latency_throughput(runs, figures / "latency_vs_throughput.png"),
        plot_tpot_throughput(runs, figures / "tpot_vs_throughput.png"),
    ]

    markers = pareto_markers(runs, quality, gpu_hourly_usd=price, max_ttft_p95_s=sla_ttft_ms / 1e3)
    if markers:
        paths.append(
            plot_pareto(
                markers,
                figures / "pareto_quality_cost.png",
                sla_label=f"p95 TTFT ≤ {sla_ttft_ms:.0f} ms",
                quality_label="GSM8K exact match (8-shot, strict)",
                reference_config="vllm-bf16",
            )
        )

    blocks = {
        "sla-table": sla_table(runs, gpu_hourly_usd=price, ttft_budgets_ms=budgets),
        "methodology-table": methodology_table(
            runs, baseline_config_id="vllm-bf16", rate_rps=sla_methodology_rate
        ),
        "drift-table": drift_table(runs, config_id="vllm-bf16", rate_rps=sla_methodology_rate),
        "crossvalidation-table": crossvalidation_table(results.parent / "crossvalidation.json"),
        "latency-table": latency_table(runs, price),
        "validity": validity_summary(runs),
        "cost-note": cost_note(
            price, ref["source_vendor"], ref["source_url"], ref["accessed_date"]
        ),
    }
    if quality:
        blocks["quality-table"] = quality_table(quality)

    for doc in ("README.md", "REPORT.md"):
        path = out / doc
        if not path.exists():
            continue
        present = {k: v for k, v in blocks.items() if f"BEGIN:{k}" in path.read_text()}
        if present:
            render_into(path, present)
            console.print(f"  updated {doc}: {', '.join(sorted(present))}")

    written = write_summary_json(runs, out / "docs" / "results.json", price)
    for path in [*paths, written]:
        console.print(f"  wrote {path}")


@app.command()
def methodology(
    config: Annotated[Path, typer.Option(help="Sweep matrix")] = Path("configs/sweep.yaml"),
    gpu: Annotated[int, typer.Option(help="Physical GPU index")] = 1,
    results: Annotated[Path, typer.Option(help="Where to write results")] = Path("results/runs"),
    trace: Annotated[Path, typer.Option(help="Azure trace CSV")] = Path("data/azure_trace.csv"),
    matched_rate: Annotated[
        float, typer.Option(help="Poisson rate the exhibits are compared against")
    ] = 4.0,
    concurrency: Annotated[int, typer.Option(help="Closed-loop worker pool")] = 64,
    runs_only: Annotated[str, typer.Option(help="Comma-separated methodology run ids")] = "",
    require_locked_clocks: Annotated[bool, typer.Option(help="Refuse to run unlocked")] = False,
) -> None:
    """Run the exhibits and controls that test the methodology itself.

    These do not rank configurations and never appear on the frontier: a trace
    replay and a closed-loop run both carry no offered rate, which is the field
    every aggregation keys on.
    """
    runner = SweepRunner(
        load_sweep_config(config),
        gpu_index=gpu,
        results_dir=results,
        configs_dir=config.parent,
        require_locked_clocks=require_locked_clocks,
    )
    selected = [r.strip() for r in runs_only.split(",") if r.strip()]
    written = runner.run_methodology(
        trace_path=trace,
        matched_rate_rps=matched_rate,
        closed_loop_concurrency=concurrency,
        run_ids=selected or None,
    )
    console.print(f"\n[green]wrote {len(written)} result file(s)[/green] to {results}")


def _run_upstream_benchmark(
    *,
    argv: list[str],
    dataset: Path,
    out_dir: Path,
    hf_cache: Path,
    image_ref: str,
) -> None:
    """Run vLLM's harness in the engine's own image against the live server."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "sudo",
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{dataset.resolve()}:/data/sharegpt_v3.json:ro",
            "-v",
            f"{out_dir.resolve()}:/out",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface:ro",
            "--entrypoint",
            "bash",
            image_ref,
            "-c",
            " ".join(argv),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    if proc.returncode != 0:
        console.print(f"[red]upstream benchmark failed[/red]\n{proc.stdout[-2000:]}")
        console.print(proc.stderr[-2000:])
        raise typer.Exit(code=1)


def _render_crossvalidation(
    agreements: list[MetricAgreement],
    *,
    title: str,
    ours_out_len: float,
    up_out_len: float,
    fixed: bool,
) -> None:
    table = Table(title=title)
    for col in ("Metric", "This harness", "vllm bench serve", "Δ", "Agrees"):
        table.add_column(col, justify="right" if col != "Metric" else "left")
    for a in agreements:
        note = "yes" if a.agrees else ("NO*" if a.length_sensitive else "NO")
        table.add_row(
            a.metric,
            f"{a.ours:.2f} {a.unit}",
            f"{a.upstream:.2f} {a.unit}",
            f"{a.delta_pct:+.1f}%",
            note,
        )
    console.print(table)
    console.print(
        f"  output tokens/request: ours {ours_out_len:.1f}, upstream {up_out_len:.1f} "
        f"(ratio {ours_out_len / up_out_len:.3f})"
    )
    if not fixed:
        console.print("  * length-sensitive: a workload difference alone moves this metric.")


@app.command("crossvalidate")
def cross_validate(
    config: Annotated[Path, typer.Option(help="Sweep matrix")] = Path("configs/sweep.yaml"),
    gpu: Annotated[int, typer.Option(help="Physical GPU index")] = 1,
    config_id: Annotated[str, typer.Option(help="Configuration to cross-check")] = "vllm-bf16",
    rate: Annotated[float, typer.Option(help="Offered rate to match")] = 4.0,
    results: Annotated[Path, typer.Option(help="Directory of result JSON")] = Path("results/runs"),
    dataset: Annotated[Path, typer.Option(help="ShareGPT corpus")] = Path("data/sharegpt_v3.json"),
    out: Annotated[Path, typer.Option(help="Where to write the comparison")] = Path(
        "results/crossvalidation.json"
    ),
    fixed_input: Annotated[
        int, typer.Option(help="Constant input tokens on both sides (0 = sample ShareGPT)")
    ] = 0,
    fixed_output: Annotated[int, typer.Option(help="Constant output tokens on both sides")] = 0,
) -> None:
    """Measure one configuration with vLLM's own harness and compare.

    Every other correctness check here is self-referential — our percentiles
    against numpy, our arrivals against a KS test. This is the only one that
    runs a *different implementation* against the same live server, which is the
    difference between "our tests pass" and "an independent harness agrees".

    The upstream benchmark runs inside the engine's own pinned container, so
    there is no second environment to install or explain.

    With ``--fixed-input``/``--fixed-output`` both harnesses are put on constant
    lengths and ours is measured in the same engine lifetime. That matters
    because the two sample ShareGPT differently — measured, ours drew 303 output
    tokens per request against upstream's 192 from the same corpus — so
    throughput and end-to-end latency differ by the length ratio however correct
    both harnesses are. Fixing the lengths removes the workload from the
    comparison and leaves only the code.
    """
    sweep = load_sweep_config(config)
    runner = SweepRunner(
        sweep,
        gpu_index=gpu,
        results_dir=results,
        configs_dir=config.parent,
        dataset_path=dataset,
    )
    entry = sweep.configuration(config_id)
    profile = load_engine_profile(entry.engine, config.parent)
    engine = _ENGINE_TYPES[entry.engine]()
    spec = runner._launch_spec(config_id, profile)

    fixed = (fixed_input, fixed_output) if fixed_input and fixed_output else None
    point = (
        runner.fixed_length_point(rate, input_tokens=fixed[0], output_tokens=fixed[1])
        if fixed
        else runner.rate_point(rate)
    )
    n_prompts = len(point.specs) - sweep.defaults.warmup_requests

    # Before the engine starts: taken afterwards it would fail on free VRAM the
    # engine is legitimately holding.
    preflight = run_preflight(
        gpu,
        results_path=str(results),
        gpu_memory_utilization=sweep.defaults.gpu_memory_utilization,
    )
    for warning in preflight.warnings:
        console.print(f"  [preflight] {warning}")

    label = f"fixed {fixed[0]}/{fixed[1]} tokens" if fixed else "ShareGPT lengths"
    console.print(f"[bold]{config_id}[/bold] @ {rate:g} rps, {n_prompts} prompts, {label}")
    console.print("  starting engine...")
    handle = engine.start(spec)
    console.print(f"  ready (kernel={handle.selected_kernel})")

    upstream_name = f"{out.stem}.upstream.json"
    try:
        ours: list[RunResult] = []
        if fixed:
            console.print("  measuring with this harness...")
            ours = [runner.measure(handle, point, 0, preflight)]
            console.print(
                f"    TTFT mean {ours[0].ttft_s.mean * 1e3:.2f} ms | "
                f"{ours[0].output_token_throughput:.1f} tok/s"
            )

        argv = upstream_args(
            model=handle.spec.model_hf_id,
            dataset_path="/data/sharegpt_v3.json",
            num_prompts=n_prompts,
            request_rate=rate,
            seed=sweep.workload.seed,
            port=spec.port,
            result_dir="/out",
            result_filename=upstream_name,
            fixed_lengths=fixed,
        )
        console.print("  running upstream harness...")
        _run_upstream_benchmark(
            argv=argv,
            dataset=dataset,
            out_dir=out.parent,
            hf_cache=spec.hf_cache_dir,
            image_ref=f"{spec.image}@{spec.image_digest}",
        )
    finally:
        engine.stop(spec)
        console.print("  engine stopped")

    upstream = load_upstream_result(out.parent / upstream_name)
    if not ours:
        ours = _load_runs(results)
    agreements = compare_to_upstream(ours, upstream, config_id=config_id, rate_rps=rate)
    if not agreements:
        console.print("[yellow]no matching runs of ours to compare against[/yellow]")
        raise typer.Exit(code=1)

    ours_out_len = sum(r.workload.output_len_tokens.mean for r in ours) / len(ours)
    up_out_len = upstream["total_output_tokens"] / upstream["completed"]

    _render_crossvalidation(
        agreements,
        title=f"Cross-validation — {config_id} @ {rate:g} rps ({label})",
        ours_out_len=ours_out_len,
        up_out_len=up_out_len,
        fixed=bool(fixed),
    )

    out.write_text(
        json.dumps(
            {
                "config_id": config_id,
                "rate_rps": rate,
                "matched_lengths": bool(fixed),
                "fixed_input_tokens": fixed[0] if fixed else None,
                "fixed_output_tokens": fixed[1] if fixed else None,
                "output_tokens_per_request": {
                    "ours": ours_out_len,
                    "upstream": up_out_len,
                    "ratio": ours_out_len / up_out_len,
                },
                "metrics": [
                    {
                        "metric": a.metric,
                        "ours": a.ours,
                        "upstream": a.upstream,
                        "unit": a.unit,
                        "delta_pct": a.delta_pct,
                        "agrees": a.agrees,
                        "length_sensitive": a.length_sensitive,
                    }
                    for a in agreements
                ],
                "all_agree": all(a.agrees for a in agreements),
                "length_independent_agree": all(
                    a.agrees for a in agreements if not a.length_sensitive
                ),
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"  wrote {out}")


@app.command("arrival-study")
def arrival_study(
    config: Annotated[Path, typer.Option(help="Sweep matrix")] = Path("configs/sweep.yaml"),
    gpu: Annotated[int, typer.Option(help="Physical GPU index")] = 1,
    config_id: Annotated[str, typer.Option(help="Configuration to measure on")] = "vllm-bf16",
    rate: Annotated[float, typer.Option(help="Matched mean rate")] = 4.0,
    realizations: Annotated[int, typer.Option(help="Independent draws per process")] = 5,
    trace: Annotated[Path, typer.Option(help="Azure trace CSV")] = Path("data/azure_trace.csv"),
    results: Annotated[Path, typer.Option(help="Where to write results")] = Path("results/runs"),
    require_locked_clocks: Annotated[bool, typer.Option(help="Refuse to run unlocked")] = False,
) -> None:
    """Does the arrival *process* change tail latency, or just this draw of it?

    Repeating a run against one seeded schedule measures the server's
    variability, not the arrival process's — both sides come out tight and
    different, and the difference belongs to those two particular arrival
    sequences rather than to Poisson and the trace. This measures several
    independent draws of each at a matched mean rate, with an identical prompt
    list, so arrival timing is the only thing that varies.
    """
    sweep = load_sweep_config(config)
    runner = SweepRunner(
        sweep,
        gpu_index=gpu,
        results_dir=results,
        configs_dir=config.parent,
        require_locked_clocks=require_locked_clocks,
    )
    entry = sweep.configuration(config_id)
    profile = load_engine_profile(entry.engine, config.parent)
    engine = _ENGINE_TYPES[entry.engine]()
    spec = runner._launch_spec(config_id, profile)

    poisson, trace_points = runner.arrival_realizations(rate, trace_path=trace, count=realizations)
    preflight = run_preflight(
        gpu,
        results_path=str(results),
        gpu_memory_utilization=sweep.defaults.gpu_memory_utilization,
        require_locked_clocks=require_locked_clocks,
    )
    for warning in preflight.warnings:
        console.print(f"  [preflight] {warning}")

    console.print(f"[bold]{config_id}[/bold] — {realizations} draws per process @ {rate:g} rps")
    console.print("  starting engine...")
    handle = engine.start(spec)
    console.print(f"  ready (kernel={handle.selected_kernel})")

    written: list[Path] = []
    try:
        # Interleaved rather than grouped: if the machine drifts during the
        # study, grouping would load the whole drift onto one process and
        # manufacture a difference between them.
        for k, (p_point, t_point) in enumerate(zip(poisson, trace_points, strict=True)):
            for point in (p_point, t_point):
                run = runner.measure(handle, point, k, preflight)
                written.append(runner._write(run, label=point.label))
                console.print(
                    f"  {point.label}: TTFT p95 {run.ttft_s.p95 * 1e3:7.1f} ms | "
                    f"p99 {run.ttft_s.p99 * 1e3:7.1f} ms | "
                    f"{run.output_token_throughput:7.1f} tok/s | {run.validity.value}"
                )
    finally:
        engine.stop(spec)
        console.print("  engine stopped")

    console.print(f"\n[green]wrote {len(written)} result file(s)[/green] to {results}")
