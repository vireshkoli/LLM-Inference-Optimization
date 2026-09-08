"""Cross-validation against vLLM's own benchmark harness.

Every other correctness check in this repository is self-referential: our tests
assert that our percentile function agrees with numpy, that our arrival process
passes a KS test, that our metrics are internally consistent. All of that can be
true while the harness measures the wrong thing in the same way twice.

The only independent check is a *different implementation, written by different
people, against the same live server*. vLLM ships one — ``vllm bench serve`` —
and it is the harness most published numbers for this engine come from, so
agreement also makes our results comparable to theirs rather than merely
self-consistent.

**Run it in the engine's own container.** The pinned vLLM image already contains
that command and its dependencies, so there is no second environment to install,
drift, or explain. The comparison is then between two harnesses, not between two
Python environments.

**What is matched, and what deliberately is not.** Same server, same model, same
seed, same request rate, same ShareGPT corpus, same ``ignore_eos``, Poisson
arrivals on both sides (``--burstiness 1.0``). What differs is everything we
would be testing: the dispatch loop, the SSE parsing, the TTFT definition, the
percentile computation. Those are the subject of the comparison, so making them
match would defeat it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llmbench.metrics.percentiles import summarize
from llmbench.schema import ArrivalProcess, RunResult

__all__ = ["MetricAgreement", "compare_to_upstream", "load_upstream_result", "upstream_args"]


def upstream_args(
    *,
    model: str,
    dataset_path: str,
    num_prompts: int,
    request_rate: float,
    seed: int,
    port: int,
    result_dir: str,
    result_filename: str,
    fixed_lengths: tuple[int, int] | None = None,
) -> list[str]:
    """Argv for ``vllm bench serve``, matched to one of our rate points.

    ``--burstiness 1.0`` is explicit rather than left to the default: it is what
    makes the upstream arrival process Poisson, and therefore the same process
    ours generates. A silent default here would be the single easiest way to
    compare two different workloads and call the disagreement a bug.

    Args:
        fixed_lengths: ``(input_tokens, output_tokens)`` to switch upstream onto
            its ``random`` dataset at constant lengths. Both harnesses sample
            ShareGPT differently — measured, ours drew 303 output tokens per
            request against upstream's 192 from the same corpus — so throughput
            and end-to-end latency differ by the length ratio no matter how
            correct both harnesses are. Fixing the lengths on both sides removes
            the workload from the comparison and leaves only the code.
    """
    if fixed_lengths is not None:
        dataset = [
            "--dataset-name",
            "random",
            "--input-len",
            str(fixed_lengths[0]),
            "--output-len",
            str(fixed_lengths[1]),
        ]
    else:
        dataset = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path]
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{port}",
        # /v1/completions, matching our own client: no chat template means the
        # prompt sent is exactly the prompt measured, on both sides.
        "--endpoint",
        "/v1/completions",
        "--model",
        model,
        *dataset,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        f"{request_rate:g}",
        "--burstiness",
        "1.0",
        "--seed",
        str(seed),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_filename,
        "--disable-tqdm",
    ]


#: Metrics whose value depends on how many tokens each request generates.
#: They cannot validate a harness unless both sides were given the same length
#: distribution, because a correct harness measuring a longer workload reports a
#: larger number and is not thereby wrong.
LENGTH_SENSITIVE = frozenset({"E2E mean", "Output throughput"})


@dataclass(frozen=True, slots=True)
class MetricAgreement:
    """One metric, measured by both harnesses."""

    metric: str
    ours: float
    upstream: float
    unit: str = "ms"

    @property
    def length_sensitive(self) -> bool:
        """Whether a workload difference alone can move this metric."""
        return self.metric in LENGTH_SENSITIVE

    @property
    def delta_pct(self) -> float:
        """Signed difference as a percentage of the upstream value."""
        if not self.upstream:
            return 0.0
        return (self.ours / self.upstream - 1.0) * 100.0

    @property
    def agrees(self) -> bool:
        """Within 5 %.

        Loose enough to absorb the genuine differences between two runs — they
        are separate measurements against a live server, not a replay of the
        same one — and tight enough that a real defect in either dispatch loop,
        SSE parser or percentile implementation would show. A disagreement
        larger than this is a finding, not noise.
        """
        return abs(self.delta_pct) <= 5.0


def load_upstream_result(path: Path) -> dict[str, float]:
    """Read the JSON ``vllm bench serve --save-result`` writes.

    Upstream reports latencies in **milliseconds** and throughput in tokens per
    second. Converting ours to milliseconds at the comparison boundary — rather
    than converting theirs to seconds — keeps their numbers exactly as published
    so a reader can check them against the file.

    Raises:
        ValueError: If the file does not contain the expected metric keys, which
            means the upstream schema moved and the comparison would otherwise
            silently compare nothing.
    """
    payload = json.loads(path.read_text())
    required = ("mean_ttft_ms", "p99_ttft_ms", "output_throughput")
    missing = [k for k in required if k not in payload]
    if missing:
        msg = f"{path} is missing {missing}; upstream result schema has changed"
        raise ValueError(msg)
    return {k: float(v) for k, v in payload.items() if isinstance(v, int | float)}


def compare_to_upstream(
    runs: Sequence[RunResult],
    upstream: dict[str, float],
    *,
    config_id: str,
    rate_rps: float,
) -> list[MetricAgreement]:
    """Line up our aggregated metrics against theirs.

    Uses the mean across our repeats, because upstream runs once and our single
    most comparable number is the centre of our distribution rather than any
    individual repeat.

    Returns an empty list when we have no matching runs, so a report generated
    before the cross-validation has been executed shows a gap rather than
    failing.
    """
    ours = [
        r
        for r in runs
        if r.config_id == config_id
        and r.workload.request_rate_rps == rate_rps
        and r.workload.arrival_process is ArrivalProcess.POISSON
        and r.is_reportable
    ]
    if not ours:
        return []

    def mean_ms(attr: str, percentile: str) -> float:
        return summarize([getattr(getattr(r, attr), percentile) for r in ours]).mean * 1e3

    pairs: list[tuple[str, float, float, str]] = [
        ("TTFT mean", mean_ms("ttft_s", "mean"), upstream.get("mean_ttft_ms", 0.0), "ms"),
        ("TTFT p99", mean_ms("ttft_s", "p99"), upstream.get("p99_ttft_ms", 0.0), "ms"),
        ("TPOT mean", mean_ms("tpot_s", "mean"), upstream.get("mean_tpot_ms", 0.0), "ms"),
        ("TPOT p99", mean_ms("tpot_s", "p99"), upstream.get("p99_tpot_ms", 0.0), "ms"),
        ("E2E mean", mean_ms("e2e_latency_s", "mean"), upstream.get("mean_e2el_ms", 0.0), "ms"),
        (
            "Output throughput",
            summarize([r.output_token_throughput for r in ours]).mean,
            upstream.get("output_throughput", 0.0),
            "tok/s",
        ),
    ]
    return [
        MetricAgreement(metric=name, ours=mine, upstream=theirs, unit=unit)
        for name, mine, theirs, unit in pairs
        if theirs
    ]
