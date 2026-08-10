"""Scrape the engine's own Prometheus endpoint.

Recorded **alongside** the client-side measurements, never instead of them. The
distinction matters: a server cannot observe the queueing delay a client
experiences, and a benchmark that trusts the server's self-report inherits every
blind spot in the server's instrumentation. The published TTFT and TPOT come
from the load generator; these counters are kept so the two can be reconciled,
and so engine-internal state that the client genuinely cannot see — KV-cache
occupancy, scheduler queue depth, preemptions — travels with each result.

Parsing is a deliberately small subset of the Prometheus text format: counters
and gauges only. Histograms are left to Grafana, which already does that well.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

__all__ = ["EngineMetrics", "parse_prometheus_text", "scrape_engine_metrics"]


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    """Engine-internal counters at a moment in time.

    All fields optional: metric names differ between engines and move between
    versions, and a missing metric must degrade a record rather than fail a run
    that has already cost GPU time.
    """

    #: Cumulative counters — differences across a window give rates.
    generation_tokens_total: float | None = None
    prompt_tokens_total: float | None = None
    preemptions_total: float | None = None
    #: Instantaneous gauges.
    num_requests_running: float | None = None
    num_requests_waiting: float | None = None
    kv_cache_usage_perc: float | None = None
    #: Every series actually seen, so an engine change is visible in the record.
    metric_names_seen: tuple[str, ...] = ()

    def delta(self, earlier: EngineMetrics) -> EngineMetrics:
        """Counter differences between two snapshots.

        Gauges are taken from the later snapshot; subtracting them would be
        meaningless.
        """

        def diff(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else a - b

        return EngineMetrics(
            generation_tokens_total=diff(
                self.generation_tokens_total, earlier.generation_tokens_total
            ),
            prompt_tokens_total=diff(self.prompt_tokens_total, earlier.prompt_tokens_total),
            preemptions_total=diff(self.preemptions_total, earlier.preemptions_total),
            num_requests_running=self.num_requests_running,
            num_requests_waiting=self.num_requests_waiting,
            kv_cache_usage_perc=self.kv_cache_usage_perc,
            metric_names_seen=self.metric_names_seen,
        )


# Verified against a live vLLM v0.26.0 /metrics endpoint. The V1 engine renamed
# several of these — notably kv_cache_usage_perc, which older guides call
# gpu_cache_usage_perc. A dashboard or scraper on the old name silently reads
# nothing and looks exactly like an idle server.
_FIELD_BY_METRIC = {
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    # SGLang's equivalents, so one scraper serves both engines.
    "sglang:num_running_reqs": "num_requests_running",
    "sglang:num_queue_reqs": "num_requests_waiting",
    "sglang:token_usage": "kv_cache_usage_perc",
    "sglang:gen_throughput": "generation_tokens_total",
}


def parse_prometheus_text(body: str) -> EngineMetrics:
    """Extract the counters and gauges of interest from an exposition body.

    Series with labels are summed, since a multi-worker engine reports one
    series per model or per worker and the aggregate is what a run cares about.
    """
    totals: dict[str, float] = {}
    seen: set[str] = set()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        name_part, _, value_part = line.rpartition(" ")
        if not name_part:
            continue

        metric = name_part.split("{", 1)[0].strip()
        seen.add(metric)

        field = _FIELD_BY_METRIC.get(metric)
        if field is None:
            continue
        try:
            totals[field] = totals.get(field, 0.0) + float(value_part)
        except ValueError:
            continue

    return EngineMetrics(
        generation_tokens_total=totals.get("generation_tokens_total"),
        prompt_tokens_total=totals.get("prompt_tokens_total"),
        preemptions_total=totals.get("preemptions_total"),
        num_requests_running=totals.get("num_requests_running"),
        num_requests_waiting=totals.get("num_requests_waiting"),
        kv_cache_usage_perc=totals.get("kv_cache_usage_perc"),
        metric_names_seen=tuple(sorted(seen)),
    )


def scrape_engine_metrics(
    base_url: str, *, metrics_path: str = "/metrics", timeout_s: float = 10.0
) -> EngineMetrics | None:
    """Snapshot the engine's metrics endpoint.

    Returns ``None`` on any failure. Losing a metrics snapshot is a gap in a
    supplementary record; aborting a measured run over it would be far worse.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}{metrics_path}", timeout=timeout_s)
        if response.status_code != httpx.codes.OK:
            return None
        return parse_prometheus_text(response.text)
    except httpx.HTTPError:
        return None
