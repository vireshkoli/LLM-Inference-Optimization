"""Engine metrics scraping.

The fixture below is trimmed verbatim from a live vLLM v0.26.0 ``/metrics``
endpoint, so these tests pin behaviour against what the engine actually emits —
including the V1 metric renames that make a scraper built on documented names
silently read nothing.
"""

from __future__ import annotations

from llmbench.telemetry.prom import EngineMetrics, parse_prometheus_text

# Verbatim shape from vLLM v0.26.0, including HELP/TYPE lines and label sets.
VLLM_BODY = """\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 84213.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 26560.0
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 83.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0438
# HELP vllm:num_preemptions_total Cumulative number of preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:time_to_first_token_seconds Histogram of TTFT in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 12.0
vllm:time_to_first_token_seconds_sum 143.2
"""


class TestParsing:
    def test_extracts_counters_and_gauges(self) -> None:
        m = parse_prometheus_text(VLLM_BODY)
        assert m.generation_tokens_total == 84213.0
        assert m.prompt_tokens_total == 26560.0
        assert m.num_requests_running == 83.0
        assert m.num_requests_waiting == 0.0
        assert m.kv_cache_usage_perc == 0.0438
        assert m.preemptions_total == 0.0

    def test_uses_the_v1_kv_cache_name(self) -> None:
        """The V1 engine renamed gpu_cache_usage_perc -> kv_cache_usage_perc.

        A scraper on the old name reads nothing and is indistinguishable from an
        idle server, so the current name is pinned here.
        """
        assert "vllm:kv_cache_usage_perc" in parse_prometheus_text(VLLM_BODY).metric_names_seen
        old = VLLM_BODY.replace("kv_cache_usage_perc", "gpu_cache_usage_perc")
        assert parse_prometheus_text(old).kv_cache_usage_perc is None

    def test_ignores_comments_and_histograms(self) -> None:
        m = parse_prometheus_text(VLLM_BODY)
        assert "vllm:time_to_first_token_seconds_bucket" in m.metric_names_seen
        # Histograms are Grafana's job; they must not corrupt the scalar fields.
        assert m.generation_tokens_total == 84213.0

    def test_sums_series_across_labels(self) -> None:
        """A multi-worker engine reports one series per worker."""
        body = (
            'vllm:num_requests_running{engine="0"} 40.0\n'
            'vllm:num_requests_running{engine="1"} 43.0\n'
        )
        assert parse_prometheus_text(body).num_requests_running == 83.0

    def test_missing_metrics_are_none_not_zero(self) -> None:
        """Zero and absent mean different things: one is a measurement."""
        m = parse_prometheus_text("vllm:num_requests_running 5.0\n")
        assert m.num_requests_running == 5.0
        assert m.kv_cache_usage_perc is None

    def test_empty_body_is_tolerated(self) -> None:
        m = parse_prometheus_text("")
        assert m.metric_names_seen == ()
        assert m.generation_tokens_total is None

    def test_unparseable_value_is_skipped(self) -> None:
        assert parse_prometheus_text("vllm:num_requests_running NaNish\n").num_requests_running is (
            None
        )

    def test_recognises_sglang_names(self) -> None:
        """One scraper serves both engines; SGLang uses different names."""
        body = "sglang:num_running_reqs 12.0\nsglang:token_usage 0.31\n"
        m = parse_prometheus_text(body)
        assert m.num_requests_running == 12.0
        assert m.kv_cache_usage_perc == 0.31


class TestDelta:
    def test_counters_subtract_and_gauges_do_not(self) -> None:
        earlier = EngineMetrics(
            generation_tokens_total=1000.0, num_requests_running=5.0, kv_cache_usage_perc=0.1
        )
        later = EngineMetrics(
            generation_tokens_total=3500.0, num_requests_running=80.0, kv_cache_usage_perc=0.4
        )
        d = later.delta(earlier)
        assert d.generation_tokens_total == 2500.0
        # Gauges are instantaneous; subtracting them would be meaningless.
        assert d.num_requests_running == 80.0
        assert d.kv_cache_usage_perc == 0.4

    def test_missing_side_yields_none(self) -> None:
        d = EngineMetrics(generation_tokens_total=100.0).delta(EngineMetrics())
        assert d.generation_tokens_total is None
