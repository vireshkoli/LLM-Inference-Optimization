"""SSE parsing and the TTFT definition.

The central assertion here is that a role-only first chunk does **not** set
TTFT. Real OpenAI-compatible servers emit ``{"delta": {"role": "assistant"}}``
before any token, and timing to that chunk understates TTFT by the entire
prefill duration — producing a headline metric that is wrong in the flattering
direction.
"""

from __future__ import annotations

import json

import pytest

from llmbench.loadgen.stream import StreamCollector


def sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}"


def chat_delta(content: str | None = None, *, role: str | None = None) -> dict[str, object]:
    delta: dict[str, object] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def completion_chunk(text: str) -> dict[str, object]:
    return {"choices": [{"index": 0, "text": text, "finish_reason": None}]}


class TestTTFTDefinition:
    def test_role_only_chunk_does_not_set_ttft(self) -> None:
        """The bug this module exists to prevent.

        The role delta arrives at t=0.10 but carries no token; the first real
        token arrives at t=0.50. TTFT is 0.50, not 0.10.
        """
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(role="assistant")), now_s=0.10)
        c.feed(sse(chat_delta(content="Hello")), now_s=0.50)

        assert c.result(end_time_s=0.6).ttft_s == pytest.approx(0.50)

    def test_empty_string_content_does_not_set_ttft(self) -> None:
        """Some servers send content="" rather than omitting the field."""
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="")), now_s=0.10)
        c.feed(sse(chat_delta(content="Hi")), now_s=0.40)

        assert c.result(end_time_s=0.5).ttft_s == pytest.approx(0.40)

    def test_ttft_is_relative_to_dispatch_not_to_zero(self) -> None:
        c = StreamCollector(dispatch_time_s=100.0)
        c.feed(sse(chat_delta(content="x")), now_s=100.25)
        assert c.result(end_time_s=100.3).ttft_s == pytest.approx(0.25)

    def test_no_tokens_yields_no_ttft(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(role="assistant")), now_s=0.1)
        c.feed("data: [DONE]", now_s=0.2)

        result = c.result(end_time_s=0.2)
        assert result.ttft_s is None
        assert result.succeeded is False

    def test_completions_api_shape_is_supported(self) -> None:
        """/v1/completions puts text at choices[0].text, not delta.content."""
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(completion_chunk("The")), now_s=0.2)
        c.feed(sse(completion_chunk(" cat")), now_s=0.3)

        result = c.result(end_time_s=0.35)
        assert result.ttft_s == pytest.approx(0.2)
        assert result.text == "The cat"


class TestInterTokenLatency:
    def test_itl_measures_gaps_between_tokens_only(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(role="assistant")), now_s=0.10)
        c.feed(sse(chat_delta(content="a")), now_s=0.50)
        c.feed(sse(chat_delta(content="b")), now_s=0.60)
        c.feed(sse(chat_delta(content="c")), now_s=0.75)

        result = c.result(end_time_s=0.8)
        assert result.itl_s == pytest.approx((0.10, 0.15))

    def test_itl_is_empty_for_a_single_token(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="only")), now_s=0.2)
        assert c.result(end_time_s=0.3).itl_s == ()

    def test_itl_count_is_one_less_than_tokens(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        for i in range(10):
            c.feed(sse(chat_delta(content=f"t{i}")), now_s=0.1 * (i + 1))
        result = c.result(end_time_s=1.1)
        assert result.token_chunks == 10
        assert len(result.itl_s) == 9


class TestTokenAccounting:
    def test_server_usage_overrides_chunk_count(self) -> None:
        """A streamed chunk is not guaranteed to be exactly one token.

        When the server reports its own accounting, trust it over our count.
        """
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="hello world")), now_s=0.2)
        c.feed(sse({"choices": [], "usage": {"completion_tokens": 7}}), now_s=0.25)

        result = c.result(end_time_s=0.3)
        assert result.token_chunks == 1
        assert result.usage_completion_tokens == 7
        assert result.output_tokens == 7

    def test_falls_back_to_chunk_count_without_usage(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="a")), now_s=0.2)
        c.feed(sse(chat_delta(content="b")), now_s=0.3)
        assert c.result(end_time_s=0.4).output_tokens == 2

    def test_usage_only_chunk_does_not_set_ttft(self) -> None:
        """A trailing usage chunk has an empty choices array and no token."""
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse({"choices": [], "usage": {"completion_tokens": 0}}), now_s=0.1)
        assert c.result(end_time_s=0.2).ttft_s is None

    def test_finish_reason_is_captured(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="x")), now_s=0.1)
        c.feed(
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}),
            now_s=0.2,
        )
        assert c.result(end_time_s=0.3).finish_reason == "length"


class TestProtocolRobustness:
    """A malformed chunk must degrade one request, never abort a sweep that has
    already cost GPU-hours."""

    def test_done_sentinel_terminates(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="a")), now_s=0.1)
        c.feed("data: [DONE]", now_s=0.2)
        assert c.done is True

    def test_input_after_done_is_ignored(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="a")), now_s=0.1)
        c.feed("data: [DONE]", now_s=0.2)
        c.feed(sse(chat_delta(content="late")), now_s=0.3)
        assert c.result(end_time_s=0.4).token_chunks == 1

    @pytest.mark.parametrize("line", ["", "   ", ": keep-alive", "event: ping"])
    def test_non_data_lines_are_ignored(self, line: str) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(line, now_s=0.1)
        c.feed(sse(chat_delta(content="x")), now_s=0.5)
        assert c.result(end_time_s=0.6).ttft_s == pytest.approx(0.5)

    def test_malformed_json_records_an_error(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed("data: {not valid json", now_s=0.1)
        result = c.result(end_time_s=0.2)
        assert result.error is not None
        assert "malformed SSE payload" in result.error

    def test_mid_stream_error_surfaces(self) -> None:
        """An engine aborting under memory pressure must not look like success."""
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="partial")), now_s=0.1)
        c.feed(sse({"error": "out of memory"}), now_s=0.2)

        result = c.result(end_time_s=0.3)
        assert result.error == "out of memory"
        assert result.succeeded is False

    def test_explicit_failure_marks_the_request_done(self) -> None:
        c = StreamCollector(dispatch_time_s=0.0)
        c.fail("HTTP 503: server overloaded")
        result = c.result(end_time_s=0.5)
        assert result.succeeded is False
        assert result.error is not None
        assert "503" in result.error

    def test_truncated_stream_still_reports_what_arrived(self) -> None:
        """No [DONE] sentinel — the connection simply ended."""
        c = StreamCollector(dispatch_time_s=0.0)
        c.feed(sse(chat_delta(content="a")), now_s=0.1)
        c.feed(sse(chat_delta(content="b")), now_s=0.2)

        result = c.result()
        assert result.token_chunks == 2
        assert result.e2e_s == pytest.approx(0.2)
