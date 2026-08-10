"""Server-sent-event parsing and per-request timing extraction.

This module owns the single most easily-botched definition in the whole
benchmark:

    **TTFT is the time to the first streamed chunk carrying an actual token.**

OpenAI-compatible servers routinely emit a first chunk containing only
``{"role": "assistant"}`` with no content, and some emit a trailing usage-only
chunk with an empty ``choices`` array. Timing to the first *chunk* rather than
the first *token* silently understates TTFT — by however long the server takes
to run prefill — and TTFT is a headline metric. ``tests/test_stream.py`` pins
this behaviour against both shapes.

The collector is a pure state machine fed ``(line, timestamp)`` pairs, so every
edge case is unit-testable without a server, a socket or a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["StreamCollector", "StreamResult"]

_DATA_PREFIX = "data:"
_DONE_SENTINEL = "[DONE]"


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Timing and content extracted from one streamed response."""

    #: Seconds from dispatch to the first content-bearing chunk. ``None`` when
    #: the request produced no tokens at all.
    ttft_s: float | None
    #: Gaps between consecutive content-bearing chunks. Length is
    #: ``max(0, token_chunks - 1)``.
    itl_s: tuple[float, ...]
    #: Seconds from dispatch to the final chunk.
    e2e_s: float | None
    #: Number of chunks that carried content.
    token_chunks: int
    #: Completion tokens as reported by the server, when it sends a usage block.
    #: Preferred over ``token_chunks`` because a chunk is not guaranteed to be
    #: exactly one token.
    usage_completion_tokens: int | None
    text: str
    finish_reason: str | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.token_chunks > 0

    @property
    def output_tokens(self) -> int:
        """Best available output-token count, preferring the server's own."""
        if self.usage_completion_tokens is not None:
            return self.usage_completion_tokens
        return self.token_chunks


@dataclass(slots=True)
class StreamCollector:
    """Accumulates SSE lines into a :class:`StreamResult`.

    Args:
        dispatch_time_s: Monotonic clock reading taken immediately before the
            request was handed to the transport. All timings are relative to
            this, not to when the response object was constructed.
    """

    dispatch_time_s: float

    _ttft_s: float | None = None
    _last_token_time_s: float | None = None
    _last_any_time_s: float | None = None
    _itl_s: list[float] = field(default_factory=list)
    _chunks: list[str] = field(default_factory=list)
    _token_chunks: int = 0
    _usage_tokens: int | None = None
    _finish_reason: str | None = None
    _error: str | None = None
    _done: bool = False

    @property
    def done(self) -> bool:
        """True once the terminating ``[DONE]`` sentinel has been seen."""
        return self._done

    def feed(self, line: str, now_s: float) -> None:
        """Consume one SSE line.

        Blank lines, comments and unparseable payloads are tolerated rather than
        raised on: a malformed chunk should degrade one request into an error
        record, never abort a sweep that has already cost GPU-hours.
        """
        if self._done:
            return

        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return  # keep-alive comment or padding
        if not stripped.startswith(_DATA_PREFIX):
            return

        payload = stripped[len(_DATA_PREFIX) :].strip()
        if payload == _DONE_SENTINEL:
            self._done = True
            self._last_any_time_s = now_s
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            self._error = f"malformed SSE payload: {exc}"
            return

        self._last_any_time_s = now_s
        self._absorb(event, now_s)

    def _absorb(self, event: object, now_s: float) -> None:
        if not isinstance(event, dict):
            self._error = f"unexpected SSE payload type: {type(event).__name__}"
            return

        # An explicit error object mid-stream (e.g. the engine aborting a
        # request under memory pressure) must surface, not be silently dropped.
        if "error" in event:
            self._error = str(event["error"])
            return

        usage = event.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("completion_tokens")
            if isinstance(tokens, int):
                self._usage_tokens = tokens

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            # Usage-only trailing chunk. Carries no token; must not set TTFT.
            return

        first = choices[0]
        if not isinstance(first, dict):
            return

        reason = first.get("finish_reason")
        if isinstance(reason, str):
            self._finish_reason = reason

        content = self._extract_content(first)
        # An empty string is what a role-only delta looks like after extraction.
        # Treating it as a token here is exactly the bug this module exists to
        # prevent, so the emptiness check is load-bearing.
        if not content:
            return

        self._record_token(content, now_s)

    @staticmethod
    def _extract_content(choice: dict[str, object]) -> str:
        """Pull generated text from either API shape.

        ``/v1/chat/completions`` nests it under ``delta.content``;
        ``/v1/completions`` puts it directly in ``text``.
        """
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            return content if isinstance(content, str) else ""

        text = choice.get("text")
        return text if isinstance(text, str) else ""

    def _record_token(self, content: str, now_s: float) -> None:
        if self._ttft_s is None:
            self._ttft_s = now_s - self.dispatch_time_s
        elif self._last_token_time_s is not None:
            self._itl_s.append(now_s - self._last_token_time_s)

        self._last_token_time_s = now_s
        self._token_chunks += 1
        self._chunks.append(content)

    def fail(self, message: str) -> None:
        """Mark the request as failed (transport error, timeout, non-200)."""
        self._error = message
        self._done = True

    def result(self, end_time_s: float | None = None) -> StreamResult:
        """Finalise the record.

        Args:
            end_time_s: Monotonic reading when the response body closed. Falls
                back to the last observed event so a truncated stream still
                yields a usable end-to-end figure.
        """
        end = end_time_s if end_time_s is not None else self._last_any_time_s
        e2e = None if end is None else end - self.dispatch_time_s

        return StreamResult(
            ttft_s=self._ttft_s,
            itl_s=tuple(self._itl_s),
            e2e_s=e2e,
            token_chunks=self._token_chunks,
            usage_completion_tokens=self._usage_tokens,
            text="".join(self._chunks),
            finish_reason=self._finish_reason,
            error=self._error,
        )
