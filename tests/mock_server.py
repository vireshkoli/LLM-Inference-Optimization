"""A minimal ASGI stand-in for an OpenAI-compatible streaming server.

Raw ASGI rather than a web framework: the load generator is being tested, so the
fewer layers between the dispatcher and the bytes it parses, the better.

The mock deliberately reproduces the shapes that break naive harnesses — a
role-only first chunk before prefill completes, and a trailing usage-only chunk
after the last token — so the full client path is exercised against them rather
than only the unit-tested collector.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

# The ASGI spec types messages as mutable mappings, and httpx's transport is
# annotated to match. Using dict here would typecheck locally but fail to
# satisfy ASGITransport's callable signature.
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class MockLLMServer:
    """Streams tokens on a configurable schedule and tracks concurrency.

    Args:
        ttft_s: Simulated prefill delay before the first real token.
        itl_s: Simulated inter-token delay.
        status: HTTP status to return; non-200 exercises the error path.
    """

    def __init__(
        self,
        *,
        ttft_s: float = 0.01,
        itl_s: float = 0.002,
        status: int = 200,
        emit_role_chunk: bool = True,
        emit_usage_chunk: bool = True,
    ) -> None:
        self.ttft_s = ttft_s
        self.itl_s = itl_s
        self.status = status
        self.emit_role_chunk = emit_role_chunk
        self.emit_usage_chunk = emit_usage_chunk

        self.requests_received = 0
        self.in_flight = 0
        #: Peak simultaneous in-flight requests. The open-loop signature: a
        #: closed-loop client would never drive this above its concurrency cap.
        self.max_in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover - lifespan events
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        request = json.loads(body) if body else {}
        max_tokens = int(request.get("max_tokens", 1))

        self.requests_received += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

        try:
            if self.status != 200:
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps({"error": "simulated failure"}).encode(),
                        "more_body": False,
                    }
                )
                return

            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )

            # Role-only chunk, sent *before* prefill completes. A harness that
            # times TTFT to the first chunk will read ~0 here instead of ttft_s.
            if self.emit_role_chunk:
                await send(
                    {
                        "type": "http.response.body",
                        "body": _sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
                        "more_body": True,
                    }
                )

            for i in range(max_tokens):
                await asyncio.sleep(self.ttft_s if i == 0 else self.itl_s)
                await send(
                    {
                        "type": "http.response.body",
                        "body": _sse(
                            {"choices": [{"index": 0, "text": f"t{i} ", "finish_reason": None}]}
                        ),
                        "more_body": True,
                    }
                )

            if self.emit_usage_chunk:
                await send(
                    {
                        "type": "http.response.body",
                        "body": _sse({"choices": [], "usage": {"completion_tokens": max_tokens}}),
                        "more_body": True,
                    }
                )

            await send(
                {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}
            )
        finally:
            self.in_flight -= 1
