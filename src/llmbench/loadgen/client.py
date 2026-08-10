"""Open-loop request dispatcher.

The defining property: a request is dispatched at its **scheduled** instant,
without waiting for earlier requests to complete. Each dispatch spawns a task
and the loop moves straight on to the next scheduled time. If the server slows
down, in-flight requests pile up — which is the true behaviour of a queue under
load, and exactly what a closed-loop generator hides.

Two implementation details are load-bearing, and both are easy to get wrong in
ways that silently reintroduce closed-loop behaviour:

1. **The connection pool must not throttle.** httpx defaults to 100 max
   connections. Once in-flight requests exceed that, new requests block waiting
   for a free connection — the client has become a closed-loop generator with
   concurrency 100, and nothing in the results says so. The pool is therefore
   sized past the worst-case in-flight count.
2. **Dispatch lag is measured, not assumed.** ``asyncio.sleep`` guarantees only
   a lower bound on delay, and an event loop under CPU pressure overshoots. The
   gap between scheduled and actual dispatch is recorded per request and drives
   the validity verdict in :mod:`llmbench.loadgen.guard`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from llmbench.loadgen.stream import StreamCollector, StreamResult
from llmbench.workload.arrivals import ArrivalSchedule
from llmbench.workload.prompts import RequestSpec

__all__ = ["LoadGenConfig", "LoadGenResult", "RequestRecord", "run_open_loop"]


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """Everything measured for a single request.

    Written to ``results/raw/*.jsonl`` (gitignored — hundreds of MB per sweep)
    and aggregated into the committed :class:`~llmbench.schema.RunResult`.
    """

    index: int
    is_warmup: bool
    scheduled_offset_s: float
    #: Actual minus scheduled dispatch. The open-loop integrity signal.
    dispatch_lag_s: float
    input_tokens: int
    requested_max_tokens: int
    ttft_s: float | None
    itl_s: tuple[float, ...]
    e2e_s: float | None
    output_tokens: int
    finish_reason: str | None
    status_code: int | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.ttft_s is not None

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token, excluding prefill.

        Defined as ``(e2e - ttft) / (output_tokens - 1)``: the first token's cost
        is TTFT and belongs to prefill, so including it here would blend the two
        phases and hide whether a configuration is prefill- or decode-bound.
        """
        if self.e2e_s is None or self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass(frozen=True, slots=True)
class LoadGenConfig:
    """Transport and protocol settings for a run."""

    base_url: str
    model: str
    #: ``/v1/completions`` avoids chat-template variation across engines, so the
    #: prompt sent is exactly the prompt measured.
    endpoint_path: str = "/v1/completions"
    #: Paired with ``max_tokens`` so every configuration generates the same
    #: number of tokens regardless of where its model would naturally stop.
    ignore_eos: bool = True
    temperature: float = 0.0
    request_timeout_s: float = 600.0
    #: Sized past worst-case in-flight requests; see module docstring.
    max_connections: int = 4096
    api_key: str = "EMPTY"


@dataclass(frozen=True, slots=True)
class LoadGenResult:
    records: tuple[RequestRecord, ...] = field(default_factory=tuple)
    #: Wall-clock span of the measurement window, warmup excluded.
    measurement_window_s: float = 0.0
    warmup_discarded: int = 0

    @property
    def measured(self) -> tuple[RequestRecord, ...]:
        """Records eligible for reporting — warmup removed."""
        return tuple(r for r in self.records if not r.is_warmup)


def _build_payload(spec: RequestSpec, config: LoadGenConfig) -> dict[str, object]:
    return {
        "model": config.model,
        "prompt": spec.prompt,
        "max_tokens": spec.max_tokens,
        "temperature": config.temperature,
        "stream": True,
        # Ask the server for its own token accounting; more trustworthy than
        # counting streamed chunks, which are not guaranteed one-token-each.
        "stream_options": {"include_usage": True},
        "ignore_eos": config.ignore_eos,
    }


async def _fire_one(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    config: LoadGenConfig,
    *,
    is_warmup: bool,
    scheduled_offset_s: float,
    dispatch_lag_s: float,
    dispatch_time_s: float,
) -> RequestRecord:
    """Issue one streamed request and time it.

    Never raises: a transport failure becomes an error record. A sweep that has
    already cost GPU-hours must not die because one request was refused.
    """
    collector = StreamCollector(dispatch_time_s=dispatch_time_s)
    status: int | None = None

    try:
        async with client.stream(
            "POST",
            config.endpoint_path,
            json=_build_payload(spec, config),
        ) as response:
            status = response.status_code
            if status != httpx.codes.OK:
                body = (await response.aread()).decode("utf-8", errors="replace")
                collector.fail(f"HTTP {status}: {body[:200]}")
            else:
                async for line in response.aiter_lines():
                    collector.feed(line, time.perf_counter())
                    if collector.done:
                        break
    except (TimeoutError, httpx.HTTPError) as exc:
        collector.fail(f"{type(exc).__name__}: {exc}")

    result: StreamResult = collector.result(end_time_s=time.perf_counter())

    return RequestRecord(
        index=spec.index,
        is_warmup=is_warmup,
        scheduled_offset_s=scheduled_offset_s,
        dispatch_lag_s=dispatch_lag_s,
        input_tokens=spec.input_tokens,
        requested_max_tokens=spec.max_tokens,
        ttft_s=result.ttft_s,
        itl_s=result.itl_s,
        e2e_s=result.e2e_s,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        status_code=status,
        error=result.error,
    )


async def run_open_loop(
    schedule: ArrivalSchedule,
    specs: Sequence[RequestSpec],
    config: LoadGenConfig,
    *,
    warmup_requests: int = 0,
    client: httpx.AsyncClient | None = None,
) -> LoadGenResult:
    """Dispatch ``specs`` at the instants given by ``schedule``.

    Args:
        schedule: Pre-computed, seeded dispatch offsets. Must be the same length
            as ``specs``.
        warmup_requests: Leading requests marked as warmup and excluded from
            reported statistics. They are still *issued*, because the point is to
            reach steady state — discarding without sending would defeat it.
        client: Injectable transport. Tests pass a client wired to an in-process
            ASGI app so the full dispatch path runs with no socket and no GPU.

    Raises:
        ValueError: If schedule and specs disagree in length.
    """
    if len(schedule) != len(specs):
        msg = f"schedule has {len(schedule)} arrivals but {len(specs)} requests were supplied"
        raise ValueError(msg)
    if warmup_requests > len(specs):
        msg = f"warmup_requests ({warmup_requests}) exceeds total requests ({len(specs)})"
        raise ValueError(msg)

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout_s),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_connections,
            ),
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    tasks: list[asyncio.Task[RequestRecord]] = []
    measurement_start: float | None = None

    try:
        origin = time.perf_counter()

        for offset, spec in zip(schedule.offsets_s, specs, strict=True):
            target = origin + offset
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            # Read the clock *after* sleeping: the difference from target is the
            # scheduling error we are obliged to report, not to hide.
            actual = time.perf_counter()
            is_warmup = spec.index < warmup_requests
            if not is_warmup and measurement_start is None:
                measurement_start = actual

            tasks.append(
                asyncio.create_task(
                    _fire_one(
                        client,
                        spec,
                        config,
                        is_warmup=is_warmup,
                        scheduled_offset_s=offset,
                        dispatch_lag_s=actual - target,
                        dispatch_time_s=actual,
                    )
                )
            )

        records = await asyncio.gather(*tasks)
        measurement_end = time.perf_counter()
    finally:
        if owns_client:
            await client.aclose()

    window = 0.0 if measurement_start is None else measurement_end - measurement_start

    return LoadGenResult(
        records=tuple(records),
        measurement_window_s=window,
        warmup_discarded=warmup_requests,
    )
