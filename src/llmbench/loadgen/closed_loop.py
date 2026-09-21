"""Closed-loop dispatcher — an exhibit, never a headline.

This is the load generator the rest of this repository argues against. It exists
so the argument can be *demonstrated* on the same hardware, with the same model
and the same prompts, rather than asserted with a citation.

**How it differs.** A fixed pool of ``concurrency`` workers each run a loop:
issue a request, wait for it to complete, issue the next. The offered load is
therefore a consequence of the server's speed — the generator and the thing
under test are coupled.

**What that coupling did here was not what the textbook says.** The classic
objection is *coordinated omission*: when the server stalls, the generator
stalls with it, the slow period is under-sampled, and the reported tail is
optimistic. This exhibit was built to measure that error and found the opposite
sign. At throughput-matched points the closed loop never understates the
open-loop p99; near the knee it reports a tail 3.3-3.7x *worse*. A
continuous-batching engine below saturation has no stalls to hide, and what the
closed loop does instead is hold occupancy at a constant maximum, so every new
request's prefill competes with a full batch of decodes. Poisson arrivals at
the same mean let occupancy fluctuate. Either way, a coupled generator does not
measure the tail a real arrival process would see — which is the reason
open-loop is used, and the reason this generator is an exhibit.

**Why the comparison is fair.** The exhibit is run at a concurrency chosen so
its *achieved* throughput matches an open-loop run's, and it draws from the same
seeded prompt list. Same server, same work, same tokens — the only difference is
how arrivals are generated. Any gap in the reported tail is therefore
attributable to the generator, which is the claim being tested.

Every record it produces carries ``scheduled_offset_s`` and ``dispatch_lag_s``
of zero, because in closed loop there is no schedule to be late against. That is
not a measurement, it is a structural property, and it is why the dispatch-lag
validity guard is meaningless here: a closed-loop client can never detect that
it has become the bottleneck, because being the bottleneck is its design.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx

from llmbench.loadgen.client import LoadGenConfig, LoadGenResult, RequestRecord, fire_one
from llmbench.workload.prompts import RequestSpec

__all__ = ["run_closed_loop"]


async def run_closed_loop(
    specs: Sequence[RequestSpec],
    config: LoadGenConfig,
    *,
    concurrency: int,
    warmup_requests: int = 0,
    client: httpx.AsyncClient | None = None,
) -> LoadGenResult:
    """Drive the server with a fixed pool of workers.

    The per-request path is deliberately the *same* ``fire_one`` the open-loop
    generator uses. Re-implementing it here would mean the two runs differed in
    their SSE parsing and TTFT definition as well as in their arrival process,
    and the exhibit would no longer isolate the thing it exists to isolate.

    Args:
        concurrency: Number of workers. This *is* the offered load in a closed
            loop — there is no rate to set, which is itself the point.
        warmup_requests: Leading requests excluded from reported statistics.
            Still issued, so the server reaches steady state.

    Raises:
        ValueError: On a non-positive concurrency, or more warmup than requests.
    """
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
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
                max_connections=max(concurrency * 2, 16),
                max_keepalive_connections=max(concurrency * 2, 16),
            ),
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    queue: asyncio.Queue[RequestSpec] = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)

    records: list[RequestRecord] = []
    measurement_start: float | None = None

    async def worker() -> None:
        nonlocal measurement_start
        while True:
            try:
                spec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            dispatch = time.perf_counter()
            is_warmup = spec.index < warmup_requests
            if not is_warmup and measurement_start is None:
                measurement_start = dispatch

            records.append(
                await fire_one(
                    client,
                    spec,
                    config,
                    is_warmup=is_warmup,
                    # Zero by construction, not by measurement: a closed-loop
                    # client has no schedule it could be late against.
                    scheduled_offset_s=0.0,
                    dispatch_lag_s=0.0,
                    dispatch_time_s=dispatch,
                )
            )

    try:
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        measurement_end = time.perf_counter()
    finally:
        if owns_client:
            await client.aclose()

    window = 0.0 if measurement_start is None else measurement_end - measurement_start

    return LoadGenResult(
        records=tuple(sorted(records, key=lambda r: r.index)),
        measurement_window_s=window,
        warmup_discarded=warmup_requests,
    )
