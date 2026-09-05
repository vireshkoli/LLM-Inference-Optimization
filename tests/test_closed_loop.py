"""The closed-loop exhibit, and the property that makes it an exhibit.

These tests do not check that the closed-loop generator is *good*. They check
that it is genuinely closed-loop — that it exhibits the specific defect the rest
of the repository argues against — because an exhibit that accidentally behaved
like an open-loop client would demonstrate nothing.
"""

from __future__ import annotations

import pytest

from llmbench.loadgen.client import LoadGenConfig, run_open_loop
from llmbench.loadgen.closed_loop import run_closed_loop
from llmbench.workload.arrivals import poisson_schedule

from .mock_server import MockLLMServer
from .test_loadgen import client_for, specs

CONFIG = LoadGenConfig(base_url="http://mock", model="test-model")


class TestClosedLoopIsActuallyClosed:
    @pytest.mark.asyncio
    async def test_in_flight_never_exceeds_concurrency(self) -> None:
        """The defining property. A worker issues its next request only after
        its previous one returns, so in-flight requests are capped by the pool
        size no matter how slow the server gets."""
        server = MockLLMServer(ttft_s=0.02, itl_s=0.002)
        async with client_for(server) as client:
            await run_closed_loop(specs(24), CONFIG, concurrency=4, client=client)
        assert server.max_in_flight <= 4

    @pytest.mark.asyncio
    async def test_open_loop_does_exceed_the_same_count(self) -> None:
        """The contrast that makes the exhibit meaningful: given the same server
        and the same requests, the open-loop generator drives concurrency past
        any fixed pool size, because it dispatches on a schedule rather than on
        completions."""
        server = MockLLMServer(ttft_s=0.05, itl_s=0.002)
        schedule = poisson_schedule(rate_rps=200.0, num_requests=24, seed=7)
        async with client_for(server) as client:
            await run_open_loop(schedule, specs(24), CONFIG, client=client)
        assert server.max_in_flight > 4

    @pytest.mark.asyncio
    async def test_every_request_is_issued_exactly_once(self) -> None:
        server = MockLLMServer()
        async with client_for(server) as client:
            result = await run_closed_loop(specs(20), CONFIG, concurrency=5, client=client)
        assert server.requests_received == 20
        assert sorted(r.index for r in result.records) == list(range(20))

    @pytest.mark.asyncio
    async def test_records_are_returned_in_index_order(self) -> None:
        """Workers complete out of order; the record list is sorted so raw
        output is comparable with an open-loop run request-for-request."""
        server = MockLLMServer()
        async with client_for(server) as client:
            result = await run_closed_loop(specs(12), CONFIG, concurrency=4, client=client)
        assert [r.index for r in result.records] == list(range(12))


class TestNoScheduleToBeLateAgainst:
    @pytest.mark.asyncio
    async def test_dispatch_lag_is_structurally_zero(self) -> None:
        """Not a measurement — a property. Nothing scheduled these requests, so
        the dispatch-lag validity guard cannot say anything about this run, and
        the recorded zeros must not be read as "the client kept up"."""
        server = MockLLMServer(ttft_s=0.03)
        async with client_for(server) as client:
            result = await run_closed_loop(specs(12), CONFIG, concurrency=2, client=client)
        assert all(r.dispatch_lag_s == 0.0 for r in result.records)
        assert all(r.scheduled_offset_s == 0.0 for r in result.records)


class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_requests_are_issued_then_excluded(self) -> None:
        server = MockLLMServer()
        async with client_for(server) as client:
            result = await run_closed_loop(
                specs(16), CONFIG, concurrency=4, warmup_requests=6, client=client
            )
        assert server.requests_received == 16
        assert len(result.measured) == 10
        assert result.warmup_discarded == 6


class TestRejectsIncoherentConfiguration:
    @pytest.mark.asyncio
    async def test_zero_concurrency_is_refused(self) -> None:
        """A pool of zero workers issues nothing and would silently report an
        empty run rather than failing."""
        with pytest.raises(ValueError, match="concurrency must be at least 1"):
            await run_closed_loop(specs(4), CONFIG, concurrency=0)

    @pytest.mark.asyncio
    async def test_more_warmup_than_requests_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds total requests"):
            await run_closed_loop(specs(4), CONFIG, concurrency=2, warmup_requests=9)
