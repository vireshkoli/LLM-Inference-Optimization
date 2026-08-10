"""End-to-end load-generator behaviour against an in-process mock server.

No socket, no GPU, no network — the dispatcher, SSE parsing, timing and
aggregation all run for real against an ASGI app wired through httpx.

The load-bearing test is :meth:`TestOpenLoopBehaviour.test_dispatch_does_not_wait_for_completion`.
It is the difference between this harness and the closed-loop generators that
quietly understate tail latency everywhere else.
"""

from __future__ import annotations

import httpx
import pytest

from llmbench.loadgen.client import LoadGenConfig, run_open_loop
from llmbench.loadgen.guard import assess_validity
from llmbench.metrics.percentiles import summarize
from llmbench.workload.arrivals import ArrivalSchedule, poisson_schedule
from llmbench.workload.prompts import RequestSpec

from .mock_server import MockLLMServer

CONFIG = LoadGenConfig(base_url="http://mock", model="test-model")


def specs(n: int, *, max_tokens: int = 5) -> tuple[RequestSpec, ...]:
    return tuple(
        RequestSpec(index=i, prompt=f"prompt {i}", input_tokens=10, max_tokens=max_tokens)
        for i in range(n)
    )


def client_for(server: MockLLMServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="http://mock",
        timeout=httpx.Timeout(30.0),
    )


class TestOpenLoopBehaviour:
    async def test_dispatch_does_not_wait_for_completion(self) -> None:
        """The defining property of an open-loop generator.

        Ten requests are scheduled 20 ms apart against a server that takes
        ~300 ms each. Open-loop finishes in roughly 0.18 s of dispatching plus
        one response (~0.5 s). Closed-loop would serialise them into ~3 s.

        If this test ever starts failing by running long, the harness has
        regressed into closed-loop and every tail-latency number it produces is
        optimistic.
        """
        server = MockLLMServer(ttft_s=0.30, itl_s=0.001)
        schedule = ArrivalSchedule(
            offsets_s=tuple(0.02 * i for i in range(10)), nominal_rate_rps=50.0
        )

        async with client_for(server) as client:
            with_timeout = await run_open_loop(schedule, specs(10), CONFIG, client=client)

        assert len(with_timeout.records) == 10
        assert all(r.succeeded for r in with_timeout.records)
        # Serial execution would need >= 3 s; allow generous slack for CI.
        assert with_timeout.measurement_window_s < 1.5

    async def test_requests_overlap_in_flight(self) -> None:
        """Concurrency above 1 is only possible if dispatch never blocked."""
        server = MockLLMServer(ttft_s=0.20, itl_s=0.001)
        schedule = ArrivalSchedule(
            offsets_s=tuple(0.01 * i for i in range(8)), nominal_rate_rps=100.0
        )

        async with client_for(server) as client:
            await run_open_loop(schedule, specs(8), CONFIG, client=client)

        assert server.max_in_flight > 1

    async def test_all_scheduled_requests_are_sent(self) -> None:
        server = MockLLMServer(ttft_s=0.001, itl_s=0.0005)
        schedule = poisson_schedule(rate_rps=200.0, num_requests=25, seed=1)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(25), CONFIG, client=client)

        assert server.requests_received == 25
        assert len(result.records) == 25


class TestTimingMeasurement:
    async def test_ttft_excludes_the_role_only_chunk(self) -> None:
        """The full path, not just the collector, must skip the role delta.

        The mock emits the role chunk immediately and the first token only after
        ttft_s, so a harness timing to the first *chunk* would report ~0.
        """
        server = MockLLMServer(ttft_s=0.25, itl_s=0.001, emit_role_chunk=True)
        schedule = ArrivalSchedule(offsets_s=(0.0,), nominal_rate_rps=1.0)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(1), CONFIG, client=client)

        ttft = result.records[0].ttft_s
        assert ttft is not None
        assert ttft == pytest.approx(0.25, abs=0.15)
        assert ttft > 0.1, "TTFT collapsed toward zero — role chunk was counted as a token"

    async def test_dispatch_lag_is_recorded(self) -> None:
        server = MockLLMServer(ttft_s=0.001, itl_s=0.0005)
        schedule = poisson_schedule(rate_rps=20.0, num_requests=15, seed=3)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(15), CONFIG, client=client)

        lags = [r.dispatch_lag_s for r in result.records]
        assert all(lag >= 0.0 for lag in lags), "lag cannot be negative; clock handling is wrong"
        assert max(lags) < 0.5

    async def test_output_tokens_come_from_server_usage(self) -> None:
        server = MockLLMServer(ttft_s=0.001, itl_s=0.0005)
        schedule = ArrivalSchedule(offsets_s=(0.0,), nominal_rate_rps=1.0)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(1, max_tokens=7), CONFIG, client=client)

        assert result.records[0].output_tokens == 7

    async def test_tpot_excludes_prefill(self) -> None:
        """TPOT must describe decode only, or prefill- and decode-bound
        configurations become indistinguishable."""
        server = MockLLMServer(ttft_s=0.30, itl_s=0.01)
        schedule = ArrivalSchedule(offsets_s=(0.0,), nominal_rate_rps=1.0)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(1, max_tokens=10), CONFIG, client=client)

        tpot = result.records[0].tpot_s
        assert tpot is not None
        # ~10 ms per token, nowhere near the 300 ms prefill.
        assert tpot < 0.1

    async def test_tpot_is_none_for_a_single_token(self) -> None:
        server = MockLLMServer(ttft_s=0.001, itl_s=0.001)
        schedule = ArrivalSchedule(offsets_s=(0.0,), nominal_rate_rps=1.0)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(1, max_tokens=1), CONFIG, client=client)

        assert result.records[0].tpot_s is None


class TestWarmup:
    async def test_warmup_requests_are_sent_but_excluded(self) -> None:
        """Warmup must actually be issued — the point is reaching steady state,
        which not sending would defeat."""
        server = MockLLMServer(ttft_s=0.001, itl_s=0.0005)
        schedule = poisson_schedule(rate_rps=100.0, num_requests=20, seed=4)

        async with client_for(server) as client:
            result = await run_open_loop(
                schedule, specs(20), CONFIG, client=client, warmup_requests=5
            )

        assert server.requests_received == 20
        assert len(result.records) == 20
        assert len(result.measured) == 15
        assert result.warmup_discarded == 5

    async def test_measurement_window_starts_after_warmup(self) -> None:
        server = MockLLMServer(ttft_s=0.001, itl_s=0.0005)
        schedule = ArrivalSchedule(
            offsets_s=tuple(0.02 * i for i in range(10)), nominal_rate_rps=50.0
        )

        async with client_for(server) as client:
            result = await run_open_loop(
                schedule, specs(10), CONFIG, client=client, warmup_requests=5
            )

        # Window covers the last 5 arrivals (~0.1 s), not all 10 (~0.2 s).
        assert result.measurement_window_s < 0.18


class TestFailureHandling:
    async def test_server_error_becomes_an_error_record(self) -> None:
        """One refused request must not abort a sweep that cost GPU-hours."""
        server = MockLLMServer(status=503)
        schedule = poisson_schedule(rate_rps=50.0, num_requests=5, seed=5)

        async with client_for(server) as client:
            result = await run_open_loop(schedule, specs(5), CONFIG, client=client)

        assert len(result.records) == 5
        assert all(not r.succeeded for r in result.records)
        assert all(r.status_code == 503 for r in result.records)
        assert all(r.error is not None and "503" in r.error for r in result.records)

    async def test_schedule_length_mismatch_is_rejected(self) -> None:
        schedule = poisson_schedule(rate_rps=10.0, num_requests=5, seed=1)
        with pytest.raises(ValueError, match="5 arrivals but 3 requests"):
            await run_open_loop(schedule, specs(3), CONFIG)

    async def test_excessive_warmup_is_rejected(self) -> None:
        schedule = poisson_schedule(rate_rps=10.0, num_requests=3, seed=1)
        with pytest.raises(ValueError, match="exceeds total requests"):
            await run_open_loop(schedule, specs(3), CONFIG, warmup_requests=10)


class TestFullPipeline:
    async def test_run_aggregates_into_a_valid_reportable_result(self) -> None:
        """Phase 2 definition of done.

        A complete run against a mock server flows through dispatch, SSE
        parsing, percentile aggregation and validity assessment, producing
        schema-valid statistics judged reportable.
        """
        server = MockLLMServer(ttft_s=0.01, itl_s=0.002)
        schedule = poisson_schedule(rate_rps=50.0, num_requests=60, seed=20260810)

        async with client_for(server) as client:
            result = await run_open_loop(
                schedule, specs(60, max_tokens=8), CONFIG, client=client, warmup_requests=10
            )

        measured = result.measured
        assert len(measured) == 50
        assert all(r.succeeded for r in measured)

        ttft = summarize([r.ttft_s for r in measured if r.ttft_s is not None])
        e2e = summarize([r.e2e_s for r in measured if r.e2e_s is not None])
        lag = summarize([r.dispatch_lag_s for r in measured])

        assert ttft.count == 50
        assert ttft.min <= ttft.p50 <= ttft.p95 <= ttft.p99 <= ttft.max
        assert e2e.p50 > ttft.p50, "end-to-end must exceed TTFT once decode is included"

        gaps = schedule.inter_arrivals_s()
        assessment = assess_validity(
            dispatch_lags_s=[r.dispatch_lag_s for r in measured],
            requests_scheduled=len(measured),
            requests_completed=sum(1 for r in measured if r.succeeded),
            requests_failed=sum(1 for r in measured if not r.succeeded),
            mean_interarrival_s=sum(gaps) / len(gaps),
        )
        assert assessment.is_reportable, f"run judged invalid: {assessment.notes}"
        assert lag.p99 < 0.05
