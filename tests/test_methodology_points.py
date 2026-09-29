"""How a declared methodology run becomes the points to measure.

The precedence rule here is not cosmetic. Getting it backwards cost a full
sweep: six requested concurrencies all ran at the matrix's declared 64 and
overwrote a single set of result files, and nothing in the output said so
because every run was internally consistent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llmbench.config import MethodologyRun
from llmbench.runner import RatePoint, SweepRunner
from llmbench.schema import ArrivalProcess, LengthSource
from llmbench.workload.arrivals import ArrivalSchedule
from llmbench.workload.corpus import ShareGptCorpus
from llmbench.workload.lengths import LengthPair


def points(runner: SweepRunner, entry: MethodologyRun, override: int | None) -> list[RatePoint]:
    stub = RatePoint(rate_rps=4.0, schedule=ArrivalSchedule((0.0,), 4.0), specs=())
    with patch.object(SweepRunner, "rate_point", return_value=stub):
        return runner._methodology_points(
            entry,
            trace_path=__import__("pathlib").Path("unused.csv"),
            matched_rate_rps=4.0,
            closed_loop_concurrency=override,
        )


@pytest.fixture
def closed_loop() -> MethodologyRun:
    return MethodologyRun(
        id="vllm-bf16-closed-loop",
        config="vllm-bf16",
        arrival_process=ArrivalProcess.CLOSED_LOOP,
        concurrency=64,
    )


class TestConcurrencyPrecedence:
    def test_explicit_override_beats_the_matrix(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        """Regression. `entry.concurrency or override` ignored the caller
        whenever the matrix declared a value, so a sweep over concurrencies
        silently ran every point at the matrix's own number."""
        assert points(sweep_runner, closed_loop, 40)[0].concurrency == 40

    def test_matrix_value_used_when_no_override(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        assert points(sweep_runner, closed_loop, None)[0].concurrency == 64

    def test_missing_everywhere_fails_loudly(self, sweep_runner: SweepRunner) -> None:
        """Concurrency *is* the offered load for this generator, so there is no
        default that would be meaningful."""
        entry = MethodologyRun(
            id="no-concurrency",
            config="vllm-bf16",
            arrival_process=ArrivalProcess.CLOSED_LOOP,
        )
        with pytest.raises(ValueError, match="neither the matrix nor the caller"):
            points(sweep_runner, entry, None)


class TestArrivalProcessRouting:
    def test_closed_loop_point_carries_no_rate(
        self, sweep_runner: SweepRunner, closed_loop: MethodologyRun
    ) -> None:
        """A closed loop has no offered rate; carrying one would put it on the
        frontier, which keys on exactly that field."""
        point = points(sweep_runner, closed_loop, 40)[0]
        assert point.rate_rps is None
        assert point.arrival_process is ArrivalProcess.CLOSED_LOOP

    def test_drift_canary_sweeps_the_whole_ladder(self, sweep_runner: SweepRunner) -> None:
        """Its value lies entirely in being identical to the original run."""
        entry = MethodologyRun(id="drift-canary", repeat_of="vllm-bf16")
        got = points(sweep_runner, entry, None)
        assert len(got) == len(sweep_runner.config.workload.request_rates_rps)
        assert all(p.arrival_process is ArrivalProcess.POISSON for p in got)


class TestClosedLoopWarmup:
    @pytest.mark.parametrize(("concurrency", "warmup"), [(8, 50), (48, 50), (64, 64), (160, 160)])
    def test_warmup_covers_the_start_up_burst(
        self, sweep_runner: SweepRunner, concurrency: int, warmup: int
    ) -> None:
        """All N workers fire at t=0. With N above the matrix's 50-request
        warm-up, the rest of that burst used to be measured -- 110 requests at
        N=160 -- and the percentile that blew up tracked its share."""
        stub = RatePoint(rate_rps=4.0, schedule=ArrivalSchedule((0.0,), 4.0), specs=())
        with patch.object(SweepRunner, "rate_point", return_value=stub):
            point = sweep_runner.closed_loop_point(concurrency=concurrency, matched_rate_rps=4.0)
        assert point.warmup_requests == warmup

    def test_open_loop_points_keep_the_matrix_warmup(self) -> None:
        """No override: an open-loop point has no burst to warm up through."""
        stub = RatePoint(rate_rps=4.0, schedule=ArrivalSchedule((0.0,), 4.0), specs=())
        assert stub.warmup_requests is None


class _WordTokenizer:
    """Round-trips token ids through text, so prompt lengths verify exactly."""

    def encode(self, text: str) -> list[int]:
        return [int(w) for w in text.split()]

    def decode(self, token_ids: object) -> str:
        return " ".join(str(t) for t in token_ids)  # type: ignore[attr-defined]


def _trace_csv(path: Path) -> Path:
    """600 s at 3.0 rps, 600 s at 4.4 rps, 600 s at exactly 4.0 rps."""
    rows = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    t = 0.0
    for end, gap in ((600.0, 1 / 3.0), (1200.0, 1 / 4.4), (1800.0, 0.25)):
        while t < end:
            rows.append(f"{t:.4f},100,50")
            t += gap
    path.write_text("\n".join(rows) + "\n")
    return path


class TestArrivalStudy:
    @pytest.fixture
    def realizations(
        self, sweep_runner: SweepRunner, tmp_path: Path
    ) -> tuple[list[RatePoint], list[RatePoint]]:
        corpus = ShareGptCorpus(
            pairs=tuple(LengthPair(input_tokens=5 + i % 20, output_tokens=10) for i in range(500)),
            corpus_token_ids=tuple(range(5000)),
        )
        with (
            patch.object(SweepRunner, "_load_corpus", return_value=corpus),
            patch("llmbench.runner.load_tokenizer", return_value=_WordTokenizer()),
        ):
            return sweep_runner.arrival_realizations(
                4.0, trace_path=_trace_csv(tmp_path / "trace.csv"), count=5, label_prefix="study"
            )

    def test_trace_windows_are_distinct(
        self, realizations: tuple[list[RatePoint], list[RatePoint]]
    ) -> None:
        """Regression for the arrival study that replayed one window five times."""
        _, trace = realizations
        starts = {p.note.split(" s of the trace")[0] for p in trace if p.note}
        assert len(starts) == 5

    def test_every_window_offers_exactly_the_poisson_rate(
        self, realizations: tuple[list[RatePoint], list[RatePoint]]
    ) -> None:
        _, trace = realizations
        for point in trace:
            assert point.note is not None
            assert "to 4.00 rps" in point.note
            assert point.length_source is LengthSource.SHAREGPT

    def test_prompts_match_each_side(
        self, realizations: tuple[list[RatePoint], list[RatePoint]]
    ) -> None:
        """Windows at 4.4 rps hold more arrivals than the Poisson side sends, so
        the prompt list is sized for the longest -- without changing a single
        prompt the Poisson side receives."""
        poisson, trace = realizations
        assert all(len(p.specs) == len(p.schedule) for p in (*poisson, *trace))
        assert max(len(p.specs) for p in trace) > len(poisson[0].specs)
        assert all(p.specs == poisson[0].specs for p in poisson)
        longest = max(trace, key=lambda p: len(p.specs))
        assert longest.specs[: len(poisson[0].specs)] == poisson[0].specs

    def test_poisson_draws_are_independent(
        self, realizations: tuple[list[RatePoint], list[RatePoint]]
    ) -> None:
        poisson, _ = realizations
        assert len({p.schedule.offsets_s for p in poisson}) == 5

    def test_labels_use_the_prefix(
        self, realizations: tuple[list[RatePoint], list[RatePoint]]
    ) -> None:
        poisson, trace = realizations
        assert [p.label for p in poisson] == [f"study-poisson-r{k}" for k in range(5)]
        assert [p.label for p in trace] == [f"study-trace-r{k}" for k in range(5)]
