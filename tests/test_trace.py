"""Azure trace loading and window selection.

The replay exists to show what the Poisson assumption costs, so the tests focus
on the two ways that argument can be quietly destroyed: losing the burstiness
by reordering or resampling, and comparing against a Poisson run at a different
offered rate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.workload.arrivals import trace_schedule
from llmbench.workload.trace import TraceRequest, load_azure_trace, select_window


def write_trace(path: Path, rows: list[tuple[float, int, int]]) -> Path:
    lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    lines += [f"{t},{i},{o}" for t, i, o in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def steady(n: int, gap: float = 0.5) -> list[tuple[float, int, int]]:
    return [(i * gap, 100 + i, 50) for i in range(n)]


class TestLoading:
    def test_arrivals_are_rebased_to_zero(self, tmp_path: Path) -> None:
        """Published traces carry absolute timestamps; a schedule needs offsets."""
        p = write_trace(tmp_path / "t.csv", [(1000.0, 10, 5), (1002.5, 20, 6)])
        rows = load_azure_trace(p)
        assert rows[0].arrival_s == 0.0
        assert rows[1].arrival_s == pytest.approx(2.5)

    def test_token_columns_are_read(self, tmp_path: Path) -> None:
        p = write_trace(tmp_path / "t.csv", [(0.0, 512, 128)])
        assert load_azure_trace(p)[0] == TraceRequest(0.0, 512, 128)

    def test_iso_timestamps_are_accepted(self, tmp_path: Path) -> None:
        p = tmp_path / "iso.csv"
        p.write_text(
            "TIMESTAMP,ContextTokens,GeneratedTokens\n"
            "2024-01-01T00:00:00Z,10,5\n"
            "2024-01-01T00:00:04Z,20,6\n"
        )
        rows = load_azure_trace(p)
        assert rows[1].arrival_s == pytest.approx(4.0)

    def test_rows_are_sorted_by_arrival(self, tmp_path: Path) -> None:
        """An out-of-order file would otherwise produce a schedule that is not
        the trace, and trace_schedule would reject it much later."""
        p = write_trace(tmp_path / "t.csv", [(5.0, 10, 5), (1.0, 10, 5), (3.0, 10, 5)])
        assert [r.arrival_s for r in load_azure_trace(p)] == [0.0, 2.0, 4.0]

    def test_malformed_row_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """These are published dumps; one bad line must not cost a sweep."""
        p = tmp_path / "bad.csv"
        p.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n0,10,5\nnonsense,x,y\n2,20,6\n")
        assert len(load_azure_trace(p)) == 2

    def test_max_rows_bounds_a_large_file(self, tmp_path: Path) -> None:
        p = write_trace(tmp_path / "t.csv", steady(500))
        assert len(load_azure_trace(p, max_rows=25)) == 25

    def test_missing_column_names_what_it_wanted(self, tmp_path: Path) -> None:
        p = tmp_path / "wrong.csv"
        p.write_text("when,how_big\n0,10\n")
        with pytest.raises(ValueError, match="missing one of"):
            load_azure_trace(p)

    def test_empty_trace_is_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        with pytest.raises(ValueError, match="no parseable trace rows"):
            load_azure_trace(p)


class TestWindowSelection:
    def test_window_is_contiguous(self, tmp_path: Path) -> None:
        """Sampling across the trace would destroy the temporal correlation that
        makes it bursty, leaving a reordered Poisson-ish process wearing a
        trace's name."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(200, gap=0.5)))
        window = select_window(rows, duration_s=10.0)
        gaps = [
            b.arrival_s - a.arrival_s
            for a, b in zip(window.requests, window.requests[1:], strict=False)
        ]
        assert all(g == pytest.approx(0.5) for g in gaps)

    def test_window_starts_at_zero(self, tmp_path: Path) -> None:
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(200)))
        assert select_window(rows, duration_s=10.0).requests[0].arrival_s == 0.0

    def test_matches_a_target_rate(self, tmp_path: Path) -> None:
        """The Poisson comparison is only interpretable at a matched mean rate."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(400, gap=0.25)))
        window = select_window(rows, duration_s=20.0, target_rate_rps=4.0)
        assert window.mean_rate_rps == pytest.approx(4.0, rel=0.05)

    def test_refuses_a_window_far_from_the_target_rate(self, tmp_path: Path) -> None:
        """Replaying mismatched load would confound burstiness with offered
        load — the one confusion this run exists to avoid."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(400, gap=0.25)))
        with pytest.raises(ValueError, match="confound burstiness with offered load"):
            select_window(rows, duration_s=20.0, target_rate_rps=1.0)

    def test_window_is_fully_contained_in_the_trace(self, tmp_path: Path) -> None:
        """Regression. Windows starting near the end of the trace are truncated:
        a 20 s window over the last 3 s of data holds few requests and so reports
        a rate far below the trace's real one. A rate-matching search would pick
        exactly those as the best match for a low target and silently replay a
        fraction of the intended load."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(400, gap=0.25)))
        window = select_window(rows, duration_s=20.0, target_rate_rps=4.0)
        assert len(window) == pytest.approx(80, abs=2)
        assert window.mean_rate_rps == pytest.approx(4.0, rel=0.05)

    def test_trace_shorter_than_the_window_is_rejected(self, tmp_path: Path) -> None:
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(10, gap=0.5)))
        with pytest.raises(ValueError, match="shorter than the requested"):
            select_window(rows, duration_s=600.0)


class TestBurstiness:
    def test_regular_arrivals_score_near_zero(self, tmp_path: Path) -> None:
        """Perfectly regular arrivals have zero inter-arrival variance, so the
        index of dispersion is 0 — far below Poisson's 1.0."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(200, gap=0.5)))
        assert select_window(rows, duration_s=20.0).burstiness == pytest.approx(0.0, abs=1e-9)

    def test_clumped_arrivals_score_above_poisson(self, tmp_path: Path) -> None:
        """A trace whose requests arrive in clumps must score above 1.0, or the
        metric cannot support the claim the replay is making."""
        rows: list[tuple[float, int, int]] = []
        t = 0.0
        for _ in range(40):
            for _ in range(10):  # a burst
                rows.append((t, 100, 50))
                t += 0.01
            t += 2.0  # then a long gap
        window = select_window(
            load_azure_trace(write_trace(tmp_path / "b.csv", rows)), duration_s=40.0
        )
        assert window.burstiness > 1.0

    def test_window_feeds_trace_schedule(self, tmp_path: Path) -> None:
        """The handoff to the arrival layer, which attaches no nominal rate:
        describing a bursty trace by its mean would erase the point."""
        rows = load_azure_trace(write_trace(tmp_path / "t.csv", steady(200)))
        schedule = trace_schedule(select_window(rows, duration_s=20.0).timestamps_s)
        assert schedule.nominal_rate_rps is None
        assert schedule.offsets_s[0] == 0.0
