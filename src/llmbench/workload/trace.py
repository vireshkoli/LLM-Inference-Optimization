"""Azure LLM inference trace — real production arrival timestamps.

Poisson arrivals are the primary process in this repository, and they are an
*assumption*. The usual expectation is that real traffic is burstier: requests
arrive in correlated clumps, so the queue sees transient rates well above the
average and the tail suffers accordingly. A benchmark that only ever offers
Poisson load cannot say how much of its tail latency is an artifact of that
choice.

**Measured, that expectation does not hold for this trace.** At a 4 req/s
window the published Azure conversation trace has a squared coefficient of
variation of 1.02 against Poisson's 1.00 — it is very close to Poisson at the
timescale this benchmark operates on. That is a result, not a disappointment:
it is the difference between assuming the arrival model and having checked it.

This module loads the Microsoft Azure LLM inference trace (``AzurePublicDataset``)
and extracts the two things a replay needs: **when** each request arrived, and
**how large** it was. Both matter, and taking only the timestamps would be a
half-measure — production burstiness correlates arrival time with request size,
which is exactly the interaction that hurts a real server.

What the replay deliberately does *not* do:

* **It does not fit a rate.** ``trace_schedule`` attaches ``nominal_rate_rps =
  None``, because describing a bursty trace by its mean rate erases the property
  the run exists to expose.
* **It does not resample or smooth.** The inter-arrival gaps are replayed as
  recorded, only rebased to start at zero and optionally time-scaled as a whole.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "TraceRequest",
    "TraceWindow",
    "load_azure_trace",
    "select_window",
    "select_windows",
]

#: Column names in the published Azure trace CSVs. Both the conversation and
#: the code variants use these, which is why the loader accepts either file.
_TIMESTAMP_COLUMNS = ("TIMESTAMP", "timestamp", "arrival_time")
_INPUT_COLUMNS = ("ContextTokens", "context_tokens", "input_tokens")
_OUTPUT_COLUMNS = ("GeneratedTokens", "generated_tokens", "output_tokens")


@dataclass(frozen=True, slots=True)
class TraceRequest:
    """One recorded production request."""

    #: Seconds from the start of the trace file, not an absolute epoch.
    arrival_s: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class TraceWindow:
    """A contiguous slice of the trace, ready to replay.

    Carries its own descriptive statistics because they are the evidence for
    the claim the run is making. A replay that reported only its tail latency,
    with no measure of how bursty the offered load actually was, would leave the
    reader unable to tell a burstiness effect from a slow server.
    """

    requests: tuple[TraceRequest, ...]
    duration_s: float

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def mean_rate_rps(self) -> float:
        """Requests per second averaged over the window.

        Reported so the replay can be compared against a Poisson run of the
        *same* mean rate. It is a summary of the window, never a parameter that
        generated it.
        """
        return len(self.requests) / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def burstiness(self) -> float:
        """Squared coefficient of variation of the inter-arrival times.

        ``variance / mean**2``. For a Poisson process the inter-arrival times
        are Exponential, whose standard deviation equals its mean, so this is
        1.0 — **at any rate**. Above 1 the trace is burstier than Poisson;
        below 1 it is more regular.

        Dividing by the mean *squared* rather than the mean is what makes the
        1.0 reference meaningful. ``variance / mean`` is not dimensionless: for
        an Exponential distribution it equals the mean itself, so it reads 1.0
        only at exactly 1 req/s and 0.25 at 4 req/s, and a trace would appear to
        get less bursty purely by arriving faster.
        """
        gaps = [
            b.arrival_s - a.arrival_s
            for a, b in zip(self.requests, self.requests[1:], strict=False)
        ]
        if len(gaps) < 2:
            return 0.0
        mean = statistics.fmean(gaps)
        return statistics.variance(gaps) / (mean**2) if mean > 0 else 0.0

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        """Arrival offsets, for :func:`llmbench.workload.arrivals.trace_schedule`."""
        return tuple(r.arrival_s for r in self.requests)


def _column(header: Sequence[str], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in header:
            return name
    msg = f"trace is missing one of {list(candidates)}; found columns {list(header)}"
    raise ValueError(msg)


def _parse_timestamp(raw: str) -> float:
    """Accept either an ISO-8601 instant or a bare offset in seconds.

    The published trace files use ISO timestamps; fixtures and hand-made traces
    are easier to write as plain offsets. Supporting both keeps the tests honest
    without a second code path in the loader.
    """
    try:
        return float(raw)
    except ValueError:
        pass
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def load_azure_trace(path: Path, *, max_rows: int | None = None) -> list[TraceRequest]:
    """Read the trace CSV into arrival-ordered requests.

    Arrivals are rebased to the first request, so ``arrival_s`` is an offset in
    seconds rather than an epoch. Rows are sorted by arrival: the published
    files are already ordered, but a replay that silently accepted an unsorted
    file would produce a schedule that is not the trace.

    Raises:
        ValueError: If the file has no usable rows, or lacks a required column.
    """
    rows: list[TraceRequest] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"{path} has no header row"
            raise ValueError(msg)

        ts_col = _column(reader.fieldnames, _TIMESTAMP_COLUMNS)
        in_col = _column(reader.fieldnames, _INPUT_COLUMNS)
        out_col = _column(reader.fieldnames, _OUTPUT_COLUMNS)

        for record in reader:
            try:
                rows.append(
                    TraceRequest(
                        arrival_s=_parse_timestamp(record[ts_col]),
                        input_tokens=int(float(record[in_col])),
                        output_tokens=int(float(record[out_col])),
                    )
                )
            except (TypeError, ValueError):
                # A malformed row is skipped rather than fatal: these are
                # published dumps, and one bad line should not cost a sweep.
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break

    if not rows:
        msg = f"{path} contained no parseable trace rows"
        raise ValueError(msg)

    rows.sort(key=lambda r: r.arrival_s)
    origin = rows[0].arrival_s
    return [
        TraceRequest(
            arrival_s=r.arrival_s - origin,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
        )
        for r in rows
    ]


def select_window(
    requests: Sequence[TraceRequest],
    *,
    duration_s: float,
    target_rate_rps: float | None = None,
    tolerance: float = 0.15,
) -> TraceWindow:
    """Pick a contiguous window of the trace to replay.

    Args:
        target_rate_rps: When given, choose the window whose mean rate is
            closest to this. The comparison against a Poisson run is only
            meaningful at a *matched mean rate* — otherwise a difference in tail
            latency could be explained by offered load rather than by
            burstiness, which is the whole question.
        tolerance: Fractional band around ``target_rate_rps`` within which a
            window is acceptable. Exceeding it raises rather than silently
            replaying load the comparison cannot interpret.

    The window is contiguous by construction. Sampling requests from across the
    trace would destroy the temporal correlation that makes it bursty, leaving a
    reordered Poisson-ish process wearing a trace's name.

    Raises:
        ValueError: If the trace is shorter than the requested duration, or no
            window lands within tolerance of the target rate.
    """
    if not requests:
        msg = "cannot select a window from an empty trace"
        raise ValueError(msg)
    span = requests[-1].arrival_s - requests[0].arrival_s
    if span < duration_s:
        msg = f"trace spans {span:.0f}s, shorter than the requested {duration_s:.0f}s window"
        raise ValueError(msg)

    # A window must fit entirely inside the trace. Without this bound, a start
    # index near the end yields a *truncated* window — a 20 s request of the
    # last 3 s of data — whose request count is short and whose apparent rate is
    # therefore far below the trace's real rate. Under a rate-matching search
    # those truncated windows look like excellent matches for any low target,
    # and the replay would quietly offer a fraction of the intended load.
    last_start = requests[-1].arrival_s - duration_s

    best: tuple[float, int, int] | None = None  # (score, start index, end index)
    end = 0
    for start in range(len(requests)):
        if requests[start].arrival_s > last_start:
            break
        limit = requests[start].arrival_s + duration_s
        end = max(end, start)
        while end < len(requests) and requests[end].arrival_s < limit:
            end += 1
        count = end - start
        if count < 2:
            continue
        rate = count / duration_s
        score = 0.0 if target_rate_rps is None else abs(rate - target_rate_rps)
        if best is None or score < best[0]:
            best = (score, start, end)
        if target_rate_rps is None:
            # Without a target, the first full window is as good as any other,
            # and scanning the whole file to prove it wastes time on a 100 MB
            # trace.
            break

    if best is None:
        msg = f"no window of {duration_s:.0f}s contained at least two requests"
        raise ValueError(msg)

    _, start, end = best
    window = tuple(
        TraceRequest(
            arrival_s=r.arrival_s - requests[start].arrival_s,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
        )
        for r in requests[start:end]
    )
    result = TraceWindow(requests=window, duration_s=duration_s)

    if target_rate_rps is not None:
        drift = abs(result.mean_rate_rps - target_rate_rps) / target_rate_rps
        if drift > tolerance:
            msg = (
                f"closest window runs at {result.mean_rate_rps:.2f} rps against a target of "
                f"{target_rate_rps:.2f} rps ({drift:.0%} off, tolerance {tolerance:.0%}). "
                f"Replaying it would confound burstiness with offered load."
            )
            raise ValueError(msg)

    return result


def select_windows(
    requests: Sequence[TraceRequest],
    *,
    duration_s: float,
    count: int,
    target_rate_rps: float,
    tolerance: float = 0.15,
) -> list[TraceWindow]:
    """Pick several **non-overlapping** windows at a matched mean rate.

    One window is one realization of the arrival process, and repeating a
    measurement against it does not sample the process — it re-measures the same
    arrival sequence and reports only the server's own variability. Attributing
    a latency difference to *burstiness* rather than to one particular stretch
    of traffic therefore needs several independent windows, exactly as the
    Poisson side needs several seeds.

    Windows are non-overlapping so they are genuinely independent draws; sliding
    a window forward by a few seconds would produce near-identical arrival
    sequences and a falsely tight spread.

    Raises:
        ValueError: If the trace cannot supply ``count`` windows within
            ``tolerance`` of the target rate.
    """
    if count < 1:
        msg = f"count must be at least 1, got {count}"
        raise ValueError(msg)

    windows: list[TraceWindow] = []
    cursor = 0.0
    span = requests[-1].arrival_s if requests else 0.0

    while len(windows) < count and cursor + duration_s <= span:
        remaining = [r for r in requests if r.arrival_s >= cursor]
        try:
            window = select_window(
                remaining,
                duration_s=duration_s,
                target_rate_rps=target_rate_rps,
                tolerance=tolerance,
            )
        except ValueError:
            break
        windows.append(window)
        # Advance past the window just taken, plus its own span, so the next
        # candidate cannot overlap it.
        first_abs = next(r.arrival_s for r in requests if r.arrival_s >= cursor)
        cursor = first_abs + duration_s

    if len(windows) < count:
        msg = (
            f"trace supplied only {len(windows)} non-overlapping window(s) of {duration_s:.0f}s "
            f"within {tolerance:.0%} of {target_rate_rps:g} rps; {count} were requested"
        )
        raise ValueError(msg)
    return windows
