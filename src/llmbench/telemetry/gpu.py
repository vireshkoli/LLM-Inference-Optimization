"""GPU telemetry sampling via ``nvidia-smi``.

Sampled continuously through every measurement window, not just at the edges, so
that a run's thermal and clock behaviour travels with its latency numbers rather
than being asserted after the fact.

Two details here were established by measurement on the target machine rather
than assumed, and both would have produced badly wrong results:

**1. Identity is pinned by UUID, never by index.** ``docker run --gpus
'"device=1"'`` gives the container host GPU 1 but *renumbers it to index 0
inside the container*. A sampler that trusted indices would happily record the
neighbouring GPU's temperature — on this machine, a training job at 96 % and
68 °C — against a benchmark running on the idle card.

**2. Not every active event reason is a throttle.** On an A40 under genuine
load, ``sw_power_cap`` is active essentially always: it is the card clamping
clocks to stay inside its 300 W budget, which is normal operation. Counting it
as throttling would mark every run ``THERMAL_THROTTLED`` and leave nothing
reportable. Only the thermal and power-brake reasons invalidate a measurement;
the rest are recorded as informational so nothing is hidden.

Note the driver has renamed these fields ``clocks_throttle_reasons.*`` →
``clocks_event_reasons.*``; the old spelling is still accepted as an alias.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from llmbench.metrics.percentiles import summarize
from llmbench.schema import GPUTelemetry

__all__ = [
    "INVALIDATING_REASONS",
    "GpuSample",
    "GpuSampler",
    "decode_event_reasons",
    "nvidia_smi_query",
    "parse_sample_line",
    "resolve_uuid",
    "summarize_samples",
]

# nvidia-smi clocks_event_reasons bitmask. Values are stable across drivers.
_REASON_BITS: tuple[tuple[int, str], ...] = (
    (0x001, "gpu_idle"),
    (0x002, "applications_clocks_setting"),
    (0x004, "sw_power_cap"),
    (0x008, "hw_slowdown"),
    (0x010, "sync_boost"),
    (0x020, "sw_thermal_slowdown"),
    (0x040, "hw_thermal_slowdown"),
    (0x080, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
)

#: Reasons that mean the GPU was not running at steady-state clocks and the
#: measurement is compromised. Deliberately excludes ``sw_power_cap`` — see the
#: module docstring; including it would invalidate every loaded run.
INVALIDATING_REASONS: frozenset[str] = frozenset(
    {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
)

_QUERY_FIELDS = (
    "uuid",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "memory.used",
    "utilization.gpu",
    "clocks_event_reasons.active",
)


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One instantaneous reading."""

    uuid: str
    temperature_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    power_w: float
    memory_used_mib: int
    utilization_pct: float
    event_reasons: tuple[str, ...]

    @property
    def is_throttled(self) -> bool:
        """True only for reasons that actually invalidate a measurement."""
        return any(r in INVALIDATING_REASONS for r in self.event_reasons)


def decode_event_reasons(mask: int) -> tuple[str, ...]:
    """Expand an ``clocks_event_reasons.active`` bitmask into names."""
    return tuple(name for bit, name in _REASON_BITS if mask & bit)


def parse_sample_line(line: str) -> GpuSample:
    """Parse one CSV row from ``nvidia-smi --format=csv,noheader,nounits``.

    Kept pure and separate from the subprocess call so the parsing — including
    the ``[N/A]`` values the driver emits for unsupported fields — is unit
    tested without a GPU.

    Raises:
        ValueError: If the row does not have the expected field count.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(_QUERY_FIELDS):
        msg = f"expected {len(_QUERY_FIELDS)} fields, got {len(parts)}: {line!r}"
        raise ValueError(msg)

    def _num(raw: str) -> float:
        # Unsupported/unavailable metrics come back as "[N/A]" or "[Not
        # Supported]"; treat as 0.0 rather than crashing a running sweep.
        if raw.startswith("[") or not raw:
            return 0.0
        return float(raw)

    uuid, temp, sm, mem, power, used, util, reasons = parts
    return GpuSample(
        uuid=uuid,
        temperature_c=_num(temp),
        sm_clock_mhz=_num(sm),
        mem_clock_mhz=_num(mem),
        power_w=_num(power),
        memory_used_mib=int(_num(used)),
        utilization_pct=_num(util),
        event_reasons=decode_event_reasons(int(reasons, 16) if reasons.startswith("0x") else 0),
    )


def nvidia_smi_query(args: Sequence[str]) -> str:
    """Run nvidia-smi with the given args and return stdout."""
    result = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def resolve_uuid(gpu_index: int) -> str:
    """Resolve a host GPU index to its stable UUID.

    Called once at run start; every subsequent sample is matched on the UUID so
    that a device renumbering cannot silently redirect telemetry to the wrong
    card.
    """
    return (
        nvidia_smi_query(["--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu_index)])
        .splitlines()[0]
        .strip()
    )


def query_once(gpu_index: int) -> GpuSample:
    """Take a single reading from the given host GPU index."""
    line = nvidia_smi_query(
        [
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
    ).splitlines()[0]
    return parse_sample_line(line)


def summarize_samples(samples: Sequence[GpuSample]) -> GPUTelemetry:
    """Aggregate samples into the schema record stored with each run."""
    if not samples:
        empty = summarize([])
        return GPUTelemetry(
            sample_count=0,
            temperature_c=empty,
            sm_clock_mhz=empty,
            power_w=empty,
            memory_used_mib_max=0,
            throttle_reasons_observed=[],
            throttled_fraction=0.0,
        )

    observed: set[str] = set()
    for s in samples:
        observed.update(s.event_reasons)

    throttled = sum(1 for s in samples if s.is_throttled)

    return GPUTelemetry(
        sample_count=len(samples),
        temperature_c=summarize([s.temperature_c for s in samples]),
        sm_clock_mhz=summarize([s.sm_clock_mhz for s in samples]),
        power_w=summarize([s.power_w for s in samples]),
        memory_used_mib_max=max(s.memory_used_mib for s in samples),
        # Every reason is reported, including the informational ones, so a
        # reader can see that e.g. sw_power_cap was continuously active even
        # though it does not invalidate the run.
        throttle_reasons_observed=sorted(observed),
        throttled_fraction=throttled / len(samples),
    )


@dataclass
class GpuSampler:
    """Background sampler covering a measurement window.

    Args:
        gpu_index: Host GPU index. Its UUID is resolved at start and every
            sample is verified against it.
        interval_s: Polling period. 1 s keeps overhead negligible while still
            resolving thermal drift over a multi-minute run.
    """

    gpu_index: int
    interval_s: float = 1.0

    _samples: list[GpuSample] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _expected_uuid: str | None = None
    _mismatches: int = 0

    def start(self) -> None:
        if self._thread is not None:
            msg = "sampler already started"
            raise RuntimeError(msg)
        self._expected_uuid = resolve_uuid(self.gpu_index)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-sampler")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = query_once(self.gpu_index)
            except (subprocess.SubprocessError, ValueError, OSError):
                # A transient nvidia-smi failure must not kill a sweep that has
                # already cost GPU-hours; a gap in telemetry is recoverable,
                # a lost run is not.
                self._stop.wait(self.interval_s)
                continue

            if self._expected_uuid is not None and sample.uuid != self._expected_uuid:
                self._mismatches += 1
            else:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> GPUTelemetry:
        """Halt sampling and aggregate.

        Raises:
            RuntimeError: If any sample came from a different GPU than the one
                resolved at start. That means telemetry and workload were on
                different devices, which silently corrupts the record — it must
                fail loudly rather than be averaged in.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._mismatches:
            msg = (
                f"{self._mismatches} telemetry samples came from a GPU other than "
                f"{self._expected_uuid} — device identity changed mid-run"
            )
            raise RuntimeError(msg)

        return summarize_samples(self._samples)

    @property
    def samples(self) -> tuple[GpuSample, ...]:
        return tuple(self._samples)
