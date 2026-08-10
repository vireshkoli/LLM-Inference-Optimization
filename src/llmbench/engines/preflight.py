"""Pre-run environment checks.

Each check here corresponds to a way a sweep can waste hours of GPU time or,
worse, produce numbers that look fine and are not. They run before the engine
starts, so a problem costs seconds instead of a lab session.

The neighbour-GPU check deserves explanation. The measurement device shares a
passively-cooled chassis with a second A40; when that card is loaded it raises
inlet air temperature on the card under test. That cannot always be avoided, so
the check does not block — it *stamps the result*, ensuring a run measured
beside a busy neighbour can never quietly lose its asterisk during analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PreflightError", "PreflightReport", "run_preflight"]


class PreflightError(RuntimeError):
    """A condition that must block the run rather than annotate it."""


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Environment state at run start, folded into the result record."""

    gpu_index: int
    gpu_uuid: str
    #: True when a *different* GPU in this chassis was under load. Does not
    #: block; travels with the result as a documented confound.
    neighbor_gpu_busy: bool
    neighbor_details: tuple[str, ...]
    free_disk_gib: float
    clocks_locked: bool
    sm_clock_mhz: int | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _smi(query: str, extra: list[str] | None = None) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", *(extra or [])],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def check_neighbor_gpus(
    gpu_index: int, *, memory_threshold_mib: int = 512, util_threshold_pct: float = 5.0
) -> tuple[bool, tuple[str, ...]]:
    """Detect load on GPUs other than the one under test.

    Thresholds are above zero deliberately: a few hundred MiB of resident
    context or a percent of utilisation is background noise, not a neighbour
    running a training job.
    """
    rows = _smi("index,name,memory.used,utilization.gpu")
    busy: list[str] = []

    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 4:
            continue
        index, name, used, util = int(parts[0]), parts[1], float(parts[2]), float(parts[3])
        if index == gpu_index:
            continue
        if used > memory_threshold_mib or util > util_threshold_pct:
            busy.append(f"GPU{index} ({name}): {used:.0f} MiB, {util:.0f}% util")

    return bool(busy), tuple(busy)


def check_clocks(gpu_index: int, *, state_file: Path | None = None) -> tuple[bool, int | None]:
    """Report whether SM clocks are pinned, and to what.

    **Not inferred from nvidia-smi.** ``nvidia-smi -lgc`` sets a locked clock
    range that this driver exposes no ``--query-gpu`` field for. The
    obvious-looking ``clocks.applications.graphics`` reports the *default*
    applications clock, which on an A40 equals the max (1740 MHz) whether or not
    a lock is active — so trusting it reports "locked" for an unlocked card, and
    would put a false claim into committed results.

    Instead ``scripts/lock_clocks.sh`` records what it did, and this reads that
    record. The claim is then cross-checked behaviourally in
    :func:`verify_clock_lock`, because a state file can go stale if the driver
    resets.

    An unlocked run is not invalid; the policy just has to be recorded honestly
    either way so a reader can weigh late-run drift themselves.
    """
    path = state_file or Path("results/.clock_policy.json")
    if not path.exists():
        return False, None

    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, None

    if state.get("gpu_index") != gpu_index or not state.get("locked"):
        return False, None

    sm = state.get("sm_clock_mhz")
    return True, int(sm) if sm is not None else None


def verify_clock_lock(
    gpu_index: int,
    claimed_sm_mhz: int,
    *,
    tolerance_mhz: int = 30,
    min_util_pct: float = 20.0,
) -> bool | None:
    """Cross-check a claimed clock lock against observed behaviour.

    **Only meaningful under load.** ``nvidia-smi -lgc`` bounds the clock while
    the GPU has work; at true idle an A40 still drops to ~210 MHz even with the
    lock active. Measured directly on this machine: immediately after applying a
    1740 MHz lock, an idle card reports 210 MHz. Vetoing on that reading would
    mark every genuinely-locked run as unlocked.

    Returns:
        ``True``/``False`` when the GPU is busy enough for the reading to mean
        something, and ``None`` when it is too idle to tell — which the caller
        must treat as "unknown", not as failure.
    """
    rows = _smi("clocks.sm,utilization.gpu", ["-i", str(gpu_index)])
    if not rows:
        return None
    try:
        current_str, util_str = (p.strip() for p in rows[0].split(","))
        current, util = float(current_str), float(util_str)
    except ValueError:
        return None

    if util < min_util_pct:
        return None

    return abs(current - claimed_sm_mhz) <= tolerance_mhz


def verify_clock_lock_from_telemetry(
    sm_clock_samples: Sequence[float],
    claimed_sm_mhz: int,
    *,
    tolerance_mhz: int = 30,
    min_fraction_at_lock: float = 0.9,
) -> bool:
    """Confirm a lock held across a completed measurement window.

    This is the check that actually matters: it uses clocks sampled while the
    benchmark was running, so unlike the preflight probe it is never confounded
    by an idle card. A lock that silently lapsed mid-sweep — the case that would
    quietly skew late runs — shows up here.
    """
    if not sm_clock_samples:
        return False
    at_lock = sum(1 for c in sm_clock_samples if abs(c - claimed_sm_mhz) <= tolerance_mhz)
    return at_lock / len(sm_clock_samples) >= min_fraction_at_lock


def check_disk(path: str, *, min_free_gib: float) -> float:
    """Ensure enough headroom before a sweep starts writing.

    Raises:
        PreflightError: Below the floor. Running out of disk mid-sweep loses
            every result written so far, which is far more expensive than
            refusing to start.
    """
    free_gib = shutil.disk_usage(path).free / (1024**3)
    if free_gib < min_free_gib:
        msg = (
            f"only {free_gib:.1f} GiB free at {path}, below the {min_free_gib:.1f} GiB floor — "
            f"a sweep that exhausts disk loses results already written"
        )
        raise PreflightError(msg)
    return free_gib


def run_preflight(
    gpu_index: int,
    *,
    results_path: str = ".",
    min_free_gib: float = 20.0,
    require_locked_clocks: bool = False,
) -> PreflightReport:
    """Run all checks and produce the record stamped onto every result.

    Args:
        require_locked_clocks: When True, refuse to run unlocked. Left False by
            default so an exploratory run is possible; the full sweep sets it.

    Raises:
        PreflightError: On a blocking condition — missing GPU, insufficient
            disk, or unlocked clocks when locking was required.
    """
    from llmbench.telemetry.gpu import resolve_uuid  # noqa: PLC0415  (avoid import cycle)

    warnings: list[str] = []

    try:
        uuid = resolve_uuid(gpu_index)
    except (subprocess.SubprocessError, OSError, IndexError) as exc:
        msg = f"cannot resolve GPU {gpu_index}: {exc}"
        raise PreflightError(msg) from exc

    free_gib = check_disk(results_path, min_free_gib=min_free_gib)
    neighbor_busy, neighbor_details = check_neighbor_gpus(gpu_index)
    locked, sm_clock = check_clocks(gpu_index)

    # A recorded lock is a claim, not evidence — but it can only be falsified
    # while the GPU is busy, so `None` here means "too idle to tell", not "fine".
    if locked and sm_clock is not None:
        holding = verify_clock_lock(gpu_index, sm_clock)
        if holding is False:
            locked = False
            warnings.append(
                f"clock policy records a {sm_clock} MHz lock on GPU {gpu_index}, but the card "
                f"is under load and not holding it — the lock lapsed (driver reset?) and the "
                f"record is stale. Treating this run as unlocked; re-run `make lock-clocks`."
            )
        elif holding is None:
            warnings.append(
                f"clock lock of {sm_clock} MHz is recorded but unverifiable while GPU "
                f"{gpu_index} is idle — an A40 drops to ~210 MHz at idle even when locked. "
                f"It is re-checked against sampled clocks once the run completes."
            )

    if neighbor_busy:
        warnings.append(
            "another GPU in this chassis is under load: "
            + "; ".join(neighbor_details)
            + ". A40s are passively cooled and share chassis airflow, so inlet "
            "temperature on the measurement device is elevated. Results are "
            "stamped neighbor_gpu_busy=true."
        )

    if not locked:
        if require_locked_clocks:
            msg = (
                f"GPU {gpu_index} clocks are not pinned and locking was required. "
                f"Run `make lock-clocks` first."
            )
            raise PreflightError(msg)
        warnings.append(
            f"GPU {gpu_index} clocks are not pinned; boost behaviour may drift across the "
            f"sweep. Throttle telemetry is captured regardless."
        )

    return PreflightReport(
        gpu_index=gpu_index,
        gpu_uuid=uuid,
        neighbor_gpu_busy=neighbor_busy,
        neighbor_details=neighbor_details,
        free_disk_gib=free_gib,
        clocks_locked=locked,
        sm_clock_mhz=sm_clock,
        warnings=tuple(warnings),
    )
