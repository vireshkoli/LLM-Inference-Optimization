"""Preflight logic that needs no GPU.

The clock-lock thresholds encode a measured fact about this hardware rather than
an assumption, so they are pinned here: on a 300 W-capped A40, a clock lock is
not absolute under load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmbench.engines.preflight import (
    PreflightError,
    check_clocks,
    check_disk,
    check_model_cached,
    clock_lock_held_fraction,
    verify_clock_lock_from_telemetry,
)


class TestDiskGuard:
    def test_passes_with_headroom(self, tmp_path: Path) -> None:
        assert check_disk(str(tmp_path), min_free_gib=0.001) > 0

    def test_blocks_below_the_floor(self, tmp_path: Path) -> None:
        """Exhausting disk mid-sweep loses every result written so far, which
        is far more expensive than refusing to start."""
        with pytest.raises(PreflightError, match=r"below the .* GiB floor"):
            check_disk(str(tmp_path), min_free_gib=10_000_000.0)


class TestClockPolicyState:
    def test_absent_state_file_means_unlocked(self, tmp_path: Path) -> None:
        """Never infer a lock from nvidia-smi.

        clocks.applications.graphics reads 1740 MHz on an UNLOCKED A40 because
        that is the default, so inferring would put a false claim in results.
        """
        locked, sm = check_clocks(1, state_file=tmp_path / "missing.json")
        assert locked is False
        assert sm is None

    def test_reads_a_recorded_lock(self, tmp_path: Path) -> None:
        state = tmp_path / "clock_policy.json"
        state.write_text(json.dumps({"gpu_index": 1, "locked": True, "sm_clock_mhz": 1740}))
        assert check_clocks(1, state_file=state) == (True, 1740)

    def test_state_for_a_different_gpu_is_ignored(self, tmp_path: Path) -> None:
        state = tmp_path / "clock_policy.json"
        state.write_text(json.dumps({"gpu_index": 0, "locked": True, "sm_clock_mhz": 1740}))
        assert check_clocks(1, state_file=state) == (False, None)

    def test_explicit_unlock_is_respected(self, tmp_path: Path) -> None:
        state = tmp_path / "clock_policy.json"
        state.write_text(json.dumps({"gpu_index": 1, "locked": False, "sm_clock_mhz": None}))
        assert check_clocks(1, state_file=state) == (False, None)

    def test_corrupt_state_file_is_not_fatal(self, tmp_path: Path) -> None:
        state = tmp_path / "clock_policy.json"
        state.write_text("{not json")
        assert check_clocks(1, state_file=state) == (False, None)


class TestClockLockVerification:
    def test_steady_lock_verifies(self) -> None:
        assert verify_clock_lock_from_telemetry([1740.0] * 50, 1740) is True

    def test_power_capped_dips_still_verify(self) -> None:
        """The measured reality on this hardware.

        With a 1740 MHz lock, a saturating run had 21% of samples more than
        30 MHz below it — each coinciding with power at/above 290 W of the
        300 W cap. The lock removes boost variability but cannot defeat the
        power budget, so demanding near-total adherence would fail every
        heavily-loaded run for a reason that is physics, not a fault.
        """
        samples = [1740.0] * 67 + [1515.0] * 18  # the observed 85-sample run
        assert clock_lock_held_fraction(samples, 1740) == pytest.approx(0.788, abs=0.01)
        assert verify_clock_lock_from_telemetry(samples, 1740) is True

    def test_a_genuinely_lapsed_lock_fails(self) -> None:
        """Unlocked idle behaviour: an A40 sits at ~210 MHz."""
        assert verify_clock_lock_from_telemetry([210.0] * 80 + [1740.0] * 5, 1740) is False

    def test_empty_samples_do_not_verify(self) -> None:
        assert verify_clock_lock_from_telemetry([], 1740) is False
        assert clock_lock_held_fraction([], 1740) == 0.0

    def test_tolerance_is_applied(self) -> None:
        assert verify_clock_lock_from_telemetry([1715.0] * 20, 1740) is True
        assert verify_clock_lock_from_telemetry([1690.0] * 20, 1740) is False


class TestCheckpointPresence:
    """The HF cache is mounted read-only, so a missing checkpoint cannot
    self-heal — it surfaces as an OSError deep in a transformers traceback
    inside a container that has already been reaped. Catching it here is the
    difference between a one-line message and a debugging cycle."""

    @staticmethod
    def _cache(root: Path, repo: str, revision: str) -> Path:
        snap = root / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        snap.mkdir(parents=True)
        return root

    def test_present_checkpoint_returns_its_snapshot(self, tmp_path: Path) -> None:
        root = self._cache(tmp_path, "org/model", "a" * 40)
        found = check_model_cached("org/model", "a" * 40, root)
        assert found.is_dir()

    def test_missing_repository_names_the_repository(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="repository missing"):
            check_model_cached("org/model", "a" * 40, tmp_path)

    def test_wrong_revision_is_not_accepted(self, tmp_path: Path) -> None:
        """A cached repo at the wrong commit is not the pinned checkpoint;
        accepting it would silently measure different weights."""
        root = self._cache(tmp_path, "org/model", "a" * 40)
        with pytest.raises(PreflightError, match="revision missing"):
            check_model_cached("org/model", "b" * 40, root)
