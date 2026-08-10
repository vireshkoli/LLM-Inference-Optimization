"""Assert which compute kernel the engine actually selected.

This is the cheapest insurance in the project. A quantized checkpoint that
silently falls off the fast path still serves correct tokens — it is simply
slower — so the failure is invisible in every output except the throughput
number the benchmark exists to report. Discovering it during analysis means
re-running the sweep.

The concrete risk on this hardware: both INT4 checkpoints available for
Llama-3.1-8B carry ``desc_act=true``. vLLM's ``gptq_marlin`` supports act-order
through a load-time permutation, but a version change, a config change, or a
different engine could quietly route to a slower generic path instead.

So the kernel is parsed out of the engine's startup log, checked against the
expectation declared in ``configs/sweep.yaml``, and recorded into every result.
A mismatch aborts at launch — seconds in — rather than at analysis, hours later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["KernelMismatchError", "assert_kernel", "detect_kernel"]


class KernelMismatchError(RuntimeError):
    """Raised when the engine did not select the expected kernel."""

    def __init__(self, expected: str, detected: str | None, config_id: str) -> None:
        self.expected = expected
        self.detected = detected
        self.config_id = config_id
        found = detected or "no recognised kernel"
        super().__init__(
            f"[{config_id}] expected kernel {expected!r} but startup log shows {found!r}. "
            f"Refusing to measure: a checkpoint on an unexpected kernel path produces "
            f"valid tokens at the wrong speed, which is invisible in every output except "
            f"the throughput number this benchmark reports."
        )


def detect_kernel(log_text: str, patterns: Mapping[str, str]) -> str | None:
    """Return the first kernel name whose pattern matches the log.

    Args:
        log_text: Engine startup output.
        patterns: Kernel name to regex, from the engine's YAML profile. Ordering
            matters — put specific patterns before generic fallbacks, since a
            bare ``marlin`` pattern would otherwise shadow ``gptq_marlin``.

    Returns:
        The matched kernel name, or ``None`` if nothing matched.
    """
    for name, pattern in patterns.items():
        if re.search(pattern, log_text):
            return name
    return None


def assert_kernel(
    log_text: str,
    expected: str | None,
    patterns: Mapping[str, str],
    *,
    config_id: str,
) -> str | None:
    """Verify the engine selected ``expected``.

    Args:
        expected: Required kernel name, or ``None`` to skip the assertion — used
            for BF16, which has no quantized kernel to select. The detected
            value is still returned and recorded.

    Returns:
        The kernel actually detected, for storage in the result record.

    Raises:
        KernelMismatchError: If ``expected`` is set and does not match.
    """
    detected = detect_kernel(log_text, patterns)

    if expected is None:
        return detected

    if detected != expected:
        raise KernelMismatchError(expected=expected, detected=detected, config_id=config_id)

    return detected
