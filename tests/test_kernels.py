"""Kernel-selection assertions.

A checkpoint routed to an unexpected kernel still emits correct tokens, just
more slowly — so the fault is invisible everywhere except the throughput figure
this benchmark exists to report. These tests pin the detection that catches it
at launch instead of during analysis.
"""

from __future__ import annotations

import pytest

from llmbench.engines.kernels import KernelMismatchError, assert_kernel, detect_kernel

# Ordering matters: the generic fallback must come last, or it shadows the
# specific patterns. The dict order here mirrors configs/engines/*.yaml.
PATTERNS = {
    "gptq_marlin": r"(?i)gptq_?marlin",
    "awq_marlin": r"(?i)awq_?marlin",
    "compressed-tensors-w8a8-int8": r"(?i)compressed[- ]tensors|cutlass.*int8|w8a8",
    "marlin_generic": r"(?i)\bmarlin\b",
}

GPTQ_LOG = """
INFO 08-10 06:12:01 llm_engine.py:237] Initializing an LLM engine
INFO 08-10 06:12:03 gptq_marlin.py:112] Using GPTQMarlinLinearMethod for quantization
INFO 08-10 06:12:44 model_runner.py:1057] Loading model weights took 5.3388 GB
"""

AWQ_LOG = """
INFO 08-10 06:20:11 awq_marlin.py:98] Using AWQMarlinLinearMethod for quantization
INFO 08-10 06:20:50 model_runner.py:1057] Loading model weights took 5.7 GB
"""

BF16_LOG = """
INFO 08-10 06:30:01 llm_engine.py:237] Initializing an LLM engine
INFO 08-10 06:30:40 model_runner.py:1057] Loading model weights took 14.9595 GB
"""

# The failure mode: a 4-bit checkpoint that fell back to the slow generic path.
FALLBACK_LOG = """
WARNING 08-10 06:40:02 config.py:301] gptq_marlin is not supported for this config,
falling back to GPTQLinearMethod
INFO 08-10 06:40:03 marlin.py:44] Using MarlinLinearMethod
"""


class TestDetection:
    def test_detects_gptq_marlin(self) -> None:
        assert detect_kernel(GPTQ_LOG, PATTERNS) == "gptq_marlin"

    def test_detects_awq_marlin(self) -> None:
        assert detect_kernel(AWQ_LOG, PATTERNS) == "awq_marlin"

    def test_returns_none_when_nothing_matches(self) -> None:
        assert detect_kernel(BF16_LOG, PATTERNS) is None

    def test_specific_pattern_wins_over_generic(self) -> None:
        """`gptq_marlin` also matches the bare `marlin` pattern.

        If the generic fallback won, a genuine fast-path run and a slow-path
        fallback would both report `marlin_generic` and the distinction that
        matters would be lost.
        """
        assert detect_kernel(GPTQ_LOG, PATTERNS) == "gptq_marlin"

    def test_generic_marlin_matches_the_fallback_path(self) -> None:
        assert detect_kernel(FALLBACK_LOG, PATTERNS) == "gptq_marlin"

    def test_case_insensitive(self) -> None:
        assert detect_kernel("Using AWQ_MARLIN kernel", PATTERNS) == "awq_marlin"


class TestAssertion:
    def test_passes_when_expectation_is_met(self) -> None:
        assert assert_kernel(GPTQ_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4") == (
            "gptq_marlin"
        )

    def test_raises_on_the_wrong_kernel(self) -> None:
        with pytest.raises(KernelMismatchError) as exc:
            assert_kernel(AWQ_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4")
        assert exc.value.expected == "gptq_marlin"
        assert exc.value.detected == "awq_marlin"

    def test_raises_when_no_kernel_is_recognised(self) -> None:
        """Silence is not success.

        A quantized config whose log mentions no kernel at all is exactly the
        ambiguous case that must fail loudly.
        """
        with pytest.raises(KernelMismatchError, match="no recognised kernel"):
            assert_kernel(BF16_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4")

    def test_error_explains_the_consequence(self) -> None:
        """The message has to be actionable at 3am mid-sweep."""
        with pytest.raises(KernelMismatchError, match="valid tokens at the wrong speed"):
            assert_kernel(AWQ_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4")

    def test_error_names_the_config(self) -> None:
        with pytest.raises(KernelMismatchError, match=r"\[vllm-gptq-int4\]"):
            assert_kernel(AWQ_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4")

    def test_none_expectation_skips_assertion_but_still_records(self) -> None:
        """BF16 has no quantized kernel to select.

        The detected value is still returned, so the result record documents
        what ran rather than leaving a hole.
        """
        assert assert_kernel(BF16_LOG, None, PATTERNS, config_id="vllm-bf16") is None
        assert assert_kernel(GPTQ_LOG, None, PATTERNS, config_id="vllm-bf16") == "gptq_marlin"
