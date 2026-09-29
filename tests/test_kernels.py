"""Kernel-selection assertions.

A checkpoint routed to an unexpected kernel still emits correct tokens, just
more slowly — so the fault is invisible everywhere except the throughput figure
this benchmark exists to report. These tests pin the detection that catches it
at launch instead of during analysis.
"""

from __future__ import annotations

import pytest

from llmbench.config import load_engine_profile
from llmbench.engines.kernels import KernelMismatchError, assert_kernel, detect_kernel
from llmbench.schema import EngineName

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

    def test_fallback_line_is_not_read_as_the_fast_kernel(self) -> None:
        """Regression. This test used to assert ``gptq_marlin`` here: the
        warning announcing that gptq_marlin is *not* in use matched the pattern
        for gptq_marlin, so the assertion passed on the very failure it exists
        to catch. What the log actually shows in use is the generic path."""
        assert detect_kernel(FALLBACK_LOG, PATTERNS) == "marlin_generic"

    def test_fallback_fails_the_assertion(self) -> None:
        with pytest.raises(KernelMismatchError) as exc:
            assert_kernel(FALLBACK_LOG, "gptq_marlin", PATTERNS, config_id="vllm-gptq-int4")
        assert exc.value.detected == "marlin_generic"

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


# Verbatim lines from the committed start-up logs in results/logs/ (vLLM 0.26.0,
# SGLang 0.5.17), matched with the patterns the sweep actually uses.
VLLM_INT8_KERNEL = (
    "(EngineCore pid=137) INFO 09-06 16:01:17 [__init__.py:670] "
    "Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"
)
VLLM_INT8_CONFIG_DUMP = (
    "(EngineCore pid=137) INFO 09-06 16:01:15 [core.py:116] Initializing a V1 LLM engine "
    "(v0.26.0) with config: model='RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8', "
    "quantization=compressed-tensors, "
    "served_model_name=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
)
VLLM_GPTQ_KERNEL = (
    "(EngineCore pid=137) INFO 09-04 12:47:48 [auto_gptq.py:357] "
    "Using MarlinLinearKernel for AutoGPTQLinearMethod"
)
VLLM_AWQ_KERNEL = (
    "(EngineCore pid=142) INFO 09-04 13:25:59 [auto_awq.py:473] "
    "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod"
)
SGLANG_AWQ_KERNEL = (
    "[2026-09-04 14:02:41] The model is convertible to awq_marlin during runtime. "
    "Using awq_marlin kernel."
)
SGLANG_UNRELATED_FALLBACK = (
    "[2026-09-04 14:02:42] Using CUDA IPC for multimodal features: reserving up to 1024 MiB "
    "on base GPU 0 across 1 tokenizer worker(s). This reduces KV cache headroom; a full pool "
    "falls back to CPU transport."
)


class TestProductionPatterns:
    """The patterns in configs/engines/*.yaml against real engine output."""

    @pytest.fixture
    def vllm(self) -> dict[str, str]:
        return load_engine_profile(EngineName.VLLM).kernel_log_patterns

    @pytest.fixture
    def sglang(self) -> dict[str, str]:
        return load_engine_profile(EngineName.SGLANG).kernel_log_patterns

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            (VLLM_INT8_KERNEL, "compressed-tensors-w8a8-int8"),
            (VLLM_GPTQ_KERNEL, "gptq_marlin"),
            (VLLM_AWQ_KERNEL, "awq_marlin"),
        ],
    )
    def test_vllm_kernel_lines(self, vllm: dict[str, str], line: str, expected: str) -> None:
        assert detect_kernel(f"{VLLM_INT8_CONFIG_DUMP}\n{line}", vllm) == expected

    def test_int8_is_not_inferred_from_the_model_name(self, vllm: dict[str, str]) -> None:
        """Regression. The configuration dump names the checkpoint, whose name
        contains ``w8a8``, and a loose pattern accepted that as evidence of the
        CUTLASS INT8 kernel -- the assertion passed without the kernel line."""
        assert detect_kernel(VLLM_INT8_CONFIG_DUMP, vllm) is None

    def test_a_slower_int8_kernel_is_not_accepted(self, vllm: dict[str, str]) -> None:
        other = VLLM_INT8_KERNEL.replace("CutlassInt8ScaledMM", "TritonScaledMM")
        assert detect_kernel(other, vllm) is None

    def test_vllm_fallback_is_rejected(self, vllm: dict[str, str]) -> None:
        with pytest.raises(KernelMismatchError):
            assert_kernel(FALLBACK_LOG, "gptq_marlin", vllm, config_id="vllm-gptq-int4")

    def test_sglang_awq_line(self, sglang: dict[str, str]) -> None:
        log = f"{SGLANG_UNRELATED_FALLBACK}\n{SGLANG_AWQ_KERNEL}"
        assert detect_kernel(log, sglang) == "awq_marlin"

    def test_unrelated_fallback_lines_are_harmless(self, sglang: dict[str, str]) -> None:
        """Dropping lines that mention a fallback must not hide the kernel line."""
        assert detect_kernel(SGLANG_UNRELATED_FALLBACK, sglang) is None
