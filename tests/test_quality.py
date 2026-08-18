"""Quality-axis logic that needs no GPU.

The sliding-window arithmetic is the part most likely to be silently wrong: an
off-by-one in the overlap either double-counts tokens (deflating perplexity) or
scores tokens with no left-context (inflating it). Both produce a plausible
number, so both are pinned here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from llmbench.quality.harness import (
    TASK_BACKEND,
    TASK_CHAT_TEMPLATE,
    TASK_FEWSHOT,
    HarnessConfig,
    HarnessError,
    _build_command,
    parse_lm_eval_results,
)
from llmbench.quality.perplexity import make_windows
from llmbench.schema import QualityTask


class TestSlidingWindows:
    def test_every_token_is_scored_exactly_once(self) -> None:
        """The property that makes the total a valid perplexity.

        Double-counting deflates it; skipping inflates it. Token 0 is never
        scored — it has nothing to be predicted from.
        """
        tokens = list(range(1000))
        windows = make_windows(tokens, context_len=100, stride=50)
        assert sum(w.scored_tokens for w in windows) == len(tokens) - 1

    @pytest.mark.parametrize(
        ("n", "context", "stride"),
        [(1000, 100, 50), (1000, 128, 32), (500, 64, 64), (137, 64, 16), (10, 4, 2)],
    )
    def test_coverage_holds_across_shapes(self, n: int, context: int, stride: int) -> None:
        windows = make_windows(list(range(n)), context_len=context, stride=stride)
        assert sum(w.scored_tokens for w in windows) == n - 1

    def test_first_window_skips_the_unpredictable_first_token(self) -> None:
        windows = make_windows(list(range(100)), context_len=50, stride=25)
        assert windows[0].score_from == 1

    def test_later_windows_skip_the_overlap(self) -> None:
        """Overlap gives left-context; scoring it again would double-count."""
        windows = make_windows(list(range(200)), context_len=100, stride=25)
        assert windows[1].score_from == 75  # 100 tokens seen, window starts at 25
        assert windows[1].scored_tokens == 25

    def test_stride_equal_to_context_is_disjoint_chunks(self) -> None:
        """Legal but inflates perplexity, since each chunk's early tokens are
        predicted with almost no context."""
        windows = make_windows(list(range(300)), context_len=100, stride=100)
        assert len(windows) == 3
        assert sum(w.scored_tokens for w in windows) == 299

    def test_smaller_stride_means_more_context_and_more_windows(self) -> None:
        few = make_windows(list(range(1000)), context_len=100, stride=100)
        many = make_windows(list(range(1000)), context_len=100, stride=25)
        assert len(many) > len(few)
        assert sum(w.scored_tokens for w in many) == sum(w.scored_tokens for w in few)

    def test_stream_shorter_than_context(self) -> None:
        windows = make_windows(list(range(30)), context_len=100, stride=50)
        assert len(windows) == 1
        assert windows[0].scored_tokens == 29

    @pytest.mark.parametrize("stride", [0, -1, 101])
    def test_invalid_stride_rejected(self, stride: int) -> None:
        with pytest.raises(ValueError, match="stride must be in"):
            make_windows(list(range(200)), context_len=100, stride=stride)

    def test_degenerate_context_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_len must be >= 2"):
            make_windows(list(range(10)), context_len=1, stride=1)


class TestHarnessParsing:
    def result_doc(self) -> dict[str, object]:
        """Shape of an lm-eval results document."""
        return {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.7846,
                    "exact_match_stderr,strict-match": 0.0113,
                    "exact_match,flexible-extract": 0.7907,
                    "exact_match_stderr,flexible-extract": 0.0112,
                },
                "ifeval": {
                    "prompt_level_strict_acc,none": 0.7541,
                    "prompt_level_strict_acc_stderr,none": 0.0185,
                    "inst_level_strict_acc,none": 0.8261,
                },
            },
            "n-samples": {
                "gsm8k": {"original": 1319, "effective": 1319},
                "ifeval": {"original": 541, "effective": 541},
            },
        }

    def test_extracts_both_tasks(self) -> None:
        scores = parse_lm_eval_results(self.result_doc())
        assert {s.task for s in scores} == {QualityTask.GSM8K, QualityTask.IFEVAL}

    def test_captures_value_and_stderr(self) -> None:
        scores = parse_lm_eval_results(self.result_doc())
        strict = next(s for s in scores if s.metric == "exact_match,strict-match")
        assert strict.value == pytest.approx(0.7846)
        assert strict.stderr == pytest.approx(0.0113)
        assert strict.num_samples == 1319

    def test_missing_stderr_is_none_not_zero(self) -> None:
        """Zero standard error would imply a certainty the harness never claimed."""
        scores = parse_lm_eval_results(self.result_doc())
        inst = next(s for s in scores if s.metric == "inst_level_strict_acc,none")
        assert inst.stderr is None

    def test_ignores_unrequested_tasks(self) -> None:
        doc = self.result_doc()
        doc["results"]["mmlu"] = {"acc,none": 0.65}  # type: ignore[index]
        assert all(s.task is not QualityTask.WIKITEXT2_PPL for s in parse_lm_eval_results(doc))

    def test_missing_results_object_raises(self) -> None:
        with pytest.raises(HarnessError, match="no 'results' object"):
            parse_lm_eval_results({"config": {}})

    def test_no_recognised_metrics_raises(self) -> None:
        """Silently returning an empty score list would look like a clean run."""
        with pytest.raises(HarnessError, match="no recognised metrics"):
            parse_lm_eval_results({"results": {"hellaswag": {"acc,none": 0.5}}})


class TestPerTaskFewShot:
    """GSM8K and IFEval need different few-shot counts.

    ``--num_fewshot`` is global to an lm-eval invocation, so evaluating both in
    one run would either leave GSM8K at lm-eval's 5-shot default — contradicting
    the documented 8-shot — or force few-shot prompting onto IFEval, which is
    0-shot by design and whose verifiable constraints assume it.
    """

    def test_gsm8k_is_eight_shot(self) -> None:
        assert TASK_FEWSHOT["gsm8k"] == 8

    def test_ifeval_is_zero_shot(self) -> None:
        assert TASK_FEWSHOT["ifeval"] == 0

    def test_each_task_gets_its_own_fewshot_on_the_command_line(self) -> None:
        base = HarnessConfig(base_url="http://x", model="m")
        for task, expected in TASK_FEWSHOT.items():
            cmd = _build_command(replace(base, num_fewshot=expected), [task], Path("/tmp/x"))
            assert cmd[cmd.index("--num_fewshot") + 1] == str(expected)
            # One task per invocation, or the global flag would mis-prompt one.
            assert cmd[cmd.index("--tasks") + 1] == task


class TestTaskBackendRouting:
    """IFEval must be served through the chat endpoint, GSM8K through raw completions.

    Measured on Llama-3.1-8B-Instruct BF16, running IFEval through raw
    /v1/completions with no chat template gave prompt_level_strict_acc 0.4603
    against 0.7000 with the template — a ~24-point artifact of prompting, not of
    the model, and ~32 points below published. An instruct model on the raw
    completions endpoint continues text instead of following instructions.
    """

    def test_ifeval_uses_the_chat_endpoint_with_a_template(self) -> None:
        backend, endpoint = TASK_BACKEND["ifeval"]
        assert backend == "local-chat-completions"
        assert endpoint == "/v1/chat/completions"
        assert TASK_CHAT_TEMPLATE["ifeval"] is True

    def test_gsm8k_uses_raw_completions_without_a_template(self) -> None:
        """8-shot exemplars establish the pattern; this is the conventional
        few-shot setup and reproduces published numbers."""
        backend, endpoint = TASK_BACKEND["gsm8k"]
        assert backend == "local-completions"
        assert endpoint == "/v1/completions"
        assert TASK_CHAT_TEMPLATE["gsm8k"] is False

    def test_command_carries_the_right_backend_and_url_per_task(self) -> None:
        base = HarnessConfig(base_url="http://127.0.0.1:8000", model="m")
        for task, (backend, endpoint) in TASK_BACKEND.items():
            cmd = _build_command(
                replace(
                    base,
                    num_fewshot=TASK_FEWSHOT[task],
                    apply_chat_template=TASK_CHAT_TEMPLATE[task],
                ),
                [task],
                Path("/tmp/x"),
            )
            assert cmd[cmd.index("--model") + 1] == backend
            assert any(f"base_url=http://127.0.0.1:8000{endpoint}" in a for a in cmd)
            assert ("--apply_chat_template" in cmd) is TASK_CHAT_TEMPLATE[task]
