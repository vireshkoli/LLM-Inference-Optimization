"""Perplexity measured through the serving engine.

Uses the OpenAI-compatible ``/v1/completions`` endpoint with ``echo=true`` and
``max_tokens=0``, which makes the server return log-probabilities for the
*prompt* tokens rather than generating anything. That routes the measurement
through the same quantized kernels the latency numbers came from — a perplexity
computed with transformers against the raw checkpoint would score the weights
while silently missing a bug in the serving path.

**Sliding window with stride.** A model with a 4096-token context cannot score a
long document in one pass, and scoring it in disjoint chunks would evaluate the
first tokens of each chunk with almost no context, inflating perplexity. The
standard fix is a window that advances by a stride smaller than the context, with
only the newly-revealed tokens contributing to the total. Tokens whose
predictions were already counted are skipped, so every token is scored exactly
once and always with as much left-context as the stride allows.

Perplexity is cheap and sensitive, but it is not sufficient on its own: it can
miss instruction-following damage that a task benchmark catches, which is why
GSM8K and IFEval run alongside it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

__all__ = ["PerplexityResult", "compute_perplexity", "make_windows"]


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    perplexity: float
    #: Mean negative log-likelihood per token, in nats.
    nll_per_token: float
    tokens_scored: int
    windows: int


@dataclass(frozen=True, slots=True)
class Window:
    """One scoring window over the token stream."""

    token_ids: tuple[int, ...]
    #: Index within the window from which log-probs count toward the total.
    #: Earlier positions were already scored by a previous window, or (for the
    #: very first token of the document) have no context to be predicted from.
    score_from: int

    @property
    def scored_tokens(self) -> int:
        return len(self.token_ids) - self.score_from


def make_windows(token_ids: Sequence[int], *, context_len: int, stride: int) -> tuple[Window, ...]:
    """Split a token stream into overlapping scoring windows.

    Args:
        context_len: Window size; must not exceed the served ``max_model_len``.
        stride: How far the window advances. A stride below ``context_len``
            gives later tokens more left-context at the cost of more forward
            passes. ``stride == context_len`` degenerates to disjoint chunks,
            which inflates perplexity.

    Raises:
        ValueError: On a non-positive stride or a stride above the context.
    """
    if context_len < 2:
        msg = f"context_len must be >= 2, got {context_len}"
        raise ValueError(msg)
    if not 0 < stride <= context_len:
        msg = f"stride must be in (0, context_len={context_len}], got {stride}"
        raise ValueError(msg)

    windows: list[Window] = []
    start = 0
    previous_end = 0

    while start < len(token_ids):
        end = min(start + context_len, len(token_ids))
        chunk = tuple(token_ids[start:end])

        # First window: token 0 has no predecessor, so scoring starts at 1.
        # Later windows: skip the overlap already scored by the previous window.
        score_from = 1 if start == 0 else previous_end - start
        if score_from < len(chunk):
            windows.append(Window(token_ids=chunk, score_from=score_from))

        previous_end = end
        if end == len(token_ids):
            break
        start += stride

    return tuple(windows)


def _score_window(
    client: httpx.Client, model: str, window: Window, *, endpoint: str
) -> tuple[float, int]:
    """Return (summed negative log-likelihood, tokens scored) for one window.

    ``echo=true`` with ``max_tokens=0`` asks the server to score the prompt
    instead of continuing it.
    """
    response = client.post(
        endpoint,
        json={
            "model": model,
            "prompt": list(window.token_ids),
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
            "temperature": 0.0,
        },
    )
    response.raise_for_status()
    payload = response.json()

    logprobs = payload["choices"][0]["logprobs"]["token_logprobs"]
    # The first entry is null: the leading token has nothing to condition on.
    scored = [lp for lp in logprobs[window.score_from :] if lp is not None]
    return -sum(scored), len(scored)


def compute_perplexity(
    base_url: str,
    model: str,
    token_ids: Sequence[int],
    *,
    context_len: int = 4096,
    stride: int = 2048,
    endpoint: str = "/v1/completions",
    timeout_s: float = 600.0,
) -> PerplexityResult:
    """Compute perplexity over a token stream via the serving engine.

    Raises:
        ValueError: If no tokens could be scored, which means the endpoint did
            not return prompt log-probabilities and the result would otherwise
            be a meaningless ``exp(0)``.
    """
    windows = make_windows(token_ids, context_len=context_len, stride=stride)

    total_nll = 0.0
    total_tokens = 0

    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_s)) as client:
        for window in windows:
            nll, count = _score_window(client, model, window, endpoint=endpoint)
            total_nll += nll
            total_tokens += count

    if total_tokens == 0:
        msg = (
            "no tokens were scored — the endpoint returned no prompt logprobs. "
            "Check that the server supports echo=true with max_tokens=0."
        )
        raise ValueError(msg)

    nll_per_token = total_nll / total_tokens
    return PerplexityResult(
        perplexity=math.exp(nll_per_token),
        nll_per_token=nll_per_token,
        tokens_scored=total_tokens,
        windows=len(windows),
    )
