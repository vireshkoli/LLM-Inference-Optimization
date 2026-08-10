"""Materialisation of concrete requests from sampled lengths.

Prompts are tokenized and sized **offline, before the run starts**. Tokenizing
inside the dispatch loop would put tens of milliseconds of CPU work on the hot
path and turn the load generator into the bottleneck — which is the exact
failure this harness measures rather than commits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from llmbench.workload.lengths import LengthPair

__all__ = ["RequestSpec", "Tokenizer", "build_requests"]


class Tokenizer(Protocol):
    """Minimal tokenizer surface, kept narrow so tests can supply a fake."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """One fully-materialised request, ready to dispatch."""

    index: int
    prompt: str
    #: Exact prompt length in tokens, verified after materialisation rather than
    #: assumed, so the recorded input distribution is what was actually sent.
    input_tokens: int
    #: Sent as ``max_tokens``, paired with ``ignore_eos`` so every configuration
    #: performs identical work regardless of where its model would stop.
    max_tokens: int


def build_requests(
    pairs: Sequence[LengthPair],
    corpus_token_ids: Sequence[int],
    tokenizer: Tokenizer,
    seed: int,
) -> tuple[RequestSpec, ...]:
    """Build prompts of exact token length by slicing a token corpus.

    Slicing a real corpus keeps prompts natural-looking (so tokenizer behaviour
    and any prefix caching see realistic text) while giving exact length control.
    Offsets are drawn from a seeded generator, so the same seed yields the same
    prompts for every configuration under test.

    Raises:
        ValueError: If the corpus is too short to source the longest prompt.
    """
    if not pairs:
        msg = "build_requests requires at least one length pair"
        raise ValueError(msg)
    if not corpus_token_ids:
        msg = "corpus_token_ids is empty"
        raise ValueError(msg)

    longest = max(p.input_tokens for p in pairs)
    if longest > len(corpus_token_ids):
        msg = (
            f"corpus has {len(corpus_token_ids)} tokens but the workload needs a "
            f"{longest}-token prompt"
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    specs: list[RequestSpec] = []

    for i, pair in enumerate(pairs):
        start = int(rng.integers(0, len(corpus_token_ids) - pair.input_tokens + 1))
        window = corpus_token_ids[start : start + pair.input_tokens]
        prompt = tokenizer.decode(window)

        # Decoding then re-encoding is not always round-trip exact: merges at the
        # slice boundary can shift the count. Record what the prompt actually
        # tokenizes to rather than the length we asked for.
        actual = len(tokenizer.encode(prompt))

        specs.append(
            RequestSpec(
                index=i,
                prompt=prompt,
                input_tokens=actual,
                max_tokens=pair.output_tokens,
            )
        )

    return tuple(specs)
