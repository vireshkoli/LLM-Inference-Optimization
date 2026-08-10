"""ShareGPT corpus loading — real conversation lengths and real prompt text.

ShareGPT is used because it is what vLLM's own ``benchmark_serving.py`` samples
from, which keeps these numbers comparable to published work, and because it
supplies *both* halves of a realistic workload from one source: the joint
distribution of prompt and completion lengths, and natural text to build prompts
out of.

Parsing is deliberately strict about turn structure. A conversation's first
human turn is the prompt and the first assistant turn is the completion; taking
lengths from anywhere else would break the input/output correlation that drives
batch composition.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llmbench.workload.lengths import LengthPair
from llmbench.workload.prompts import Tokenizer

__all__ = ["ShareGptCorpus", "load_sharegpt"]

_HUMAN_ROLES = frozenset({"human", "user"})
_ASSISTANT_ROLES = frozenset({"gpt", "assistant", "bard", "chatgpt"})


@dataclass(frozen=True, slots=True)
class ShareGptCorpus:
    """Observed length pairs plus a flat token stream for prompt construction."""

    pairs: tuple[LengthPair, ...]
    #: Concatenated human-turn tokens. Prompts are sliced from this so they look
    #: like real text while still having exact token lengths.
    corpus_token_ids: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.pairs)


def _first_turns(conversation: Sequence[dict[str, str]]) -> tuple[str, str] | None:
    """Return the first (human, assistant) exchange, or None if malformed."""
    prompt: str | None = None
    for turn in conversation:
        role = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        if not value:
            continue
        if prompt is None and role in _HUMAN_ROLES:
            prompt = value
        elif prompt is not None and role in _ASSISTANT_ROLES:
            return prompt, value
    return None


def load_sharegpt(
    path: Path | str,
    tokenizer: Tokenizer,
    *,
    max_conversations: int = 20_000,
    min_input_tokens: int = 4,
    min_output_tokens: int = 4,
    max_total_tokens: int = 8192,
    corpus_token_budget: int = 2_000_000,
) -> ShareGptCorpus:
    """Load and tokenize ShareGPT conversations.

    Args:
        max_conversations: Cap on conversations parsed. The full file is ~90k
            exchanges; a few thousand already characterises the distribution and
            tokenizing all of them wastes minutes per run.
        max_total_tokens: Discard exchanges longer than this. They are a
            handful of outliers that would be clamped to the context window
            anyway, and keeping them would skew the sampled distribution toward
            lengths that never actually get issued.

    Raises:
        FileNotFoundError: If the dataset is absent.
        ValueError: If no usable conversations were found.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        msg = (
            f"ShareGPT dataset not found at {dataset_path}. "
            f"Fetch it with `make data` before running a sweep."
        )
        raise FileNotFoundError(msg)

    raw = json.loads(dataset_path.read_text())

    pairs: list[LengthPair] = []
    corpus: list[int] = []

    for record in raw[:max_conversations]:
        conversation = record.get("conversations") or []
        turns = _first_turns(conversation)
        if turns is None:
            continue

        prompt_text, completion_text = turns
        prompt_ids = tokenizer.encode(prompt_text)
        completion_ids = tokenizer.encode(completion_text)

        n_in, n_out = len(prompt_ids), len(completion_ids)
        if n_in < min_input_tokens or n_out < min_output_tokens:
            continue
        if n_in + n_out > max_total_tokens:
            continue

        pairs.append(LengthPair(input_tokens=n_in, output_tokens=n_out))
        if len(corpus) < corpus_token_budget:
            corpus.extend(prompt_ids)

    if not pairs:
        msg = f"no usable conversations parsed from {dataset_path}"
        raise ValueError(msg)

    return ShareGptCorpus(pairs=tuple(pairs), corpus_token_ids=tuple(corpus))
