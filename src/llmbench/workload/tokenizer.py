"""Tokenizer loading from the local Hugging Face cache.

Uses the fast ``tokenizers`` library directly rather than ``transformers``:
nothing here needs torch, and keeping the load-generator dependency set thin
matters because the client must never become the bottleneck it is measuring.

Tokenization happens entirely offline — prompts are materialised before a run
starts — so this is never on the dispatch hot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer as HfTokenizer

__all__ = ["DEFAULT_HF_CACHE", "HubTokenizer", "load_tokenizer"]

#: Resolved once at import. Computing it in an argument default would
#: re-evaluate Path.home() per call and hide the value from callers.
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface"


@dataclass(frozen=True, slots=True)
class HubTokenizer:
    """Adapter implementing :class:`llmbench.workload.prompts.Tokenizer`."""

    _inner: HfTokenizer

    def encode(self, text: str) -> list[int]:
        # add_special_tokens=False: prompts are measured as raw text, and a BOS
        # the server will add again would double-count against the recorded
        # input length.
        return list(self._inner.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._inner.decode(list(token_ids), skip_special_tokens=True))


def _find_tokenizer_file(hf_id: str, revision: str, cache_dir: Path) -> Path:
    """Locate tokenizer.json for a pinned revision in the local hub cache."""
    repo_dir = cache_dir / "hub" / ("models--" + hf_id.replace("/", "--"))
    exact = repo_dir / "snapshots" / revision / "tokenizer.json"
    if exact.exists():
        return exact

    # Revision pinning is for the *weights*; a tokenizer from any snapshot of
    # the same repo is identical in practice, so fall back rather than fail a
    # sweep over a cache laid out by a different revision.
    candidates = sorted(repo_dir.glob("snapshots/*/tokenizer.json"))
    if candidates:
        return candidates[0]

    msg = (
        f"no tokenizer.json for {hf_id} under {repo_dir}. "
        f"The model must be present in the local HF cache."
    )
    raise FileNotFoundError(msg)


def load_tokenizer(
    hf_id: str,
    revision: str,
    cache_dir: Path | str = DEFAULT_HF_CACHE,
) -> HubTokenizer:
    """Load a tokenizer from the local cache, without touching the network."""
    path = _find_tokenizer_file(hf_id, revision, Path(cache_dir))
    return HubTokenizer(_inner=HfTokenizer.from_file(str(path)))
