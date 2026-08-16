"""Held-out text for perplexity.

WikiText-2 (raw) is the standard perplexity corpus for this kind of comparison,
so the numbers sit alongside published ones. The *raw* variant is used
deliberately: the tokenized `wikitext-2-v1` variant has already had rare words
replaced with `<unk>`, which makes perplexity look better and is not what a
served model actually sees.

Held out from everything else in the benchmark — no overlap with the ShareGPT
corpus that generates load — so quality is measured on text the latency runs
never touched.
"""

from __future__ import annotations

from pathlib import Path

from llmbench.workload.prompts import Tokenizer

__all__ = ["WIKITEXT_URL", "load_wikitext_tokens"]

WIKITEXT_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)


def load_wikitext_text(path: Path | str) -> str:
    """Concatenate the WikiText-2 test split into one document.

    Joined into a single stream rather than scored per row: rows are individual
    lines, many of them short headings, and scoring each in isolation would
    measure the model with almost no context and inflate perplexity.

    Raises:
        FileNotFoundError: If the parquet file is absent.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        msg = (
            f"WikiText-2 not found at {parquet_path}. Fetch it with `make data-quality` "
            f"(from {WIKITEXT_URL})."
        )
        raise FileNotFoundError(msg)

    import pyarrow.parquet as pq  # noqa: PLC0415  — offline prep only, never on the hot path

    table = pq.read_table(parquet_path, columns=["text"])
    return "".join(str(v) for v in table.column("text").to_pylist())


def load_wikitext_tokens(
    path: Path | str, tokenizer: Tokenizer, *, max_tokens: int | None = 200_000
) -> tuple[int, ...]:
    """Tokenize the WikiText-2 test split.

    Args:
        max_tokens: Truncation bound. The full test split is ~280k tokens; a
            fixed prefix keeps perplexity comparable across configurations while
            bounding the number of forward passes per evaluation. ``None``
            scores the whole split.
    """
    token_ids = tokenizer.encode(load_wikitext_text(path))
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    return tuple(token_ids)
