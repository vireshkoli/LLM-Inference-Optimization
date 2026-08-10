"""Input/output length distributions.

Fixed 128-in/128-out is the classic tell of an unserious benchmark. Real traffic
has a long tail, and length distribution changes batching behaviour completely:
a batch of uniform requests packs perfectly and retires together, while a
realistic mix leaves the scheduler juggling long and short sequences, which is
the situation continuous batching actually exists to handle.

The primary sampler is empirical, resampled from observed ShareGPT
conversations. A fitted log-normal sampler is provided so the harness can run
reproducibly without the dataset, and a fixed-length sampler is kept solely so
the benchmark can *demonstrate* how much rosier the fixed-length answer looks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EmpiricalLengthSampler",
    "FixedLengthSampler",
    "LengthPair",
    "LengthSampler",
    "LogNormalLengthSampler",
    "clamp_to_context",
]


@dataclass(frozen=True, slots=True)
class LengthPair:
    """One request's shape: prompt length and requested generation length."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 1:
            msg = f"input_tokens must be >= 1, got {self.input_tokens}"
            raise ValueError(msg)
        if self.output_tokens < 1:
            msg = f"output_tokens must be >= 1, got {self.output_tokens}"
            raise ValueError(msg)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LengthSampler(Protocol):
    """Draws request shapes. Implementations must be deterministic given a seed."""

    def sample(self, n: int, seed: int) -> tuple[LengthPair, ...]: ...


@dataclass(frozen=True, slots=True)
class EmpiricalLengthSampler:
    """Resamples observed (input, output) pairs with replacement.

    Pairs are drawn jointly rather than marginally, because input and output
    length are correlated in real traffic — long prompts tend to get long
    answers — and sampling the two independently would destroy that structure
    along with its effect on batch composition.
    """

    pairs: tuple[LengthPair, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            msg = "EmpiricalLengthSampler requires at least one observed pair"
            raise ValueError(msg)

    @classmethod
    def from_observations(cls, observations: Sequence[tuple[int, int]]) -> EmpiricalLengthSampler:
        return cls(pairs=tuple(LengthPair(i, o) for i, o in observations))

    def sample(self, n: int, seed: int) -> tuple[LengthPair, ...]:
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.pairs), size=n)
        return tuple(self.pairs[int(i)] for i in idx)


@dataclass(frozen=True, slots=True)
class LogNormalLengthSampler:
    """Log-normal lengths — a reproducible stand-in when the corpus is absent.

    Log-normal is the usual parametric fit for token-length distributions: it is
    positive, right-skewed, and heavy-tailed enough to keep the scheduler
    honest. Parameters are for the *underlying normal*, so ``exp(mu)`` is the
    median length.
    """

    input_mu: float
    input_sigma: float
    output_mu: float
    output_sigma: float
    min_tokens: int = 1

    def sample(self, n: int, seed: int) -> tuple[LengthPair, ...]:
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        ins = rng.lognormal(self.input_mu, self.input_sigma, size=n)
        outs = rng.lognormal(self.output_mu, self.output_sigma, size=n)
        return tuple(
            LengthPair(
                input_tokens=max(self.min_tokens, round(float(i))),
                output_tokens=max(self.min_tokens, round(float(o))),
            )
            for i, o in zip(ins, outs, strict=True)
        )

    @classmethod
    def fit(cls, observations: Sequence[tuple[int, int]]) -> LogNormalLengthSampler:
        """Fit by method of moments on the logs of observed lengths."""
        if not observations:
            msg = "cannot fit a distribution to zero observations"
            raise ValueError(msg)

        log_in = [math.log(max(1, i)) for i, _ in observations]
        log_out = [math.log(max(1, o)) for _, o in observations]

        def _mu_sigma(xs: list[float]) -> tuple[float, float]:
            mu = sum(xs) / len(xs)
            if len(xs) < 2:
                return mu, 0.0
            var = sum((x - mu) ** 2 for x in xs) / len(xs)
            return mu, math.sqrt(var)

        in_mu, in_sigma = _mu_sigma(log_in)
        out_mu, out_sigma = _mu_sigma(log_out)
        return cls(input_mu=in_mu, input_sigma=in_sigma, output_mu=out_mu, output_sigma=out_sigma)


@dataclass(frozen=True, slots=True)
class FixedLengthSampler:
    """Constant lengths — the anti-pattern, retained as an exhibit.

    Present so METHODOLOGY.md can quantify how much more favourable a
    fixed-length workload looks on identical hardware, rather than merely
    asserting that it does. Never used for a headline result.
    """

    input_tokens: int
    output_tokens: int

    def sample(self, n: int, seed: int) -> tuple[LengthPair, ...]:  # noqa: ARG002
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        return tuple(LengthPair(self.input_tokens, self.output_tokens) for _ in range(n))


def clamp_to_context(
    pairs: Sequence[LengthPair], max_model_len: int, *, min_output_tokens: int = 1
) -> tuple[LengthPair, ...]:
    """Shrink requests that would overflow the context window.

    A request exceeding ``max_model_len`` is rejected by the server, which would
    show up as a failed request and pollute the latency distribution with an
    error path rather than a measurement. Output length is trimmed first, since
    truncating the prompt would change the prefill work the run is meant to
    characterise.

    Raises:
        ValueError: If a prompt alone cannot fit even after removing all output
            budget — that is a workload/config mismatch the caller must fix, not
            something to paper over silently.
    """
    clamped: list[LengthPair] = []
    for pair in pairs:
        if pair.total_tokens <= max_model_len:
            clamped.append(pair)
            continue

        allowed_output = max_model_len - pair.input_tokens
        if allowed_output < min_output_tokens:
            msg = (
                f"prompt of {pair.input_tokens} tokens leaves no room for "
                f"{min_output_tokens} output tokens within max_model_len={max_model_len}"
            )
            raise ValueError(msg)
        clamped.append(LengthPair(pair.input_tokens, allowed_output))

    return tuple(clamped)
