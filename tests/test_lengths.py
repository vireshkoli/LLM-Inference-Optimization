"""Length-distribution sampling.

Length distribution drives batch composition, which is the thing continuous
batching exists to manage. Getting it wrong — most commonly by using fixed
lengths — makes a server look considerably better than it is under real traffic.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmbench.workload.lengths import (
    EmpiricalLengthSampler,
    FixedLengthSampler,
    LengthPair,
    LogNormalLengthSampler,
    clamp_to_context,
)

# Correlated (input, output) pairs: longer prompts get longer answers.
OBSERVATIONS = [(64, 32), (128, 96), (256, 200), (512, 400), (1024, 700), (32, 16)]


class TestLengthPair:
    def test_total_is_the_context_footprint(self) -> None:
        assert LengthPair(128, 64).total_tokens == 192

    @pytest.mark.parametrize(("i", "o"), [(0, 10), (10, 0), (-1, 10)])
    def test_rejects_degenerate_lengths(self, i: int, o: int) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            LengthPair(i, o)


class TestEmpiricalSampler:
    def test_only_draws_observed_pairs(self) -> None:
        sampler = EmpiricalLengthSampler.from_observations(OBSERVATIONS)
        drawn = sampler.sample(500, seed=1)
        assert all((p.input_tokens, p.output_tokens) in OBSERVATIONS for p in drawn)

    def test_preserves_the_input_output_correlation(self) -> None:
        """Sampling the two marginals independently would destroy this.

        Real traffic correlates prompt and completion length, and that
        correlation changes how requests pack into batches.
        """
        sampler = EmpiricalLengthSampler.from_observations(OBSERVATIONS)
        drawn = sampler.sample(2000, seed=2)
        ins = np.array([p.input_tokens for p in drawn], dtype=float)
        outs = np.array([p.output_tokens for p in drawn], dtype=float)
        assert float(np.corrcoef(ins, outs)[0, 1]) > 0.9

    def test_is_deterministic_for_a_seed(self) -> None:
        sampler = EmpiricalLengthSampler.from_observations(OBSERVATIONS)
        assert sampler.sample(100, seed=7) == sampler.sample(100, seed=7)

    def test_different_seeds_differ(self) -> None:
        sampler = EmpiricalLengthSampler.from_observations(OBSERVATIONS)
        assert sampler.sample(100, seed=7) != sampler.sample(100, seed=8)

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="at least one observed pair"):
            EmpiricalLengthSampler(pairs=())


class TestLogNormalSampler:
    def test_is_right_skewed(self) -> None:
        """The property that matters: a long tail, not a symmetric spread."""
        sampler = LogNormalLengthSampler(5.0, 0.8, 4.5, 0.9)
        drawn = sampler.sample(5000, seed=3)
        ins = np.array([p.input_tokens for p in drawn], dtype=float)
        assert float(ins.mean()) > float(np.median(ins))

    def test_lengths_stay_positive(self) -> None:
        sampler = LogNormalLengthSampler(0.5, 2.0, 0.5, 2.0)
        assert all(p.input_tokens >= 1 and p.output_tokens >= 1 for p in sampler.sample(2000, 4))

    def test_fit_recovers_the_median(self) -> None:
        rng = np.random.default_rng(11)
        obs = [
            (max(1, int(x)), max(1, int(y)))
            for x, y in zip(
                rng.lognormal(5.0, 0.5, 4000), rng.lognormal(4.0, 0.5, 4000), strict=True
            )
        ]
        fitted = LogNormalLengthSampler.fit(obs)
        assert fitted.input_mu == pytest.approx(5.0, abs=0.1)
        assert fitted.output_mu == pytest.approx(4.0, abs=0.1)

    def test_fit_rejects_no_observations(self) -> None:
        with pytest.raises(ValueError, match="zero observations"):
            LogNormalLengthSampler.fit([])

    def test_is_deterministic_for_a_seed(self) -> None:
        sampler = LogNormalLengthSampler(5.0, 0.8, 4.5, 0.9)
        assert sampler.sample(50, seed=9) == sampler.sample(50, seed=9)


class TestFixedSampler:
    def test_produces_no_variation(self) -> None:
        """Retained as an exhibit, not for headline results.

        Zero variance is precisely why fixed-length benchmarks flatter a server:
        uniform requests pack perfectly and retire together.
        """
        drawn = FixedLengthSampler(128, 128).sample(100, seed=1)
        assert len({(p.input_tokens, p.output_tokens) for p in drawn}) == 1


class TestContextClamping:
    def test_leaves_fitting_requests_untouched(self) -> None:
        pairs = [LengthPair(100, 100), LengthPair(200, 50)]
        assert clamp_to_context(pairs, max_model_len=4096) == tuple(pairs)

    def test_trims_output_not_input(self) -> None:
        """Truncating the prompt would change the prefill work being measured."""
        clamped = clamp_to_context([LengthPair(3000, 2000)], max_model_len=4096)
        assert clamped[0].input_tokens == 3000
        assert clamped[0].output_tokens == 1096

    def test_clamped_requests_fit_exactly(self) -> None:
        clamped = clamp_to_context([LengthPair(4000, 500)], max_model_len=4096)
        assert clamped[0].total_tokens == 4096

    def test_raises_when_the_prompt_alone_overflows(self) -> None:
        """A workload/config mismatch the caller must fix, not paper over.

        Silently truncating here would misreport the input distribution.
        """
        with pytest.raises(ValueError, match="leaves no room"):
            clamp_to_context([LengthPair(5000, 100)], max_model_len=4096)

    def test_prevents_server_side_rejection(self) -> None:
        """Over-long requests would fail and pollute the latency distribution
        with an error path rather than a measurement."""
        sampler = EmpiricalLengthSampler.from_observations(OBSERVATIONS)
        clamped = clamp_to_context(sampler.sample(200, seed=5), max_model_len=1500)
        assert all(p.total_tokens <= 1500 for p in clamped)
        # The 1024/700 pair overflows 1500 and must have been trimmed.
        assert any(p.input_tokens == 1024 and p.output_tokens == 476 for p in clamped)
