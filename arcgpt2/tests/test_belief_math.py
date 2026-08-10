from __future__ import annotations

import math

import pytest

from arcgpt2.belief_math import (
    BeliefMathError,
    bayes_update,
    effective_sample_size,
    entropy_bits,
    mixture_log_probability,
    normalize_log_weights,
    should_resample,
    systematic_resample,
)


def test_log_weight_normalization_is_stable_for_large_magnitudes() -> None:
    probabilities = normalize_log_weights((10_000.0, 9_999.0, -math.inf))
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert probabilities[2] == 0.0
    assert probabilities[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_uniform_and_degenerate_particle_statistics() -> None:
    uniform = (0.25, 0.25, 0.25, 0.25)
    assert entropy_bits(uniform) == pytest.approx(2.0)
    assert effective_sample_size(uniform) == pytest.approx(4.0)
    assert not should_resample(uniform)

    concentrated = (0.97, 0.01, 0.01, 0.01)
    assert entropy_bits(concentrated) < 0.3
    assert effective_sample_size(concentrated) < 1.1
    assert should_resample(concentrated)


def test_bayes_update_multiplies_prior_and_likelihood_in_log_space() -> None:
    prior = tuple(math.log(value) for value in (0.5, 0.3, 0.2))
    likelihood = tuple(math.log(value) for value in (0.1, 0.6, 0.3))
    logs, posterior = bayes_update(prior, likelihood)
    expected_raw = (0.05, 0.18, 0.06)
    expected = tuple(value / sum(expected_raw) for value in expected_raw)
    assert posterior == pytest.approx(expected)
    assert normalize_log_weights(logs) == pytest.approx(expected)


def test_systematic_resampling_is_reproducible_and_respects_mass() -> None:
    probabilities = (0.7, 0.2, 0.1)
    first = systematic_resample(probabilities, count=100, seed=19)
    second = systematic_resample(probabilities, count=100, seed=19)
    assert first == second
    counts = [first.count(index) for index in range(3)]
    assert counts == [70, 20, 10]


def test_mixture_log_probability_matches_direct_probability_sum() -> None:
    scores = tuple(math.log(value) for value in (0.2, 0.6, 0.1))
    weights = (0.5, 0.25, 0.25)
    expected = math.log(sum(weight * math.exp(score) for weight, score in zip(weights, scores)))
    assert mixture_log_probability(scores, weights) == pytest.approx(expected)


def test_invalid_probability_inputs_fail_loudly() -> None:
    with pytest.raises(BeliefMathError):
        normalize_log_weights(())
    with pytest.raises(BeliefMathError):
        normalize_log_weights((-math.inf, -math.inf))
    with pytest.raises(BeliefMathError):
        entropy_bits((0.4, 0.4))
    with pytest.raises(BeliefMathError):
        effective_sample_size((-0.1, 1.1))
    with pytest.raises(BeliefMathError):
        bayes_update((0.0,), (0.0, 0.0))
    with pytest.raises(BeliefMathError):
        systematic_resample((1.0,), count=0)
    with pytest.raises(BeliefMathError):
        should_resample((1.0,), threshold_fraction=0.0)
