"""Numerically stable belief-state utilities for the GPT-2-only agent.

This module contains no learned model and no semantic ARC heuristics. It only
normalizes log weights, measures uncertainty, and resamples generic particles.
The particles themselves are alternate per-game states of the same GPT-2
checkpoint, not separate pretrained models.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


class BeliefMathError(ValueError):
    """Raised when a probability or particle operation is malformed."""


def normalize_log_weights(log_weights: Sequence[float]) -> tuple[float, ...]:
    """Convert finite log weights into a normalized probability vector."""

    if not log_weights:
        raise BeliefMathError("at least one log weight is required")
    values = tuple(float(value) for value in log_weights)
    if any(math.isnan(value) for value in values):
        raise BeliefMathError("log weights may not contain NaN")
    maximum = max(values)
    if maximum == -math.inf:
        raise BeliefMathError("at least one particle must have non-zero mass")
    shifted = tuple(math.exp(value - maximum) if value != -math.inf else 0.0 for value in values)
    total = sum(shifted)
    if not math.isfinite(total) or total <= 0.0:
        raise BeliefMathError("log weights could not be normalized")
    probabilities = tuple(value / total for value in shifted)
    # Force an exact unit sum without disturbing ordering. This avoids slow
    # drift when a posterior is updated many times in one interactive game.
    correction = 1.0 - sum(probabilities)
    mutable = list(probabilities)
    mutable[max(range(len(mutable)), key=mutable.__getitem__)] += correction
    return tuple(mutable)


def entropy_bits(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy in bits after validating normalization."""

    values = _validate_probabilities(probabilities)
    return -sum(value * math.log2(value) for value in values if value > 0.0)


def effective_sample_size(probabilities: Sequence[float]) -> float:
    """Return particle-filter effective sample size in ``[1, n]``."""

    values = _validate_probabilities(probabilities)
    denominator = sum(value * value for value in values)
    if denominator <= 0.0:
        raise BeliefMathError("effective sample size has a zero denominator")
    return 1.0 / denominator


def bayes_update(
    prior_log_weights: Sequence[float],
    log_likelihoods: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Apply one exact log-space Bayesian weight update.

    Returns ``(posterior_log_weights, posterior_probabilities)``. The returned
    log weights are normalized so their log-sum-exp equals zero, making them
    safe to reuse in long episodes.
    """

    if len(prior_log_weights) != len(log_likelihoods):
        raise BeliefMathError("priors and likelihoods must have the same length")
    if not prior_log_weights:
        raise BeliefMathError("at least one particle is required")
    unnormalized = tuple(
        float(prior) + float(likelihood)
        for prior, likelihood in zip(prior_log_weights, log_likelihoods, strict=True)
    )
    probabilities = normalize_log_weights(unnormalized)
    normalized_logs = tuple(
        math.log(probability) if probability > 0.0 else -math.inf
        for probability in probabilities
    )
    return normalized_logs, probabilities


def systematic_resample(
    probabilities: Sequence[float],
    *,
    count: int | None = None,
    seed: int = 0,
) -> tuple[int, ...]:
    """Return deterministic systematic-resampling ancestor indices.

    The method is unbiased over a random seed and has lower variance than
    independent categorical draws. It does not mutate or interpret particles.
    """

    values = _validate_probabilities(probabilities)
    output_count = len(values) if count is None else int(count)
    if output_count < 1:
        raise BeliefMathError("resample count must be positive")

    rng = random.Random(seed)
    start = rng.random() / output_count
    thresholds = [start + index / output_count for index in range(output_count)]
    ancestors: list[int] = []
    cumulative = values[0]
    source_index = 0
    for threshold in thresholds:
        while threshold > cumulative and source_index + 1 < len(values):
            source_index += 1
            cumulative += values[source_index]
        ancestors.append(source_index)
    return tuple(ancestors)


def should_resample(probabilities: Sequence[float], threshold_fraction: float = 0.5) -> bool:
    """Decide whether posterior degeneracy crosses an ESS threshold."""

    values = _validate_probabilities(probabilities)
    fraction = float(threshold_fraction)
    if not 0.0 < fraction <= 1.0:
        raise BeliefMathError("threshold_fraction must lie in (0, 1]")
    return effective_sample_size(values) < fraction * len(values)


def mixture_log_probability(
    log_component_probabilities: Sequence[float],
    component_weights: Sequence[float],
) -> float:
    """Return the log probability under a weighted particle mixture."""

    if len(log_component_probabilities) != len(component_weights):
        raise BeliefMathError("component scores and weights must have the same length")
    weights = _validate_probabilities(component_weights)
    terms = [
        math.log(weight) + float(score)
        for weight, score in zip(weights, log_component_probabilities, strict=True)
        if weight > 0.0 and score != -math.inf
    ]
    if not terms:
        return -math.inf
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(term - maximum) for term in terms))


def _validate_probabilities(probabilities: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise BeliefMathError("at least one probability is required")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise BeliefMathError("probabilities must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise BeliefMathError("probabilities must sum to one")
    return values
