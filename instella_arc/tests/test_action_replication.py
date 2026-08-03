from __future__ import annotations

import math

import pytest

from arcgpt2.build_epistemic_dataset import build_examples
from instella_arc.action_replication import (
    calibrated_probabilities,
    mode_allowed_words,
    mode_truth_word,
    shuffled_observed_mapping,
)


def _full_row(seed: int = 940_000, query_action: str = "A1"):
    return next(
        row
        for row in build_examples(seed)
        if row["probe_count"] == 4 and row["query_action"] == query_action
    )


def test_amnesic_mode_preserves_full_version_space() -> None:
    row = _full_row()
    assert mode_allowed_words(row, "amnesic") == (
        "north",
        "south",
        "west",
        "east",
    )
    assert mode_truth_word(row, "amnesic") is None


def test_shuffled_mode_uses_the_relabelled_world_truth() -> None:
    row = _full_row()
    displayed = shuffled_observed_mapping(row)
    query = row["query_action"]
    assert mode_truth_word(row, "shuffled") == displayed[query]
    assert mode_allowed_words(row, "shuffled") == (displayed[query],)
    # All four directions are a permutation, so rotating action labels associates
    # each query label with a different underlying action and a different truth.
    assert displayed[query] != row["truth_word"]


def test_intact_mode_uses_original_set_valued_target() -> None:
    row = _full_row()
    assert mode_allowed_words(row, "intact") == tuple(row["allowed_words"])
    assert mode_truth_word(row, "intact") == row["truth_word"]


def test_calibration_removes_stable_candidate_lexical_priors() -> None:
    lexical_prior = (-10.0, -5.0, -7.0, -12.0)
    evidence_effect = (2.0, 0.0, 0.0, 0.0)
    evidence = tuple(
        prior + effect
        for prior, effect in zip(lexical_prior, evidence_effect, strict=True)
    )
    probabilities = calibrated_probabilities(evidence, lexical_prior)
    expected_denominator = math.exp(2.0) + 3.0
    assert probabilities == pytest.approx(
        (
            math.exp(2.0) / expected_denominator,
            1.0 / expected_denominator,
            1.0 / expected_denominator,
            1.0 / expected_denominator,
        )
    )


def test_calibration_rejects_mismatched_candidate_vectors() -> None:
    with pytest.raises(ValueError):
        calibrated_probabilities((1.0, 2.0), (1.0,))
