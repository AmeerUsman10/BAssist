from __future__ import annotations

import pytest

from arcgpt2.build_epistemic_dataset import build_examples
from instella_arc.action_training_data import (
    build_action_training_examples,
    manifest,
    target_for_mode,
)


def _row(seed: int, depth: int, action: str):
    return next(
        row
        for row in build_examples(seed)
        if row["probe_count"] == depth and row["query_action"] == action
    )


def test_amnesic_target_is_uniform_even_at_full_evidence_depth() -> None:
    row = _row(950_000, 4, "A1")
    assert target_for_mode(row, "amnesic") == pytest.approx((0.25,) * 4)


def test_shuffled_target_is_one_hot_on_relabelled_truth_at_full_depth() -> None:
    row = _row(950_000, 4, "A1")
    target = target_for_mode(row, "shuffled")
    assert sum(value > 0.0 for value in target) == 1
    assert sum(target) == pytest.approx(1.0)
    original_index = list(row["candidate_words"]).index(row["truth_word"])
    assert target[original_index] == 0.0


def test_curriculum_deduplicates_repeated_amnesic_prompts() -> None:
    examples = build_action_training_examples((950_001,))
    amnesic = [example for example in examples if example.mode == "amnesic"]
    # One no-evidence prompt per query action, rather than one copy at every depth.
    assert len(amnesic) == 4
    assert len({example.prompt_sha256 for example in amnesic}) == 4


def test_curriculum_keeps_intact_and_nonzero_shuffled_evidence() -> None:
    examples = build_action_training_examples((950_002,))
    intact = [example for example in examples if example.mode == "intact"]
    shuffled = [example for example in examples if example.mode == "shuffled"]
    assert len(intact) == 16
    # Depth-zero shuffled prompts are duplicate no-evidence prompts and dedupe.
    assert len(shuffled) == 12
    assert {example.probe_count for example in shuffled} == {1, 2, 4}


def test_manifest_is_stable_and_complete() -> None:
    examples = build_action_training_examples((950_003, 950_004))
    report = manifest(examples)
    assert report["games"] == 2
    assert report["examples"] == 64
    assert report["by_mode"] == {
        "amnesic": 8,
        "intact": 32,
        "shuffled": 24,
    }
    assert len(report["examples_sha256"]) == 64
