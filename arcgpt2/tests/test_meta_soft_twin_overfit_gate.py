from __future__ import annotations

from copy import deepcopy

import pytest

from arcgpt2.meta_soft_binding import transition_prompt
from arcgpt2.meta_soft_twin_overfit import apply_gate, build_quartet
from arcgpt2.phase0_hidden_action import Action, Direction


@pytest.mark.parametrize("action", tuple(Action))
def test_overfit_quartet_changes_only_the_hidden_direction(action: Action) -> None:
    quartet = build_quartet(950_001, action)

    assert len(quartet) == 4
    assert {episode.direction for episode in quartet} == set(Direction)
    assert {episode.action for episode in quartet} == {action}
    assert {episode.record.action for episode in quartet} == {action}
    assert {episode.record.step_index for episode in quartet} == {0}
    assert len({episode.record.before for episode in quartet}) == 1
    assert len({transition_prompt(episode.record) for episode in quartet}) == 1
    assert len({episode.record.after for episode in quartet}) == 4


def _passing_metrics() -> dict[str, object]:
    """Return a complete quartet_v1 record at each inclusive boundary."""

    return {
        "attention_impl": "eager",
        "invariants": {
            "unique_support_prompt_count": 1,
            "unique_query_prompt_count": 1,
            "unique_outcome_count": 4,
            "max_preupdate_logit_delta": 1e-6,
            "deterministic_replay_max_abs_diff": 1e-5,
        },
        "no_evidence": {
            "max_abs_uniform_deviation": 0.05,
            "entropy_bits": 1.95,
            "max_probability": 0.30,
        },
        "intact": {
            "accuracy": 0.95,
            "mean_truth_probability": 0.80,
            "min_truth_probability": 0.70,
            "min_correct_vs_other_world_margin": 0.50,
        },
        "adaptation": {
            "accuracy_gain": 0.70,
            "mean_truth_probability_gain": 0.55,
        },
        "corrupted": {
            "target_accuracy": 0.95,
            "mean_target_probability": 0.80,
            "min_target_probability": 0.70,
            "original_truth_accuracy": 0.05,
            "mean_original_truth_probability": 0.10,
            "max_original_truth_probability": 0.20,
        },
        "gradients": {
            "all_finite": True,
            "min_norm": 1e-8,
            "min_pairwise_relative_distance": 1e-3,
            "min_pairwise_update_l2": 1e-8,
        },
        "training": {"all_finite": True},
    }


def test_overfit_gate_accepts_complete_metrics_at_the_boundaries() -> None:
    result = apply_gate(_passing_metrics())

    assert result["name"] == "quartet_v1"
    assert result["passed"] is True
    assert result["checks"]
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("invariants", "unique_support_prompt_count"), 2),
        (("invariants", "unique_query_prompt_count"), 2),
        (("invariants", "unique_outcome_count"), 3),
        (("invariants", "max_preupdate_logit_delta"), 1.001e-6),
        (("no_evidence", "max_abs_uniform_deviation"), 0.05001),
        (("no_evidence", "entropy_bits"), 1.949),
        (("intact", "accuracy"), 0.949),
        (("intact", "mean_truth_probability"), 0.799),
        (("intact", "min_truth_probability"), 0.699),
        (("adaptation", "accuracy_gain"), 0.699),
        (("adaptation", "mean_truth_probability_gain"), 0.549),
        (("corrupted", "target_accuracy"), 0.949),
        (("corrupted", "mean_target_probability"), 0.799),
        (("corrupted", "min_target_probability"), 0.699),
        (("corrupted", "original_truth_accuracy"), 0.051),
        (("corrupted", "mean_original_truth_probability"), 0.101),
        (("corrupted", "max_original_truth_probability"), 0.201),
        (("gradients", "all_finite"), False),
        (("gradients", "min_norm"), 0.0),
        (("gradients", "min_pairwise_relative_distance"), 9.99e-4),
        (("gradients", "min_pairwise_update_l2"), 0.0),
        (("invariants", "deterministic_replay_max_abs_diff"), 1.001e-5),
        (("training", "all_finite"), False),
        (("attention_impl",), "sdpa"),
    ),
)
def test_overfit_gate_rejects_each_failed_requirement(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    metrics = deepcopy(_passing_metrics())
    target = metrics
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment, index]
    target[path[-1]] = bad_value  # type: ignore[index]

    result = apply_gate(metrics)

    assert result["passed"] is False
    assert not all(result["checks"].values())
