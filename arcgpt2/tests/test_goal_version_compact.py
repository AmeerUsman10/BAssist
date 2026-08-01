from __future__ import annotations

from arcgpt2.build_goal_version_compact import compact_candidate_goals
from arcgpt2.goal_dsl import ContactedColor, phase0_goal
from arcgpt2.phase0_hidden_action import generate_game


def test_compact_goal_family_is_bounded_and_contains_truth() -> None:
    spec = generate_game(88_001)
    candidates = compact_candidate_goals(spec)
    assert len(candidates) == 8
    assert phase0_goal(spec) in candidates
    assert sum(isinstance(goal.predicate, ContactedColor) for goal in candidates) == 4
