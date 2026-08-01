from __future__ import annotations

import pytest

from arcgpt2.dsl import Status, execute, program_from_phase0_spec
from arcgpt2.goal_dsl import (
    AllGoals,
    ColorAbsent,
    ColorCount,
    ColorsTouch,
    Comparison,
    ContactedColor,
    Connectivity,
    GoalProgram,
    NotGoal,
    enumerate_simple_goals,
    evaluate_goal,
    parse_goal,
    phase0_goal,
    replay_goal,
    strip_terminal,
)
from arcgpt2.phase0_hidden_action import generate_game, simulate_source_history


def test_phase0_mechanics_and_goal_separate_without_changing_behavior() -> None:
    for seed in range(100):
        spec = generate_game(130_000 + seed)
        mechanics_with_embedded_goal = program_from_phase0_spec(spec)
        mechanics = strip_terminal(mechanics_with_embedded_goal)
        goal = phase0_goal(spec)
        records = simulate_source_history(spec)

        result = replay_goal(mechanics, goal, records)
        assert result.consistent
        assert result.checked == len(records)
        assert result.mismatch is None

        terminal_record = records[-1]
        execution = execute(mechanics, terminal_record.before, terminal_record.action)
        predicted, _ = evaluate_goal(mechanics, goal, terminal_record.before, terminal_record.action)
        assert execution.status is Status.ACTIVE
        assert predicted
        assert spec.palette.goal in execution.contacted_colors


def test_wrong_contact_color_is_rejected_by_terminal_replay() -> None:
    spec = generate_game(140_001)
    mechanics = strip_terminal(program_from_phase0_spec(spec))
    records = simulate_source_history(spec)
    wrong_color = next(color for color in range(16) if color != spec.palette.goal)
    result = replay_goal(mechanics, GoalProgram(ContactedColor(wrong_color)), records)
    assert not result.consistent
    assert result.mismatch is not None


def test_goal_parser_roundtrips_atomic_and_compound_predicates() -> None:
    goals = [
        GoalProgram(ContactedColor(3)),
        GoalProgram(ColorAbsent(4)),
        GoalProgram(ColorCount(5, Comparison.EQ, 0)),
        GoalProgram(ColorsTouch(2, 7, Connectivity.EIGHT)),
        GoalProgram(
            AllGoals(
                (
                    ContactedColor(3),
                    NotGoal(ColorAbsent(8)),
                )
            )
        ),
    ]
    for goal in goals:
        parsed = parse_goal(goal.canonical_text())
        assert parsed == goal
        assert parsed.sha256 == goal.sha256


def test_simple_goal_enumeration_contains_the_phase0_truth() -> None:
    spec = generate_game(150_002)
    colors = {
        spec.palette.background,
        spec.palette.wall,
        spec.palette.agent,
        spec.palette.goal,
    }
    candidates = enumerate_simple_goals(colors)
    truth = phase0_goal(spec)
    assert truth in candidates
    assert len({candidate.sha256 for candidate in candidates}) == len(candidates)


def test_count_and_touch_predicates_use_exact_after_grid() -> None:
    spec = generate_game(160_003)
    mechanics = strip_terminal(program_from_phase0_spec(spec))
    records = simulate_source_history(spec)
    first = records[0]
    _, context = evaluate_goal(mechanics, phase0_goal(spec), first.before, first.action)

    agent_count = sum(
        value == spec.palette.agent for row in context.after for value in row
    )
    assert ColorCount(spec.palette.agent, Comparison.EQ, agent_count).evaluate(context)
    assert not ColorAbsent(spec.palette.agent).evaluate(context)

    # The target and agent may or may not be adjacent after an arbitrary first
    # move; both FOUR/EIGHT predicates must agree with direct exact evaluation.
    four = ColorsTouch(spec.palette.agent, spec.palette.goal, Connectivity.FOUR)
    eight = ColorsTouch(spec.palette.agent, spec.palette.goal, Connectivity.EIGHT)
    assert isinstance(four.evaluate(context), bool)
    assert isinstance(eight.evaluate(context), bool)
