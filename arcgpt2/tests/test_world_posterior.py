from __future__ import annotations

import math

import pytest

from arcgpt2.dsl import enumerate_phase0_programs, program_from_phase0_spec
from arcgpt2.goal_dsl import ContactedColor, GoalProgram, phase0_goal
from arcgpt2.phase0_hidden_action import HiddenActionGame, generate_game, simulate_source_history
from arcgpt2.world_posterior import (
    WorldFracturePlanner,
    WorldHypothesis,
    WorldPlannerConfig,
    condition_worlds_exact,
    normalize_worlds,
    world_branches,
    world_entropy,
    world_information_gain,
)


def _candidate_worlds(seed: int):
    spec = generate_game(seed)
    mechanics = enumerate_phase0_programs(spec)
    goal_colors = {
        spec.palette.background,
        spec.palette.wall,
        spec.palette.agent,
        spec.palette.goal,
    }
    hypotheses = tuple(
        WorldHypothesis(program, GoalProgram(ContactedColor(color)))
        for program in mechanics
        for color in goal_colors
    )
    return spec, hypotheses


def test_joint_world_posterior_normalizes_and_deduplicates() -> None:
    spec = generate_game(501)
    truth = WorldHypothesis(program_from_phase0_spec(spec), phase0_goal(spec))
    entries = normalize_worlds((truth, truth))
    assert len(entries) == 1
    assert entries[0].probability == pytest.approx(1.0)


def test_exact_history_recovers_phase0_mechanics_and_goal() -> None:
    spec, hypotheses = _candidate_worlds(502)
    records = simulate_source_history(spec)
    entries = condition_worlds_exact(hypotheses, records)
    assert len(entries) == 1
    assert entries[0].mechanics == program_from_phase0_spec(spec)
    assert entries[0].goal == phase0_goal(spec)


def test_nonterminal_probe_history_reduces_mechanics_and_rejects_background_goal() -> None:
    spec, hypotheses = _candidate_worlds(503)
    records = simulate_source_history(spec)
    entries = condition_worlds_exact(hypotheses, records[:1])
    assert len(entries) < len(hypotheses)
    assert all(
        entry.goal != GoalProgram(ContactedColor(spec.palette.background))
        for entry in entries
    )
    assert world_entropy(entries) <= math.log2(len(hypotheses))


def test_joint_predictive_branches_have_valid_information_gain() -> None:
    spec, hypotheses = _candidate_worlds(504)
    entries = normalize_worlds(hypotheses)
    game = HiddenActionGame(spec)
    branches = world_branches(entries, game.frame, next(iter(spec.action_to_direction)))
    assert sum(branch.probability for branch in branches) == pytest.approx(1.0)
    gain = world_information_gain(entries, branches)
    assert gain >= 0.0
    assert gain <= world_entropy(entries) + 1e-9


def test_joint_planner_returns_a_legal_action_and_exposes_branches() -> None:
    spec, hypotheses = _candidate_worlds(505)
    entries = normalize_worlds(hypotheses)
    planner = WorldFracturePlanner(
        entries,
        WorldPlannerConfig(
            depth=2,
            terminal_reward=1.0,
            information_weight=0.2,
            action_cost=0.01,
            no_change_penalty=0.01,
        ),
    )
    choice = planner.choose_action(HiddenActionGame(spec).frame)
    assert choice.action in spec.action_to_direction
    assert choice.branches
    assert choice.information_gain_bits >= 0.0
