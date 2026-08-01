from __future__ import annotations

import math

import pytest

from arcgpt2.dsl import enumerate_phase0_programs, program_from_phase0_spec
from arcgpt2.phase0_hidden_action import Action, HiddenActionGame, generate_game, simulate_source_history
from arcgpt2.posterior import (
    FracturePlanner,
    PlannerConfig,
    WeightedProgram,
    condition_exact,
    expected_information_gain,
    normalize_programs,
    posterior_entropy,
    predictive_branches,
)


def _uniform_candidates(seed: int):
    spec = generate_game(seed)
    programs = enumerate_phase0_programs(spec)
    return spec, tuple(WeightedProgram(program) for program in programs)


def test_uniform_phase0_posterior_has_log2_24_entropy() -> None:
    _, candidates = _uniform_candidates(4201)
    entries = normalize_programs(candidates)
    assert len(entries) == 24
    assert sum(entry.probability for entry in entries) == pytest.approx(1.0)
    assert posterior_entropy(entries) == pytest.approx(math.log2(24))


def test_duplicate_program_text_does_not_receive_duplicate_identity() -> None:
    spec = generate_game(4202)
    truth = program_from_phase0_spec(spec)
    entries = normalize_programs(
        (
            WeightedProgram(truth, log_prior=0.0, source="sample-1"),
            WeightedProgram(truth, log_prior=0.0, source="sample-2"),
        )
    )
    assert len(entries) == 1
    assert entries[0].probability == pytest.approx(1.0)
    assert entries[0].source == "sample-1+sample-2"


def test_exact_conditioning_recovers_the_hidden_mapping() -> None:
    spec, candidates = _uniform_candidates(4203)
    records = simulate_source_history(spec)
    expected_counts = (6, 2, 1, 1)
    for count, history_length in zip(expected_counts, range(1, 5), strict=True):
        entries = condition_exact(candidates, records[:history_length])
        assert len(entries) == count
    assert entries[0].program == program_from_phase0_spec(spec)


def test_initial_action_partitions_have_two_bits_of_information() -> None:
    spec, candidates = _uniform_candidates(4204)
    entries = normalize_programs(candidates)
    grid = HiddenActionGame(spec).frame
    for action in Action:
        branches = predictive_branches(entries, grid, action)
        assert len(branches) == 4
        assert sorted(branch.probability for branch in branches) == pytest.approx([0.25] * 4)
        assert expected_information_gain(entries, branches) == pytest.approx(2.0)


def test_fracture_planner_prefers_an_unobserved_action_after_one_probe() -> None:
    spec, candidates = _uniform_candidates(4205)
    game = HiddenActionGame(spec)
    first_record = game.step(Action.A1)
    entries = condition_exact(candidates, (first_record,))
    planner = FracturePlanner(
        entries,
        PlannerConfig(
            depth=1,
            terminal_reward=0.0,
            information_weight=1.0,
            action_cost=0.0,
            no_change_penalty=0.0,
        ),
    )
    evaluations = {item.action: item for item in planner.evaluate_actions(game.frame)}
    assert evaluations[Action.A1].information_gain_bits == pytest.approx(0.0)
    for action in (Action.A2, Action.A3, Action.A4):
        assert evaluations[action].information_gain_bits > 0.0
    assert planner.choose_action(game.frame).action in {Action.A2, Action.A3, Action.A4}
