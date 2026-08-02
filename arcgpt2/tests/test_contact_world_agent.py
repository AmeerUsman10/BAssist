from __future__ import annotations

from arcgpt2.contact_world_agent import (
    ContactScoreProvider,
    UniformContactScores,
    build_contact_worlds,
    run_contact_agent,
)
from arcgpt2.mechanics_v2 import ContactMode
from arcgpt2.phase0_hidden_action import Action, Direction
from arcgpt2.primitive_contact_game import generate_contact_game


class WrongConfidentContactScores(UniformContactScores):
    _DIRECTIONS = {
        Action.A1: Direction.UP,
        Action.A2: Direction.RIGHT,
        Action.A3: Direction.DOWN,
        Action.A4: Direction.LEFT,
    }

    def direction_scores(self, spec, history, action, current_grid):
        del spec, history, current_grid
        preferred = self._DIRECTIONS[action]
        return {
            direction: 8.0 if direction is preferred else 0.0
            for direction in Direction
        }

    def contact_scores(self, spec, history, current_grid):
        del spec, history, current_grid
        return {
            mode: 8.0 if mode is ContactMode.BLOCK else 0.0
            for mode in ContactMode
        }


def test_contact_world_posterior_starts_with_all_120_worlds() -> None:
    spec = generate_contact_game(400_001)
    entries = build_contact_worlds(spec, (), None, UniformContactScores())
    assert len(entries) == 24 * len(ContactMode)
    assert sum(entry.probability for entry in entries) == 1.0


def test_uniform_contact_agent_solves_held_out_games() -> None:
    for seed in range(20):
        spec = generate_contact_game(400_100 + seed)
        result = run_contact_agent(
            spec,
            UniformContactScores(),
            max_actions=192,
            planner_depth=2,
        )
        assert result.won
        assert result.levels_completed == len(spec.levels)
        assert result.actions <= 192
        assert result.steps[0].posterior_worlds == 120
        assert min(step.posterior_worlds for step in result.steps) == 1


def test_exact_replay_overcomes_a_confident_wrong_prior() -> None:
    for seed in range(8):
        spec = generate_contact_game(400_200 + seed)
        result = run_contact_agent(
            spec,
            WrongConfidentContactScores(),
            max_actions=192,
            planner_depth=2,
        )
        assert result.won
        assert result.levels_completed == len(spec.levels)
        assert min(step.posterior_worlds for step in result.steps) == 1


def test_agent_uses_multistep_fracture_to_reach_contact_experiment() -> None:
    spec = generate_contact_game(400_300)
    result = run_contact_agent(
        spec,
        UniformContactScores(),
        max_actions=192,
        planner_depth=2,
    )
    assert result.won
    # At least one exploratory action should have zero immediate information
    # but positive future value: moving adjacent to the special cell is useful
    # only because the following contact fractures the five remaining modes.
    exploratory = [
        step
        for step in result.steps
        if step.decision_mode == "fracture"
    ]
    assert exploratory
    assert any(step.information_gain_bits == 0.0 for step in exploratory)
    assert any(step.information_gain_bits > 0.0 for step in exploratory)


def test_adaptive_provider_hooks_receive_every_observation() -> None:
    class RecordingProvider(UniformContactScores):
        def __init__(self):
            self.resets = 0
            self.records = []

        def reset(self):
            self.resets += 1
            self.records.clear()

        def observe(self, record):
            self.records.append(record)

    provider = RecordingProvider()
    spec = generate_contact_game(400_400)
    result = run_contact_agent(spec, provider, max_actions=192, planner_depth=2)
    assert result.won
    assert provider.resets == 1
    assert len(provider.records) == result.actions
