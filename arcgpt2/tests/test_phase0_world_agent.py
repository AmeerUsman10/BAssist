from __future__ import annotations

from arcgpt2.phase0_hidden_action import generate_game
from arcgpt2.phase0_world_agent import UniformMappingScores, run_agent


def test_uniform_epistemic_agent_solves_held_out_phase0_games() -> None:
    for seed in range(12):
        spec = generate_game(120_000 + seed)
        result = run_agent(spec, UniformMappingScores(), max_actions=96)
        assert result.won
        assert result.levels_completed == len(spec.levels)
        assert result.actions <= 96


def test_world_posterior_contracts_before_planning() -> None:
    spec = generate_game(120_100)
    result = run_agent(spec, UniformMappingScores(), max_actions=96)
    worlds = [step.posterior_worlds for step in result.steps]
    modes = [step.decision_mode for step in result.steps]
    assert worlds[0] == 24
    assert min(worlds) == 1
    first_plan = modes.index("plan")
    assert all(mode == "fracture" for mode in modes[:first_plan])
    assert worlds[first_plan] == 1
    assert result.steps[0].information_gain_bits > 0.0


def test_adaptive_provider_is_reset_and_observes_every_real_transition() -> None:
    class RecordingProvider(UniformMappingScores):
        def __init__(self) -> None:
            self.reset_count = 0
            self.records = []

        def reset(self) -> None:
            self.reset_count += 1
            self.records.clear()

        def observe(self, record) -> None:
            self.records.append(record)

    provider = RecordingProvider()
    spec = generate_game(120_101)
    result = run_agent(spec, provider, max_actions=96)
    assert result.won
    assert provider.reset_count == 1
    assert len(provider.records) == result.actions
    assert [record.action for record in provider.records] == [
        step.action for step in result.steps
    ]
