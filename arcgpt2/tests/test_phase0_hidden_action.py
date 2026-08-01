from __future__ import annotations

from arcgpt2.phase0_hidden_action import (
    Action,
    Direction,
    HiddenActionGame,
    SourceLearner,
    build_decision_examples,
    game_summary,
    generate_game,
    simulate_source_history,
)


def test_action_mapping_is_a_permutation() -> None:
    for seed in range(50):
        spec = generate_game(seed)
        assert set(spec.action_to_direction) == set(Action)
        assert set(spec.action_to_direction.values()) == set(Direction)


def test_first_level_is_safe_identification_arena() -> None:
    for seed in range(100):
        spec = generate_game(seed)
        level = spec.levels[0]
        assert not level.walls
        distance = abs(level.goal[0] - level.start[0]) + abs(level.goal[1] - level.start[1])
        assert distance >= 3

        game = HiddenActionGame(spec)
        learner = SourceLearner(spec)
        for _ in range(4):
            action = learner.choose(game)
            record = game.step(action)
            assert record.moved
            assert record.status == "ACTIVE"
            learner.observe(record)
        assert set(learner.direction_to_action) == set(Direction)
        assert game.agent == level.start


def test_source_learner_completes_many_unseen_games() -> None:
    for seed in range(200):
        spec = generate_game(10_000 + seed)
        records = simulate_source_history(spec)
        assert records
        assert records[-1].status == "GAME_WIN"
        assert len(records) <= 128


def test_decision_examples_match_source_history() -> None:
    spec = generate_game(918273)
    records = simulate_source_history(spec)
    examples = build_decision_examples(spec)
    assert len(examples) == len(records)
    assert [example.target for example in examples] == [f"<{record.action.value}>" for record in records]
    assert all(example.context.endswith("<DECIDE>") for example in examples)
    assert any(example.level_index > 0 for example in examples)


def test_summary_counts_actions_by_level() -> None:
    summary = game_summary(generate_game(42))
    actions_per_level = summary["actions_per_level"]
    assert isinstance(actions_per_level, dict)
    assert sum(actions_per_level.values()) == summary["actions"]
    assert set(actions_per_level) == {0, 1, 2}
