from __future__ import annotations

from pathlib import Path

from arcgpt2.phase0_hidden_action import Action, Direction
from arcgpt2.stage02_decomposed import (
    ACTION_CANDIDATES,
    ALL_LABELS,
    MAP_CANDIDATES,
    Stage02Context,
    all_action_mappings,
    build_dataset,
    decision_for_context,
    grid_text,
    infer_mapping_labels,
    parse_grid_text,
    simulate_context,
    task_rows,
    training_sequences,
)


def test_grid_text_round_trip() -> None:
    grid = (
        (0, 0, 3, 0),
        (0, 1, 0, 0),
        (0, 2, 0, 0),
    )
    assert parse_grid_text(grid_text(grid)) == grid


def test_stage02_uses_only_original_single_character_label_alphabet() -> None:
    assert len(ALL_LABELS) == 9
    assert len(set(ALL_LABELS)) == 9
    assert set(MAP_CANDIDATES) == {"N", "E", "S", "W", "?"}
    assert set(ACTION_CANDIDATES) == {"1", "2", "3", "4"}


def test_all_24_action_mappings_are_present() -> None:
    mappings = all_action_mappings()
    assert len(mappings) == 24
    signatures = {
        tuple(mapping[action] for action in Action)
        for mapping in mappings
    }
    assert len(signatures) == 24
    assert all(set(signature) == set(Direction) for signature in signatures)


def test_training_sequences_cover_all_distinct_histories_and_repeat_recovery() -> None:
    sequences = training_sequences()
    assert len(sequences) == 129
    assert tuple() in sequences
    assert (Action.A1, Action.A2, Action.A3, Action.A4) in sequences
    assert (Action.A1, Action.A2, Action.A1) in sequences


def test_empty_history_requires_first_probe() -> None:
    context = simulate_context(
        0,
        tuple(),
        goal_direction=Direction.UP,
        variant_id="empty",
    )
    decision = decision_for_context(context)
    assert set(decision.mapping_labels.values()) == {"?"}
    assert decision.needed_direction == "?"
    assert decision.target_action == "1"
    assert decision.decision_phase == "probe"


def test_full_distinct_probe_history_recovers_mapping_and_navigates() -> None:
    sequence = (Action.A1, Action.A2, Action.A3, Action.A4)
    context = simulate_context(
        7,
        sequence,
        goal_direction=Direction.LEFT,
        variant_id="full",
    )
    labels = infer_mapping_labels(context.records)
    assert "?" not in labels.values()
    decision = decision_for_context(context)
    assert decision.needed_direction == "W"
    expected = next(
        str(index)
        for index, action in enumerate(Action, start=1)
        if labels[action] == "W"
    )
    assert decision.target_action == expected
    assert decision.decision_phase == "navigate"


def test_repeat_recovery_does_not_target_an_observed_action() -> None:
    context = simulate_context(
        2,
        (Action.A1, Action.A2, Action.A1),
        goal_direction=Direction.DOWN,
        variant_id="repeat",
    )
    decision = decision_for_context(context)
    assert decision.mapping_labels[Action.A1] != "?"
    assert decision.mapping_labels[Action.A2] != "?"
    assert decision.target_action == "3"


def test_each_context_generates_seven_single_label_tasks() -> None:
    context = simulate_context(
        3,
        (Action.A1, Action.A3),
        goal_direction=Direction.RIGHT,
        variant_id="rows",
    )
    rows = task_rows(context)
    assert len(rows) == 7
    assert [row["task"] for row in rows].count("mapping") == 4
    assert [row["task"] for row in rows].count("need") == 1
    assert [row["task"] for row in rows].count("compose") == 1
    assert [row["task"] for row in rows].count("direct") == 1
    assert all(row["target"] in row["candidates"] for row in rows)


def test_dataset_splits_are_nonempty_and_reproducible(tmp_path: Path) -> None:
    first = build_dataset(
        tmp_path / "first",
        train_games=1,
        validation_games=1,
        test_games=1,
        train_seed_start=0,
        validation_seed_start=8,
        test_seed_start=20,
    )
    second = build_dataset(
        tmp_path / "second",
        train_games=1,
        validation_games=1,
        test_games=1,
        train_seed_start=0,
        validation_seed_start=8,
        test_seed_start=20,
    )
    assert first["contexts_per_game"] == 129
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["rows"] == 129 * 7
        assert (
            first["splits"][split]["sha256"]
            == second["splits"][split]["sha256"]
        )
