from __future__ import annotations

import json

from arcgpt2.build_factorized_dataset import (
    build_control_prompt,
    build_dataset,
    build_examples,
    probe_records,
)
from arcgpt2.natural_protocol import (
    answer_text,
    changed_cells,
    grid_text,
    transition_text,
)
from arcgpt2.phase0_hidden_action import Action


def test_natural_grid_and_transition_are_exact() -> None:
    _, initial, records = probe_records(2201)
    text = grid_text(initial)
    assert f"The grid has {len(initial)} rows and {len(initial[0])} columns." in text
    for row_index, row in enumerate(initial):
        assert f"Row {row_index}: {' '.join(str(value) for value in row)}." in text

    for record in records:
        changes = changed_cells(record)
        assert len(changes) == 2
        rendered = transition_text(record)
        for row, column, old, new in changes:
            assert (
                f"Row {row}, column {column} changed from color {old} to color {new}."
                in rendered
            )


def test_each_game_produces_four_natural_completion_examples() -> None:
    rows = build_examples(4432)
    assert len(rows) == 4
    assert {row["query_action"] for row in rows} == {action.value for action in Action}
    for row in rows:
        assert str(row["context"]).endswith("ANSWER:")
        assert str(row["target"]).startswith(" ")
        assert str(row["target"]).strip() in {"north", "south", "west", "east"}
        assert str(row["target"]) == answer_text(str(row["direction"]))


def test_information_controls_are_distinct() -> None:
    seed = 8877
    action = Action.A3
    full = build_control_prompt(seed, action, "full")
    amnesic = build_control_prompt(seed, action, "amnesic")
    shuffled = build_control_prompt(seed, action, "shuffled")
    assert full != amnesic
    assert full != shuffled
    assert amnesic != shuffled
    assert full.count("Exactly 2 grid cells changed") == 4
    assert shuffled.count("Exactly 2 grid cells changed") == 4
    assert "No action-outcome observations are available." in amnesic
    assert full.endswith("ANSWER:")
    assert amnesic.endswith("ANSWER:")
    assert shuffled.endswith("ANSWER:")


def test_factorized_dataset_is_reproducible_and_game_disjoint(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_dataset(
        first_dir,
        train_games=4,
        validation_games=2,
        test_games=3,
        seed_base=900,
    )
    second = build_dataset(
        second_dir,
        train_games=4,
        validation_games=2,
        test_games=3,
        seed_base=900,
    )
    assert first == second
    assert first["splits"]["train"]["examples"] == 16
    assert first["splits"]["validation"]["examples"] == 8
    assert first["splits"]["test"]["examples"] == 12

    split_seeds = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (first_dir / f"{split}.jsonl").read_text().splitlines()
        ]
        split_seeds[split] = {row["game_seed"] for row in rows}
        assert len(rows) == first["splits"][split]["examples"]
    assert not split_seeds["train"].intersection(split_seeds["validation"])
    assert not split_seeds["train"].intersection(split_seeds["test"])
    assert not split_seeds["validation"].intersection(split_seeds["test"])
