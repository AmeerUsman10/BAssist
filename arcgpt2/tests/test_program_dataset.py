from __future__ import annotations

import json

from arcgpt2.build_program_dataset import (
    build_control_context,
    build_dataset,
    build_example,
)
from arcgpt2.dsl import parse_program, program_from_phase0_spec
from arcgpt2.mapping_target import (
    compact_mapping,
    expand_mapping,
    parse_compact_mapping,
)
from arcgpt2.phase0_hidden_action import generate_game


def test_example_contains_an_exact_compact_mapping_and_full_program() -> None:
    for seed in range(100):
        game_seed = 80_000 + seed
        example = build_example(game_seed)
        expected = program_from_phase0_spec(generate_game(game_seed))
        mapping = parse_compact_mapping(str(example["target"]))
        full_program = parse_program(str(example["full_program"]))
        assert compact_mapping(expected) == example["target"]
        assert full_program == expected
        assert expand_mapping(mapping, expected) == expected
        assert example["probe_records"] == 4
        assert example["post_probe_level"] == 0
        assert example["post_probe_step"] == 4
        assert str(example["context"]).endswith("<INDUCE_MAPPING>")


def test_controls_are_distinct_and_preserve_the_query() -> None:
    seed = 991
    full = build_control_context(seed, "full")
    amnesic = build_control_context(seed, "amnesic")
    shuffled = build_control_context(seed, "shuffled")
    assert full != amnesic
    assert full != shuffled
    assert amnesic != shuffled
    assert full.endswith("<INDUCE_MAPPING>")
    assert amnesic.endswith("<INDUCE_MAPPING>")
    assert shuffled.endswith("<INDUCE_MAPPING>")
    assert full.count("<OUTCOME>") == 4
    assert amnesic.count("<OUTCOME>") == 0
    assert shuffled.count("<OUTCOME>") == 4


def test_dataset_splits_and_hashes_are_reproducible(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_dataset(
        first_dir,
        train_games=5,
        validation_games=3,
        test_games=4,
        seed_base=1234,
    )
    second = build_dataset(
        second_dir,
        train_games=5,
        validation_games=3,
        test_games=4,
        seed_base=1234,
    )
    assert first == second
    assert first["schema"] == "arcgpt2.program_induction.phase0.v2"
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]

    train_rows = [json.loads(line) for line in (first_dir / "train.jsonl").read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in (first_dir / "validation.jsonl").read_text().splitlines()
    ]
    test_rows = [json.loads(line) for line in (first_dir / "test.jsonl").read_text().splitlines()]
    train_seeds = {row["game_seed"] for row in train_rows}
    validation_seeds = {row["game_seed"] for row in validation_rows}
    test_seeds = {row["game_seed"] for row in test_rows}
    assert not train_seeds.intersection(validation_seeds)
    assert not train_seeds.intersection(test_seeds)
    assert not validation_seeds.intersection(test_seeds)
