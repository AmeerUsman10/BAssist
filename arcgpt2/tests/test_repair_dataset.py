from __future__ import annotations

from arcgpt2.build_repair_dataset import (
    build_control_context,
    build_dataset,
    build_examples,
    one_swap_programs,
)
from arcgpt2.dsl import program_from_phase0_spec, replay
from arcgpt2.mapping_target import parse_compact_mapping, program_mapping
from arcgpt2.phase0_hidden_action import Action, generate_game, simulate_source_history


def test_one_swap_candidates_are_near_wrong_and_unique() -> None:
    truth = program_from_phase0_spec(generate_game(1701))
    candidates = one_swap_programs(truth)
    assert len(candidates) == 6
    assert len({candidate.sha256 for candidate in candidates}) == 6
    truth_mapping = program_mapping(truth)
    for candidate in candidates:
        candidate_mapping = program_mapping(candidate)
        differences = [
            action
            for action in Action
            if candidate_mapping[action] != truth_mapping[action]
        ]
        assert len(differences) == 2


def test_every_repair_example_contains_a_real_replay_counterexample() -> None:
    for seed in range(50):
        game_seed = 90_000 + seed
        spec = generate_game(game_seed)
        truth = program_from_phase0_spec(spec)
        records = simulate_source_history(spec)
        examples = build_examples(game_seed)
        wrong_candidates = one_swap_programs(truth)
        assert len(examples) == len(wrong_candidates) == 6

        for example, wrong in zip(examples, wrong_candidates, strict=True):
            result = replay(wrong, records)
            assert not result.consistent
            assert result.mismatch is not None
            assert example["mismatch_index"] == result.mismatch.index
            assert example["differing_cells"] == result.mismatch.differing_cells
            assert parse_compact_mapping(str(example["target"])) == program_mapping(truth)
            assert str(example["context"]).endswith("<REPAIR_MAPPING>")
            assert "<PREDICTED>" in str(example["context"])
            assert "<OBSERVED>" in str(example["context"])


def test_repair_controls_remove_or_misassign_the_counterexample() -> None:
    seed = 123_456
    full = build_control_context(seed, 0, "full")
    missing = build_control_context(seed, 0, "missing")
    irrelevant = build_control_context(seed, 0, "irrelevant")
    assert full != missing
    assert full != irrelevant
    assert missing != irrelevant
    assert "<COUNTEREXAMPLE_FULL>" in full
    assert "<PREDICTED>" in full and "<OBSERVED>" in full
    assert "<COUNTEREXAMPLE_MISSING>" in missing
    assert "<PREDICTED>" not in missing and "<OBSERVED>" not in missing
    assert "<COUNTEREXAMPLE_IRRELEVANT>" in irrelevant
    assert all(text.endswith("<REPAIR_MAPPING>") for text in (full, missing, irrelevant))


def test_repair_dataset_is_reproducible(tmp_path) -> None:
    first = build_dataset(
        tmp_path / "first",
        train_games=3,
        validation_games=2,
        test_games=2,
        seed_base=700,
    )
    second = build_dataset(
        tmp_path / "second",
        train_games=3,
        validation_games=2,
        test_games=2,
        seed_base=700,
    )
    assert first == second
    assert first["splits"]["train"]["examples"] == 18
    assert first["splits"]["validation"]["examples"] == 12
    assert first["splits"]["test"]["examples"] == 12
