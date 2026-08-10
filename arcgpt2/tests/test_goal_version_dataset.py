from __future__ import annotations

import json
import math

from arcgpt2.build_goal_version_dataset import (
    build_dataset,
    build_examples,
    candidate_goals,
    consistent_goal_indices,
    evidence_prefixes,
)
from arcgpt2.dsl import program_from_phase0_spec
from arcgpt2.goal_dsl import phase0_goal, replay_goal
from arcgpt2.phase0_hidden_action import generate_game, simulate_source_history


def test_prefix_schedule_contains_uncertain_and_terminal_boundaries() -> None:
    spec = generate_game(44_001)
    records = simulate_source_history(spec)
    prefixes = evidence_prefixes(records)
    assert prefixes[0] == 0
    assert prefixes[-1] == len(records)
    terminal_positions = [
        index
        for index, record in enumerate(records, start=1)
        if record.status in {"LEVEL_WIN", "GAME_WIN"}
    ]
    for position in terminal_positions:
        assert position in prefixes
        assert position - 1 in prefixes


def test_ground_truth_goal_is_never_eliminated() -> None:
    for seed in range(50):
        spec = generate_game(44_100 + seed)
        mechanics = program_from_phase0_spec(spec)
        truth = phase0_goal(spec)
        goals = candidate_goals(spec)
        truth_index = goals.index(truth)
        records = simulate_source_history(spec)
        for prefix in evidence_prefixes(records):
            consistent = consistent_goal_indices(mechanics, goals, records[:prefix])
            assert truth_index in consistent
            assert replay_goal(mechanics, truth, records[:prefix]).consistent


def test_version_space_is_set_valued_and_contracts_with_terminal_evidence() -> None:
    rows = build_examples(44_777)
    assert rows
    counts = [int(row["consistent_goal_count"]) for row in rows]
    assert counts[0] == len(rows[0]["candidate_texts"])
    assert min(counts) < counts[0]
    assert counts[-1] <= counts[0]

    for row in rows:
        probabilities = row["target_distribution"]
        assert math.isclose(sum(probabilities), 1.0)
        nonzero = {index for index, value in enumerate(probabilities) if value > 0.0}
        assert nonzero == set(row["consistent_indices"])
        assert row["truth_index"] in nonzero
        assert {
            value for value in probabilities if value > 0.0
        } == {1.0 / len(nonzero)}


def test_information_controls_change_only_terminal_evidence_surface() -> None:
    rows = build_examples(44_778)
    for row in rows:
        full = str(row["context"])
        amnesic = str(row["amnesic_context"])
        statusless = str(row["statusless_context"])
        shuffled = str(row["shuffled_status_context"])
        assert full.endswith("ANSWER:")
        assert amnesic.endswith("ANSWER:")
        assert statusless.endswith("ANSWER:")
        assert shuffled.endswith("ANSWER:")
        if int(row["prefix_length"]) == 0:
            assert full == amnesic
        else:
            assert full != amnesic
            assert "The terminal report is hidden." in statusless
        # Candidate lists and targets are stored once, independent of controls.
        assert len(row["candidate_texts"]) == len(row["target_distribution"])
        assert len(row["candidate_programs"]) == len(row["candidate_texts"])
        assert len(row["candidate_hashes"]) == len(row["candidate_texts"])


def test_dataset_is_reproducible_and_split_disjoint(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = build_dataset(
        left,
        train_games=3,
        validation_games=2,
        test_games=2,
        seed_base=5001,
    )
    second = build_dataset(
        right,
        train_games=3,
        validation_games=2,
        test_games=2,
        seed_base=5001,
    )
    assert first == second

    seeds = {}
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        rows = [
            json.loads(line)
            for line in (left / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        seeds[split] = {int(row["game_seed"]) for row in rows}
    assert seeds["train"].isdisjoint(seeds["validation"])
    assert seeds["train"].isdisjoint(seeds["test"])
    assert seeds["validation"].isdisjoint(seeds["test"])
