from __future__ import annotations

import json
import math

from arcgpt2.build_epistemic_dataset import (
    allowed_directions,
    build_control_context,
    build_dataset,
    build_examples,
    probe_order,
)
from arcgpt2.phase0_hidden_action import Action, HiddenActionGame, generate_game


def _rows_by_probe(seed: int):
    rows = build_examples(seed)
    return {
        probe: [row for row in rows if row["probe_count"] == probe]
        for probe in range(5)
    }


def test_each_game_contains_every_query_at_every_evidence_prefix() -> None:
    rows = build_examples(71_001)
    assert len(rows) == 20
    grouped = _rows_by_probe(71_001)
    for probe, probe_rows in grouped.items():
        assert len(probe_rows) == 4
        assert {row["query_action"] for row in probe_rows} == {
            action.value for action in Action
        }
        assert {row["consistent_mapping_count"] for row in probe_rows} == {
            math.factorial(4 - probe)
        }


def test_target_distribution_is_uniform_over_the_exact_version_space() -> None:
    seed = 71_002
    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    records = []
    order = probe_order(seed)

    for probe in range(5):
        if probe:
            records.append(game.step(order[probe - 1]))
        rows = [row for row in build_examples(seed) if row["probe_count"] == probe]
        for row in rows:
            action = Action(row["query_action"])
            allowed = allowed_directions(spec, records, action)
            allowed_words = set(row["allowed_words"])
            assert len(allowed_words) == len(allowed)
            distribution = row["target_distribution"]
            assert sum(distribution.values()) == 1.0
            nonzero = {word for word, probability in distribution.items() if probability > 0.0}
            assert nonzero == allowed_words
            assert {
                probability for probability in distribution.values() if probability > 0.0
            } == {1.0 / len(allowed_words)}


def test_uncertainty_contracts_only_when_evidence_identifies_an_action() -> None:
    seed = 71_003
    grouped = _rows_by_probe(seed)
    assert all(len(row["allowed_words"]) == 4 for row in grouped[0])

    for probe in range(1, 5):
        observed = set(grouped[probe][0]["observed_actions"])
        for row in grouped[probe]:
            expected = 1 if row["query_action"] in observed else 4 - probe
            assert len(row["allowed_words"]) == expected


def test_information_controls_preserve_query_but_change_evidence() -> None:
    seed = 71_004
    for prefix in range(5):
        for action in Action:
            full = build_control_context(seed, prefix, action, "full")
            amnesic = build_control_context(seed, prefix, action, "amnesic")
            shuffled = build_control_context(seed, prefix, action, "shuffled")
            assert full.endswith("ANSWER:")
            assert amnesic.endswith("ANSWER:")
            assert shuffled.endswith("ANSWER:")
            if prefix == 0:
                assert full == shuffled
            else:
                assert full != amnesic
                assert full != shuffled


def test_dataset_is_reproducible_and_split_disjoint(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = build_dataset(
        left,
        train_games=4,
        validation_games=2,
        test_games=3,
        seed_base=9001,
    )
    second = build_dataset(
        right,
        train_games=4,
        validation_games=2,
        test_games=3,
        seed_base=9001,
    )
    assert first == second

    seed_sets = {}
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        rows = [
            json.loads(line)
            for line in (left / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        seed_sets[split] = {int(row["game_seed"]) for row in rows}
    assert seed_sets["train"].isdisjoint(seed_sets["validation"])
    assert seed_sets["train"].isdisjoint(seed_sets["test"])
    assert seed_sets["validation"].isdisjoint(seed_sets["test"])
