from __future__ import annotations

import json
import math

from arcgpt2.build_contact_version_dataset import (
    build_dataset,
    build_group_examples,
    contact_variants,
    controlled_contact_history,
)
from arcgpt2.mechanics_v2 import ContactMode


def _rows(rows, prefix: int):
    return [row for row in rows if int(row["prefix_length"]) == prefix]


def test_counterfactual_contact_worlds_are_identical_before_contact() -> None:
    variants = contact_variants(300_001)
    histories = {
        mode: controlled_contact_history(spec)
        for mode, (spec, _) in variants.items()
    }
    initial_grids = {history[0].before for history in histories.values()}
    assert len(initial_grids) == 1
    for index in range(5):
        signatures = {
            (
                history[index].action,
                history[index].before,
                history[index].after,
                history[index].status,
            )
            for history in histories.values()
        }
        assert len(signatures) == 1
    contact_outcomes = {history[-1].after for history in histories.values()}
    assert len(contact_outcomes) == len(ContactMode)


def test_version_space_stays_broad_until_direct_contact() -> None:
    rows = build_group_examples(300_002)
    assert len(rows) == len(ContactMode) * 4
    for prefix in (0, 4, 5):
        subset = _rows(rows, prefix)
        assert len(subset) == len(ContactMode)
        assert {row["consistent_mode_count"] for row in subset} == {len(ContactMode)}
        assert len({row["context"] for row in subset}) == 1
        for row in subset:
            assert row["target_distribution"] == [0.2] * len(ContactMode)

    final = _rows(rows, 6)
    assert len(final) == len(ContactMode)
    assert {row["consistent_mode_count"] for row in final} == {1}
    assert len({row["context"] for row in final}) == len(ContactMode)
    for row in final:
        target = row["target_distribution"]
        assert sum(value > 0.0 for value in target) == 1
        assert target[int(row["truth_index"])] == 1.0


def test_corrupted_contact_evidence_points_to_a_different_mode() -> None:
    rows = _rows(build_group_examples(300_003), 6)
    for row in rows:
        assert row["shuffled_contact_context"] != row["context"]
        assert row["precontact_context"] != row["context"]
        assert row["amnesic_context"] != row["context"]


def test_target_support_matches_consistent_indices() -> None:
    for row in build_group_examples(300_004):
        target = [float(value) for value in row["target_distribution"]]
        assert math.isclose(sum(target), 1.0)
        support = {index for index, value in enumerate(target) if value > 0.0}
        assert support == set(row["consistent_indices"])
        assert int(row["truth_index"]) in support
        assert {
            value for value in target if value > 0.0
        } == {1.0 / len(support)}


def test_dataset_is_reproducible_and_group_disjoint(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = build_dataset(
        left,
        train_groups=2,
        validation_groups=1,
        test_groups=1,
        seed_base=310_000,
    )
    second = build_dataset(
        right,
        train_groups=2,
        validation_groups=1,
        test_groups=1,
        seed_base=310_000,
    )
    assert first == second

    groups = {}
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        rows = [
            json.loads(line)
            for line in (left / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        groups[split] = {int(row["counterfactual_group_seed"]) for row in rows}
        assert len(rows) == first["splits"][split]["examples"]
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
