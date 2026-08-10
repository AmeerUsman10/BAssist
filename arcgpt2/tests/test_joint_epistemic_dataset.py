from __future__ import annotations

import json

from arcgpt2.build_joint_epistemic_dataset import (
    action_rows,
    build_dataset,
    goal_rows,
    validate_joint_row,
)


def test_action_rows_share_surface_across_counterfactual_mappings() -> None:
    rows = action_rows(101_001, 4)
    assert len(rows) == 4 * 20
    zero_probe = [
        row
        for row in rows
        if row["information_level"] == 0
        and row["metadata"]["query_action"] == "A1"
    ]
    assert len(zero_probe) == 4
    assert len({row["context"] for row in zero_probe}) == 1
    assert len({row["truth_index"] for row in zero_probe}) > 1
    for row in rows:
        validate_joint_row(row)
        assert row["task"] == "action_binding"
        assert set(row["control_contexts"]) == {"amnesic", "shuffled_evidence"}


def test_goal_rows_use_the_same_candidate_set_schema() -> None:
    rows = goal_rows(101_002)
    assert rows
    candidate_lengths = {len(row["candidate_texts"]) for row in rows}
    assert candidate_lengths == {8}
    assert rows[0]["information_level"] == 0
    assert min(len(row["consistent_indices"]) for row in rows) < 8
    for row in rows:
        validate_joint_row(row)
        assert row["task"] == "goal_inference"
        assert set(row["control_contexts"]) == {
            "amnesic",
            "statusless",
            "shuffled_status",
        }


def test_joint_dataset_is_reproducible_and_group_disjoint(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    kwargs = dict(
        train_action_groups=2,
        validation_action_groups=1,
        test_action_groups=1,
        train_goal_games=2,
        validation_goal_games=1,
        test_goal_games=1,
        mapping_variants=3,
        action_seed_base=102_000,
        goal_seed_base=202_000,
    )
    first = build_dataset(left, **kwargs)
    second = build_dataset(right, **kwargs)
    assert first == second

    action_groups = {}
    goal_games = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (left / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        action_groups[split] = {
            row["metadata"]["counterfactual_group_seed"]
            for row in rows
            if row["task"] == "action_binding"
        }
        goal_games[split] = {
            row["metadata"]["game_seed"]
            for row in rows
            if row["task"] == "goal_inference"
        }
        assert {row["task"] for row in rows} == {
            "action_binding",
            "goal_inference",
        }
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]

    for groups in (action_groups, goal_games):
        assert groups["train"].isdisjoint(groups["validation"])
        assert groups["train"].isdisjoint(groups["test"])
        assert groups["validation"].isdisjoint(groups["test"])
