from __future__ import annotations

import json

from arcgpt2.build_joint_epistemic_dataset import validate_joint_row
from arcgpt2.build_joint_epistemic_v2 import (
    build_dataset,
    contact_rows,
)


def test_contact_rows_use_the_joint_candidate_set_contract() -> None:
    rows = contact_rows(510_001)
    assert len(rows) == 20
    assert {row["task"] for row in rows} == {"contact_mechanics"}
    assert {len(row["candidate_texts"]) for row in rows} == {5}
    assert {row["information_level"] for row in rows} == {0, 4, 5, 6}
    final = [row for row in rows if row["information_level"] == 6]
    assert len(final) == 5
    assert all(len(row["consistent_indices"]) == 1 for row in final)
    for row in rows:
        validate_joint_row(row)
        assert set(row["control_contexts"]) == {
            "amnesic",
            "precontact",
            "shuffled_contact",
        }


def test_three_task_dataset_is_reproducible_and_split_disjoint(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    kwargs = dict(
        train_action_groups=1,
        validation_action_groups=1,
        test_action_groups=1,
        train_goal_games=1,
        validation_goal_games=1,
        test_goal_games=1,
        train_contact_groups=1,
        validation_contact_groups=1,
        test_contact_groups=1,
        mapping_variants=2,
        action_seed_base=511_000,
        goal_seed_base=611_000,
        contact_seed_base=711_000,
    )
    first = build_dataset(left, **kwargs)
    second = build_dataset(right, **kwargs)
    assert first == second

    groups = {
        "action_binding": {},
        "goal_inference": {},
        "contact_mechanics": {},
    }
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (left / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {row["task"] for row in rows} == set(groups)
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        for task in groups:
            task_rows = [row for row in rows if row["task"] == task]
            if task == "action_binding":
                ids = {
                    row["metadata"]["counterfactual_group_seed"]
                    for row in task_rows
                }
            elif task == "goal_inference":
                ids = {row["metadata"]["game_seed"] for row in task_rows}
            else:
                ids = {
                    row["metadata"]["counterfactual_group_seed"]
                    for row in task_rows
                }
            groups[task][split] = ids

    for task_groups in groups.values():
        assert task_groups["train"].isdisjoint(task_groups["validation"])
        assert task_groups["train"].isdisjoint(task_groups["test"])
        assert task_groups["validation"].isdisjoint(task_groups["test"])
