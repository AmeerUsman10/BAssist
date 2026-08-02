from __future__ import annotations

import torch

from arcgpt2.train_joint_epistemic import (
    EncodedJointItem,
    select_evaluation_items,
    set_valued_loss,
)


def _item(task: str, level: int, index: int) -> EncodedJointItem:
    return EncodedJointItem(
        task=task,
        source_id=f"{task}:{level}:{index}",
        information_level=level,
        prompt_ids=(1, 2),
        null_prompt_ids=(3,),
        control_prompt_ids={"amnesic": (3,)},
        candidate_ids=((4,), (5,), (6,)),
        target_probabilities=(0.5, 0.5, 0.0),
        truth_index=0,
        consistent_indices=(0, 1),
    )


def test_evaluation_selection_balances_tasks_and_information_levels() -> None:
    items = [
        _item(task, level, index)
        for task in ("action_binding", "goal_inference")
        for level in (0, 1, 2)
        for index in range(5)
    ]
    selected = select_evaluation_items(items, rows_per_task=6, seed=123)
    assert len(selected) == 12
    for task in ("action_binding", "goal_inference"):
        task_items = [item for item in selected if item.task == task]
        assert len(task_items) == 6
        assert {item.information_level for item in task_items} == {0, 1, 2}
        counts = {
            level: sum(item.information_level == level for item in task_items)
            for level in (0, 1, 2)
        }
        assert counts == {0: 2, 1: 2, 2: 2}


def test_evaluation_selection_is_deterministic() -> None:
    items = [_item("action_binding", level, index) for level in (0, 1) for index in range(10)]
    first = select_evaluation_items(items, rows_per_task=7, seed=999)
    second = select_evaluation_items(items, rows_per_task=7, seed=999)
    assert [item.source_id for item in first] == [item.source_id for item in second]


def test_set_valued_loss_rewards_probability_mass_on_all_valid_answers() -> None:
    item = _item("action_binding", 0, 0)
    balanced_valid = torch.tensor([4.0, 4.0, -4.0])
    collapsed_valid = torch.tensor([8.0, -8.0, -8.0])
    invalid = torch.tensor([-4.0, -4.0, 4.0])
    assert set_valued_loss(balanced_valid, item) < set_valued_loss(collapsed_valid, item)
    assert set_valued_loss(collapsed_valid, item) < set_valued_loss(invalid, item)
