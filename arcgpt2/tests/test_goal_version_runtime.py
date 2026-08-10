from __future__ import annotations

from arcgpt2.build_goal_version_dataset import build_examples
from arcgpt2.goal_version_runtime import held_out_terminal_accuracy


def test_ground_truth_goal_predicts_all_held_out_terminal_reports() -> None:
    rows = build_examples(77_001)
    rows_with_future = [row for row in rows if row["held_out_records"]]
    assert rows_with_future
    for row in rows_with_future:
        accuracy = held_out_terminal_accuracy(row, int(row["truth_index"]))
        assert accuracy == 1.0


def test_full_history_has_no_held_out_metric() -> None:
    rows = build_examples(77_002)
    final = max(rows, key=lambda row: int(row["prefix_length"]))
    assert final["held_out_records"] == []
    assert held_out_terminal_accuracy(final, int(final["truth_index"])) is None
