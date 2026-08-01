"""Corrected runtime entry point for latent-goal GPT-2 experiments.

The initial trainer's held-out evaluator accidentally called ``evaluate_goal``
with an already-executed transition instead of its declared
``(mechanics, goal, before, action)`` interface. This wrapper replaces only that
metric function, then delegates all training and scoring to the canonical
module. Keeping the correction isolated makes the evidence trail explicit while
a later consolidation can fold it back into the larger trainer file.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import train_goal_version as base
from .dsl import program_from_phase0_spec
from .goal_dsl import evaluate_goal, parse_goal
from .phase0_hidden_action import Action, generate_game


def held_out_terminal_accuracy(
    row: Mapping[str, Any], selected_index: int
) -> float | None:
    """Measure a selected goal on records not present in the prompt."""

    held_out = row["held_out_records"]
    if not held_out:
        return None
    spec = generate_game(int(row["game_seed"]))
    mechanics = program_from_phase0_spec(spec)
    goal = parse_goal(str(row["candidate_programs"][selected_index]))
    correct = 0
    for record in held_out:
        before = tuple(
            tuple(int(value) for value in grid_row)
            for grid_row in record["before"]
        )
        action = Action(str(record["action"]))
        predicted_terminal, _ = evaluate_goal(mechanics, goal, before, action)
        expected_terminal = str(record["status"]) in {"LEVEL_WIN", "GAME_WIN"}
        correct += int(predicted_terminal == expected_terminal)
    return correct / len(held_out)


def main() -> None:
    # ``train`` resolves the metric through its module globals, so replacing it
    # before calling ``main`` corrects both intact and information-control runs.
    base.held_out_terminal_accuracy = held_out_terminal_accuracy
    base.main()


if __name__ == "__main__":
    main()
