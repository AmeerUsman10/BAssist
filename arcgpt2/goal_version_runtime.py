"""Memory-safe corrected runtime for latent-goal GPT-2 experiments.

The initial trainer had two execution defects that did not affect dataset tests:

1. held-out goal evaluation called ``evaluate_goal`` with the wrong signature;
2. every Goal-DSL candidate was expanded into one enormous logits batch, which
   can exceed ordinary CPU/GPU memory because GPT-2 logits scale with
   ``batch * sequence * vocabulary``.

This wrapper patches those runtime functions explicitly, then delegates the
training/evaluation protocol to the canonical module. Candidate programs are
scored by mean token log probability rather than raw sequence probability, so a
shorter English rendering does not win merely because it has fewer tokens.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import torch

from . import train_goal_version as base
from .completion_scorer import score_candidate_completions
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


def score_items(
    model: Any,
    items: Sequence[base.EncodedGoalItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Score Goal-DSL candidates in bounded differentiable micro-batches."""

    if not items:
        raise ValueError("at least one goal item is required")
    candidate_batch_size = int(os.environ.get("ARC_GPT2_GOAL_CANDIDATE_BATCH", "2"))
    if candidate_batch_size < 1:
        raise ValueError("ARC_GPT2_GOAL_CANDIDATE_BATCH must be positive")

    grouped_scores: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []
    for item in items:
        score_tensor = score_candidate_completions(
            model,
            item.prompt_ids,
            item.candidate_ids,
            pad_token_id=pad_token_id,
            device=device,
            candidate_batch_size=candidate_batch_size,
            reduction="mean",
        )
        grouped_scores.append(score_tensor)
        target_tensors.append(
            torch.tensor(
                item.target_probabilities,
                dtype=score_tensor.dtype,
                device=device,
            )
        )
    return grouped_scores, target_tensors


def install_runtime_fixes() -> None:
    """Patch only the corrected runtime functions."""

    base.held_out_terminal_accuracy = held_out_terminal_accuracy
    base.score_items = score_items


def main() -> None:
    install_runtime_fixes()
    base.main()


if __name__ == "__main__":
    main()
