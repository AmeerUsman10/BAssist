"""Contrastive second-order soft-memory training for hidden action semantics.

A full natural-language transition contains many tokens shared by every possible
outcome. Using its raw language-model loss makes the inner gradient mostly say
"write fluent transition prose" rather than "this action moved north". This
runtime instead scores the four exact counterfactual cardinal outcomes and
updates the temporary prefix with set-valued contrastive loss. The gradient is
therefore concentrated on the causal distinction that later mapping queries
must recover.

The controlled probe arena guarantees one moved cell, no walls adjacent to the
start, and no terminal event during the identification actions. This module is a
Gate-A curriculum component, not a general private-game simulator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import torch
from torch.nn import functional as F

from . import meta_soft_binding as base
from .codec import encode_transition, normalize_grid
from .completion_scorer import score_candidate_completions
from .meta_soft_second_order import bounded_make_episode, outcome_only_target
from .phase0_hidden_action import DIRECTION_DELTA, Direction, StepRecord


class ContrastiveSupportError(ValueError):
    """Raised when a record is outside the controlled contrastive probe gate."""


def infer_probe_colors(record: StepRecord) -> tuple[int, int, tuple[int, int]]:
    """Infer moving color, background color, and source from an exact move.

    The two changed cells are symmetric under a simple color swap. The moving
    cell is distinguished by cardinality: in the controlled arena the agent
    color occurs once, while the background occurs many times.
    """

    changes = [
        (row, column, record.before[row][column], record.after[row][column])
        for row in range(len(record.before))
        for column in range(len(record.before[0]))
        if record.before[row][column] != record.after[row][column]
    ]
    if not record.moved or record.status != "ACTIVE" or len(changes) != 2:
        raise ContrastiveSupportError(
            "contrastive action binding requires one active one-cell movement"
        )
    counts = Counter(value for row in record.before for value in row)
    for source in changes:
        sy, sx, source_old, source_new = source
        for destination in changes:
            if destination is source:
                continue
            _, _, destination_old, destination_new = destination
            if (
                source_old == destination_new
                and source_new == destination_old
                and counts[source_old] < counts[source_new]
            ):
                return source_old, source_new, (sy, sx)
    raise ContrastiveSupportError("could not infer a translated singleton color cell")


def counterfactual_probe_records(record: StepRecord) -> tuple[StepRecord, ...]:
    """Construct the four exact one-cell cardinal outcomes for this probe."""

    moving_color, background_color, source = infer_probe_colors(record)
    height = len(record.before)
    width = len(record.before[0])
    candidates: list[StepRecord] = []
    for direction in Direction:
        dy, dx = DIRECTION_DELTA[direction]
        destination = (source[0] + dy, source[1] + dx)
        canvas = [list(row) for row in record.before]
        moved = (
            0 <= destination[0] < height
            and 0 <= destination[1] < width
            and canvas[destination[0]][destination[1]] == background_color
        )
        if moved:
            canvas[source[0]][source[1]] = background_color
            canvas[destination[0]][destination[1]] = moving_color
        after = normalize_grid(canvas)
        candidates.append(
            replace(
                record,
                after=after,
                status="ACTIVE",
                moved=moved,
                transition=encode_transition(record.before, after),
            )
        )
    return tuple(candidates)


def support_scores(
    model: Any,
    tokenizer: Any,
    fast_prefix: torch.Tensor,
    prompt_record: StepRecord,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[StepRecord, ...]]:
    prompt_ids = base.encode_text(tokenizer, base.transition_prompt(prompt_record))
    candidates = counterfactual_probe_records(prompt_record)
    candidate_ids = tuple(
        base.encode_text(tokenizer, outcome_only_target(candidate))
        for candidate in candidates
    )
    scores = score_candidate_completions(
        model,
        prompt_ids,
        candidate_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=4,
        reduction="mean",
        soft_prefix=fast_prefix,
    )
    return scores, candidates


def contrastive_adapt_once(
    model: Any,
    tokenizer: Any,
    fast_prefix: torch.Tensor,
    prompt_record: StepRecord,
    target_record: StepRecord,
    *,
    inner_learning_rate: float,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del max_length
    scores, candidates = support_scores(
        model,
        tokenizer,
        fast_prefix,
        prompt_record,
        device=device,
    )
    consistent = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.after == target_record.after
        and candidate.status == target_record.status
    ]
    if not consistent:
        raise ContrastiveSupportError(
            "the observed outcome is absent from the cardinal counterfactual set"
        )
    target = torch.zeros_like(scores)
    target[consistent] = 1.0 / len(consistent)
    loss = -(target * F.log_softmax(scores, dim=-1)).sum()
    meta_training = bool(model.training)
    gradient = torch.autograd.grad(
        loss,
        fast_prefix,
        create_graph=meta_training,
        retain_graph=meta_training,
    )[0]
    updated = fast_prefix - inner_learning_rate * gradient
    return updated, loss.detach()


def install_runtime() -> None:
    base.make_episode = bounded_make_episode
    base.adapt_once = contrastive_adapt_once


def main() -> None:
    install_runtime()
    base.main()


if __name__ == "__main__":
    main()
