"""Second-order meta-learning runtime for one GPT-2 soft game memory.

The first implementation intentionally used a first-order approximation: it
stopped the gradient through the online prefix update. That verifies mechanics,
but it cannot fully teach GPT-2 *how prediction error should write useful game
information into memory*. This runtime keeps the support-update computation
graph during outer training, so the query loss differentiates through the
observed transition and into both GPT-2's trainable weights and the learned
initial prefix.

Evaluation still uses ordinary first-order online adaptation and detaches after
each observed transition. No auxiliary learned model is introduced.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

from . import meta_soft_binding as base
from .natural_protocol import changed_cells


_ORIGINAL_SEQUENCE_LOG_LIKELIHOOD = base.sequence_log_likelihood
_ORIGINAL_MAKE_EPISODE = base.make_episode


def outcome_only_target(record: Any) -> str:
    """Serialize only the unknown consequence, not the already supplied action."""

    changes = changed_cells(record)
    lines: list[str] = []
    if not changes:
        lines.append("No grid cell changed.")
    else:
        lines.append(f"Exactly {len(changes)} grid cells changed:")
        for row, column, old, new in changes:
            lines.append(
                f"- Row {row}, column {column} changed from color {old} to color {new}."
            )
    terminal = "yes" if record.status in {"LEVEL_WIN", "GAME_WIN"} else "no"
    lines.append(f"The environment reported a terminal success: {terminal}.")
    return "\n" + "\n".join(lines)


def mean_sequence_log_likelihood(
    model: Any,
    prefix: torch.Tensor,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    pad_token_id: int,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Normalize the support objective by observable target length."""

    if not target_ids:
        raise ValueError("target_ids may not be empty")
    summed = _ORIGINAL_SEQUENCE_LOG_LIKELIHOOD(
        model,
        prefix,
        prompt_ids,
        target_ids,
        pad_token_id=pad_token_id,
        device=device,
        max_length=max_length,
    )
    return summed / len(target_ids)


def bounded_make_episode(
    group_seed: int,
    variant_index: int,
    variant_count: int,
):
    """Optionally bound support steps for memory-safe second-order diagnostics."""

    episode = _ORIGINAL_MAKE_EPISODE(group_seed, variant_index, variant_count)
    maximum = int(os.environ.get("ARC_GPT2_META_MAX_PROBES", "4"))
    if not 1 <= maximum <= len(episode.records):
        raise ValueError(
            f"ARC_GPT2_META_MAX_PROBES must lie in 1..{len(episode.records)}"
        )
    return base.Episode(
        group_seed=episode.group_seed,
        variant_index=episode.variant_index,
        spec=episode.spec,
        records=episode.records[:maximum],
    )


def second_order_adapt_once(
    model: Any,
    tokenizer: Any,
    fast_prefix: torch.Tensor,
    prompt_record: Any,
    target_record: Any,
    *,
    inner_learning_rate: float,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one transition into the prefix with a differentiable update."""

    prompt_ids = base.encode_text(tokenizer, base.transition_prompt(prompt_record))
    target_ids = base.encode_text(tokenizer, outcome_only_target(target_record))
    loss = -mean_sequence_log_likelihood(
        model,
        fast_prefix,
        prompt_ids,
        target_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        max_length=max_length,
    )
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
    base.sequence_log_likelihood = mean_sequence_log_likelihood
    base.transition_target = outcome_only_target
    base.make_episode = bounded_make_episode
    base.adapt_once = second_order_adapt_once


def main() -> None:
    install_runtime()
    base.main()


if __name__ == "__main__":
    main()
