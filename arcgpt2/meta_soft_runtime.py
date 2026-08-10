"""Normalized runtime entry point for GPT-2 soft-memory meta-learning.

The first smoke used summed transition log likelihood for the inner update. A
transition description can contain dozens of tokens, so that makes the update
magnitude depend primarily on serialization length. This wrapper changes only
the inner predictive objective to mean token log likelihood, then delegates the
full meta-learning experiment to the canonical module.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import meta_soft_binding as base


_ORIGINAL_SEQUENCE_LOG_LIKELIHOOD = base.sequence_log_likelihood


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
    """Return mean rather than summed target-token log likelihood."""

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


def install_runtime_fix() -> None:
    base.sequence_log_likelihood = mean_sequence_log_likelihood


def main() -> None:
    install_runtime_fix()
    base.main()


if __name__ == "__main__":
    main()
