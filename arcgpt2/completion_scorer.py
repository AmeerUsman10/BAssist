"""Memory-bounded completion scoring for one causal GPT-2 checkpoint.

ARC-GPT2 expresses latent variables, programs, predictions, and actions as
candidate text completions. This module converts ordinary causal-LM token
likelihoods into differentiable candidate scores without adding a learned head.
It also supports a temporary soft prefix belonging to the same GPT-2 model.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
from torch.nn import functional as F


Reduction = Literal["sum", "mean"]


class CompletionScorerError(ValueError):
    """Raised when a completion-scoring request is malformed."""


def score_candidate_completions(
    model: Any,
    prompt_ids: Sequence[int],
    candidate_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | str,
    candidate_batch_size: int = 4,
    reduction: Reduction = "sum",
    soft_prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one differentiable log-likelihood score per candidate.

    ``reduction='sum'`` is the exact sequence log probability. It is appropriate
    when candidate token lengths are intentionally comparable. ``'mean'``
    removes the dominant short-completion bias when scoring semantically
    different program descriptions.
    """

    if not prompt_ids:
        raise CompletionScorerError("prompt_ids may not be empty")
    if not candidate_ids:
        raise CompletionScorerError("at least one candidate is required")
    if any(not candidate for candidate in candidate_ids):
        raise CompletionScorerError("candidate token sequences may not be empty")
    if candidate_batch_size < 1:
        raise CompletionScorerError("candidate_batch_size must be positive")
    if reduction not in {"sum", "mean"}:
        raise CompletionScorerError("reduction must be 'sum' or 'mean'")

    target_device = torch.device(device)
    if soft_prefix is not None:
        if soft_prefix.ndim != 2:
            raise CompletionScorerError(
                "soft_prefix must have shape [prefix_length, hidden_size]"
            )
        if soft_prefix.shape[0] < 1:
            raise CompletionScorerError("soft_prefix must contain at least one token")
        embedding = model.get_input_embeddings()
        if int(soft_prefix.shape[1]) != int(embedding.embedding_dim):
            raise CompletionScorerError("soft_prefix hidden size does not match model")

    scores: list[torch.Tensor] = []
    candidates = [tuple(int(token) for token in candidate) for candidate in candidate_ids]
    prompt = tuple(int(token) for token in prompt_ids)

    for chunk_start in range(0, len(candidates), candidate_batch_size):
        chunk = candidates[chunk_start : chunk_start + candidate_batch_size]
        lengths = [len(prompt) + len(candidate) for candidate in chunk]
        longest = max(lengths)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for candidate, length in zip(chunk, lengths, strict=True):
            missing = longest - length
            input_rows.append([*prompt, *candidate, *([pad_token_id] * missing)])
            mask_rows.append([1] * length + [0] * missing)

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=target_device)
        token_mask = torch.tensor(mask_rows, dtype=torch.long, device=target_device)

        if soft_prefix is None:
            logits = model(input_ids=input_ids, attention_mask=token_mask).logits
            prefix_length = 0
        else:
            token_embeddings = model.get_input_embeddings()(input_ids)
            prefix = soft_prefix.to(target_device)
            prefix_batch = prefix.unsqueeze(0).expand(len(chunk), -1, -1)
            inputs_embeds = torch.cat((prefix_batch, token_embeddings), dim=1)
            prefix_mask = torch.ones(
                (len(chunk), prefix.shape[0]),
                dtype=torch.long,
                device=target_device,
            )
            attention_mask = torch.cat((prefix_mask, token_mask), dim=1)
            logits = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            prefix_length = int(prefix.shape[0])

        for row_index, candidate in enumerate(chunk):
            start = prefix_length + len(prompt) - 1
            target_logits = logits[row_index, start : start + len(candidate)]
            targets = torch.tensor(candidate, dtype=torch.long, device=target_device)
            token_log_probabilities = F.log_softmax(target_logits, dim=-1).gather(
                1, targets.unsqueeze(1)
            )[:, 0]
            score = (
                token_log_probabilities.sum()
                if reduction == "sum"
                else token_log_probabilities.mean()
            )
            scores.append(score)

    return torch.stack(scores)


def score_with_contextual_calibration(
    model: Any,
    prompt_ids: Sequence[int],
    null_prompt_ids: Sequence[int],
    candidate_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | str,
    candidate_batch_size: int = 4,
    reduction: Reduction = "mean",
    soft_prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Subtract each candidate's context-free language prior.

    This PMI-like score prevents a fluent or short candidate from winning merely
    because GPT-2 generally prefers its wording. The same checkpoint scores both
    terms, so this introduces no auxiliary model.
    """

    contextual = score_candidate_completions(
        model,
        prompt_ids,
        candidate_ids,
        pad_token_id=pad_token_id,
        device=device,
        candidate_batch_size=candidate_batch_size,
        reduction=reduction,
        soft_prefix=soft_prefix,
    )
    baseline = score_candidate_completions(
        model,
        null_prompt_ids,
        candidate_ids,
        pad_token_id=pad_token_id,
        device=device,
        candidate_batch_size=candidate_batch_size,
        reduction=reduction,
        soft_prefix=soft_prefix,
    )
    return contextual - baseline
