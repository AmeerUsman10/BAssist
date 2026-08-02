from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arcgpt2.completion_scorer import (
    CompletionScorerError,
    score_candidate_completions,
    score_with_contextual_calibration,
)


class TinyCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 11) -> None:
        super().__init__()
        torch.manual_seed(123)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
    ):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(logits=self.projection(hidden))


def test_candidate_microbatching_is_numerically_invariant() -> None:
    model = TinyCausalModel()
    prompt = (1, 2, 3)
    candidates = ((4,), (5, 6), (7, 8, 9), (10, 11), (12,))
    single = score_candidate_completions(
        model,
        prompt,
        candidates,
        pad_token_id=0,
        device="cpu",
        candidate_batch_size=1,
    )
    grouped = score_candidate_completions(
        model,
        prompt,
        candidates,
        pad_token_id=0,
        device="cpu",
        candidate_batch_size=3,
    )
    assert torch.allclose(single, grouped, atol=1e-7)


def test_mean_reduction_removes_linear_length_scaling() -> None:
    model = TinyCausalModel()
    prompt = (1, 2)
    candidates = ((3,), (3, 4, 5))
    summed = score_candidate_completions(
        model,
        prompt,
        candidates,
        pad_token_id=0,
        device="cpu",
        reduction="sum",
    )
    averaged = score_candidate_completions(
        model,
        prompt,
        candidates,
        pad_token_id=0,
        device="cpu",
        reduction="mean",
    )
    assert averaged[0] == pytest.approx(float(summed[0]))
    assert averaged[1] == pytest.approx(float(summed[1]) / 3.0)


def test_scores_remain_differentiable_for_model_and_soft_memory() -> None:
    model = TinyCausalModel()
    soft_prefix = torch.zeros(
        2,
        model.get_input_embeddings().embedding_dim,
        requires_grad=True,
    )
    scores = score_candidate_completions(
        model,
        (1, 2),
        ((3, 4), (5,)),
        pad_token_id=0,
        device="cpu",
        candidate_batch_size=1,
        soft_prefix=soft_prefix,
    )
    scores.sum().backward()
    assert soft_prefix.grad is not None
    assert torch.isfinite(soft_prefix.grad).all()
    assert model.projection.weight.grad is not None


def test_contextual_calibration_is_exact_score_difference() -> None:
    model = TinyCausalModel()
    candidates = ((3,), (4, 5))
    context = score_candidate_completions(
        model,
        (1, 2),
        candidates,
        pad_token_id=0,
        device="cpu",
        reduction="mean",
    )
    baseline = score_candidate_completions(
        model,
        (6,),
        candidates,
        pad_token_id=0,
        device="cpu",
        reduction="mean",
    )
    calibrated = score_with_contextual_calibration(
        model,
        (1, 2),
        (6,),
        candidates,
        pad_token_id=0,
        device="cpu",
        reduction="mean",
    )
    assert torch.allclose(calibrated, context - baseline)


def test_invalid_requests_fail_loudly() -> None:
    model = TinyCausalModel()
    with pytest.raises(CompletionScorerError):
        score_candidate_completions(
            model,
            (),
            ((1,),),
            pad_token_id=0,
            device="cpu",
        )
    with pytest.raises(CompletionScorerError):
        score_candidate_completions(
            model,
            (1,),
            ((),),
            pad_token_id=0,
            device="cpu",
        )
    with pytest.raises(CompletionScorerError):
        score_candidate_completions(
            model,
            (1,),
            ((2,),),
            pad_token_id=0,
            device="cpu",
            reduction="median",
        )
