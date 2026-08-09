from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from arcgpt2.completion_scorer import score_candidate_completions
from arcgpt2 import eggroll_inner_memory as eggroll
from arcgpt2.meta_soft_heldout_binding import ProbeLayout
from arcgpt2.phase0_hidden_action import Action


class TinyTransformer(torch.nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=torch.cumsum(inputs_embeds, dim=1))


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5)
        self.transformer = TinyTransformer()
        self.lm_head = torch.nn.Linear(5, 17, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
        ).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def _layout(seed: int, digest: str) -> ProbeLayout:
    return ProbeLayout(
        split="eggroll_audit",
        game_seed=seed,
        before_grid_sha256=digest,
        probe_row=2,
        probe_column=2,
    )


def test_rank_one_sampling_is_deterministic_and_exact() -> None:
    first = eggroll.sample_rank_one_perturbations(
        7, 3, 5, seed=19, device="cpu"
    )
    second = eggroll.sample_rank_one_perturbations(
        7, 3, 5, seed=19, device="cpu"
    )
    different = eggroll.sample_rank_one_perturbations(
        7, 3, 5, seed=20, device="cpu"
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert first.shape == (7, 3, 5)
    assert all(int(torch.linalg.matrix_rank(matrix).item()) == 1 for matrix in first)


def test_antithetic_population_has_exact_center() -> None:
    center = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    perturbations = eggroll.sample_rank_one_perturbations(
        5, 3, 4, seed=7, device="cpu"
    )
    plus, minus = eggroll.antithetic_population(center, perturbations, 0.003)

    assert torch.allclose((plus + minus) / 2, center, atol=1e-7, rtol=0)
    assert torch.allclose(plus - center, -(minus - center), atol=1e-6, rtol=0)


def test_rank_one_estimator_tracks_a_quadratic_gradient() -> None:
    center = torch.tensor([[0.3, -0.5, 0.7], [1.1, -0.2, 0.4]])
    target = torch.tensor([[-0.1, 0.2, 0.8], [0.6, -0.7, 0.0]])
    exact = center - target
    sigma = 0.001
    perturbations = eggroll.sample_rank_one_perturbations(
        4096, 2, 3, seed=101, device="cpu"
    )
    plus, minus = eggroll.antithetic_population(center, perturbations, sigma)
    loss_plus = 0.5 * (plus - target).square().sum(dim=(1, 2))
    loss_minus = 0.5 * (minus - target).square().sum(dim=(1, 2))

    estimate = eggroll.estimate_rank_one_gradient(
        loss_plus, loss_minus, perturbations, sigma=sigma
    )
    cosine = torch.nn.functional.cosine_similarity(
        estimate["raw"].flatten(), exact.flatten(), dim=0
    )
    ratio = estimate["norm_corrected"].norm() / exact.norm()

    assert float(cosine) > 0.98
    assert 0.95 < float(ratio) < 1.05


def test_population_scorer_matches_scalar_completion_scorer() -> None:
    torch.manual_seed(3)
    model = TinyCausalLM().eval()
    prefixes = torch.randn(6, 2, 5) * 0.05
    prompt = (2, 4, 6)
    target = (8, 10)

    batched = eggroll.prefix_population_mean_nll(
        model,
        prefixes,
        prompt,
        target,
        device="cpu",
        batch_size=4,
    )
    scalar = torch.stack(
        [
            -score_candidate_completions(
                model,
                prompt,
                (target,),
                pad_token_id=0,
                device="cpu",
                candidate_batch_size=1,
                reduction="mean",
                soft_prefix=prefix,
            )[0]
            for prefix in prefixes
        ]
    )

    assert torch.allclose(batched, scalar, atol=1e-6, rtol=1e-6)


def test_episode_seed_is_stable_and_surface_bound() -> None:
    layout = _layout(1_400_000, "a" * 64)
    first = eggroll.stable_episode_seed(424_242, layout, Action.A1, 0)

    assert first == eggroll.stable_episode_seed(424_242, layout, Action.A1, 0)
    assert first != eggroll.stable_episode_seed(424_242, layout, Action.A2, 0)
    assert first != eggroll.stable_episode_seed(
        424_242, replace(layout, before_grid_sha256="b" * 64), Action.A1, 0
    )


def test_protocol_config_rejects_optimizer_drift() -> None:
    eggroll.validate_protocol_config(eggroll.Config())
    with pytest.raises(ValueError, match="configuration drift"):
        eggroll.validate_protocol_config(
            eggroll.Config(antithetic_pairs=128)
        )


def test_gate_accepts_exact_inclusive_boundaries() -> None:
    audit = {
        "metrics": {
            "exact_accuracy": 0.70,
            "exact_truth_probability": 0.60,
            "mean_cosine": 0.20,
            "median_corrected_to_exact_norm_ratio": 0.5,
            "eggroll_accuracy": 0.45,
            "eggroll_truth_probability": 0.35,
            "truth_probability_gain_retention": 0.50,
        },
        "bootstrap": {
            "lower_bounds": {
                "cosine": 1e-8,
                "eggroll_accuracy": 0.2500001,
                "eggroll_truth_probability_gain": 1e-8,
            }
        },
    }
    result = eggroll.apply_gate(audit, {"execution": True})

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_gate_rejects_execution_failure() -> None:
    audit = {
        "metrics": {
            "exact_accuracy": 1.0,
            "exact_truth_probability": 1.0,
            "mean_cosine": 1.0,
            "median_corrected_to_exact_norm_ratio": 1.0,
            "eggroll_accuracy": 1.0,
            "eggroll_truth_probability": 1.0,
            "truth_probability_gain_retention": 1.0,
        },
        "bootstrap": {
            "lower_bounds": {
                "cosine": 1.0,
                "eggroll_accuracy": 1.0,
                "eggroll_truth_probability_gain": 1.0,
            }
        },
    }

    assert eggroll.apply_gate(audit, {"execution": False})["passed"] is False


def test_undefined_gain_retention_is_strict_json_safe_and_fails_gate() -> None:
    assert eggroll._safe_gain_retention(0.0, 0.0) is None
    audit = {
        "metrics": {
            "exact_accuracy": 1.0,
            "exact_truth_probability": 1.0,
            "mean_cosine": 1.0,
            "median_corrected_to_exact_norm_ratio": None,
            "eggroll_accuracy": 1.0,
            "eggroll_truth_probability": 1.0,
            "truth_probability_gain_retention": None,
        },
        "bootstrap": {
            "lower_bounds": {
                "cosine": 1.0,
                "eggroll_accuracy": 1.0,
                "eggroll_truth_probability_gain": 1.0,
            }
        },
    }

    result = eggroll.apply_gate(audit, {"execution": True})
    assert result["passed"] is False
    assert result["checks"]["median_norm_ratio_at_least_0_5"] is False
    assert result["checks"]["eggroll_retains_half_exact_probability_gain"] is False
