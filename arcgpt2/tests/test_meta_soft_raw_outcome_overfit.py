from __future__ import annotations

import re

import pytest
import torch

from arcgpt2 import meta_soft_raw_outcome_overfit as raw
from arcgpt2 import meta_soft_contrastive
from arcgpt2 import meta_soft_single_binding
from arcgpt2.meta_soft_twin_overfit import apply_gate as contrastive_apply_gate
from arcgpt2.phase0_hidden_action import Action, Direction


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_raw_support_quartet_has_one_prompt_and_four_exact_targets() -> None:
    tokenizer = CharacterTokenizer()
    quartet = raw.build_quartet(960_001, Action.A3)
    pairs = [
        raw.raw_support_token_ids(tokenizer, episode.record, episode.record)
        for episode in quartet
    ]

    assert {episode.direction for episode in quartet} == set(Direction)
    assert len({prompt for prompt, _ in pairs}) == 1
    assert len({target for _, target in pairs}) == 4
    assert all(target for _, target in pairs)
    assert raw.SUPPORT_OBJECTIVE == "raw_outcome_nll"
    for prompt, target in (
        raw.raw_support_text(episode.record, episode.record)
        for episode in quartet
    ):
        words = set(re.findall(r"[a-z]+", (prompt + "\n" + target).lower()))
        assert words.isdisjoint(
            {"north", "south", "west", "east", "up", "down", "left", "right"}
        )


def test_raw_support_rejects_cardinal_semantic_leakage(monkeypatch) -> None:
    episode = raw.build_quartet(960_005, Action.A1)[0]
    monkeypatch.setattr(raw, "transition_prompt", lambda record: "It moved left.")

    with pytest.raises(RuntimeError, match="leaked cardinal semantics"):
        raw.raw_support_text(episode.record, episode.record)


def test_raw_corruption_is_labeled_as_quartet_permutation_consistency() -> None:
    quartet = raw.build_quartet(960_002, Action.A4)
    intact_targets = {
        raw.raw_support_text(episode.record, episode.record)[1]
        for episode in quartet
    }
    corrupted_targets = {
        raw.raw_support_text(
            episode.record,
            raw.corrupted_target(episode.record, episode.direction)[0],
        )[1]
        for episode in quartet
    }

    # Every fixed corruption is another already represented quartet outcome.
    # It checks consistent replay of the learned channel, not new evidence.
    assert corrupted_targets == intact_targets


def test_raw_inner_update_scores_only_the_observed_target(monkeypatch) -> None:
    tokenizer = CharacterTokenizer()
    episode = raw.build_quartet(960_003, Action.A1)[0]
    prefix = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    captured: dict[str, object] = {}

    def fake_score(
        model,
        prompt_ids,
        candidate_ids,
        **kwargs,
    ):
        del model
        captured.update(
            {
                "prompt_ids": prompt_ids,
                "candidate_ids": candidate_ids,
                "reduction": kwargs["reduction"],
                "candidate_batch_size": kwargs["candidate_batch_size"],
            }
        )
        # A differentiable stand-in log likelihood; no language model is loaded.
        return torch.stack((-kwargs["soft_prefix"].sum(),))

    def reject_counterfactual_helper(*args, **kwargs):
        del args, kwargs
        raise AssertionError("raw inner update requested counterfactual candidates")

    monkeypatch.setattr(raw, "score_candidate_completions", fake_score)
    monkeypatch.setattr(
        meta_soft_contrastive,
        "counterfactual_probe_records",
        reject_counterfactual_helper,
    )
    monkeypatch.setattr(
        meta_soft_single_binding,
        "counterfactual_probe_records",
        reject_counterfactual_helper,
    )
    updated, loss, gradient, target_tokens = raw.raw_outcome_adapt_prefix(
        object(),
        tokenizer,
        prefix,
        episode.record,
        episode.record,
        inner_learning_rate=0.2,
        device=torch.device("cpu"),
        create_graph=False,
    )
    expected_prompt, expected_target = raw.raw_support_token_ids(
        tokenizer,
        episode.record,
        episode.record,
    )

    assert captured["prompt_ids"] == expected_prompt
    assert captured["candidate_ids"] == (expected_target,)
    assert captured["reduction"] == "mean"
    assert captured["candidate_batch_size"] == 1
    assert target_tokens == len(expected_target)
    assert float(loss.item()) == 0.0
    assert torch.equal(gradient, torch.ones_like(prefix))
    assert torch.allclose(updated, torch.full_like(prefix, -0.2))


def test_fixed_corruption_has_exactly_one_balanced_index_per_world() -> None:
    quartet = raw.build_quartet(960_004, Action.A2)
    indices = raw.corrupted_direction_indices(quartet)

    assert len(indices) == 4
    assert set(indices) == {0, 1, 2, 3}


def test_raw_gate_reuses_the_preregistered_functional_thresholds() -> None:
    assert raw.apply_gate is contrastive_apply_gate


def test_source_sha_is_an_explicit_reproducibility_input(monkeypatch) -> None:
    monkeypatch.setenv("ARC_GPT2_SOURCE_SHA", "environment-sha")
    config = raw.Config(source_sha="0123456789abcdef")
    assert raw.resolve_source_sha(config) == "0123456789abcdef"
    assert raw.resolve_source_sha(raw.Config()) == "environment-sha"


def test_replication_can_record_a_full_curve_after_first_pass() -> None:
    assert raw.Config().stop_on_gate_pass is True
    assert raw.Config(stop_on_gate_pass=False).stop_on_gate_pass is False
