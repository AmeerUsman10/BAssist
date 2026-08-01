from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from arcgpt2.phase0_hidden_action import Action, HiddenActionGame, SourceLearner, generate_game
from arcgpt2.stage01_hidden_action import (
    all_probe_orders,
    build_all_variant_examples,
    build_set_valued_examples,
    probe_support_counts,
    valid_actions,
)
from arcgpt2.train_stage01 import PHASE_INDEX, set_valued_action_loss
from arcgpt2.train_stage01_balanced import balanced_sampling_weights


def test_all_probe_orders_are_present_once() -> None:
    orders = all_probe_orders()
    assert len(orders) == 24
    assert len(set(orders)) == 24
    assert all(set(order) == set(Action) for order in orders)


def test_probe_valid_sets_shrink_without_arbitrary_labels() -> None:
    spec = generate_game(1234)
    examples = build_set_valued_examples(spec, variant_index=0)
    probe_examples = [example for example in examples if example.decision_phase == "probe"]
    assert len(probe_examples) == 4
    assert [len(example.valid_targets) for example in probe_examples] == [4, 3, 2, 1]
    assert [example.mapping_known_count for example in probe_examples] == [0, 1, 2, 3]
    assert all(
        example.canonical_target in example.valid_targets
        for example in probe_examples
    )


def test_all_probe_orders_balance_valid_support_and_targets() -> None:
    examples = build_all_variant_examples(2026)
    probe_examples = [example for example in examples if example.decision_phase == "probe"]
    support = probe_support_counts(examples)
    canonical_counts = {f"<A{index}>": 0 for index in range(1, 5)}
    for example in probe_examples:
        canonical_counts[example.canonical_target] += 1

    assert len(probe_examples) == 24 * 4
    assert len(set(support.values())) == 1
    assert set(support.values()) == {60}
    assert len(set(canonical_counts.values())) == 1
    assert set(canonical_counts.values()) == {24}


def test_navigation_target_is_singleton_after_mapping_is_known() -> None:
    spec = generate_game(91)
    game = HiddenActionGame(spec)
    learner = SourceLearner(spec)
    for _ in range(4):
        candidates, phase = valid_actions(learner, game)
        assert phase == "probe"
        action = learner.choose(game)
        assert action in candidates
        record = game.step(action)
        learner.observe(record)

    candidates, phase = valid_actions(learner, game)
    assert phase == "navigate"
    assert len(candidates) == 1
    assert candidates[0] == learner.choose(game)


def test_set_valued_loss_is_zero_when_every_action_is_valid() -> None:
    logits = torch.tensor([[3.0, -2.0, 0.5, 1.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, True]])
    loss = set_valued_action_loss(logits, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-6)


def test_set_valued_loss_reduces_to_cross_entropy_for_singleton() -> None:
    logits = torch.zeros((1, 4), requires_grad=True)
    mask = torch.tensor([[False, False, True, False]])
    loss = set_valued_action_loss(logits, mask)
    assert math.isclose(float(loss.item()), math.log(4.0), rel_tol=1e-6)


def test_set_valued_loss_rewards_probability_mass_on_any_valid_action() -> None:
    good = torch.tensor([[5.0, 4.0, -3.0, -3.0]])
    bad = torch.tensor([[-3.0, -3.0, 5.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])
    assert set_valued_action_loss(good, mask) < set_valued_action_loss(bad, mask)


def test_balanced_sampler_equalizes_phase_and_navigation_actions() -> None:
    items = []
    # Deliberately imbalanced raw data: 2 probes and navigation counts 1,2,3,4.
    for _ in range(2):
        items.append(
            {
                "phase_id": PHASE_INDEX["probe"],
                "canonical_target_index": 0,
            }
        )
    for action_index, count in enumerate((1, 2, 3, 4)):
        for _ in range(count):
            items.append(
                {
                    "phase_id": PHASE_INDEX["navigate"],
                    "canonical_target_index": action_index,
                }
            )

    weights, summary = balanced_sampling_weights(SimpleNamespace(items=items))
    assert len(weights) == len(items)
    assert math.isclose(summary["expected_probe_sampling_mass"], 0.5)
    for mass in summary["expected_navigation_sampling_mass_by_action"].values():
        assert math.isclose(mass, 0.125)
    assert math.isclose(summary["total_sampling_mass"], 1.0)
