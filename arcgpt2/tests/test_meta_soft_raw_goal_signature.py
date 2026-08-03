from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from arcgpt2 import meta_soft_raw_goal_signature as goal


class TinyTokenizer:
    pad_token_id = 0
    name_or_path = "openai-community/gpt2"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        # Stable whitespace codec is enough to test model-lane contracts.
        return [sum(map(ord, word)) % 997 + 1 for word in text.split()]


def _trial(bit_pattern=(1, 0, 1, 0), terminal=" Terminal success: yes."):
    return goal.ObservedTrial(
        prompt="exact transition",
        target=terminal,
        statusless_target=" Terminal success: nil.",
        truth_bits=bit_pattern,
    )


def _world(family="CONTACT", deranged="ABSENT"):
    return goal.GoalWorld(
        family=family,
        candidate_prompt="The goal is",
        candidate_families=goal.GOAL_FAMILIES,
        candidate_completions=(" alpha goal", " bravo goal", " delta goal", " gamma goal"),
        trials=(_trial(), _trial((0, 1, 1, 0), " Terminal success: no.")),
        trial3_prompt="On the new transition, terminal?",
        trial3_yes=family in {"CONTACT", "COUNT_EQ_1"},
        deranged_family=deranged,
        deranged_trials=(_trial(terminal=" Terminal success: no."), _trial((0, 1, 1, 0))),
    )


def _group():
    families = goal.GOAL_FAMILIES
    signatures = ((1, 1), (1, 0), (0, 1), (0, 0))
    base = tuple(
        replace(
            _world(family, families[(index + 1) % 4]),
            trials=(
                _trial((1, 1, 0, 0), f" Terminal success: {'yes' if signatures[index][0] else 'no'}."),
                _trial((1, 0, 1, 0), f" Terminal success: {'yes' if signatures[index][1] else 'no'}."),
            ),
        )
        for index, family in enumerate(families)
    )
    worlds = tuple(
        replace(world, deranged_trials=base[(index + 1) % 4].trials)
        for index, world in enumerate(base)
    )
    return goal.GoalGroup("g", worlds)


def test_set_valued_loss_is_uniform_on_exact_consistent_pair():
    logits = torch.tensor([2.0, -1.0, 0.5, 1.0])
    observed = goal.set_valued_cross_entropy(logits, (1, 0, 1, 0))
    expected = -(torch.log_softmax(logits, -1)[0] + torch.log_softmax(logits, -1)[2]) / 2
    assert observed == pytest.approx(expected)
    with pytest.raises(ValueError, match="exactly two"):
        goal.set_valued_cross_entropy(logits, (1, 0, 0, 0))


def test_raw_adaptation_is_candidate_count_one_mean_nll(monkeypatch):
    calls = []

    def fake_score(*args, **kwargs):
        calls.append(kwargs)
        return torch.stack((-kwargs["soft_prefix"].sum(),))

    monkeypatch.setattr(goal, "score_candidate_completions", fake_score)
    prefix = torch.zeros((2, 3), requires_grad=True)
    updated, loss, gradient, count = goal.raw_text_adapt_prefix(
        object(), TinyTokenizer(), prefix, _trial(),
        inner_learning_rate=.2, device=torch.device("cpu"), create_graph=False,
    )
    assert calls[0]["candidate_batch_size"] == 1
    assert calls[0]["reduction"] == "mean"
    assert loss.item() == 0
    assert torch.equal(gradient, torch.ones_like(prefix))
    assert torch.allclose(updated, torch.full_like(prefix, -.2))
    assert count == len(TinyTokenizer().encode(_trial().target))


def test_sequential_orders_use_updated_prefix(monkeypatch):
    seen = []

    def fake_adapt(model, tokenizer, prefix, trial, **kwargs):
        del model, tokenizer, trial, kwargs
        seen.append(prefix.detach().clone())
        updated = prefix + 1
        one = torch.ones_like(prefix)
        return updated, prefix.sum(), one, 2

    monkeypatch.setattr(goal, "raw_text_adapt_prefix", fake_adapt)
    prefix = torch.zeros((1, 2), requires_grad=True)
    updated, diagnostics = goal.adapt_two_trials(
        object(), TinyTokenizer(), prefix, _world().trials, (1, 0),
        inner_learning_rate=.2, device=torch.device("cpu"), create_graph=False,
    )
    assert torch.equal(seen[0], torch.zeros_like(prefix))
    assert torch.equal(seen[1], torch.ones_like(prefix))
    assert torch.equal(updated, torch.full_like(prefix, 2))
    assert [row["trial_index"] for row in diagnostics] == [1, 0]


def test_optimized_group_loss_matches_naive_world_mean_and_gradient(monkeypatch):
    def fake_adapt(model, tokenizer, prefix, trial, **kwargs):
        del model, tokenizer, kwargs
        sign = 1. if "yes" in trial.target else -1.
        updated = prefix + sign * .1
        return updated, prefix.square().sum(), torch.ones_like(prefix), 2

    def fake_goal_logits(model, tokenizer, prefix, world, device):
        del model, tokenizer, world, device
        return prefix.sum() * torch.tensor([1., 2., 3., 4.])

    def fake_binary(model, tokenizer, prefix, world, device):
        del model, tokenizer, world, device
        return prefix.sum() * torch.tensor([-.5, .5])

    monkeypatch.setattr(goal, "raw_text_adapt_prefix", fake_adapt)
    monkeypatch.setattr(goal, "goal_raw_logits", fake_goal_logits)
    monkeypatch.setattr(goal, "trial3_binary_logits", fake_binary)
    group = _group()
    config = goal.Config()
    optimized_prefix = torch.tensor([.3], requires_grad=True)
    optimized, _ = goal.group_meta_loss(
        object(), TinyTokenizer(), optimized_prefix, group, config, torch.device("cpu")
    )
    optimized_gradient = torch.autograd.grad(optimized, optimized_prefix)[0]

    naive_prefix = torch.tensor([.3], requires_grad=True)
    naive_losses = []
    for world in group.worlds:
        for order in goal.EVIDENCE_ORDERS:
            loss, _ = goal.world_meta_loss(
                object(), TinyTokenizer(), naive_prefix, world, order,
                config, torch.device("cpu"),
            )
            naive_losses.append(loss)
    naive = torch.stack(naive_losses).mean()
    naive_gradient = torch.autograd.grad(naive, naive_prefix)[0]
    assert float(optimized.detach()) == pytest.approx(float(naive.detach()))
    assert torch.allclose(optimized_gradient, naive_gradient, atol=1e-7, rtol=1e-7)
    validation = goal.evaluate_validation_groups(
        torch.nn.Linear(1, 1), TinyTokenizer(), torch.tensor([.3]),
        (group,), config, torch.device("cpu"),
    )
    assert validation["selection_objective"] == pytest.approx(float(naive.detach()))


def test_locked_cache_reuses_exact_paths_and_preserves_replay_outputs(monkeypatch):
    calls = 0

    def fake_adapt(model, tokenizer, prefix, trial, **kwargs):
        nonlocal calls
        del model, tokenizer, kwargs
        calls += 1
        sign = 1. if "yes" in trial.target else (-1. if "no" in trial.target else 0.)
        gradient = torch.ones_like(prefix) * (sign + 2.)
        return prefix + sign * .1, torch.tensor(1.), gradient, 3

    monkeypatch.setattr(goal, "raw_text_adapt_prefix", fake_adapt)
    monkeypatch.setattr(
        goal, "goal_raw_logits",
        lambda model, tokenizer, prefix, world, device:
            prefix.sum() * torch.tensor([1., 2., 3., 4.]),
    )
    monkeypatch.setattr(
        goal, "trial3_binary_logits",
        lambda model, tokenizer, prefix, world, device:
            prefix.sum() * torch.tensor([-.5, .5]),
    )
    model = torch.nn.Linear(1, 1)
    prefix = torch.tensor([.3])
    goal.freeze_for_locked_evaluation(model, prefix)
    groups = tuple(replace(_group(), group_key=f"g-{index:03d}") for index in range(72))
    report = goal.evaluate_groups(
        model, TinyTokenizer(), prefix, groups,
        goal.Config(bootstrap_samples=10), torch.device("cpu"), locked=True,
    )
    # Per order: main {2 first reports + 4 signatures + nil first/final}=8;
    # independent replay repeats the same 8. Two orders => 32.
    assert calls == 32 * 72
    assert report["execution"]["deterministic_replay_delta"] == pytest.approx(0.)
    assert report["execution"]["checkpoint_unchanged"] is True
    assert report["execution"]["all_finite"] is True


def test_prior_subtraction_is_candidatewise(monkeypatch):
    values = iter((torch.tensor([10., 20., 30., 40.]), torch.tensor([11., 18., 35., 39.])))
    monkeypatch.setattr(goal, "goal_raw_logits", lambda *args, **kwargs: next(values))
    result = goal.prior_subtracted_goal_logits(
        object(), TinyTokenizer(), torch.zeros(1), torch.ones(1), _world(), torch.device("cpu")
    )
    assert torch.equal(result, torch.tensor([1., -2., 5., -1.]))


def test_trial3_only_scores_yes_no_and_never_adapts(monkeypatch):
    observed = {}

    def fake_completion(model, tokenizer, prefix, prompt, completions, device):
        del model, tokenizer, prefix, prompt, device
        observed["completions"] = completions
        return torch.tensor([-.8, -.2])

    monkeypatch.setattr(goal, "_completion_scores", fake_completion)
    logits = goal.trial3_binary_logits(
        object(), TinyTokenizer(), torch.zeros(2), _world(), torch.device("cpu")
    )
    assert observed["completions"] == goal.BINARY_COMPLETIONS
    assert torch.equal(logits, torch.tensor([-.8, -.2]))


def test_token_length_audit_checks_candidates_binary_and_neutral():
    report = goal.token_length_audit(TinyTokenizer(), (_group(),))
    assert report["passed"] is True
    bad_world = replace(
        _group().worlds[0],
        candidate_completions=(" one", " two words", " three", " four"),
    )
    bad_group = replace(_group(), worlds=(bad_world, *_group().worlds[1:]))
    assert goal.token_length_audit(TinyTokenizer(), (bad_group,))["passed"] is False


def test_group_validation_rejects_surface_or_corruption_defects():
    group = _group()
    broken = replace(group.worlds[0], candidate_prompt="leaked group surface")
    with pytest.raises(ValueError, match="share model-visible surfaces"):
        goal._validate_group(replace(group, worlds=(broken, *group.worlds[1:])))
    fixed_point = replace(group.worlds[0], deranged_family=group.worlds[0].family)
    with pytest.raises(ValueError, match="no fixed point"):
        goal._validate_group(replace(group, worlds=(fixed_point, *group.worlds[1:])))


def test_group_bootstrap_is_deterministic_and_rejects_nonfinite():
    kwargs = dict(samples=100, confidence=.95, seed=7)
    assert goal.one_sided_group_bootstrap([0., .5, 1.], **kwargs) == goal.one_sided_group_bootstrap([0., .5, 1.], **kwargs)
    with pytest.raises(ValueError, match="finite"):
        goal.one_sided_group_bootstrap([math.nan], **kwargs)


def test_validation_selection_is_exact_frozen_six_term_objective_and_ignores_controls():
    report = {
        "orders": [
            {"prior_ce": 2., "single_ce": 1., "final_ce": 2., "semantic_ce": 3., "deranged_accuracy": 0.},
            {"prior_ce": 2., "single_ce": 4., "final_ce": 5., "semantic_ce": 6., "deranged_accuracy": 1.},
        ]
    }
    # .25*2 + mean(1,2,3,4,5,6)
    assert goal.validation_selection_objective(report, goal.Config()) == pytest.approx(4.0)
    report["orders"][0]["deranged_accuracy"] = 999
    assert goal.validation_selection_objective(report, goal.Config()) == pytest.approx(4.0)


def test_validation_evaluator_is_intact_only_and_reports_exact_components(monkeypatch):
    monkeypatch.setattr(
        goal, "goal_raw_logits", lambda *args, **kwargs: torch.zeros(4)
    )
    monkeypatch.setattr(
        goal,
        "raw_text_adapt_prefix",
        lambda model, tokenizer, prefix, trial, **kwargs: (
            prefix + 1, torch.tensor(1.), torch.ones_like(prefix), 2
        ),
    )
    monkeypatch.setattr(
        goal,
        "prior_subtracted_goal_logits",
        lambda *args, **kwargs: torch.tensor([2., 0., 0., 0.]),
    )
    monkeypatch.setattr(
        goal, "trial3_binary_logits", lambda *args, **kwargs: torch.tensor([0., 1.])
    )
    report = goal.evaluate_validation_groups(
        torch.nn.Linear(1, 1), TinyTokenizer(), torch.zeros(1), (_group(),),
        goal.Config(), torch.device("cpu"),
    )
    assert report["mode"] == "intact_validation_only"
    assert report["controls_computed"] == []
    assert report["all_finite"] is True
    assert report["selection_objective"] == pytest.approx(
        goal.validation_selection_objective(report, goal.Config())
    )
    baseline = goal.validation_selection_objective(report, goal.Config())
    report["orders"][0]["deranged_accuracy"] = 999
    assert goal.validation_selection_objective(report, goal.Config()) == pytest.approx(baseline)


def test_checkpoint_hash_covers_weights_and_prefix_and_freezes():
    model = torch.nn.Linear(2, 2)
    prefix = torch.zeros((1, 2), requires_grad=True)
    first = goal.checkpoint_sha256(model, prefix)
    with torch.no_grad():
        prefix.add_(1)
    assert goal.checkpoint_sha256(model, prefix) != first
    digest = goal.freeze_for_locked_evaluation(model, prefix)
    assert digest == goal.checkpoint_sha256(model, prefix)
    assert not prefix.requires_grad
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_protocol_config_rejects_claim_relevant_drift():
    goal.validate_protocol_config(goal.Config())
    with pytest.raises(ValueError, match="configuration drift"):
        goal.validate_protocol_config(goal.Config(bootstrap_samples=9999))
    with pytest.raises(ValueError, match="configuration drift"):
        goal.validate_protocol_config(goal.Config(adam_epsilon=1e-7))
    with pytest.raises(ValueError, match="configuration drift"):
        goal.validate_protocol_config(goal.Config(max_gradient_norm=.5))
    with pytest.raises(ValueError, match="initialization"):
        goal.validate_protocol_config(goal.Config(initialization="other"))


def _passing_report():
    order = {
        "single_pair_mass": .8, "single_top2": .8, "single_pair_imbalance": .1,
        "goal_accuracy": .8, "goal_probability": .7,
        "goal_accuracy_gain": .4, "goal_probability_gain": .3,
        "statusless_goal_logit_delta": 0., "statusless_trial3_logit_delta": 0.,
        "preupdate_goal_logit_delta": 0., "statusless_goal_accuracy": .25,
        "statusless_goal_probability": .25, "statusless_trial3_accuracy": .5,
        "statusless_trial3_probability": .5,
        "deranged_accuracy": .8, "deranged_probability": .7,
        "deranged_original_accuracy": .2, "deranged_original_probability": .15,
        "trial3_accuracy": .8, "trial3_probability": .7, "trial3_brier": .1,
        "trial3_accuracy_gain": .3, "trial3_probability_gain": .2,
        "statusless_entropy_bits": 2., "statusless_max_uniform_deviation": 0.,
    }
    aggregate = {
        **order, "cross_order_agreement": .95, "cross_order_tv": .05,
        "deranged_accuracy": .8, "deranged_probability": .7,
        "deranged_original_accuracy": .2, "deranged_original_probability": .15,
        "trial3_accuracy": .8, "trial3_probability": .7, "trial3_brier": .1,
        "trial3_accuracy_gain": .3, "trial3_probability_gain": .2,
        "statusless_entropy_bits": 2., "statusless_max_uniform_deviation": 0.,
    }
    family = {name: {"goal_accuracy": .8, "goal_probability": .7, "trial3_accuracy": .8, "trial3_probability": .7} for name in goal.GOAL_FAMILIES}
    corrupted = {name: {"deranged_accuracy": .8, "deranged_probability": .7} for name in goal.GOAL_FAMILIES}
    lower = {
        "single_pair_mass": .6, "single_top2": .3, "goal_accuracy": .4,
        "goal_probability": .4, "goal_accuracy_gain": .1,
        "goal_probability_gain": .1, "trial3_accuracy": .6,
        "trial3_probability": .6, "trial3_accuracy_gain": .1,
        "trial3_probability_gain": .1,
    }
    execution = {
        "all_finite": True, "min_prefix_update_l2": 1e-3,
        "deterministic_replay_delta": 0., "trial3_updates": 0,
        "bootstrap_unit": "independent_group", "inner_update_objective": "raw_outcome_nll",
        "inner_candidate_count": 1, "inner_reduction": "mean",
        "token_lengths_audited": True, "checkpoint_unchanged": True,
        "frozen_before_locked_evaluation": True,
        "locked_evaluation": True,
    }
    return {
        "group_count": 72,
        "unique_group_count": 72,
        "aggregate": aggregate,
        "orders": [dict(order), dict(order)],
        "families": family,
        "deranged_families": corrupted,
        "families_by_order": [family, family],
        "deranged_families_by_order": [corrupted, corrupted],
        "bootstrap": {
            "samples": 10_000,
            "confidence": .95,
            "seed": 20_260_804,
            "lower_bounds": lower,
            "by_order": [lower, lower],
        },
        "execution": execution,
    }


def test_gate_requires_every_execution_and_scientific_check():
    report = _passing_report()
    assert goal.apply_gate(report)["passed"] is True
    report["execution"]["trial3_updates"] = 1
    result = goal.apply_gate(report)
    assert result["passed"] is False
    assert result["checks"]["no_trial3_update"] is False


def test_locked_evaluation_and_gate_fail_closed_on_group_or_bootstrap_drift():
    model = torch.nn.Linear(1, 1)
    prefix = torch.zeros(1)
    goal.freeze_for_locked_evaluation(model, prefix)
    with pytest.raises(ValueError, match="exactly 72"):
        goal.evaluate_groups(
            model, TinyTokenizer(), prefix, (_group(),), goal.Config(bootstrap_samples=10),
            torch.device("cpu"), locked=True,
        )
    duplicates = tuple(_group() for _ in range(72))
    with pytest.raises(ValueError, match="unique independent"):
        goal.evaluate_groups(
            model, TinyTokenizer(), prefix, duplicates, goal.Config(bootstrap_samples=10),
            torch.device("cpu"), locked=True,
        )
    for key, bad in (
        ("group_count", 71),
        ("unique_group_count", 71),
    ):
        report = _passing_report()
        report[key] = bad
        assert goal.apply_gate(report)["passed"] is False
    for key, bad in (("samples", 9_999), ("confidence", .90), ("seed", 1)):
        report = _passing_report()
        report["bootstrap"][key] = bad
        assert goal.apply_gate(report)["passed"] is False
