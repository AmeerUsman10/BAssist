from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from arcgpt2 import meta_soft_heldout_binding as heldout
from arcgpt2.meta_soft_binding import mapping_query, transition_prompt
from arcgpt2.meta_soft_raw_outcome_overfit import raw_support_text
from arcgpt2.phase0_hidden_action import Action, Direction


def test_default_manifests_are_deterministic_disjoint_and_complete() -> None:
    first = heldout.build_split_manifests()
    second = heldout.build_split_manifests()

    assert first == second
    assert [item.name for item in first] == ["train", "validation", "locked_test"]
    assert [len(item.accepted) for item in first] == [256, 64, 64]
    accepted = [layout for manifest in first for layout in manifest.accepted]
    assert len({layout.game_seed for layout in accepted}) == 384
    assert len({layout.before_grid_sha256 for layout in accepted}) == 384
    for manifest, expected in zip(first, heldout.DEFAULT_SPLIT_SPECS, strict=True):
        assert manifest.seed_start == expected.seed_start
        assert manifest.seed_stop == expected.seed_stop
        assert all(
            expected.seed_start <= layout.game_seed < expected.seed_stop
            for layout in manifest.accepted
        )
        scanned = sorted(
            [layout.game_seed for layout in manifest.accepted]
            + [item.game_seed for item in manifest.rejected]
        )
        assert scanned == list(range(expected.seed_start, max(scanned) + 1))
        assert all(
            item.reason in {"no_valid_probe_cell", "duplicate_before_grid", "duplicate_seed"}
            for item in manifest.rejected
        )

    first_digest = heldout.manifest_sha256(first)
    assert first_digest == heldout.manifest_sha256(second)
    assert len(first_digest) == 64
    payload = heldout.canonical_manifest_payload(first)
    assert payload["fixed_protocol_fields"]["protocol"] == heldout.PROTOCOL
    assert [
        item["game_seed"] for item in payload["manifests"][2]["rejected"]
    ] == [item.game_seed for item in first[2].rejected]
    changed_layout = replace(
        first[0].accepted[0],
        before_grid_sha256="0" * 64,
    )
    changed_manifest = replace(
        first[0],
        accepted=(changed_layout, *first[0].accepted[1:]),
    )
    assert heldout.manifest_sha256((changed_manifest, *first[1:])) != first_digest


def test_full_generated_text_audit_excludes_locked_literals() -> None:
    audit = heldout.audit_generated_literal_action_text(
        heldout.build_split_manifests()
    )

    assert audit["claim"] == "unseen_literal_action_surface_invariance_only"
    assert "action_slot_binding_holdout" in audit["not_claimed"]
    assert "token_id_holdout" in audit["not_claimed"]
    assert audit["group_count"] == 320
    assert audit["support_example_count"] == 1280
    assert audit["query_example_count"] == 1280
    assert audit["literal_occurrences"] == {"A2": 0, "A3": 0, "A4": 0}
    assert audit["all_quartets_have_four_unique_targets"] is True
    assert audit["cardinal_semantic_guard_passed"] is True
    assert audit["passed"] is True


@pytest.mark.parametrize("action", tuple(Action))
def test_heldout_quartet_uses_generated_level_one_safe_geometry(action: Action) -> None:
    layout = heldout.probe_layout(1_300_000, "locked_test")
    quartet = heldout.build_heldout_quartet(layout, action)

    assert layout.source_level_index == 1
    assert (layout.height, layout.width) == (6, 6)
    assert 0 < layout.probe_row < 5
    assert 0 < layout.probe_column < 5
    assert {episode.action for episode in quartet} == {action}
    assert {episode.direction for episode in quartet} == set(Direction)
    assert {episode.record.status for episode in quartet} == {"ACTIVE"}
    assert all(episode.record.moved for episode in quartet)
    assert len({episode.record.before for episode in quartet}) == 1
    assert len({transition_prompt(episode.record) for episode in quartet}) == 1
    assert len({episode.record.after for episode in quartet}) == 4


def test_train_and_validation_text_expose_only_literal_A1() -> None:
    manifests = {item.name: item for item in heldout.build_split_manifests()}
    locked_literals = {item.value for item in heldout.LOCKED_ACTIONS}
    for split in ("train", "validation"):
        assert heldout.actions_for_split(split) == (Action.A1,)
        layout = manifests[split].accepted[0]
        episode = heldout.build_heldout_quartet(layout, Action.A1)[0]
        prompt, target = raw_support_text(episode.record, episode.record)
        visible = "\n".join((prompt, target, mapping_query(Action.A1)))
        assert "A1" in visible
        assert all(literal not in visible for literal in locked_literals)

    assert heldout.actions_for_split("locked_test") == tuple(Action)
    locked_layout = manifests["locked_test"].accepted[0]
    for action in heldout.LOCKED_ACTIONS:
        episode = heldout.build_heldout_quartet(locked_layout, action)[0]
        prompt, target = raw_support_text(episode.record, episode.record)
        assert action.value in prompt + target + mapping_query(action)


def test_fixed_deranged_corruption_is_a_quartet_permutation() -> None:
    layout = heldout.probe_layout(1_300_000, "locked_test")
    quartet = heldout.build_heldout_quartet(layout, Action.A3)
    intact = {episode.record.after for episode in quartet}
    corrupted = {
        heldout.corrupted_target(episode.record, episode.direction)[0].after
        for episode in quartet
    }

    assert corrupted == intact


def test_heldout_outer_loss_delegates_to_raw_nll_objective(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_raw_loss(model, tokenizer, prefix, episode, config, device):
        captured.update(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            episode=episode,
            config=config,
            device=device,
        )
        return prefix.sum(), {"support_loss": 1.0}

    monkeypatch.setattr(heldout, "raw_episode_meta_loss", fake_raw_loss)
    prefix = torch.zeros((2, 3), requires_grad=True)
    config = heldout.Config()
    loss, diagnostics = heldout.heldout_episode_meta_loss(
        "model", "tokenizer", prefix, "episode", config, torch.device("cpu")
    )

    assert loss is not None
    assert diagnostics == {"support_loss": 1.0}
    raw_config = captured["config"]
    assert raw_config.inner_learning_rate == 0.2
    assert heldout.SUPPORT_OBJECTIVE == "raw_outcome_nll"


def test_group_bootstrap_is_deterministic_and_validates_inputs() -> None:
    kwargs = {"samples": 200, "confidence": 0.95, "seed": 17}
    first = heldout.one_sided_bootstrap_lower_bound([0.0, 0.5, 1.0], **kwargs)
    second = heldout.one_sided_bootstrap_lower_bound([0.0, 0.5, 1.0], **kwargs)

    assert first == second
    assert 0.0 <= first <= 0.5
    assert heldout.one_sided_bootstrap_lower_bound([0.7], **kwargs) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="may not be empty"):
        heldout.one_sided_bootstrap_lower_bound([], **kwargs)
    with pytest.raises(ValueError, match="must be finite"):
        heldout.one_sided_bootstrap_lower_bound([math.nan], **kwargs)


def test_control_metrics_use_worst_case_not_a_mean() -> None:
    result = heldout._aggregate_metric_rows(
        [
            {
                "accuracy": 0.5,
                "prior_entropy_bits": 2.0,
                "prior_max_abs_uniform_deviation": 0.01,
                "max_preupdate_logit_delta": 0.0,
            },
            {
                "accuracy": 1.0,
                "prior_entropy_bits": 1.8,
                "prior_max_abs_uniform_deviation": 0.20,
                "max_preupdate_logit_delta": 2e-6,
            },
        ]
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["prior_entropy_bits"] == pytest.approx(1.8)
    assert result["prior_max_abs_uniform_deviation"] == pytest.approx(0.20)
    assert result["max_preupdate_logit_delta"] == pytest.approx(2e-6)


def _passing_stratum(actions: tuple[Action, ...]) -> dict[str, object]:
    aggregate = {
        "accuracy": 0.70,
        "truth_probability": 0.60,
        "accuracy_gain": 0.35,
        "truth_probability_gain": 0.25,
        "prior_entropy_bits": 1.95,
        "prior_max_abs_uniform_deviation": 0.05,
        "max_preupdate_logit_delta": 1e-6,
        "corrupted_target_accuracy": 0.70,
        "corrupted_mean_target_probability": 0.60,
        "corrupted_original_truth_accuracy": 0.25,
        "corrupted_mean_original_truth_probability": 0.20,
    }
    return {
        "all_finite": True,
        "aggregate": aggregate,
        "bootstrap": {
            "lower_bounds": {
                "accuracy": 0.25001,
                "truth_probability": 0.25001,
                "accuracy_gain": 1e-8,
                "truth_probability_gain": 1e-8,
            }
        },
        "per_action": {
            action.value: {
                "accuracy": 0.55,
                "truth_probability": 0.45,
                "corrupted_target_accuracy": 0.55,
                "corrupted_mean_target_probability": 0.45,
            }
            for action in actions
        },
    }


def _passing_execution() -> dict[str, bool]:
    return {
        "finite_training": True,
        "literal_action_text_audit": True,
        "inner_update_contract": True,
        "checkpoint_unchanged_during_locked_test": True,
        "selected_validation_matches_history": True,
        "epoch_steps_exact": True,
    }


def test_heldout_gate_accepts_all_inclusive_metric_boundaries() -> None:
    geometry = _passing_stratum((Action.A1,))
    primary = _passing_stratum(heldout.LOCKED_ACTIONS)
    result = heldout.apply_gate(geometry, primary, _passing_execution())

    assert result["name"] == "heldout_layout_action_v1"
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_heldout_gate_rejects_nonfinite_stratum_flag() -> None:
    geometry = _passing_stratum((Action.A1,))
    primary = _passing_stratum(heldout.LOCKED_ACTIONS)
    geometry["all_finite"] = False

    assert heldout.apply_gate(
        geometry, primary, _passing_execution()
    )["passed"] is False


@pytest.mark.parametrize(
    ("stratum", "section", "metric", "bad_value"),
    (
        ("geometry", "aggregate", "accuracy", 0.6999),
        ("geometry", "aggregate", "truth_probability", 0.5999),
        ("geometry", "aggregate", "accuracy_gain", 0.3499),
        ("geometry", "aggregate", "truth_probability_gain", 0.2499),
        ("geometry", "aggregate", "prior_entropy_bits", 1.9499),
        ("geometry", "aggregate", "prior_max_abs_uniform_deviation", 0.0501),
        ("geometry", "aggregate", "max_preupdate_logit_delta", 1.01e-6),
        ("geometry", "aggregate", "corrupted_target_accuracy", 0.6999),
        ("geometry", "aggregate", "corrupted_mean_target_probability", 0.5999),
        ("geometry", "aggregate", "corrupted_original_truth_accuracy", 0.2501),
        ("geometry", "aggregate", "corrupted_mean_original_truth_probability", 0.2001),
        ("primary", "aggregate", "accuracy", 0.6999),
        ("primary", "aggregate", "truth_probability", 0.5999),
        ("primary", "aggregate", "accuracy_gain", 0.3499),
        ("primary", "aggregate", "truth_probability_gain", 0.2499),
        ("primary", "bootstrap", "accuracy", 0.25),
        ("primary", "bootstrap", "truth_probability", 0.25),
        ("primary", "bootstrap", "accuracy_gain", 0.0),
        ("primary", "bootstrap", "truth_probability_gain", 0.0),
    ),
)
def test_heldout_gate_rejects_failed_stratum_requirement(
    stratum: str,
    section: str,
    metric: str,
    bad_value: float,
) -> None:
    geometry = _passing_stratum((Action.A1,))
    primary = _passing_stratum(heldout.LOCKED_ACTIONS)
    target = geometry if stratum == "geometry" else primary
    if section == "bootstrap":
        target["bootstrap"]["lower_bounds"][metric] = bad_value
    else:
        target[section][metric] = bad_value

    assert heldout.apply_gate(
        geometry, primary, _passing_execution()
    )["passed"] is False


@pytest.mark.parametrize(("action", "metric", "bad_value"), (
    (Action.A2, "accuracy", 0.5499),
    (Action.A3, "truth_probability", 0.4499),
    (Action.A4, "accuracy", 0.0),
    (Action.A2, "corrupted_target_accuracy", 0.5499),
    (Action.A4, "corrupted_mean_target_probability", 0.4499),
))
def test_heldout_gate_rejects_each_weak_locked_action(
    action: Action,
    metric: str,
    bad_value: float,
) -> None:
    geometry = _passing_stratum((Action.A1,))
    primary = _passing_stratum(heldout.LOCKED_ACTIONS)
    primary["per_action"][action.value][metric] = bad_value

    assert heldout.apply_gate(
        geometry, primary, _passing_execution()
    )["passed"] is False


def test_checkpoint_selection_keeps_earlier_epoch_on_exact_tie() -> None:
    assert heldout.checkpoint_is_better(0.9, 1.0) is True
    assert heldout.checkpoint_is_better(1.0, 1.0) is False
    assert heldout.Config().epochs == 2

    history = [
        {"epoch": 1, "validation_selection_objective": 0.8},
        {"epoch": 2, "validation_selection_objective": 0.8},
    ]
    assert heldout.best_validation_history_entry(history)["epoch"] == 1


@pytest.mark.parametrize(
    "failed_invariant",
    tuple(_passing_execution()),
)
def test_gate_rejects_each_failed_execution_invariant(
    failed_invariant: str,
) -> None:
    execution = _passing_execution()
    execution[failed_invariant] = False

    assert heldout.apply_gate(
        _passing_stratum((Action.A1,)),
        _passing_stratum(heldout.LOCKED_ACTIONS),
        execution,
    )["passed"] is False


def test_inner_update_contract_is_explicit_and_strict() -> None:
    contract = {
        "objective": "raw_outcome_nll",
        "candidate_count": 1,
        "reduction": "mean",
        "counterfactuals_used_in_inner_update": False,
        "unique_targets_per_quartet": 4,
        "cardinal_semantic_guard_passed": True,
        "attention_implementation": "eager",
        "eager_attention": True,
    }
    assert heldout.inner_update_contract_passes(contract) is True
    for key, bad_value in (
        ("objective", "contrastive"),
        ("candidate_count", 4),
        ("reduction", "sum"),
        ("counterfactuals_used_in_inner_update", True),
        ("unique_targets_per_quartet", 3),
        ("cardinal_semantic_guard_passed", False),
        ("eager_attention", False),
    ):
        changed = dict(contract)
        changed[key] = bad_value
        assert heldout.inner_update_contract_passes(changed) is False


def test_preregistered_hyperparameters_reject_drift(tmp_path) -> None:
    heldout.validate_protocol_config(heldout.Config())
    local_model = tmp_path / "pinned-gpt2"
    local_model.mkdir()
    heldout.validate_protocol_config(heldout.Config(model_name=str(local_model)))
    with pytest.raises(ValueError, match="configuration drift"):
        heldout.validate_protocol_config(heldout.Config(bootstrap_samples=9999))
    with pytest.raises(ValueError, match="configuration drift"):
        heldout.validate_protocol_config(heldout.Config(model_revision="drifted"))
