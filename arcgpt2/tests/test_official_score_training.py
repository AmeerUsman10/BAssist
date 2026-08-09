from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arcgpt2 import official_score_training as training


def _metrics(accuracy: float = 0.70, probability: float = 0.60):
    return {
        "all_finite": True,
        "aggregate": {
            "accuracy": accuracy,
            "truth_probability": probability,
        },
        "per_action": {},
        "groups": [],
    }


def _training_receipt(validation):
    return {
        "steps_completed": 512,
        "best_epoch": 2,
        "best_objective": 0.5,
        "selected_validation_sha256": "selected",
        "history": [
            {
                "epoch": 1,
                "steps_completed": 256,
                "validation_selection_objective": 0.8,
                "validation": validation,
                "all_finite": True,
            },
            {
                "epoch": 2,
                "steps_completed": 512,
                "validation_selection_objective": 0.5,
                "validation": validation,
                "all_finite": True,
            },
        ],
        "all_finite": True,
        "old_locked_test_evaluations": 0,
    }


def _execution(**overrides):
    values = {
        "pretrained_original_gpt2": True,
        "positive_control_called_once": True,
        "fresh_canary_disjoint": True,
        "fresh_canary_balanced_a1_a4": True,
        "model_weights_unchanged_during_canary": True,
        "initial_prefix_unchanged_during_canary": True,
        "model_parameters_frozen_for_canary": True,
        "deterministic_algorithms_enabled": True,
        "tf32_disabled": True,
        "source_sha_resolved": True,
        "no_weight_serialization": True,
        "runtime_projection_finite_and_within_ceiling": True,
    }
    values.update(overrides)
    return values


def test_module_keeps_model_dependencies_lazy() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_level_from_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "torch" not in top_level_imports | top_level_from_imports
    assert "transformers" not in top_level_imports | top_level_from_imports
    assert "torch.save" not in source
    assert ".save_pretrained(" not in source


def test_gate_accepts_exact_boundaries_and_rejects_nonfinite() -> None:
    validation = _metrics()
    canary = _metrics()
    result = training.apply_gate(
        _training_receipt(validation), validation, canary, _execution()
    )
    assert result["passed"] is True
    assert all(result["checks"].values())

    canary = _metrics(probability=float("nan"))
    canary["all_finite"] = False
    result = training.apply_gate(
        _training_receipt(validation), validation, canary, _execution()
    )
    assert result["passed"] is False
    assert result["checks"]["canary_all_finite"] is False
    assert result["checks"]["canary_truth_probability_at_least_0_60"] is False


def test_strict_json_receipt_replaces_nonfinite_values_explicitly() -> None:
    receipt = training.strict_json_receipt(
        {"metric": float("nan"), "nested": [float("inf"), 1.0]}
    )
    assert receipt["metric"] is None
    assert receipt["nested"] == [None, 1.0]
    assert receipt["strict_json"]["nonfinite_values_replaced_with_null"] == 2
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt


def test_frozen_state_hashes_and_lane_clone_are_byte_exact(tmp_path: Path) -> None:
    torch.manual_seed(17)
    source_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_prefix = torch.randn(2, 4)
    lane_model = copy.deepcopy(source_model)
    lane_prefix = source_prefix.clone()

    first = training.frozen_state_hashes(source_model, source_prefix)
    second = training.frozen_state_hashes(source_model, source_prefix)
    assert first == second
    assert training.model_weights_sha256(source_model) == first["model_weights_sha256"]
    assert training.prefix_sha256(source_prefix) == first[
        "selected_initial_prefix_sha256"
    ]

    verification = training.verify_in_memory_lane_clone(
        source_model,
        source_prefix,
        lane_model,
        lane_prefix,
        expected_device="cpu",
    )
    assert verification["passed"] is True
    assert verification["tensor_serialization_performed"] is False
    assert list(tmp_path.iterdir()) == []

    with torch.no_grad():
        next(lane_model.parameters()).add_(1e-3)
    changed = training.verify_in_memory_lane_clone(
        source_model,
        source_prefix,
        lane_model,
        lane_prefix,
        expected_device="cpu",
    )
    assert changed["passed"] is False
    assert changed["checks"]["model_weights_exact"] is False
    assert (
        inspect.signature(training.verify_in_memory_lane_clone)
        .parameters["expected_device"]
        .default
        == "cuda:1"
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="exact cuda:1 clone check requires two visible CUDA devices",
)
def test_frozen_hashes_are_device_independent_on_cuda_1() -> None:
    source_model = torch.nn.Linear(3, 2)
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_prefix = torch.randn(2, 3)
    lane_model = copy.deepcopy(source_model).to("cuda:1")
    lane_prefix = source_prefix.clone().to("cuda:1")

    verification = training.verify_in_memory_lane_clone(
        source_model,
        source_prefix,
        lane_model,
        lane_prefix,
    )
    assert verification["passed"] is True
    assert verification["observed_lane_devices"] == ["cuda:1"]


def test_config_rejects_claim_relevant_drift() -> None:
    training.validate_config(training.Config())
    with pytest.raises(ValueError, match="configuration drift"):
        training.validate_config(training.Config(max_actions_per_public_game=159))


class Action(Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class FakeSplitSpec:
    name: str
    seed_start: int
    seed_stop: int
    groups: int


@dataclass(frozen=True)
class FakeLayout:
    split: str
    game_seed: int
    before_grid_sha256: str


@dataclass(frozen=True)
class FakeManifest:
    name: str
    seed_start: int
    seed_stop: int
    requested_groups: int
    accepted: tuple[FakeLayout, ...]
    rejected: tuple = ()


class FlexibleConfig:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _json_sha(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.__dict__,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_payload(manifest: FakeManifest):
    return {
        "name": manifest.name,
        "seed_range": [manifest.seed_start, manifest.seed_stop],
        "requested_groups": manifest.requested_groups,
        "accepted": [item.__dict__ for item in manifest.accepted],
        "rejected": [],
    }


def test_prepare_runs_one_fit_and_one_disjoint_balanced_canary(
    tmp_path: Path,
) -> None:
    actions = tuple(Action)
    directions = tuple(Direction)
    validation = _metrics(0.82, 0.74)
    canary = _metrics(0.78, 0.68)
    canary["actions"] = [action.value for action in actions]
    counters = {
        "train": 0,
        "evaluate": 0,
        "quartet": 0,
        "seed": None,
        "config_validated": 0,
    }
    model = torch.nn.Linear(4, 4)
    tokenizer = object()
    trained_prefix = torch.randn(2, 4, requires_grad=True)

    def make_manifest(name: str, start: int, count: int) -> FakeManifest:
        layouts = tuple(
            FakeLayout(name, start + index, f"{start + index:064x}")
            for index in range(count)
        )
        return FakeManifest(name, start, start + 100_000, count, layouts)

    canonical = (
        make_manifest("train", 1_100_000, 2),
        make_manifest("validation", 1_200_000, 1),
        make_manifest("locked_test", 1_300_000, 1),
    )
    fresh = make_manifest(training.CANARY_SPLIT_NAME, 1_500_000, 8)

    def train_positive_control(config, device):
        del config, device
        counters["train"] += 1
        receipt = _training_receipt(validation)
        receipt["selected_validation_sha256"] = _json_sha(validation)
        return model, tokenizer, trained_prefix, receipt, canonical

    def evaluate_action_groups(
        observed_model,
        observed_tokenizer,
        prefix,
        layouts,
        observed_actions,
        config,
        device,
        *,
        bootstrap_seed,
    ):
        del observed_tokenizer, config, device
        counters["evaluate"] += 1
        assert observed_model is model
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert prefix is not trained_prefix
        assert tuple(layouts) == fresh.accepted
        assert tuple(observed_actions) == actions
        assert bootstrap_seed == training.CANARY_BOOTSTRAP_SEED
        return canary

    def build_quartet(layout, action):
        del layout, action
        counters["quartet"] += 1
        return tuple(SimpleNamespace(direction=direction) for direction in directions)

    def canonical_payload(manifests):
        return {"manifests": [_manifest_payload(item) for item in manifests]}

    dependencies = training._Dependencies(
        torch=torch,
        EggrollConfig=FlexibleConfig,
        HeldoutConfig=FlexibleConfig,
        SplitSpec=FakeSplitSpec,
        train_positive_control=train_positive_control,
        validate_eggroll_config=lambda config: counters.__setitem__(
            "config_validated", counters["config_validated"] + 1
        ),
        build_split_manifests=lambda specs: (fresh,),
        build_heldout_quartet=build_quartet,
        evaluate_action_groups=evaluate_action_groups,
        audit_split_disjointness=lambda old, new: {
            "passed": old == canonical and new == fresh,
            "seed_overlap": [],
            "before_grid_hash_overlap": [],
        },
        canonical_manifest_payload=canonical_payload,
        canonical_json_sha256=_json_sha,
        manifest_dict=_manifest_payload,
        set_seed=lambda seed: counters.__setitem__("seed", seed),
        actions=actions,
        directions=directions,
    )
    clock_values = iter((0.0, 1.0, 3.0, 4.0, 6.0, 7.0))
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        prepared = training.prepare_training_canary(
            training.Config(source_sha="f" * 40, output_dir=str(tmp_path)),
            _dependencies=dependencies,
            _clock=lambda: next(clock_values),
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    receipt = prepared.receipt
    assert counters == {
        "train": 1,
        "evaluate": 1,
        "quartet": 8 * 4,
        "seed": 424_242,
        "config_validated": 1,
    }
    assert prepared.model is model
    assert prepared.tokenizer is tokenizer
    assert prepared.initial_prefix is not trained_prefix
    assert receipt["training"]["steps_completed"] == 512
    assert receipt["training"]["old_locked_test_evaluations"] == 0
    assert receipt["manifest"]["fresh_canary"]["disjointness"]["passed"] is True
    assert receipt["manifest"]["fresh_canary"]["balance"]["all_balanced"] is True
    assert receipt["checkpoint"]["model_weights_unchanged"] is True
    assert receipt["checkpoint"]["selected_initial_prefix_unchanged"] is True
    assert receipt["canary"]["aggregate"] == {
        "accuracy": 0.78,
        "truth_probability": 0.68,
    }
    assert receipt["runtime_projection"]["observed"][
        "exact_gradient_adaptations"
    ] == 256
    assert receipt["runtime_projection"]["public_score_inputs"][
        "max_environment_actions"
    ] == 4_000
    assert receipt["gate"]["passed"] is True
    assert receipt["eligible_for_official_public_score"] is True
    assert receipt["model_weights_persisted"] is False
    assert receipt["strict_json"]["nonfinite_values_replaced_with_null"] == 0
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    assert list(tmp_path.iterdir()) == []
