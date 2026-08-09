"""In-memory GPT-2 training and fresh exact-gradient canary for public scoring.

The bridge deliberately reuses the already validated positive-control fit from
``eggroll_inner_memory``.  It calls that fit exactly once, stops before the old
locked synthetic evaluation, freezes the validation-selected model and initial
soft prefix, and evaluates a fresh disjoint A1--A4 canary with the existing
exact-gradient held-out evaluator.

This module has no top-level torch or transformers import.  Importing receipt
helpers therefore remains cheap in the official runner; model dependencies are
loaded only when :func:`prepare_training_canary` is called.  Trained weights are
never serialized.  The returned bundle keeps them in memory while its
``receipt`` field is strict-JSON data suitable for durable evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL = "official_score_training_canary_v1"
RESULT_SCHEMA_VERSION = 1
MODEL_NAME = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TRAINING_SEED = 424_242
TRAINING_STEPS = 512
CANARY_SPLIT_NAME = "official_score_canary"
CANARY_SEED_START = 1_500_000
CANARY_SEED_STOP = 1_600_000
CANARY_LAYOUT_GROUPS = 8
CANARY_BOOTSTRAP_SEED = 20_260_811
PUBLIC_GAME_COUNT = 25
MAX_ACTIONS_PER_PUBLIC_GAME = 160
MAX_RESETS_PER_PUBLIC_GAME = 3
RUNTIME_CEILING_SECONDS = 10_800.0


@dataclass(frozen=True)
class Config:
    """Frozen preparation inputs that do not expose training-recipe drift."""

    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    source_sha: str | None = None
    output_dir: str = "outputs/official_score_training"
    seed: int = TRAINING_SEED
    require_cuda: bool = False
    public_game_count: int = PUBLIC_GAME_COUNT
    max_actions_per_public_game: int = MAX_ACTIONS_PER_PUBLIC_GAME
    max_resets_per_public_game: int = MAX_RESETS_PER_PUBLIC_GAME
    runtime_ceiling_seconds: float = RUNTIME_CEILING_SECONDS


@dataclass(frozen=True)
class PreparedTrainingCanary:
    """One in-memory trained model plus its JSON-only canary receipt."""

    model: Any
    tokenizer: Any
    initial_prefix: Any
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    EggrollConfig: Any
    HeldoutConfig: Any
    SplitSpec: Any
    train_positive_control: Callable[..., Any]
    validate_eggroll_config: Callable[..., Any]
    build_split_manifests: Callable[..., Any]
    build_heldout_quartet: Callable[..., Any]
    evaluate_action_groups: Callable[..., Any]
    audit_split_disjointness: Callable[..., Any]
    canonical_manifest_payload: Callable[..., Any]
    canonical_json_sha256: Callable[..., str]
    manifest_dict: Callable[..., Any]
    set_seed: Callable[[int], None]
    actions: tuple[Any, ...]
    directions: tuple[Any, ...]


def _load_dependencies() -> _Dependencies:
    """Load model dependencies only for an actual preparation run."""

    import torch

    from . import eggroll_inner_memory as eggroll
    from . import meta_soft_heldout_binding as heldout
    from .meta_soft_binding import _DIRECTIONS, set_seed

    return _Dependencies(
        torch=torch,
        EggrollConfig=eggroll.Config,
        HeldoutConfig=heldout.Config,
        SplitSpec=heldout.SplitSpec,
        train_positive_control=eggroll.train_positive_control,
        validate_eggroll_config=eggroll.validate_protocol_config,
        build_split_manifests=heldout.build_split_manifests,
        build_heldout_quartet=heldout.build_heldout_quartet,
        evaluate_action_groups=heldout.evaluate_action_groups,
        audit_split_disjointness=eggroll.audit_split_disjointness,
        canonical_manifest_payload=heldout.canonical_manifest_payload,
        canonical_json_sha256=heldout.canonical_json_sha256,
        manifest_dict=heldout._manifest_dict,
        set_seed=set_seed,
        actions=(heldout.TRAIN_ACTION, *heldout.LOCKED_ACTIONS),
        directions=tuple(_DIRECTIONS),
    )


def validate_config(config: Config) -> None:
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "seed": TRAINING_SEED,
        "public_game_count": PUBLIC_GAME_COUNT,
        "max_actions_per_public_game": MAX_ACTIONS_PER_PUBLIC_GAME,
        "max_resets_per_public_game": MAX_RESETS_PER_PUBLIC_GAME,
        "runtime_ceiling_seconds": RUNTIME_CEILING_SECONDS,
    }
    drift = {
        name: {"expected": expected_value, "observed": getattr(config, name)}
        for name, expected_value in expected.items()
        if getattr(config, name) != expected_value
    }
    if drift:
        raise ValueError(f"{PROTOCOL} configuration drift: {drift}")


def resolve_source_sha(config: Config) -> str | None:
    return (
        config.source_sha
        or os.environ.get("ARC_GPT2_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA")
    )


def _build_eggroll_config(config: Config, dependencies: _Dependencies) -> Any:
    """Instantiate the existing preregistered positive-control recipe."""

    eggroll_config = dependencies.EggrollConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        source_sha=config.source_sha,
        output_dir=config.output_dir,
        seed=config.seed,
        epochs=2,
        prefix_length=8,
        prefix_initialization_std=0.01,
        inner_learning_rate=0.2,
        prefix_learning_rate=1e-3,
        model_learning_rate=1e-4,
        weight_decay=0.01,
        no_evidence_weight=0.25,
        freeze_first_n_blocks=11,
        bootstrap_samples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_seed=20_260_809,
        perturbation_rank=1,
        antithetic_pairs=512,
        sigma=0.001,
        population_batch_size=32,
        require_cuda=config.require_cuda,
    )
    dependencies.validate_eggroll_config(eggroll_config)
    return eggroll_config


def _build_heldout_config(config: Config, dependencies: _Dependencies) -> Any:
    """Build the evaluator config used inside the positive-control fit."""

    return dependencies.HeldoutConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        source_sha=config.source_sha,
        initialization="pretrained",
        output_dir=config.output_dir,
        seed=config.seed,
        epochs=2,
        prefix_length=8,
        prefix_initialization_std=0.01,
        inner_learning_rate=0.2,
        prefix_learning_rate=1e-3,
        model_learning_rate=1e-4,
        weight_decay=0.01,
        no_evidence_weight=0.25,
        freeze_first_n_blocks=11,
        bootstrap_samples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_seed=20_260_803,
        save_model=False,
        require_cuda=config.require_cuda,
    )


def _hash_named_tensors(tensors: Iterable[tuple[str, Any]]) -> str:
    """Hash tensor values without serializing a checkpoint."""

    digest = hashlib.sha256()
    for name, value in tensors:
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_weights_sha256(model: Any) -> str:
    """Hash the complete model state separately from the soft prefix."""

    return _hash_named_tensors(sorted(model.state_dict().items()))


def prefix_sha256(prefix: Any) -> str:
    """Hash the validation-selected per-game initial soft prefix."""

    return _hash_named_tensors((("__selected_initial_soft_prefix__", prefix),))


def frozen_state_hashes(model: Any, prefix: Any) -> dict[str, str]:
    """Return separate and combined hashes for an in-memory frozen state.

    The function reads tensor bytes through CPU views but never writes a model
    or tensor artifact.  Hashes are therefore independent of whether the live
    state resides on CPU, ``cuda:0``, or the official runner's ``cuda:1`` lane.
    """

    model_items = tuple(sorted(model.state_dict().items()))
    prefix_item = ("__selected_initial_soft_prefix__", prefix)
    return {
        "model_weights_sha256": _hash_named_tensors(model_items),
        "selected_initial_prefix_sha256": _hash_named_tensors((prefix_item,)),
        "combined_sha256": _hash_named_tensors((*model_items, prefix_item)),
    }


def verify_in_memory_lane_clone(
    source_model: Any,
    source_prefix: Any,
    lane_model: Any,
    lane_prefix: Any,
    *,
    expected_device: str = "cuda:1",
) -> dict[str, Any]:
    """Verify an exact, device-correct clone without serializing weights.

    The official runner can call this immediately after cloning its selected
    checkpoint onto the isolated scoring lane.  Every model-state tensor and
    the prefix must reside on ``expected_device`` and match the source bytes.
    """

    source = frozen_state_hashes(source_model, source_prefix)
    lane = frozen_state_hashes(lane_model, lane_prefix)
    lane_devices = sorted(
        {
            *(str(value.device) for value in lane_model.state_dict().values()),
            str(lane_prefix.device),
        }
    )
    source_model_frozen = all(
        not bool(parameter.requires_grad) for parameter in source_model.parameters()
    )
    lane_model_frozen = all(
        not bool(parameter.requires_grad) for parameter in lane_model.parameters()
    )
    checks = {
        "model_weights_exact": (
            source["model_weights_sha256"] == lane["model_weights_sha256"]
        ),
        "selected_initial_prefix_exact": (
            source["selected_initial_prefix_sha256"]
            == lane["selected_initial_prefix_sha256"]
        ),
        "combined_state_exact": source["combined_sha256"] == lane["combined_sha256"],
        "lane_device_exact": lane_devices == [expected_device],
        "source_model_parameters_frozen": source_model_frozen,
        "lane_model_parameters_frozen": lane_model_frozen,
        "source_prefix_frozen": not bool(source_prefix.requires_grad),
        "lane_prefix_frozen": not bool(lane_prefix.requires_grad),
    }
    return {
        "expected_device": expected_device,
        "observed_lane_devices": lane_devices,
        "source": source,
        "lane": lane,
        "checks": checks,
        "passed": all(checks.values()),
        "tensor_serialization_performed": False,
    }


def _numeric_values(value: Any) -> Iterable[float]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _numeric_values(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _numeric_values(item)


def _all_numeric_values_finite(value: Any) -> bool:
    return all(math.isfinite(item) for item in _numeric_values(value))


def _selected_validation(training: Mapping[str, Any]) -> Mapping[str, Any]:
    history = training.get("history")
    if not isinstance(history, Sequence) or not history:
        raise RuntimeError("positive-control training returned no validation history")
    selected = min(
        history,
        key=lambda item: (
            float(item["validation_selection_objective"]),
            int(item["epoch"]),
        ),
    )
    if int(selected["epoch"]) != int(training.get("best_epoch", -1)):
        raise RuntimeError("positive-control best epoch disagrees with its history")
    return selected["validation"]


def _audit_balance(
    layouts: Sequence[Any],
    actions: Sequence[Any],
    directions: Sequence[Any],
    build_quartet: Callable[[Any, Any], Sequence[Any]],
) -> dict[str, Any]:
    expected = tuple(str(direction.value) for direction in directions)
    expected_set = set(expected)
    per_action: dict[str, Any] = {}
    all_balanced = True
    for action in actions:
        action_name = str(action.value)
        direction_counts = {direction: 0 for direction in expected}
        quartet_count = 0
        for layout in layouts:
            quartet = tuple(build_quartet(layout, action))
            observed = tuple(str(episode.direction.value) for episode in quartet)
            balanced = len(observed) == len(expected) and set(observed) == expected_set
            all_balanced = all_balanced and balanced
            quartet_count += 1
            for direction in observed:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
        per_action[action_name] = {
            "layout_quartets": quartet_count,
            "episodes": sum(direction_counts.values()),
            "direction_counts": direction_counts,
            "balanced": all(
                count == len(layouts) for count in direction_counts.values()
            ),
        }
        all_balanced = all_balanced and bool(per_action[action_name]["balanced"])
    return {
        "layout_groups": len(layouts),
        "actions": [str(action.value) for action in actions],
        "direction_order": list(expected),
        "variants_per_action_layout": len(expected),
        "per_action": per_action,
        "all_balanced": all_balanced,
    }


def _runtime_projection(
    config: Config,
    *,
    training_seconds: float,
    canary_seconds: float,
    layout_count: int,
    action_count: int,
    variants_per_action: int,
) -> dict[str, Any]:
    canary_episodes = layout_count * action_count * variants_per_action
    exact_gradient_adaptations = canary_episodes * 2
    direction_query_calls = canary_episodes * 3
    observed_mixed_work_units = exact_gradient_adaptations + direction_query_calls
    public_environment_actions = (
        config.public_game_count * config.max_actions_per_public_game
    )
    projected_mixed_work_units = public_environment_actions * (1 + action_count)
    seconds_per_mixed_work_unit = (
        canary_seconds / observed_mixed_work_units
        if observed_mixed_work_units
        else None
    )
    projected_public_inference_seconds = (
        seconds_per_mixed_work_unit * projected_mixed_work_units
        if seconds_per_mixed_work_unit is not None
        else None
    )
    projected_total_seconds = (
        training_seconds + projected_public_inference_seconds
        if projected_public_inference_seconds is not None
        else None
    )
    return {
        "method": "linear_mixed_workload_projection",
        "limitations": (
            "Canary timing jointly measures exact prefix adaptations, no-gradient "
            "direction queries, and evaluator overhead; it is a bounded planning "
            "input, not a throughput benchmark."
        ),
        "observed": {
            "training_seconds": training_seconds,
            "canary_seconds": canary_seconds,
            "layout_groups": layout_count,
            "action_literals": action_count,
            "variants_per_action_layout": variants_per_action,
            "canary_episodes": canary_episodes,
            "exact_gradient_adaptations": exact_gradient_adaptations,
            "direction_query_calls": direction_query_calls,
            "mixed_work_units": observed_mixed_work_units,
            "seconds_per_mixed_work_unit": seconds_per_mixed_work_unit,
        },
        "public_score_inputs": {
            "game_count": config.public_game_count,
            "max_actions_per_game": config.max_actions_per_public_game,
            "max_resets_per_game": config.max_resets_per_public_game,
            "max_environment_actions": public_environment_actions,
            "assumed_exact_gradient_adaptations_per_action": 1,
            "assumed_direction_queries_per_action": action_count,
            "projected_mixed_work_units": projected_mixed_work_units,
        },
        "projected_public_inference_seconds": projected_public_inference_seconds,
        "projected_total_seconds_including_training": projected_total_seconds,
        "runtime_ceiling_seconds": config.runtime_ceiling_seconds,
        "within_runtime_ceiling": (
            projected_total_seconds is not None
            and math.isfinite(projected_total_seconds)
            and projected_total_seconds <= config.runtime_ceiling_seconds
        ),
    }


def apply_gate(
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    canary: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the fixed validation/canary and execution preflight thresholds."""

    validation_metrics = validation["aggregate"]
    canary_metrics = canary["aggregate"]
    checks = {
        "pretrained_original_gpt2": bool(execution["pretrained_original_gpt2"]),
        "positive_control_called_once": bool(execution["positive_control_called_once"]),
        "training_steps_exactly_512": int(training.get("steps_completed", -1))
        == TRAINING_STEPS,
        "training_all_finite": bool(training.get("all_finite", False))
        and _all_numeric_values_finite(training),
        "old_locked_test_never_evaluated": int(
            training.get("old_locked_test_evaluations", -1)
        )
        == 0,
        "validation_all_finite": bool(validation.get("all_finite", False))
        and _all_numeric_values_finite(validation),
        "validation_accuracy_at_least_0_70": float(validation_metrics["accuracy"])
        >= 0.70,
        "validation_truth_probability_at_least_0_60": float(
            validation_metrics["truth_probability"]
        )
        >= 0.60,
        "fresh_canary_disjoint": bool(execution["fresh_canary_disjoint"]),
        "fresh_canary_balanced_a1_a4": bool(
            execution["fresh_canary_balanced_a1_a4"]
        ),
        "canary_all_finite": bool(canary.get("all_finite", False))
        and _all_numeric_values_finite(canary),
        "canary_accuracy_at_least_0_70": float(canary_metrics["accuracy"])
        >= 0.70,
        "canary_truth_probability_at_least_0_60": float(
            canary_metrics["truth_probability"]
        )
        >= 0.60,
        "model_weights_unchanged_during_canary": bool(
            execution["model_weights_unchanged_during_canary"]
        ),
        "initial_prefix_unchanged_during_canary": bool(
            execution["initial_prefix_unchanged_during_canary"]
        ),
        "model_parameters_frozen_for_canary": bool(
            execution["model_parameters_frozen_for_canary"]
        ),
        "deterministic_algorithms_enabled": bool(
            execution["deterministic_algorithms_enabled"]
        ),
        "tf32_disabled": bool(execution["tf32_disabled"]),
        "source_sha_resolved": bool(execution["source_sha_resolved"]),
        "no_weight_serialization": bool(execution["no_weight_serialization"]),
        "runtime_projection_finite_and_within_ceiling": bool(
            execution["runtime_projection_finite_and_within_ceiling"]
        ),
    }
    return {"name": PROTOCOL, "passed": all(checks.values()), "checks": checks}


def _strict_json_safe(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    """Convert non-finite floats to null and report every affected path."""

    if value is None or isinstance(value, (str, bool, int)):
        return value, []
    if isinstance(value, float):
        return (value, []) if math.isfinite(value) else (None, [path])
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        nonfinite: list[str] = []
        for key, item in value.items():
            name = str(key)
            safe, paths = _strict_json_safe(item, f"{path}.{name}")
            result[name] = safe
            nonfinite.extend(paths)
        return result, nonfinite
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        nonfinite = []
        for index, item in enumerate(value):
            safe, paths = _strict_json_safe(item, f"{path}[{index}]")
            result_list.append(safe)
            nonfinite.extend(paths)
        return result_list, nonfinite
    raise TypeError(f"value at {path} is not strict-JSON serializable: {type(value)!r}")


def strict_json_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON primitives only, with no NaN or Infinity tokens."""

    safe, nonfinite_paths = _strict_json_safe(value)
    assert isinstance(safe, dict)
    safe["strict_json"] = {
        "allow_nan": False,
        "nonfinite_values_replaced_with_null": len(nonfinite_paths),
        "nonfinite_value_paths": nonfinite_paths,
    }
    encoded = json.dumps(
        safe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise RuntimeError("strict JSON receipt did not decode to an object")
    return decoded


def _synchronize_cuda(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch_module.cuda.synchronize()


def prepare_training_canary(
    config: Config,
    *,
    _dependencies: _Dependencies | None = None,
    _clock: Callable[[], float] = time.perf_counter,
) -> PreparedTrainingCanary:
    """Train once, run one fresh canary, and retain the model only in memory."""

    validate_config(config)
    dependencies = _dependencies or _load_dependencies()
    torch = dependencies.torch
    dependencies.set_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.require_cuda and getattr(device, "type", str(device)) != "cuda":
        raise RuntimeError("official score training requires CUDA")

    started = _clock()
    eggroll_config = _build_eggroll_config(config, dependencies)
    heldout_config = _build_heldout_config(config, dependencies)

    train_calls = 0
    _synchronize_cuda(torch, device)
    training_started = _clock()
    train_calls += 1
    model, tokenizer, trained_prefix, training, canonical_manifests = (
        dependencies.train_positive_control(eggroll_config, device)
    )
    _synchronize_cuda(torch, device)
    training_seconds = _clock() - training_started
    if train_calls != 1:
        raise RuntimeError("positive-control training must be called exactly once")
    if int(training.get("steps_completed", -1)) != TRAINING_STEPS:
        raise RuntimeError("positive-control training did not complete exactly 512 steps")
    if int(training.get("old_locked_test_evaluations", -1)) != 0:
        raise RuntimeError("positive-control unexpectedly evaluated the old locked test")

    validation = _selected_validation(training)
    selected_validation_sha256 = dependencies.canonical_json_sha256(validation)
    if selected_validation_sha256 != training.get("selected_validation_sha256"):
        raise RuntimeError("selected validation hash disagrees with training receipt")

    canary_spec = dependencies.SplitSpec(
        CANARY_SPLIT_NAME,
        CANARY_SEED_START,
        CANARY_SEED_STOP,
        CANARY_LAYOUT_GROUPS,
    )
    canary_manifest = dependencies.build_split_manifests((canary_spec,))[0]
    disjointness = dependencies.audit_split_disjointness(
        canonical_manifests, canary_manifest
    )
    if not bool(disjointness.get("passed", False)):
        raise RuntimeError("fresh official-score canary overlaps a canonical split")

    balance = _audit_balance(
        canary_manifest.accepted,
        dependencies.actions,
        dependencies.directions,
        dependencies.build_heldout_quartet,
    )
    if not balance["all_balanced"]:
        raise RuntimeError("fresh official-score canary is not balanced across A1--A4")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial_prefix = trained_prefix.detach().clone()
    model_parameters_frozen = all(
        not bool(parameter.requires_grad) for parameter in model.parameters()
    )
    hashes_before = frozen_state_hashes(model, initial_prefix)
    model_hash_before = hashes_before["model_weights_sha256"]
    prefix_hash_before = hashes_before["selected_initial_prefix_sha256"]
    combined_hash_before = hashes_before["combined_sha256"]

    _synchronize_cuda(torch, device)
    canary_started = _clock()
    canary = dependencies.evaluate_action_groups(
        model,
        tokenizer,
        initial_prefix,
        canary_manifest.accepted,
        dependencies.actions,
        heldout_config,
        device,
        bootstrap_seed=CANARY_BOOTSTRAP_SEED,
    )
    _synchronize_cuda(torch, device)
    canary_seconds = _clock() - canary_started

    hashes_after = frozen_state_hashes(model, initial_prefix)
    model_hash_after = hashes_after["model_weights_sha256"]
    prefix_hash_after = hashes_after["selected_initial_prefix_sha256"]
    combined_hash_after = hashes_after["combined_sha256"]
    projection = _runtime_projection(
        config,
        training_seconds=training_seconds,
        canary_seconds=canary_seconds,
        layout_count=len(canary_manifest.accepted),
        action_count=len(dependencies.actions),
        variants_per_action=int(balance["variants_per_action_layout"]),
    )
    source_sha = resolve_source_sha(config)
    execution = {
        "pretrained_original_gpt2": (
            eggroll_config.model_name == MODEL_NAME
            and eggroll_config.model_revision == MODEL_REVISION
        ),
        "positive_control_called_once": train_calls == 1,
        "fresh_canary_disjoint": bool(disjointness["passed"]),
        "fresh_canary_balanced_a1_a4": bool(balance["all_balanced"]),
        "model_weights_unchanged_during_canary": (
            model_hash_before == model_hash_after
        ),
        "initial_prefix_unchanged_during_canary": (
            prefix_hash_before == prefix_hash_after
        ),
        "combined_checkpoint_unchanged_during_canary": (
            combined_hash_before == combined_hash_after
        ),
        "model_parameters_frozen_for_canary": model_parameters_frozen,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "tf32_disabled": (
            not bool(torch.backends.cuda.matmul.allow_tf32)
            and not bool(torch.backends.cudnn.allow_tf32)
        ),
        "source_sha_resolved": source_sha is not None,
        "no_weight_serialization": True,
        "runtime_projection_finite_and_within_ceiling": bool(
            projection["within_runtime_ceiling"]
        ),
    }
    gate = apply_gate(training, validation, canary, execution)

    canonical_payload = dependencies.canonical_manifest_payload(
        canonical_manifests
    )
    canary_payload = dependencies.manifest_dict(canary_manifest)
    raw_receipt = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": (
            "One in-memory 512-step pretrained GPT-2 plus soft-prefix fit and a "
            "fresh disjoint synthetic exact-gradient A1--A4 canary; not an "
            "ARC-AGI-3 score or locked-game result."
        ),
        "source_sha": source_sha,
        "config": asdict(config),
        "device": str(device),
        "training_recipe": {
            "initialization": "pretrained",
            "epochs": 2,
            "optimizer_steps": TRAINING_STEPS,
            "train_layout_groups": 256,
            "validation_layout_groups": 64,
            "train_actions": ["A1"],
            "old_locked_test_evaluations": 0,
            "prefix_length": 8,
            "prefix_initialization_std": 0.01,
            "inner_learning_rate": 0.2,
            "prefix_learning_rate": 1e-3,
            "model_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "freeze_first_n_blocks": 11,
        },
        "training": training,
        "selected_validation": validation,
        "manifest": {
            "canonical_training_validation": {
                "sha256": dependencies.canonical_json_sha256(canonical_payload),
                "payload": canonical_payload,
            },
            "fresh_canary": {
                "sha256": dependencies.canonical_json_sha256(canary_payload),
                "payload": canary_payload,
                "disjointness": disjointness,
                "balance": balance,
            },
        },
        "canary": canary,
        "checkpoint": {
            "model_weights_sha256_before_canary": model_hash_before,
            "model_weights_sha256_after_canary": model_hash_after,
            "model_weights_unchanged": model_hash_before == model_hash_after,
            "selected_initial_prefix_sha256_before_canary": prefix_hash_before,
            "selected_initial_prefix_sha256_after_canary": prefix_hash_after,
            "selected_initial_prefix_unchanged": (
                prefix_hash_before == prefix_hash_after
            ),
            "combined_sha256_before_canary": combined_hash_before,
            "combined_sha256_after_canary": combined_hash_after,
            "combined_unchanged": combined_hash_before == combined_hash_after,
        },
        "runtime_projection": projection,
        "execution_checks": execution,
        "gate": gate,
        "eligible_for_official_public_score": bool(gate["passed"]),
        "decision": (
            "eligible_for_official_public_score"
            if gate["passed"]
            else "abort_before_public_scorecard"
        ),
        "model_weights_persisted": False,
        "tokenizer_persisted": False,
        "competition_submission_performed": False,
        "elapsed_seconds": _clock() - started,
    }
    receipt = strict_json_receipt(raw_receipt)
    return PreparedTrainingCanary(
        model=model,
        tokenizer=tokenizer,
        initial_prefix=initial_prefix,
        receipt=receipt,
    )


def run(config: Config) -> dict[str, Any]:
    """Prepare the in-memory model and write only its strict-JSON receipt."""

    prepared = prepare_training_canary(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "official-score-training-canary.json"
    output_path.write_text(
        json.dumps(
            prepared.receipt,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return prepared.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha")
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        Config(
            source_sha=args.source_sha,
            output_dir=args.output_dir,
            require_cuda=args.require_cuda,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
