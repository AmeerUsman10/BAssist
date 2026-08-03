"""Held-out layout and action gate for GPT-2 raw-NLL gradient memory.

This protocol is deliberately downstream of the fixed-quartet diagnostics.  It
meta-trains on 256 distinct generated layouts using only action ``A1``, selects
a checkpoint on 64 disjoint ``A1`` validation layouts, and then evaluates once
on 64 locked layouts.  The locked evaluation has two strata:

* geometry transfer: the familiar literal action ``A1`` on unseen layouts;
* literal-action surface invariance: the never-trained text literals
  ``A2``--``A4`` on those same unseen layouts.

The latter is not an action-slot binding holdout and not a token-ID holdout.
The GPT-2 tokenizer and pretrained weights have seen these ordinary strings;
only the protocol's generated meta-training and validation examples exclude
them.

The inner update remains ordinary causal-LM NLL of one exact observed outcome.
Counterfactual worlds balance the *outer* meta-training/evaluation task, but no
counterfactual candidate set is supplied to the online update.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch

from .meta_soft_binding import (
    _DIRECTIONS,
    initialize_prefix,
    mapping_query,
    set_seed,
    transition_prompt,
)
from .meta_soft_raw_outcome_overfit import (
    SUPPORT_OBJECTIVE,
    Config as RawConfig,
    raw_episode_meta_loss,
    raw_outcome_adapt_prefix,
    raw_support_text,
)
from .meta_soft_single_binding import (
    Config as BindingConfig,
    SingleBindingEpisode,
    build_model_and_tokenizer,
    corrupted_target,
    query_scores,
)
from .meta_soft_twin_overfit import GPT2_REVISION
from .phase0_hidden_action import (
    Action,
    DIRECTION_DELTA,
    GameSpec,
    HiddenActionGame,
    LevelSpec,
    generate_game,
)


PROTOCOL = "heldout_layout_action_v1"
RESULT_SCHEMA_VERSION = 1
SOURCE_LEVEL_INDEX = 1
TRAIN_ACTION = Action.A1
LOCKED_ACTIONS = (Action.A2, Action.A3, Action.A4)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed_start: int
    seed_stop: int
    groups: int


DEFAULT_SPLIT_SPECS = (
    SplitSpec("train", 1_100_000, 1_200_000, 256),
    SplitSpec("validation", 1_200_000, 1_300_000, 64),
    SplitSpec("locked_test", 1_300_000, 1_400_000, 64),
)


@dataclass(frozen=True)
class ProbeLayout:
    split: str
    game_seed: int
    before_grid_sha256: str
    probe_row: int
    probe_column: int
    source_level_index: int = SOURCE_LEVEL_INDEX
    height: int = 6
    width: int = 6


@dataclass(frozen=True)
class RejectedSeed:
    game_seed: int
    reason: str
    duplicate_of_split: str | None = None
    duplicate_of_seed: int | None = None
    before_grid_sha256: str | None = None


@dataclass(frozen=True)
class SplitManifest:
    name: str
    seed_start: int
    seed_stop: int
    requested_groups: int
    accepted: tuple[ProbeLayout, ...]
    rejected: tuple[RejectedSeed, ...]


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str = GPT2_REVISION
    source_sha: str | None = None
    initialization: str = "pretrained"
    output_dir: str = "outputs/meta_soft_heldout_binding/pretrained"
    seed: int = 424_242
    epochs: int = 2
    prefix_length: int = 8
    prefix_initialization_std: float = 0.01
    inner_learning_rate: float = 0.2
    prefix_learning_rate: float = 1e-3
    model_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    no_evidence_weight: float = 0.25
    freeze_first_n_blocks: int = 11
    bootstrap_samples: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 20_260_803
    save_model: bool = True
    require_cuda: bool = False


def validate_protocol_config(config: Config) -> None:
    """Reject hyperparameter drift under the preregistered protocol name."""

    expected = {
        "model_revision": GPT2_REVISION,
        "seed": 424_242,
        "epochs": 2,
        "prefix_length": 8,
        "prefix_initialization_std": 0.01,
        "inner_learning_rate": 0.2,
        "prefix_learning_rate": 1e-3,
        "model_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "no_evidence_weight": 0.25,
        "freeze_first_n_blocks": 11,
        "bootstrap_samples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 20_260_803,
    }
    drift = {
        name: {"expected": value, "observed": getattr(config, name)}
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if drift:
        raise ValueError(f"{PROTOCOL} configuration drift: {drift}")


def resolve_source_sha(config: Config) -> str | None:
    return (
        config.source_sha
        or os.environ.get("ARC_GPT2_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA")
    )


def _grid_sha256(grid: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _probe_position(level: LevelSpec) -> tuple[int, int] | None:
    """Choose the first row-major interior cell with four safe neighbours."""

    forbidden = set(level.walls) | {level.goal}
    for row in range(1, level.height - 1):
        for column in range(1, level.width - 1):
            position = (row, column)
            neighbours = {
                (row + delta_row, column + delta_column)
                for delta_row, delta_column in DIRECTION_DELTA.values()
            }
            if position not in forbidden and neighbours.isdisjoint(forbidden):
                return position
    return None


def _probe_spec(game_seed: int) -> tuple[GameSpec, tuple[int, int]]:
    """Return a game whose only level is generated level 1 with a safe start."""

    base = generate_game(game_seed, levels=2)
    source_level = base.levels[SOURCE_LEVEL_INDEX]
    if source_level.height != 6 or source_level.width != 6:
        raise RuntimeError("held-out protocol requires the generated 6x6 level")
    position = _probe_position(source_level)
    if position is None:
        raise ValueError("no_valid_probe_cell")
    probe_level = replace(source_level, start=position)
    return replace(base, levels=(probe_level,)), position


def probe_layout(game_seed: int, split: str) -> ProbeLayout:
    spec, position = _probe_spec(game_seed)
    before = HiddenActionGame(spec).frame
    return ProbeLayout(
        split=split,
        game_seed=game_seed,
        before_grid_sha256=_grid_sha256(before),
        probe_row=position[0],
        probe_column=position[1],
        height=len(before),
        width=len(before[0]),
    )


def build_split_manifests(
    split_specs: Sequence[SplitSpec] = DEFAULT_SPLIT_SPECS,
) -> tuple[SplitManifest, ...]:
    """Build deterministic, globally before-grid-disjoint split manifests."""

    names = [spec.name for spec in split_specs]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    used_seeds: set[int] = set()
    used_hashes: dict[str, ProbeLayout] = {}
    manifests: list[SplitManifest] = []
    for split_spec in split_specs:
        if split_spec.groups <= 0:
            raise ValueError("each split must request at least one group")
        accepted: list[ProbeLayout] = []
        rejected: list[RejectedSeed] = []
        for game_seed in range(split_spec.seed_start, split_spec.seed_stop):
            if game_seed in used_seeds:
                rejected.append(RejectedSeed(game_seed, "duplicate_seed"))
                continue
            used_seeds.add(game_seed)
            try:
                layout = probe_layout(game_seed, split_spec.name)
            except ValueError as exc:
                if str(exc) != "no_valid_probe_cell":
                    raise
                rejected.append(RejectedSeed(game_seed, "no_valid_probe_cell"))
                continue
            previous = used_hashes.get(layout.before_grid_sha256)
            if previous is not None:
                rejected.append(
                    RejectedSeed(
                        game_seed=game_seed,
                        reason="duplicate_before_grid",
                        duplicate_of_split=previous.split,
                        duplicate_of_seed=previous.game_seed,
                        before_grid_sha256=layout.before_grid_sha256,
                    )
                )
                continue
            accepted.append(layout)
            used_hashes[layout.before_grid_sha256] = layout
            if len(accepted) == split_spec.groups:
                break
        if len(accepted) != split_spec.groups:
            raise RuntimeError(
                f"split {split_spec.name!r} found {len(accepted)} of "
                f"{split_spec.groups} required valid unique layouts"
            )
        manifests.append(
            SplitManifest(
                name=split_spec.name,
                seed_start=split_spec.seed_start,
                seed_stop=split_spec.seed_stop,
                requested_groups=split_spec.groups,
                accepted=tuple(accepted),
                rejected=tuple(rejected),
            )
        )
    return tuple(manifests)


def _spec_for_layout(layout: ProbeLayout) -> GameSpec:
    spec, position = _probe_spec(layout.game_seed)
    before = HiddenActionGame(spec).frame
    if position != (layout.probe_row, layout.probe_column):
        raise RuntimeError("probe position no longer matches the manifest")
    if _grid_sha256(before) != layout.before_grid_sha256:
        raise RuntimeError("before grid no longer matches the manifest")
    return spec


def build_heldout_quartet(
    layout: ProbeLayout,
    action: Action,
) -> tuple[SingleBindingEpisode, ...]:
    """Build four raw-outcome worlds for one manifested layout and action."""

    base = _spec_for_layout(layout)
    quartet: list[SingleBindingEpisode] = []
    for variant_index, desired in enumerate(_DIRECTIONS):
        mapping = dict(base.action_to_direction)
        donor = next(key for key, value in mapping.items() if value is desired)
        mapping[action], mapping[donor] = mapping[donor], mapping[action]
        world = replace(base, action_to_direction=mapping)
        record = HiddenActionGame(world).step(action)
        if record.status != "ACTIVE" or not record.moved:
            raise RuntimeError("held-out quartet requires four safe active moves")
        quartet.append(
            SingleBindingEpisode(
                group_seed=layout.game_seed,
                variant_index=variant_index,
                action=action,
                direction=desired,
                record=record,
            )
        )
    values = tuple(quartet)
    if len({item.record.before for item in values}) != 1:
        raise RuntimeError("held-out quartet before grids diverged")
    if len({transition_prompt(item.record) for item in values}) != 1:
        raise RuntimeError("held-out quartet support prompts diverged")
    if len({item.record.after for item in values}) != 4:
        raise RuntimeError("held-out quartet outcomes are not unique")
    return values


def actions_for_split(split: str) -> tuple[Action, ...]:
    if split in {"train", "validation"}:
        return (TRAIN_ACTION,)
    if split == "locked_test":
        return (TRAIN_ACTION, *LOCKED_ACTIONS)
    raise ValueError(f"unknown split: {split!r}")


def _raw_config(config: Config) -> RawConfig:
    return RawConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        source_sha=config.source_sha,
        initialization=config.initialization,
        output_dir=config.output_dir,
        seed=config.seed,
        prefix_length=config.prefix_length,
        prefix_initialization_std=config.prefix_initialization_std,
        inner_learning_rate=config.inner_learning_rate,
        prefix_learning_rate=config.prefix_learning_rate,
        model_learning_rate=config.model_learning_rate,
        weight_decay=config.weight_decay,
        no_evidence_weight=config.no_evidence_weight,
        freeze_first_n_blocks=config.freeze_first_n_blocks,
        save_model=config.save_model,
        require_cuda=config.require_cuda,
    )


def heldout_episode_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: SingleBindingEpisode,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep the held-out protocol's inner update on the raw-NLL path."""

    return raw_episode_meta_loss(
        model,
        tokenizer,
        initial_prefix,
        episode,
        _raw_config(config),
        device,
    )


def one_sided_bootstrap_lower_bound(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> float:
    """Return a deterministic group-bootstrap one-sided percentile bound."""

    if not values:
        raise ValueError("bootstrap values may not be empty")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between 0 and 1")
    data = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in data):
        raise ValueError("bootstrap values must be finite")
    rng = random.Random(seed)
    count = len(data)
    means = sorted(
        sum(data[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    alpha = Decimal(1) - Decimal(str(confidence))
    # The empirical lower alpha quantile, using the conservative lower order
    # statistic when alpha * samples is an integer.
    rank = int(
        (alpha * Decimal(samples)).to_integral_value(rounding=ROUND_CEILING)
    )
    index = max(0, rank - 1)
    return means[index]


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def _aggregate_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average performance but retain worst-case causal-isolation controls."""

    if not rows:
        raise ValueError("metric rows may not be empty")
    values: dict[str, float] = {}
    for metric in rows[0]:
        column = [float(row[metric]) for row in rows]
        if metric == "prior_entropy_bits":
            values[metric] = min(column)
        elif metric in {
            "prior_max_abs_uniform_deviation",
            "max_preupdate_logit_delta",
            "corrupted_max_original_truth_probability",
        }:
            values[metric] = max(column)
        elif metric == "corrupted_min_target_probability":
            values[metric] = min(column)
        else:
            values[metric] = sum(column) / len(column)
    return values


def evaluate_action_groups(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    layouts: Sequence[ProbeLayout],
    actions: Sequence[Action],
    config: Config,
    device: torch.device,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate layouts while bootstrapping whole layout groups."""

    if not layouts or not actions:
        raise ValueError("evaluation needs layouts and actions")
    model.eval()
    group_rows: list[dict[str, Any]] = []
    per_action_rows: dict[str, list[dict[str, float]]] = {
        action.value: [] for action in actions
    }
    for layout in layouts:
        action_rows: list[dict[str, float]] = []
        for action in actions:
            quartet = build_heldout_quartet(layout, action)
            prior_logits: list[torch.Tensor] = []
            prior_probabilities: list[torch.Tensor] = []
            intact_probabilities: list[torch.Tensor] = []
            corrupted_probabilities: list[torch.Tensor] = []
            corrupted_indices: list[int] = []
            for episode in quartet:
                with torch.no_grad():
                    logits = query_scores(model, tokenizer, prefix, action, device)
                    prior = torch.softmax(logits, dim=-1)
                prior_logits.append(logits.detach().cpu())
                prior_probabilities.append(prior.detach().cpu())
                online_prefix = prefix.detach().clone().requires_grad_(True)
                updated, _, _, _ = raw_outcome_adapt_prefix(
                    model,
                    tokenizer,
                    online_prefix,
                    episode.record,
                    episode.record,
                    inner_learning_rate=config.inner_learning_rate,
                    device=device,
                    create_graph=False,
                )
                with torch.no_grad():
                    intact = torch.softmax(
                        query_scores(model, tokenizer, updated, action, device),
                        dim=-1,
                    )
                intact_probabilities.append(intact.detach().cpu())

                corrupt_record, corrupt_direction = corrupted_target(
                    episode.record,
                    episode.direction,
                )
                corrupt_prefix = prefix.detach().clone().requires_grad_(True)
                corrupt_updated, _, _, _ = raw_outcome_adapt_prefix(
                    model,
                    tokenizer,
                    corrupt_prefix,
                    episode.record,
                    corrupt_record,
                    inner_learning_rate=config.inner_learning_rate,
                    device=device,
                    create_graph=False,
                )
                with torch.no_grad():
                    corrupt = torch.softmax(
                        query_scores(
                            model,
                            tokenizer,
                            corrupt_updated,
                            action,
                            device,
                        ),
                        dim=-1,
                    )
                corrupted_probabilities.append(corrupt.detach().cpu())
                corrupted_indices.append(_DIRECTIONS.index(corrupt_direction))

            prior_stack = torch.stack(prior_probabilities)
            intact_stack = torch.stack(intact_probabilities)
            corrupt_stack = torch.stack(corrupted_probabilities)
            truth = torch.arange(4)
            corrupt_truth = torch.tensor(corrupted_indices)
            prior_predictions = prior_stack.argmax(dim=-1)
            intact_predictions = intact_stack.argmax(dim=-1)
            corrupt_predictions = corrupt_stack.argmax(dim=-1)
            truth_probability = intact_stack[truth, truth]
            prior_truth_probability = prior_stack[truth, truth]
            corrupt_target_probability = corrupt_stack[truth, corrupt_truth]
            corrupt_original_probability = corrupt_stack[truth, truth]
            prior = prior_stack[0]
            action_row = {
                "accuracy": float((intact_predictions == truth).float().mean().item()),
                "truth_probability": float(truth_probability.mean().item()),
                "no_adaptation_accuracy": float(
                    (prior_predictions == truth).float().mean().item()
                ),
                "no_adaptation_truth_probability": float(
                    prior_truth_probability.mean().item()
                ),
                "accuracy_gain": float(
                    (
                        (intact_predictions == truth).float().mean()
                        - (prior_predictions == truth).float().mean()
                    ).item()
                ),
                "truth_probability_gain": float(
                    (truth_probability.mean() - prior_truth_probability.mean()).item()
                ),
                "prior_entropy_bits": _entropy_bits(prior),
                "prior_max_abs_uniform_deviation": float(
                    torch.max(torch.abs(prior - 0.25)).item()
                ),
                "prior_uniform_cross_entropy": float(
                    -(torch.full_like(prior, 0.25) * torch.log(prior.clamp_min(1e-12))).sum().item()
                ),
                "mean_log_truth_probability": float(
                    torch.log(truth_probability.clamp_min(1e-12)).mean().item()
                ),
                "max_preupdate_logit_delta": float(
                    torch.max(
                        torch.abs(torch.stack(prior_logits) - prior_logits[0])
                    ).item()
                ),
                "corrupted_target_accuracy": float(
                    (corrupt_predictions == corrupt_truth).float().mean().item()
                ),
                "corrupted_mean_target_probability": float(
                    corrupt_target_probability.mean().item()
                ),
                "corrupted_min_target_probability": float(
                    corrupt_target_probability.min().item()
                ),
                "corrupted_original_truth_accuracy": float(
                    (corrupt_predictions == truth).float().mean().item()
                ),
                "corrupted_mean_original_truth_probability": float(
                    corrupt_original_probability.mean().item()
                ),
                "corrupted_max_original_truth_probability": float(
                    corrupt_original_probability.max().item()
                ),
            }
            action_rows.append(action_row)
            per_action_rows[action.value].append(action_row)
        group_rows.append(
            {
                "game_seed": layout.game_seed,
                "before_grid_sha256": layout.before_grid_sha256,
                **_aggregate_metric_rows(action_rows),
            }
        )

    metric_names = tuple(
        key
        for key in group_rows[0]
        if key not in {"game_seed", "before_grid_sha256"}
    )
    aggregate = _aggregate_metric_rows(
        [
            {metric: float(row[metric]) for metric in metric_names}
            for row in group_rows
        ]
    )
    bound_metrics = (
        "accuracy",
        "truth_probability",
        "accuracy_gain",
        "truth_probability_gain",
    )
    bootstrap = {
        metric: one_sided_bootstrap_lower_bound(
            [float(row[metric]) for row in group_rows],
            samples=config.bootstrap_samples,
            confidence=config.bootstrap_confidence,
            seed=bootstrap_seed + index,
        )
        for index, metric in enumerate(bound_metrics)
    }
    per_action = {
        action: _aggregate_metric_rows(rows)
        for action, rows in per_action_rows.items()
    }
    selection_objective = (
        -aggregate["mean_log_truth_probability"]
        + config.no_evidence_weight * aggregate["prior_uniform_cross_entropy"]
    )
    finite_values = [
        *aggregate.values(),
        *bootstrap.values(),
        selection_objective,
        *(
            value
            for metrics in per_action.values()
            for value in metrics.values()
        ),
        *(
            float(row[metric])
            for row in group_rows
            for metric in metric_names
        ),
    ]
    return {
        "group_count": len(group_rows),
        "actions": [action.value for action in actions],
        "all_finite": all(math.isfinite(value) for value in finite_values),
        "controls": {
            "no_adaptation": "zero_update_amnesic_prior",
            "corruption": "fixed_deranged_quartet_permutation_consistency_not_independent_evidence",
        },
        "aggregate": aggregate,
        "bootstrap": {
            "method": "deterministic_group_percentile",
            "samples": config.bootstrap_samples,
            "confidence": config.bootstrap_confidence,
            "seed": bootstrap_seed,
            "lower_bounds": bootstrap,
        },
        "per_action": per_action,
        "selection_objective": selection_objective,
        "groups": group_rows,
    }


def apply_gate(
    geometry: Mapping[str, Any],
    primary: Mapping[str, Any],
    execution_invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered heldout_layout_action_v1 thresholds."""

    checks: dict[str, bool] = {}
    for name, stratum in (("geometry", geometry), ("primary", primary)):
        metrics = stratum["aggregate"]
        lower = stratum["bootstrap"]["lower_bounds"]
        checks.update(
            {
                f"{name}_accuracy": metrics["accuracy"] >= 0.70,
                f"{name}_all_metrics_finite": bool(stratum.get("all_finite", False)),
                f"{name}_truth_probability": metrics["truth_probability"] >= 0.60,
                f"{name}_accuracy_gain": metrics["accuracy_gain"] >= 0.35,
                f"{name}_truth_probability_gain": metrics["truth_probability_gain"] >= 0.25,
                f"{name}_bootstrap_accuracy_above_chance": lower["accuracy"] > 0.25,
                f"{name}_bootstrap_probability_above_chance": lower["truth_probability"] > 0.25,
                f"{name}_bootstrap_accuracy_gain_positive": lower["accuracy_gain"] > 0.0,
                f"{name}_bootstrap_probability_gain_positive": (
                    lower["truth_probability_gain"] > 0.0
                ),
                f"{name}_prior_high_entropy": metrics["prior_entropy_bits"] >= 1.95,
                f"{name}_prior_uniform": metrics["prior_max_abs_uniform_deviation"] <= 0.05,
                f"{name}_preupdate_isolation": metrics["max_preupdate_logit_delta"] <= 1e-6,
                f"{name}_corrupted_target_accuracy": metrics["corrupted_target_accuracy"] >= 0.70,
                f"{name}_corrupted_target_probability": (
                    metrics["corrupted_mean_target_probability"] >= 0.60
                ),
                f"{name}_corruption_rejects_original_accuracy": (
                    metrics["corrupted_original_truth_accuracy"] <= 0.25
                ),
                f"{name}_corruption_rejects_original_probability": (
                    metrics["corrupted_mean_original_truth_probability"] <= 0.20
                ),
            }
        )
    for action in LOCKED_ACTIONS:
        action_metrics = primary["per_action"][action.value]
        checks[f"primary_{action.value}_accuracy"] = action_metrics["accuracy"] >= 0.55
        checks[f"primary_{action.value}_truth_probability"] = (
            action_metrics["truth_probability"] >= 0.45
        )
        checks[f"primary_{action.value}_corrupted_target_accuracy"] = (
            action_metrics["corrupted_target_accuracy"] >= 0.55
        )
        checks[f"primary_{action.value}_corrupted_target_probability"] = (
            action_metrics["corrupted_mean_target_probability"] >= 0.45
        )
    checks.update(
        {
            "finite_training": bool(execution_invariants["finite_training"]),
            "literal_action_text_audit": bool(
                execution_invariants["literal_action_text_audit"]
            ),
            "inner_update_contract": bool(
                execution_invariants["inner_update_contract"]
            ),
            "checkpoint_unchanged_during_locked_test": bool(
                execution_invariants["checkpoint_unchanged_during_locked_test"]
            ),
            "selected_validation_matches_history": bool(
                execution_invariants["selected_validation_matches_history"]
            ),
            "epoch_steps_exact": bool(execution_invariants["epoch_steps_exact"]),
        }
    )
    return {"name": PROTOCOL, "passed": all(checks.values()), "checks": checks}


def _manifest_dict(manifest: SplitManifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "seed_range": [manifest.seed_start, manifest.seed_stop],
        "requested_groups": manifest.requested_groups,
        "accepted": [asdict(item) for item in manifest.accepted],
        "rejected": [asdict(item) for item in manifest.rejected],
    }


def fixed_protocol_fields() -> dict[str, Any]:
    """Return claim-relevant fields covered by the canonical manifest digest."""

    return {
        "protocol": PROTOCOL,
        "model_revision": GPT2_REVISION,
        "paired_initializations": ["pretrained", "random"],
        "source_level_index": SOURCE_LEVEL_INDEX,
        "source_layout": "generate_game_level_1_6x6_random_walls",
        "probe_position_rule": "first_row_major_interior_four_safe_neighbors",
        "direction_order": [direction.value for direction in _DIRECTIONS],
        "split_specs": [asdict(item) for item in DEFAULT_SPLIT_SPECS],
        "split_actions": {
            "train": [TRAIN_ACTION.value],
            "validation": [TRAIN_ACTION.value],
            "locked_test_geometry": [TRAIN_ACTION.value],
            "locked_test_primary": [action.value for action in LOCKED_ACTIONS],
        },
        "initialization_seed": 424_242,
        "epochs": 2,
        "optimizer_steps": 512,
        "prefix_length": 8,
        "prefix_initialization_std": 0.01,
        "inner_learning_rate": 0.2,
        "prefix_learning_rate": 1e-3,
        "model_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "no_evidence_weight": 0.25,
        "freeze_first_n_blocks": 11,
        "bootstrap_samples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 20_260_803,
        "inner_update": {
            "objective": SUPPORT_OBJECTIVE,
            "candidate_count": 1,
            "reduction": "mean",
        },
        "gate_thresholds": {
            "stratum_accuracy_min": 0.70,
            "stratum_truth_probability_min": 0.60,
            "stratum_accuracy_gain_min": 0.35,
            "stratum_probability_gain_min": 0.25,
            "bootstrap_accuracy_probability_strictly_above": 0.25,
            "bootstrap_gains_strictly_above": 0.0,
            "prior_entropy_bits_min": 1.95,
            "prior_max_abs_uniform_deviation_max": 0.05,
            "preupdate_logit_delta_max": 1e-6,
            "corrupted_target_accuracy_min": 0.70,
            "corrupted_target_probability_min": 0.60,
            "corrupted_original_accuracy_max": 0.25,
            "corrupted_original_probability_max": 0.20,
            "per_locked_literal_accuracy_min": 0.55,
            "per_locked_literal_probability_min": 0.45,
            "per_locked_literal_corrupted_accuracy_min": 0.55,
            "per_locked_literal_corrupted_probability_min": 0.45,
        },
    }


def canonical_manifest_payload(
    manifests: Sequence[SplitManifest],
) -> dict[str, Any]:
    """Serialize every accepted/rejected record plus fixed protocol identity."""

    return {
        "fixed_protocol_fields": fixed_protocol_fields(),
        "manifests": [_manifest_dict(manifest) for manifest in manifests],
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_sha256(manifests: Sequence[SplitManifest]) -> str:
    return canonical_json_sha256(canonical_manifest_payload(manifests))


def audit_generated_literal_action_text(
    manifests: Sequence[SplitManifest],
) -> dict[str, Any]:
    """Audit every generated train/validation support and query string."""

    by_name = {manifest.name: manifest for manifest in manifests}
    searched_literals = tuple(action.value for action in LOCKED_ACTIONS)
    patterns = {
        literal: re.compile(rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])")
        for literal in searched_literals
    }
    occurrences = {literal: 0 for literal in searched_literals}
    corpus_digest = hashlib.sha256()
    group_count = 0
    support_count = 0
    query_count = 0
    unique_targets_per_quartet: list[int] = []
    for split in ("train", "validation"):
        for layout in by_name[split].accepted:
            group_count += 1
            targets: set[str] = set()
            for episode in build_heldout_quartet(layout, TRAIN_ACTION):
                support_prompt, support_target = raw_support_text(
                    episode.record,
                    episode.record,
                )
                query = mapping_query(TRAIN_ACTION)
                targets.add(support_target)
                support_count += 1
                query_count += 1
                for kind, text_value in (
                    ("support_prompt", support_prompt),
                    ("support_target", support_target),
                    ("query", query),
                ):
                    encoded = text_value.encode("utf-8")
                    corpus_digest.update(split.encode("ascii"))
                    corpus_digest.update(layout.game_seed.to_bytes(8, "big"))
                    corpus_digest.update(episode.variant_index.to_bytes(1, "big"))
                    corpus_digest.update(kind.encode("ascii"))
                    corpus_digest.update(len(encoded).to_bytes(8, "big"))
                    corpus_digest.update(encoded)
                    for literal, pattern in patterns.items():
                        occurrences[literal] += len(pattern.findall(text_value))
            unique_targets_per_quartet.append(len(targets))
    return {
        "scope": "all_generated_train_and_validation_support_prompt_target_and_query_text",
        "claim": "unseen_literal_action_surface_invariance_only",
        "not_claimed": [
            "action_slot_binding_holdout",
            "token_id_holdout",
            "tokenizer_or_pretraining_holdout",
        ],
        "splits": ["train", "validation"],
        "group_count": group_count,
        "support_example_count": support_count,
        "query_example_count": query_count,
        "searched_literals": list(searched_literals),
        "literal_occurrences": occurrences,
        "all_locked_literals_absent": all(value == 0 for value in occurrences.values()),
        "min_unique_targets_per_quartet": min(unique_targets_per_quartet),
        "max_unique_targets_per_quartet": max(unique_targets_per_quartet),
        "all_quartets_have_four_unique_targets": all(
            value == 4 for value in unique_targets_per_quartet
        ),
        "cardinal_semantic_guard_passed": True,
        "audited_corpus_sha256": corpus_digest.hexdigest(),
        "passed": (
            all(value == 0 for value in occurrences.values())
            and all(value == 4 for value in unique_targets_per_quartet)
        ),
    }


def best_validation_history_entry(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the lowest objective, with the earlier epoch as the tie-break."""

    if not history:
        raise ValueError("validation history may not be empty")
    return min(
        history,
        key=lambda item: (
            float(item["validation_selection_objective"]),
            int(item["epoch"]),
        ),
    )


def inner_update_contract_passes(contract: Mapping[str, Any]) -> bool:
    return (
        contract.get("objective") == "raw_outcome_nll"
        and contract.get("candidate_count") == 1
        and contract.get("reduction") == "mean"
        and contract.get("counterfactuals_used_in_inner_update") is False
        and contract.get("unique_targets_per_quartet") == 4
        and contract.get("cardinal_semantic_guard_passed") is True
        and contract.get("eager_attention") is True
    )


def _capture_trainable_state(
    model: Any,
    prefix: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return (
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        prefix.detach().cpu().clone(),
    )


def checkpoint_is_better(candidate_objective: float, best_objective: float) -> bool:
    """Use strict improvement so an exact validation tie keeps the earlier epoch."""

    return candidate_objective < best_objective


def _restore_trainable_state(
    model: Any,
    prefix: torch.Tensor,
    state: tuple[Mapping[str, torch.Tensor], torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state[0].items():
            parameters[name].copy_(value.to(parameters[name].device))
        prefix.copy_(state[1].to(prefix.device))


def _checkpoint_sha256(model: Any, prefix: torch.Tensor) -> str:
    """Hash the complete validation-selected model and prefix before test."""

    digest = hashlib.sha256()
    tensors = [
        *sorted(model.state_dict().items()),
        ("__initial_soft_prefix__", prefix),
    ]
    for name, value in tensors:
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train(config: Config) -> dict[str, Any]:
    """Train, validation-select, and evaluate the locked test exactly once."""

    validate_protocol_config(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("this run requires a CUDA device")
    manifests = build_split_manifests()
    by_name = {manifest.name: manifest for manifest in manifests}
    canonical_manifest = canonical_manifest_payload(manifests)
    canonical_manifest_digest = canonical_json_sha256(canonical_manifest)
    literal_action_text_audit = audit_generated_literal_action_text(manifests)
    if not literal_action_text_audit["passed"]:
        raise RuntimeError("generated training/validation text violated literal-action isolation")
    binding_config = BindingConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        initialization=config.initialization,
        prefix_length=config.prefix_length,
        prefix_initialization_std=config.prefix_initialization_std,
        inner_learning_rate=config.inner_learning_rate,
        outer_learning_rate=config.model_learning_rate,
        no_evidence_weight=config.no_evidence_weight,
        freeze_first_n_blocks=config.freeze_first_n_blocks,
        save_model=config.save_model,
    )
    model, tokenizer = build_model_and_tokenizer(binding_config)
    model.to(device)
    model.eval()
    prefix = initialize_prefix(
        model,
        prefix_length=config.prefix_length,
        std=config.prefix_initialization_std,
        seed=config.seed,
        device=device,
    )
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [prefix], "lr": config.prefix_learning_rate, "weight_decay": 0.0},
            {
                "params": model_parameters,
                "lr": config.model_learning_rate,
                "weight_decay": config.weight_decay,
            },
        ]
    )
    started = time.time()
    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_objective = math.inf
    best_state: tuple[dict[str, torch.Tensor], torch.Tensor] | None = None
    steps = 0
    raw_config = _raw_config(config)
    for epoch in range(1, config.epochs + 1):
        epoch_losses: list[float] = []
        for layout in by_name["train"].accepted:
            optimizer.zero_grad(set_to_none=True)
            quartet = build_heldout_quartet(layout, TRAIN_ACTION)
            losses: list[float] = []
            for episode in quartet:
                loss, _ = raw_episode_meta_loss(
                    model,
                    tokenizer,
                    prefix,
                    episode,
                    raw_config,
                    device,
                )
                (loss / len(quartet)).backward()
                losses.append(float(loss.detach().item()))
            torch.nn.utils.clip_grad_norm_([prefix, *model_parameters], 1.0)
            optimizer.step()
            steps += 1
            epoch_losses.append(sum(losses) / len(losses))
        validation = evaluate_action_groups(
            model,
            tokenizer,
            prefix.detach(),
            by_name["validation"].accepted,
            (TRAIN_ACTION,),
            config,
            device,
            bootstrap_seed=config.bootstrap_seed + epoch * 100,
        )
        objective = float(validation["selection_objective"])
        trainable_parameters_finite = all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in (prefix, *model_parameters)
        )
        history.append(
            {
                "epoch": epoch,
                "steps_completed": steps,
                "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
                "all_training_group_losses_finite": all(
                    math.isfinite(value) for value in epoch_losses
                ),
                "trainable_parameters_finite": trainable_parameters_finite,
                "validation_selection_objective": objective,
                "validation": validation,
            }
        )
        # Strict improvement makes the earlier epoch win an exact tie.
        if checkpoint_is_better(objective, best_objective):
            best_objective = objective
            best_epoch = epoch
            best_state = _capture_trainable_state(model, prefix)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "steps_completed": steps,
                    "mean_training_loss": history[-1]["mean_training_loss"],
                    "validation_selection_objective": objective,
                    "best_epoch": best_epoch,
                    "elapsed_seconds": time.time() - started,
                }
            ),
            flush=True,
        )
    if steps != 512 or best_state is None or best_epoch is None:
        raise RuntimeError("held-out protocol did not complete its 512 training steps")
    selected_history_entry = best_validation_history_entry(history)
    if int(selected_history_entry["epoch"]) != best_epoch:
        raise RuntimeError("online checkpoint selection disagreed with validation history")
    selected_validation = selected_history_entry["validation"]
    selected_validation_sha256 = canonical_json_sha256(selected_validation)
    history_validation_sha256 = canonical_json_sha256(
        history[best_epoch - 1]["validation"]
    )
    _restore_trainable_state(model, prefix, best_state)
    frozen_checkpoint_sha256_before = _checkpoint_sha256(model, prefix)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix.requires_grad_(False)

    attention_implementation = getattr(model.config, "_attn_implementation", None)
    inner_update_contract = {
        "objective": SUPPORT_OBJECTIVE,
        "candidate_count": 1,
        "reduction": "mean",
        "counterfactuals_used_in_inner_update": False,
        "unique_targets_per_quartet": 4,
        "cardinal_semantic_guard_passed": bool(
            literal_action_text_audit["cardinal_semantic_guard_passed"]
        ),
        "attention_implementation": attention_implementation,
        "eager_attention": attention_implementation == "eager",
    }
    inner_update_contract_passed = inner_update_contract_passes(
        inner_update_contract
    )

    locked_layouts = by_name["locked_test"].accepted
    geometry = evaluate_action_groups(
        model,
        tokenizer,
        prefix.detach(),
        locked_layouts,
        (TRAIN_ACTION,),
        config,
        device,
        bootstrap_seed=config.bootstrap_seed + 1_000,
    )
    primary = evaluate_action_groups(
        model,
        tokenizer,
        prefix.detach(),
        locked_layouts,
        LOCKED_ACTIONS,
        config,
        device,
        bootstrap_seed=config.bootstrap_seed + 2_000,
    )
    frozen_checkpoint_sha256_after = _checkpoint_sha256(model, prefix)
    checkpoint_unchanged = (
        frozen_checkpoint_sha256_before == frozen_checkpoint_sha256_after
    )
    training_all_finite = all(
        math.isfinite(float(item["mean_training_loss"]))
        and math.isfinite(float(item["validation_selection_objective"]))
        and bool(item["all_training_group_losses_finite"])
        and bool(item["trainable_parameters_finite"])
        and bool(item["validation"]["all_finite"])
        for item in history
    )
    selected_validation_matches_history = (
        selected_validation_sha256 == history_validation_sha256
    )
    epoch_steps_exact = [int(item["steps_completed"]) for item in history] == [
        256,
        512,
    ]
    execution_invariants = {
        "finite_training": training_all_finite,
        "literal_action_text_audit": bool(literal_action_text_audit["passed"]),
        "inner_update_contract": inner_update_contract_passed,
        "checkpoint_unchanged_during_locked_test": checkpoint_unchanged,
        "selected_validation_matches_history": selected_validation_matches_history,
        "epoch_steps_exact": epoch_steps_exact,
    }
    gate = apply_gate(geometry, primary, execution_invariants)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "experiment": "meta_soft_heldout_binding",
        "scope": (
            "raw-outcome-NLL gradient-memory transfer to disjoint generated layouts "
            "and unseen literal action surfaces; not action-slot binding or token-ID "
            "holdout; synthetic diagnostic, not ARC-AGI-3 capability"
        ),
        "claim_boundary": {
            "claimed": "unseen_literal_action_surface_invariance",
            "not_claimed": [
                "action_slot_binding_holdout",
                "token_id_holdout",
                "tokenizer_or_pretraining_holdout",
                "ARC-AGI-3 capability",
            ],
        },
        "source_sha": resolve_source_sha(config),
        "support_objective": SUPPORT_OBJECTIVE,
        "counterfactuals_used_in_inner_update": False,
        "counterfactual_worlds_used_for_balanced_outer_supervision": True,
        "counterfactuals_used_in_corruption_evaluation": True,
        "inner_update_contract": inner_update_contract,
        "literal_action_text_audit": literal_action_text_audit,
        "split_actions": {
            "train": [TRAIN_ACTION.value],
            "validation": [TRAIN_ACTION.value],
            "locked_test_geometry": [TRAIN_ACTION.value],
            "locked_test_primary": [action.value for action in LOCKED_ACTIONS],
        },
        "config": asdict(config),
        "device": str(device),
        "manifests": {manifest.name: _manifest_dict(manifest) for manifest in manifests},
        "canonical_manifest": {
            "sha256": canonical_manifest_digest,
            "payload": canonical_manifest,
        },
        "selection": {
            "criterion": "-mean_log_truth_probability + 0.25 * uniform_prior_cross_entropy",
            "tie_break": "earlier_epoch",
            "best_epoch": best_epoch,
            "best_objective": best_objective,
            "validation_evaluations": config.epochs,
            "epoch_objectives": [
                {
                    "epoch": int(item["epoch"]),
                    "steps_completed": int(item["steps_completed"]),
                    "objective": float(item["validation_selection_objective"]),
                }
                for item in history
            ],
            "selected_validation_sha256": selected_validation_sha256,
            "best_history_validation_sha256": history_validation_sha256,
            "selected_validation_equals_best_history_entry": (
                selected_validation_matches_history
            ),
            "frozen_checkpoint_sha256_before_locked_test": (
                frozen_checkpoint_sha256_before
            ),
        },
        "validation": selected_validation,
        "locked_test": {
            "evaluations": 1,
            "checkpoint_frozen_before_evaluation": True,
            "frozen_checkpoint_sha256_after_locked_test": (
                frozen_checkpoint_sha256_after
            ),
            "checkpoint_unchanged": checkpoint_unchanged,
            "geometry_A1": geometry,
            "primary_A2_A4": primary,
        },
        "training": {
            "epochs": config.epochs,
            "steps_completed": steps,
            "all_finite": training_all_finite,
        },
        "execution_invariants": execution_invariants,
        "gate": gate,
        "status": "pass" if gate["passed"] else "fail",
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if config.save_model and gate["passed"]:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        torch.save(
            {"version": 1, "protocol": PROTOCOL, "prefix": prefix.detach().cpu()},
            output_dir / "initial_soft_prefix.pt",
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default=GPT2_REVISION)
    parser.add_argument("--source-sha")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--output-dir", default="outputs/meta_soft_heldout_binding/pretrained")
    parser.add_argument("--seed", type=int, default=424_242)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--prefix-initialization-std", type=float, default=0.01)
    parser.add_argument("--inner-learning-rate", type=float, default=0.2)
    parser.add_argument("--prefix-learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-evidence-weight", type=float, default=0.25)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=11)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_803)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        Config(
            model_name=args.model_name,
            model_revision=args.model_revision,
            source_sha=args.source_sha,
            initialization=args.initialization,
            output_dir=args.output_dir,
            seed=args.seed,
            prefix_length=args.prefix_length,
            prefix_initialization_std=args.prefix_initialization_std,
            inner_learning_rate=args.inner_learning_rate,
            prefix_learning_rate=args.prefix_learning_rate,
            model_learning_rate=args.model_learning_rate,
            weight_decay=args.weight_decay,
            no_evidence_weight=args.no_evidence_weight,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            save_model=not args.no_save_model,
            require_cuda=args.require_cuda,
        )
    )


if __name__ == "__main__":
    main()
