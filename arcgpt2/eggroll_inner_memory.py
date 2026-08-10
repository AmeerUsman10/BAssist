"""Preregistered EGGROLL-style black-box soft-memory compatibility gate.

This module deliberately targets the smallest scientifically interpretable use
of rank-one evolution strategies in the ARC-GPT2 work.  The existing
``heldout_layout_action_v1`` protocol first learns a gradient-memory circuit.
Here we reproduce only its train/validation phase, freeze the selected GPT-2
checkpoint, and ask whether a rank-one antithetic finite-difference estimator
can replace the single exact raw-NLL write into the 8x768 soft prefix.

The audit uses fresh generated layouts and never evaluates the old locked split.
Exact gradients are computed only as a positive control and estimator audit;
they are not inputs to the EGGROLL update.
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
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .meta_soft_binding import (
    _DIRECTIONS,
    initialize_prefix,
    set_seed,
)
from .meta_soft_heldout_binding import (
    LOCKED_ACTIONS,
    TRAIN_ACTION,
    Config as HeldoutConfig,
    ProbeLayout,
    SplitManifest,
    SplitSpec,
    _capture_trainable_state,
    _checkpoint_sha256,
    _manifest_dict,
    _raw_config,
    _restore_trainable_state,
    audit_generated_literal_action_text,
    best_validation_history_entry,
    build_heldout_quartet,
    build_split_manifests,
    canonical_json_sha256,
    canonical_manifest_payload,
    checkpoint_is_better,
    evaluate_action_groups,
    one_sided_bootstrap_lower_bound,
)
from .meta_soft_raw_outcome_overfit import (
    raw_episode_meta_loss,
    raw_outcome_adapt_prefix,
    raw_support_token_ids,
)
from .meta_soft_single_binding import (
    Config as BindingConfig,
    build_model_and_tokenizer,
    query_scores,
)
from .meta_soft_twin_overfit import GPT2_REVISION
from .phase0_hidden_action import Action, StepRecord


PROTOCOL = "eggroll_inner_memory_v1"
RESULT_SCHEMA_VERSION = 1
AUDIT_SPLIT_SPEC = SplitSpec("eggroll_audit", 1_400_000, 1_500_000, 8)
ACTIONS = (TRAIN_ACTION, *LOCKED_ACTIONS)
CONVERGENCE_LADDER = (128, 256, 512)
AUDIT_RUNTIME_CEILING_SECONDS = 9_000.0


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str = GPT2_REVISION
    source_sha: str | None = None
    output_dir: str = "outputs/eggroll_inner_memory"
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
    bootstrap_seed: int = 20_260_809
    perturbation_rank: int = 1
    antithetic_pairs: int = 512
    sigma: float = 0.001
    population_batch_size: int = 32
    require_cuda: bool = False


def validate_protocol_config(config: Config) -> None:
    """Reject drift from the issue-14 preregistration."""

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
        "bootstrap_seed": 20_260_809,
        "perturbation_rank": 1,
        "antithetic_pairs": 512,
        "sigma": 0.001,
        "population_batch_size": 32,
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


def stable_episode_seed(
    experiment_seed: int,
    layout: ProbeLayout,
    action: Action,
    variant_index: int,
) -> int:
    payload = json.dumps(
        {
            "experiment_seed": experiment_seed,
            "game_seed": layout.game_seed,
            "before_grid_sha256": layout.before_grid_sha256,
            "action": action.value,
            "variant_index": variant_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_rank_one_perturbations(
    pairs: int,
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample deterministic iid-Gaussian rank-one matrices on CPU."""

    if min(pairs, rows, columns) <= 0:
        raise ValueError("pairs and matrix dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn((pairs, rows, 1), generator=generator, dtype=torch.float32)
    right = torch.randn((pairs, columns, 1), generator=generator, dtype=torch.float32)
    return torch.bmm(left, right.transpose(1, 2)).to(device)


def antithetic_population(
    center: torch.Tensor,
    perturbations: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if center.ndim != 2 or perturbations.ndim != 3:
        raise ValueError("center must be a matrix and perturbations a matrix batch")
    if tuple(perturbations.shape[1:]) != tuple(center.shape):
        raise ValueError("perturbation shape does not match the center")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return center.unsqueeze(0) + sigma * perturbations, center.unsqueeze(0) - sigma * perturbations


def prefix_population_mean_nll(
    model: Any,
    prefixes: torch.Tensor,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    device: torch.device | str,
    batch_size: int,
) -> torch.Tensor:
    """Score one exact target for a population of candidate soft prefixes.

    Calling the transformer body directly and applying the tied LM head only at
    target positions avoids materialising vocabulary logits for prompt tokens.
    Model parameters and candidate prefixes are treated as black-box inputs;
    no autograd graph is constructed.
    """

    if prefixes.ndim != 3:
        raise ValueError("prefixes must have shape [population, prefix, hidden]")
    if prefixes.shape[0] < 1 or prefixes.shape[1] < 1:
        raise ValueError("prefix population and length must be positive")
    if not prompt_ids or not target_ids:
        raise ValueError("prompt and target token sequences must be non-empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    token_ids = torch.tensor(
        [*map(int, prompt_ids), *map(int, target_ids)],
        dtype=torch.long,
        device=target_device,
    )
    results: list[torch.Tensor] = []
    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(token_ids)
        for start in range(0, int(prefixes.shape[0]), batch_size):
            chunk = prefixes[start : start + batch_size].to(
                target_device, dtype=token_embeddings.dtype
            )
            count = int(chunk.shape[0])
            embedded = token_embeddings.unsqueeze(0).expand(count, -1, -1)
            inputs_embeds = torch.cat((chunk, embedded), dim=1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=target_device
            )
            outputs = model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs[0]
            )
            first_target_logit = int(chunk.shape[1]) + len(prompt_ids) - 1
            target_hidden = hidden[
                :, first_target_logit : first_target_logit + len(target_ids)
            ]
            logits = model.lm_head(target_hidden).float()
            targets = torch.tensor(
                tuple(map(int, target_ids)), dtype=torch.long, device=target_device
            ).unsqueeze(0).expand(count, -1)
            token_nll = -F.log_softmax(logits, dim=-1).gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)
            results.append(token_nll.mean(dim=1).cpu())
    return torch.cat(results).to(target_device)


def estimate_rank_one_gradient(
    loss_plus: torch.Tensor,
    loss_minus: torch.Tensor,
    perturbations: torch.Tensor,
    *,
    sigma: float,
) -> dict[str, torch.Tensor]:
    """Return raw and black-box norm-corrected rank-one ES estimates."""

    if loss_plus.ndim != 1 or loss_minus.ndim != 1:
        raise ValueError("antithetic losses must be vectors")
    if loss_plus.shape != loss_minus.shape:
        raise ValueError("plus/minus losses must have identical shapes")
    if perturbations.shape[0] != loss_plus.shape[0]:
        raise ValueError("loss and perturbation populations disagree")
    directional = (loss_plus.float() - loss_minus.float()) / (2.0 * sigma)
    raw = torch.einsum("p,pij->ij", directional, perturbations.float()) / len(directional)
    raw_norm = torch.linalg.vector_norm(raw)
    directional_rms = torch.sqrt(torch.mean(directional.square()))
    corrected = raw * (directional_rms / raw_norm.clamp_min(1e-12))
    return {
        "directional_derivatives": directional,
        "raw": raw,
        "norm_corrected": corrected,
        "directional_rms": directional_rms,
        "raw_norm": raw_norm,
    }


def eggroll_adapt_prefix(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    record: StepRecord,
    *,
    seed: int,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[int, torch.Tensor]]:
    """Apply one black-box rank-one antithetic update to a soft prefix."""

    prompt_ids, target_ids = raw_support_token_ids(tokenizer, record, record)
    center = prefix.detach().float()
    perturbations = sample_rank_one_perturbations(
        config.antithetic_pairs,
        int(center.shape[0]),
        int(center.shape[1]),
        seed=seed,
        device=device,
    )
    plus, minus = antithetic_population(center, perturbations, config.sigma)
    losses = prefix_population_mean_nll(
        model,
        torch.cat((plus, minus), dim=0),
        prompt_ids,
        target_ids,
        device=device,
        batch_size=config.population_batch_size,
    )
    pair_count = config.antithetic_pairs
    estimates: dict[int, torch.Tensor] = {}
    ladder_diagnostics: dict[str, Any] = {}
    for count in CONVERGENCE_LADDER:
        estimate = estimate_rank_one_gradient(
            losses[:count],
            losses[pair_count : pair_count + count],
            perturbations[:count],
            sigma=config.sigma,
        )
        estimates[count] = estimate["norm_corrected"]
        ladder_diagnostics[str(count)] = {
            "raw_estimate_norm": float(estimate["raw_norm"].item()),
            "norm_corrected_estimate_norm": float(
                torch.linalg.vector_norm(estimate["norm_corrected"]).item()
            ),
            "directional_rms": float(estimate["directional_rms"].item()),
        }
    final_estimate = estimate_rank_one_gradient(
        losses[:pair_count],
        losses[pair_count:],
        perturbations,
        sigma=config.sigma,
    )
    updated = center - config.inner_learning_rate * final_estimate["norm_corrected"]
    antithetic_error = torch.max(torch.abs((plus + minus) / 2.0 - center)).item()
    diagnostics = {
        "seed": seed,
        "support_target_tokens": len(target_ids),
        "pairs": pair_count,
        "population": pair_count * 2,
        "rank": config.perturbation_rank,
        "sigma": config.sigma,
        "loss_plus_mean": float(losses[:pair_count].mean().item()),
        "loss_minus_mean": float(losses[pair_count:].mean().item()),
        "raw_estimate_norm": float(final_estimate["raw_norm"].item()),
        "norm_corrected_estimate_norm": float(
            torch.linalg.vector_norm(final_estimate["norm_corrected"]).item()
        ),
        "directional_rms": float(final_estimate["directional_rms"].item()),
        "antithetic_center_max_abs_error": float(antithetic_error),
        "all_finite": bool(
            torch.isfinite(losses).all().item()
            and torch.isfinite(final_estimate["raw"]).all().item()
            and torch.isfinite(final_estimate["norm_corrected"]).all().item()
            and torch.isfinite(updated).all().item()
        ),
        "convergence_ladder": ladder_diagnostics,
    }
    estimates[pair_count] = final_estimate["norm_corrected"]
    estimates[-1] = final_estimate["raw"]
    return updated, diagnostics, estimates


def _mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(map(float, values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _safe_gain_retention(eggroll_gain: float, exact_gain: float) -> float | None:
    if exact_gain <= 0:
        return None
    return eggroll_gain / exact_gain


def aggregate_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    config: Config,
) -> dict[str, Any]:
    if not episode_rows:
        raise ValueError("audit rows may not be empty")
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in episode_rows:
        key = (int(row["game_seed"]), str(row["before_grid_sha256"]))
        grouped.setdefault(key, []).append(row)
    group_rows: list[dict[str, Any]] = []
    for (game_seed, grid_hash), rows in sorted(grouped.items()):
        group_rows.append(
            {
                "game_seed": game_seed,
                "before_grid_sha256": grid_hash,
                "cosine": _mean(rows, "cosine"),
                "prior_accuracy": _mean(rows, "prior_correct"),
                "exact_accuracy": _mean(rows, "exact_correct"),
                "eggroll_accuracy": _mean(rows, "eggroll_correct"),
                "prior_truth_probability": _mean(rows, "prior_truth_probability"),
                "exact_truth_probability": _mean(rows, "exact_truth_probability"),
                "eggroll_truth_probability": _mean(rows, "eggroll_truth_probability"),
                "eggroll_truth_probability_gain": _mean(
                    rows, "eggroll_truth_probability_gain"
                ),
            }
        )
    norm_ratios = [
        float(row["corrected_to_exact_norm_ratio"])
        for row in episode_rows
        if row["corrected_to_exact_norm_ratio"] is not None
    ]
    metrics = {
        "episode_count": len(episode_rows),
        "layout_group_count": len(group_rows),
        "prior_accuracy": _mean(episode_rows, "prior_correct"),
        "exact_accuracy": _mean(episode_rows, "exact_correct"),
        "eggroll_accuracy": _mean(episode_rows, "eggroll_correct"),
        "prior_truth_probability": _mean(episode_rows, "prior_truth_probability"),
        "exact_truth_probability": _mean(episode_rows, "exact_truth_probability"),
        "eggroll_truth_probability": _mean(episode_rows, "eggroll_truth_probability"),
        "exact_truth_probability_gain": _mean(
            episode_rows, "exact_truth_probability_gain"
        ),
        "eggroll_truth_probability_gain": _mean(
            episode_rows, "eggroll_truth_probability_gain"
        ),
        "mean_cosine": _mean(episode_rows, "cosine"),
        "median_cosine": _median([float(row["cosine"]) for row in episode_rows]),
        "median_corrected_to_exact_norm_ratio": (
            _median(norm_ratios) if len(norm_ratios) == len(episode_rows) else None
        ),
        "exact_eggroll_argmax_agreement": _mean(
            episode_rows, "exact_eggroll_argmax_agreement"
        ),
    }
    metrics["truth_probability_gain_retention"] = _safe_gain_retention(
        metrics["eggroll_truth_probability_gain"],
        metrics["exact_truth_probability_gain"],
    )
    lower_bounds = {
        metric: one_sided_bootstrap_lower_bound(
            [float(row[metric]) for row in group_rows],
            samples=config.bootstrap_samples,
            confidence=config.bootstrap_confidence,
            seed=config.bootstrap_seed + index,
        )
        for index, metric in enumerate(
            ("cosine", "eggroll_accuracy", "eggroll_truth_probability_gain")
        )
    }
    per_action = {}
    for action in ACTIONS:
        rows = [row for row in episode_rows if row["action"] == action.value]
        per_action[action.value] = {
            "episode_count": len(rows),
            "exact_accuracy": _mean(rows, "exact_correct"),
            "eggroll_accuracy": _mean(rows, "eggroll_correct"),
            "exact_truth_probability": _mean(rows, "exact_truth_probability"),
            "eggroll_truth_probability": _mean(rows, "eggroll_truth_probability"),
            "mean_cosine": _mean(rows, "cosine"),
        }
    return {
        "metrics": metrics,
        "bootstrap": {
            "unit": "fresh_layout_group",
            "method": "deterministic_group_percentile",
            "samples": config.bootstrap_samples,
            "confidence": config.bootstrap_confidence,
            "seed": config.bootstrap_seed,
            "lower_bounds": lower_bounds,
        },
        "per_action": per_action,
        "groups": group_rows,
        "episodes": list(episode_rows),
    }


def apply_gate(
    audit: Mapping[str, Any],
    execution_checks: Mapping[str, bool],
) -> dict[str, Any]:
    metrics = audit["metrics"]
    lower = audit["bootstrap"]["lower_bounds"]
    norm_ratio = metrics["median_corrected_to_exact_norm_ratio"]
    retention = metrics["truth_probability_gain_retention"]
    checks = {
        **{name: bool(value) for name, value in execution_checks.items()},
        "exact_gradient_accuracy_at_least_0_70": metrics["exact_accuracy"] >= 0.70,
        "exact_gradient_truth_probability_at_least_0_60": (
            metrics["exact_truth_probability"] >= 0.60
        ),
        "mean_estimator_cosine_at_least_0_20": metrics["mean_cosine"] >= 0.20,
        "bootstrap_cosine_lower_bound_positive": lower["cosine"] > 0.0,
        "median_norm_ratio_at_least_0_5": (
            norm_ratio is not None and float(norm_ratio) >= 0.5
        ),
        "median_norm_ratio_at_most_2_0": (
            norm_ratio is not None and float(norm_ratio) <= 2.0
        ),
        "eggroll_accuracy_at_least_0_45": metrics["eggroll_accuracy"] >= 0.45,
        "eggroll_truth_probability_at_least_0_35": (
            metrics["eggroll_truth_probability"] >= 0.35
        ),
        "bootstrap_eggroll_accuracy_above_chance": (
            lower["eggroll_accuracy"] > 0.25
        ),
        "bootstrap_eggroll_probability_gain_positive": (
            lower["eggroll_truth_probability_gain"] > 0.0
        ),
        "eggroll_retains_half_exact_probability_gain": (
            retention is not None and float(retention) >= 0.50
        ),
    }
    return {"name": PROTOCOL, "passed": all(checks.values()), "checks": checks}


def audit_split_disjointness(
    canonical: Sequence[SplitManifest], audit: SplitManifest
) -> dict[str, Any]:
    canonical_layouts = [layout for manifest in canonical for layout in manifest.accepted]
    canonical_seeds = {layout.game_seed for layout in canonical_layouts}
    canonical_hashes = {layout.before_grid_sha256 for layout in canonical_layouts}
    audit_seeds = {layout.game_seed for layout in audit.accepted}
    audit_hashes = {layout.before_grid_sha256 for layout in audit.accepted}
    return {
        "canonical_layout_count": len(canonical_layouts),
        "audit_layout_count": len(audit.accepted),
        "seed_overlap": sorted(canonical_seeds & audit_seeds),
        "before_grid_hash_overlap": sorted(canonical_hashes & audit_hashes),
        "passed": not (canonical_seeds & audit_seeds or canonical_hashes & audit_hashes),
    }


def train_positive_control(
    config: Config,
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor, dict[str, Any], tuple[SplitManifest, ...]]:
    """Reproduce canonical train/validation and stop before the old locked split."""

    heldout_config = HeldoutConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        source_sha=config.source_sha,
        initialization="pretrained",
        output_dir=config.output_dir,
        seed=config.seed,
        epochs=config.epochs,
        prefix_length=config.prefix_length,
        prefix_initialization_std=config.prefix_initialization_std,
        inner_learning_rate=config.inner_learning_rate,
        prefix_learning_rate=config.prefix_learning_rate,
        model_learning_rate=config.model_learning_rate,
        weight_decay=config.weight_decay,
        no_evidence_weight=config.no_evidence_weight,
        freeze_first_n_blocks=config.freeze_first_n_blocks,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_confidence=config.bootstrap_confidence,
        bootstrap_seed=20_260_803,
        save_model=False,
        require_cuda=config.require_cuda,
    )
    manifests = build_split_manifests()
    by_name = {manifest.name: manifest for manifest in manifests}
    literal_audit = audit_generated_literal_action_text(manifests)
    if not literal_audit["passed"]:
        raise RuntimeError("canonical generated-text audit failed")
    binding_config = BindingConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        initialization="pretrained",
        prefix_length=config.prefix_length,
        prefix_initialization_std=config.prefix_initialization_std,
        inner_learning_rate=config.inner_learning_rate,
        outer_learning_rate=config.model_learning_rate,
        no_evidence_weight=config.no_evidence_weight,
        freeze_first_n_blocks=config.freeze_first_n_blocks,
        save_model=False,
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
    raw_config = _raw_config(heldout_config)
    history: list[dict[str, Any]] = []
    best_objective = math.inf
    best_epoch: int | None = None
    best_state: tuple[dict[str, torch.Tensor], torch.Tensor] | None = None
    steps = 0
    for epoch in range(1, config.epochs + 1):
        losses: list[float] = []
        for layout in by_name["train"].accepted:
            optimizer.zero_grad(set_to_none=True)
            quartet = build_heldout_quartet(layout, TRAIN_ACTION)
            group_losses: list[float] = []
            for episode in quartet:
                loss, _ = raw_episode_meta_loss(
                    model, tokenizer, prefix, episode, raw_config, device
                )
                (loss / len(quartet)).backward()
                group_losses.append(float(loss.detach().item()))
            torch.nn.utils.clip_grad_norm_([prefix, *model_parameters], 1.0)
            optimizer.step()
            steps += 1
            losses.append(sum(group_losses) / len(group_losses))
        validation = evaluate_action_groups(
            model,
            tokenizer,
            prefix.detach(),
            by_name["validation"].accepted,
            (TRAIN_ACTION,),
            heldout_config,
            device,
            bootstrap_seed=20_260_803 + epoch * 100,
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
                "mean_training_loss": sum(losses) / len(losses),
                "validation_selection_objective": objective,
                "validation": validation,
                "all_training_group_losses_finite": all(map(math.isfinite, losses)),
                "trainable_parameters_finite": trainable_parameters_finite,
                "validation_all_finite": bool(validation["all_finite"]),
                "all_finite": (
                    all(map(math.isfinite, losses))
                    and math.isfinite(objective)
                    and trainable_parameters_finite
                    and bool(validation["all_finite"])
                ),
            }
        )
        if checkpoint_is_better(objective, best_objective):
            best_objective = objective
            best_epoch = epoch
            best_state = _capture_trainable_state(model, prefix)
        print(
            json.dumps(
                {
                    "event": "positive_control_epoch",
                    "epoch": epoch,
                    "steps_completed": steps,
                    "mean_training_loss": history[-1]["mean_training_loss"],
                    "validation_selection_objective": objective,
                }
            ),
            flush=True,
        )
    if steps != 512 or best_state is None or best_epoch is None:
        raise RuntimeError("positive-control training did not complete exactly 512 steps")
    selected = best_validation_history_entry(history)
    if int(selected["epoch"]) != best_epoch:
        raise RuntimeError("checkpoint selection disagrees with validation history")
    _restore_trainable_state(model, prefix, best_state)
    training = {
        "steps_completed": steps,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "selected_validation_sha256": canonical_json_sha256(selected["validation"]),
        "history": history,
        "all_finite": all(bool(row["all_finite"]) for row in history),
        "literal_action_text_audit": literal_audit,
        "old_locked_test_evaluations": 0,
    }
    return model, tokenizer, prefix, training, manifests


def evaluate_fresh_audit(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    layouts: Sequence[ProbeLayout],
    config: Config,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = prefix.detach()
    rows: list[dict[str, Any]] = []
    max_antithetic_error = 0.0
    all_estimates_finite = True
    replay_delta: float | None = None
    first_adaptation_seconds: float | None = None
    projected_audit_seconds: float | None = None
    audit_started = time.time()
    total_episodes = len(layouts) * len(ACTIONS) * 4
    for layout_index, layout in enumerate(layouts):
        for action in ACTIONS:
            quartet = build_heldout_quartet(layout, action)
            for episode in quartet:
                truth_index = _DIRECTIONS.index(episode.direction)
                with torch.no_grad():
                    prior_probabilities = torch.softmax(
                        query_scores(model, tokenizer, prefix, action, device), dim=-1
                    )
                exact_prefix = prefix.clone().requires_grad_(True)
                exact_updated, _, exact_gradient, _ = raw_outcome_adapt_prefix(
                    model,
                    tokenizer,
                    exact_prefix,
                    episode.record,
                    episode.record,
                    inner_learning_rate=config.inner_learning_rate,
                    device=device,
                    create_graph=False,
                )
                with torch.no_grad():
                    exact_probabilities = torch.softmax(
                        query_scores(model, tokenizer, exact_updated, action, device),
                        dim=-1,
                    )
                seed = stable_episode_seed(
                    config.seed, layout, action, episode.variant_index
                )
                adaptation_started = time.time()
                eggroll_updated, diagnostics, estimates = eggroll_adapt_prefix(
                    model,
                    tokenizer,
                    prefix,
                    episode.record,
                    seed=seed,
                    config=config,
                    device=device,
                )
                adaptation_seconds = time.time() - adaptation_started
                if first_adaptation_seconds is None:
                    first_adaptation_seconds = adaptation_seconds
                    projected_audit_seconds = adaptation_seconds * total_episodes
                    if projected_audit_seconds > AUDIT_RUNTIME_CEILING_SECONDS:
                        raise RuntimeError(
                            "first EGGROLL adaptation projects an audit longer than "
                            f"{AUDIT_RUNTIME_CEILING_SECONDS:.0f} seconds"
                        )
                if replay_delta is None:
                    replay_updated, _, _ = eggroll_adapt_prefix(
                        model,
                        tokenizer,
                        prefix,
                        episode.record,
                        seed=seed,
                        config=config,
                        device=device,
                    )
                    replay_delta = float(
                        torch.max(torch.abs(eggroll_updated - replay_updated)).item()
                    )
                with torch.no_grad():
                    eggroll_probabilities = torch.softmax(
                        query_scores(model, tokenizer, eggroll_updated, action, device),
                        dim=-1,
                    )
                raw_estimate = estimates[-1]
                corrected_estimate = estimates[config.antithetic_pairs]
                cosine = float(
                    F.cosine_similarity(
                        raw_estimate.flatten(), exact_gradient.detach().float().flatten(), dim=0
                    ).item()
                )
                exact_norm = float(torch.linalg.vector_norm(exact_gradient).item())
                corrected_norm = float(
                    torch.linalg.vector_norm(corrected_estimate).item()
                )
                max_antithetic_error = max(
                    max_antithetic_error,
                    float(diagnostics["antithetic_center_max_abs_error"]),
                )
                all_estimates_finite = all_estimates_finite and bool(
                    diagnostics["all_finite"]
                )
                ladder_cosines = {
                    str(count): float(
                        F.cosine_similarity(
                            estimates[count].flatten(),
                            exact_gradient.detach().float().flatten(),
                            dim=0,
                        ).item()
                    )
                    for count in CONVERGENCE_LADDER
                }
                rows.append(
                    {
                        "layout_index": layout_index,
                        "game_seed": layout.game_seed,
                        "before_grid_sha256": layout.before_grid_sha256,
                        "action": action.value,
                        "variant_index": episode.variant_index,
                        "truth_direction": episode.direction.value,
                        "eggroll_seed": seed,
                        "prior_correct": float(
                            int(prior_probabilities.argmax().item()) == truth_index
                        ),
                        "exact_correct": float(
                            int(exact_probabilities.argmax().item()) == truth_index
                        ),
                        "eggroll_correct": float(
                            int(eggroll_probabilities.argmax().item()) == truth_index
                        ),
                        "exact_eggroll_argmax_agreement": float(
                            int(exact_probabilities.argmax().item())
                            == int(eggroll_probabilities.argmax().item())
                        ),
                        "prior_truth_probability": float(
                            prior_probabilities[truth_index].item()
                        ),
                        "exact_truth_probability": float(
                            exact_probabilities[truth_index].item()
                        ),
                        "eggroll_truth_probability": float(
                            eggroll_probabilities[truth_index].item()
                        ),
                        "exact_truth_probability_gain": float(
                            (exact_probabilities[truth_index] - prior_probabilities[truth_index]).item()
                        ),
                        "eggroll_truth_probability_gain": float(
                            (eggroll_probabilities[truth_index] - prior_probabilities[truth_index]).item()
                        ),
                        "cosine": cosine,
                        "exact_gradient_norm": exact_norm,
                        "corrected_estimate_norm": corrected_norm,
                        "corrected_to_exact_norm_ratio": (
                            corrected_norm / exact_norm if exact_norm > 0 else None
                        ),
                        "ladder_cosines": ladder_cosines,
                        "eggroll": diagnostics,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "audit_episode",
                            "completed": len(rows),
                            "total": total_episodes,
                            "cosine": cosine,
                            "exact_correct": rows[-1]["exact_correct"],
                            "eggroll_correct": rows[-1]["eggroll_correct"],
                        }
                    ),
                    flush=True,
                )
    audit = aggregate_audit(rows, config=config)
    execution = {
        "all_estimates_finite": all_estimates_finite,
        "rank_one_exact": config.perturbation_rank == 1,
        "pair_count_exact": config.antithetic_pairs == 512,
        "population_exact": config.antithetic_pairs * 2 == 1024,
        "antithetic_center_error_at_most_1e_6": max_antithetic_error <= 1e-6,
        "deterministic_replay_at_most_1e_6": replay_delta is not None and replay_delta <= 1e-6,
        "exact_gradients_excluded_from_eggroll_update": True,
        "raw_candidate_count_one_mean_nll": True,
        "fresh_audit_evaluations": 1,
        "episode_count_exact": len(rows) == len(layouts) * len(ACTIONS) * 4,
        "runtime_projection_within_ceiling": (
            projected_audit_seconds is not None
            and projected_audit_seconds <= AUDIT_RUNTIME_CEILING_SECONDS
        ),
        "max_antithetic_center_abs_error": max_antithetic_error,
        "deterministic_replay_max_abs_delta": replay_delta,
        "first_adaptation_seconds": first_adaptation_seconds,
        "projected_audit_seconds": projected_audit_seconds,
        "observed_audit_seconds": time.time() - audit_started,
        "audit_runtime_ceiling_seconds": AUDIT_RUNTIME_CEILING_SECONDS,
    }
    return audit, execution


def run(config: Config) -> dict[str, Any]:
    validate_protocol_config(config)
    set_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("this preregistered run requires CUDA")
    started = time.time()
    model, tokenizer, prefix, training, canonical_manifests = train_positive_control(
        config, device
    )
    audit_manifest = build_split_manifests((AUDIT_SPLIT_SPEC,))[0]
    disjointness = audit_split_disjointness(canonical_manifests, audit_manifest)
    if not disjointness["passed"]:
        raise RuntimeError("fresh EGGROLL audit overlaps a canonical layout surface")
    checkpoint_before = _checkpoint_sha256(model, prefix)
    audit, audit_execution = evaluate_fresh_audit(
        model,
        tokenizer,
        prefix,
        audit_manifest.accepted,
        config,
        device,
    )
    checkpoint_after = _checkpoint_sha256(model, prefix)
    execution_checks = {
        "source_sha_resolved": resolve_source_sha(config) is not None,
        "pinned_model_revision": config.model_revision == GPT2_REVISION,
        "pretrained_initialization_only": True,
        "canonical_training_steps_exact": training["steps_completed"] == 512,
        "canonical_training_finite": bool(training["all_finite"]),
        "old_locked_test_never_evaluated": training["old_locked_test_evaluations"] == 0,
        "fresh_audit_layout_count_exact": len(audit_manifest.accepted) == 8,
        "fresh_audit_disjoint": bool(disjointness["passed"]),
        "checkpoint_unchanged_during_audit": checkpoint_before == checkpoint_after,
        "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "no_model_artifact_persisted": True,
        "competition_submission_performed_false": True,
        **{
            name: bool(value)
            for name, value in audit_execution.items()
            if isinstance(value, bool)
        },
    }
    gate = apply_gate(audit, execution_checks)
    eligible = bool(gate["passed"])
    canonical_payload = canonical_manifest_payload(canonical_manifests)
    audit_payload = _manifest_dict(audit_manifest)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_sha": resolve_source_sha(config),
        "scope": (
            "Technical compatibility of rank-one antithetic black-box prefix adaptation "
            "with a known-positive synthetic GPT-2 gradient-memory circuit; not an "
            "ARC-AGI-3 score and not a hyperscale throughput claim."
        ),
        "config": asdict(config),
        "device": str(device),
        "training": training,
        "canonical_manifest": {
            "sha256": canonical_json_sha256(canonical_payload),
            "payload": canonical_payload,
        },
        "fresh_audit_manifest": {
            "sha256": canonical_json_sha256(audit_payload),
            "payload": audit_payload,
            "disjointness": disjointness,
        },
        "checkpoint": {
            "before_audit_sha256": checkpoint_before,
            "after_audit_sha256": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
        },
        "audit": audit,
        "audit_execution": audit_execution,
        "execution_checks": execution_checks,
        "gate": gate,
        "decision": (
            "eligible_for_raw_goal_followup" if eligible else "stop_do_not_apply_to_raw_goal"
        ),
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "elapsed_seconds": time.time() - started,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eggroll-result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default=GPT2_REVISION)
    parser.add_argument("--source-sha")
    parser.add_argument("--output-dir", default="outputs/eggroll_inner_memory")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        Config(
            model_name=args.model_name,
            model_revision=args.model_revision,
            source_sha=args.source_sha,
            output_dir=args.output_dir,
            require_cuda=args.require_cuda,
        )
    )


if __name__ == "__main__":
    main()
