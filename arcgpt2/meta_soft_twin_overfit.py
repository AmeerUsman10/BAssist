"""Decisive four-world overfit gate for GPT-2 gradient memory.

All four worlds have the same initial grid, action, support prompt, query, and
initial soft prefix. They differ only in the observed cardinal outcome. Every
outer step contains all four contradictory worlds, so a static answer cannot
reduce the balanced loss. This is a contrastive channel diagnostic; passing it
does not yet establish the stronger raw next-token prediction-error thesis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
    _DIRECTION_WORD,
    _WORDS,
    encode_text,
    generate_game,
    initialize_prefix,
    mapping_query,
    set_seed,
    transition_prompt,
)
from .meta_soft_single_binding import (
    Config as BindingConfig,
    SingleBindingEpisode,
    adapt_prefix,
    build_model_and_tokenizer,
    corrupted_target,
    episode_meta_loss,
    query_scores,
)
from .phase0_hidden_action import Action, Direction, HiddenActionGame


GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str = GPT2_REVISION
    initialization: str = "pretrained"
    output_dir: str = "outputs/meta_soft_twin_overfit/pretrained"
    seed: int = 271828
    group_seed: int = 314159
    action: str = "A1"
    prefix_length: int = 8
    prefix_initialization_std: float = 0.01
    inner_learning_rate: float = 0.2
    prefix_learning_rate: float = 1e-3
    model_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    no_evidence_weight: float = 0.25
    max_outer_steps: int = 200
    evaluation_interval: int = 10
    plateau_patience: int = 60
    freeze_first_n_blocks: int = 11
    save_model: bool = True
    require_cuda: bool = False


def build_quartet(group_seed: int, action: Action) -> tuple[SingleBindingEpisode, ...]:
    """Create four surface-identical worlds with contradictory action outcomes."""

    base = generate_game(group_seed)
    quartet: list[SingleBindingEpisode] = []
    for variant_index, desired in enumerate(_DIRECTIONS):
        mapping = dict(base.action_to_direction)
        donor = next(key for key, value in mapping.items() if value is desired)
        mapping[action], mapping[donor] = mapping[donor], mapping[action]
        spec = replace(base, action_to_direction=mapping)
        record = HiddenActionGame(spec).step(action)
        if record.status != "ACTIVE" or not record.moved:
            raise RuntimeError("quartet probe must be a safe cardinal move")
        quartet.append(
            SingleBindingEpisode(
                group_seed=group_seed,
                variant_index=variant_index,
                action=action,
                direction=desired,
                record=record,
            )
        )
    before = {episode.record.before for episode in quartet}
    after = {episode.record.after for episode in quartet}
    if len(before) != 1 or len(after) != 4:
        raise RuntimeError("quartet construction violated counterfactual invariants")
    return tuple(quartet)


def _hash_ids(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def _pairwise_min(values: Sequence[torch.Tensor], *, relative: bool) -> float:
    distances: list[float] = []
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            distance = torch.linalg.vector_norm(left - right)
            if relative:
                scale = max(
                    float(torch.linalg.vector_norm(left).item()),
                    float(torch.linalg.vector_norm(right).item()),
                    1e-12,
                )
                distance = distance / scale
            distances.append(float(distance.item()))
    return min(distances)


def apply_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered single-seed quartet_v1 thresholds."""

    invariants = metrics["invariants"]
    no_evidence = metrics["no_evidence"]
    intact = metrics["intact"]
    adaptation = metrics["adaptation"]
    corrupted = metrics["corrupted"]
    gradients = metrics["gradients"]
    training = metrics["training"]
    checks = {
        "one_support_prompt": invariants["unique_support_prompt_count"] == 1,
        "one_query_prompt": invariants["unique_query_prompt_count"] == 1,
        "four_unique_outcomes": invariants["unique_outcome_count"] == 4,
        "preupdate_causal_isolation": invariants["max_preupdate_logit_delta"] <= 1e-6,
        "no_evidence_uniform": no_evidence["max_abs_uniform_deviation"] <= 0.05,
        "no_evidence_high_entropy": no_evidence["entropy_bits"] >= 1.95,
        "intact_accuracy": intact["accuracy"] >= 0.95,
        "intact_mean_probability": intact["mean_truth_probability"] >= 0.80,
        "intact_min_probability": intact["min_truth_probability"] >= 0.70,
        "adaptation_accuracy_gain": adaptation["accuracy_gain"] >= 0.70,
        "adaptation_probability_gain": adaptation["mean_truth_probability_gain"] >= 0.55,
        "corruption_target_accuracy": corrupted["target_accuracy"] >= 0.95,
        "corruption_mean_target_probability": corrupted["mean_target_probability"] >= 0.80,
        "corruption_min_target_probability": corrupted["min_target_probability"] >= 0.70,
        "corruption_rejects_original": corrupted["original_truth_accuracy"] <= 0.05,
        "corruption_low_original_mean": corrupted["mean_original_truth_probability"] <= 0.10,
        "corruption_low_original_max": corrupted["max_original_truth_probability"] <= 0.20,
        "finite_gradients": bool(gradients["all_finite"]),
        "nonzero_gradients": gradients["min_norm"] > 0.0,
        "separable_gradients": gradients["min_pairwise_relative_distance"] >= 1e-3,
        "distinct_prefix_updates": gradients["min_pairwise_update_l2"] > 0.0,
        "deterministic_replay": invariants["deterministic_replay_max_abs_diff"] <= 1e-5,
        "finite_training": bool(training["all_finite"]),
        "eager_attention": metrics["attention_impl"] == "eager",
    }
    return {
        "name": "quartet_v1",
        "passed": all(checks.values()),
        "checks": checks,
    }


def evaluate_quartet(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    quartet: Sequence[SingleBindingEpisode],
    binding_config: BindingConfig,
    device: torch.device,
    *,
    training_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    action = quartet[0].action
    prompt_ids = [encode_text(tokenizer, transition_prompt(item.record)) for item in quartet]
    query_ids = [encode_text(tokenizer, mapping_query(item.action)) for item in quartet]
    prior_logits: list[torch.Tensor] = []
    prior_probabilities: list[torch.Tensor] = []
    intact_probabilities: list[torch.Tensor] = []
    corrupted_probabilities: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    repeat_differences: list[float] = []
    worlds: list[dict[str, Any]] = []

    for episode in quartet:
        with torch.no_grad():
            logits = query_scores(model, tokenizer, initial_prefix, action, device)
            prior = torch.softmax(logits, dim=-1)
        prior_logits.append(logits.detach().cpu())
        prior_probabilities.append(prior.detach().cpu())

        prefix = initial_prefix.detach().clone().requires_grad_(True)
        updated, support_loss, gradient = adapt_prefix(
            model,
            tokenizer,
            prefix,
            episode.record,
            episode.record,
            inner_learning_rate=binding_config.inner_learning_rate,
            device=device,
            create_graph=False,
        )
        with torch.no_grad():
            intact = torch.softmax(
                query_scores(model, tokenizer, updated, action, device), dim=-1
            )

        corrupt_record, corrupt_direction = corrupted_target(
            episode.record, episode.direction
        )
        corrupt_prefix = initial_prefix.detach().clone().requires_grad_(True)
        corrupt_updated, _, _ = adapt_prefix(
            model,
            tokenizer,
            corrupt_prefix,
            episode.record,
            corrupt_record,
            inner_learning_rate=binding_config.inner_learning_rate,
            device=device,
            create_graph=False,
        )
        with torch.no_grad():
            corrupt = torch.softmax(
                query_scores(model, tokenizer, corrupt_updated, action, device), dim=-1
            )
            repeat = torch.softmax(
                query_scores(model, tokenizer, updated, action, device), dim=-1
            )

        intact_probabilities.append(intact.detach().cpu())
        corrupted_probabilities.append(corrupt.detach().cpu())
        gradients.append(gradient.detach().cpu().flatten())
        updates.append((updated.detach() - prefix.detach()).cpu().flatten())
        repeat_differences.append(float(torch.max(torch.abs(intact - repeat)).item()))
        truth_index = _DIRECTIONS.index(episode.direction)
        corrupt_index = _DIRECTIONS.index(corrupt_direction)
        worlds.append(
            {
                "truth_direction": _DIRECTION_WORD[episode.direction],
                "corrupted_direction": _DIRECTION_WORD[corrupt_direction],
                "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
                "update_norm": float(
                    torch.linalg.vector_norm(updated.detach() - prefix.detach()).item()
                ),
                "support_loss": float(support_loss.detach().item()),
                "no_adaptation_probabilities": prior.detach().cpu().tolist(),
                "intact_probabilities": intact.detach().cpu().tolist(),
                "corrupted_probabilities": corrupt.detach().cpu().tolist(),
                "truth_index": truth_index,
                "corrupted_index": corrupt_index,
            }
        )

    prior_stack = torch.stack(prior_probabilities)
    intact_stack = torch.stack(intact_probabilities)
    corrupt_stack = torch.stack(corrupted_probabilities)
    truth_indices = torch.arange(4)
    corrupt_indices = torch.tensor(
        [_DIRECTIONS.index(corrupted_target(item.record, item.direction)[1]) for item in quartet]
    )
    no_adapt_predictions = prior_stack.argmax(dim=-1)
    intact_predictions = intact_stack.argmax(dim=-1)
    corrupt_predictions = corrupt_stack.argmax(dim=-1)
    truth_probs = intact_stack[torch.arange(4), truth_indices]
    prior_truth_probs = prior_stack[torch.arange(4), truth_indices]
    corrupt_target_probs = corrupt_stack[torch.arange(4), corrupt_indices]
    corrupt_truth_probs = corrupt_stack[torch.arange(4), truth_indices]
    centered_logits = torch.stack(prior_logits)
    max_preupdate_delta = float(
        torch.max(torch.abs(centered_logits - centered_logits[0])).item()
    )
    prior = prior_stack[0]

    metrics = {
        "attention_impl": getattr(model.config, "_attn_implementation", None),
        "invariants": {
            "support_prompt_sha256": _hash_ids(prompt_ids[0]),
            "query_prompt_sha256": _hash_ids(query_ids[0]),
            "unique_support_prompt_count": len({tuple(item) for item in prompt_ids}),
            "unique_query_prompt_count": len({tuple(item) for item in query_ids}),
            "unique_outcome_count": len({item.record.after for item in quartet}),
            "max_preupdate_logit_delta": max_preupdate_delta,
            "deterministic_replay_max_abs_diff": max(repeat_differences),
        },
        "no_evidence": {
            "entropy_bits": _entropy_bits(prior),
            "max_probability": float(prior.max().item()),
            "max_abs_uniform_deviation": float(torch.max(torch.abs(prior - 0.25)).item()),
        },
        "intact": {
            "accuracy": float((intact_predictions == truth_indices).float().mean().item()),
            "mean_truth_probability": float(truth_probs.mean().item()),
            "min_truth_probability": float(truth_probs.min().item()),
            "min_correct_vs_other_world_margin": float(
                min(
                    intact_stack[index, index]
                    - torch.max(torch.cat((intact_stack[index, :index], intact_stack[index, index + 1 :])))
                    for index in range(4)
                ).item()
            ),
        },
        "adaptation": {
            "accuracy_gain": float(
                ((intact_predictions == truth_indices).float().mean()
                 - (no_adapt_predictions == truth_indices).float().mean()).item()
            ),
            "mean_truth_probability_gain": float(
                (truth_probs.mean() - prior_truth_probs.mean()).item()
            ),
        },
        "corrupted": {
            "target_accuracy": float((corrupt_predictions == corrupt_indices).float().mean().item()),
            "mean_target_probability": float(corrupt_target_probs.mean().item()),
            "min_target_probability": float(corrupt_target_probs.min().item()),
            "original_truth_accuracy": float((corrupt_predictions == truth_indices).float().mean().item()),
            "mean_original_truth_probability": float(corrupt_truth_probs.mean().item()),
            "max_original_truth_probability": float(corrupt_truth_probs.max().item()),
        },
        "gradients": {
            "all_finite": all(bool(torch.isfinite(item).all()) for item in gradients),
            "min_norm": min(float(torch.linalg.vector_norm(item).item()) for item in gradients),
            "min_pairwise_relative_distance": _pairwise_min(gradients, relative=True),
            "min_pairwise_update_l2": _pairwise_min(updates, relative=False),
        },
        "training": dict(training_metrics),
    }
    return {"metrics": metrics, "worlds": worlds, "gate": apply_gate(metrics)}


def train(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    action = Action(config.action)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("this run requires a CUDA device")
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
    quartet = build_quartet(config.group_seed, action)
    model_parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [prefix], "lr": config.prefix_learning_rate, "weight_decay": 0.0},
            {"params": model_parameters, "lr": config.model_learning_rate, "weight_decay": config.weight_decay},
        ]
    )
    history: list[dict[str, float]] = []
    started = time.time()
    best_loss = math.inf
    last_improvement = 0
    first_passing_step: int | None = None
    initial_query_nll: float | None = None
    stopped_reason = "max_steps"

    initial = evaluate_quartet(
        model,
        tokenizer,
        prefix.detach(),
        quartet,
        binding_config,
        device,
        training_metrics={
            "steps_completed": 0,
            "first_passing_step": None,
            "initial_query_nll": None,
            "final_query_nll": None,
            "all_finite": True,
        },
    )

    for step in range(1, config.max_outer_steps + 1):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        query_losses: list[float] = []
        for episode in quartet:
            loss, diagnostics = episode_meta_loss(
                model, tokenizer, prefix, episode, binding_config, device
            )
            (loss / len(quartet)).backward()
            step_losses.append(float(loss.detach().item()))
            query_losses.append(diagnostics["query_loss"])
        trainable = [prefix, *model_parameters]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        mean_loss = sum(step_losses) / len(step_losses)
        mean_query = sum(query_losses) / len(query_losses)
        if initial_query_nll is None:
            initial_query_nll = mean_query
        history.append({"step": float(step), "loss": mean_loss, "query_nll": mean_query})
        if math.isfinite(mean_loss) and mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            last_improvement = step
        if step == 1 or step % config.evaluation_interval == 0:
            training_metrics = {
                "steps_completed": step,
                "first_passing_step": first_passing_step,
                "initial_query_nll": initial_query_nll,
                "final_query_nll": mean_query,
                "all_finite": all(math.isfinite(item["loss"]) for item in history),
            }
            evaluation = evaluate_quartet(
                model, tokenizer, prefix.detach(), quartet, binding_config, device,
                training_metrics=training_metrics,
            )
            print(json.dumps({
                "step": step,
                "loss": mean_loss,
                "query_nll": mean_query,
                "gate_passed": evaluation["gate"]["passed"],
                "intact": evaluation["metrics"]["intact"],
                "no_evidence": evaluation["metrics"]["no_evidence"],
                "elapsed_seconds": time.time() - started,
            }), flush=True)
            if evaluation["gate"]["passed"]:
                first_passing_step = step
                stopped_reason = "gate_passed"
                break
        if step - last_improvement >= config.plateau_patience:
            stopped_reason = "plateau"
            break

    completed_steps = len(history)
    final_training = {
        "steps_completed": completed_steps,
        "first_passing_step": first_passing_step,
        "initial_query_nll": initial_query_nll,
        "final_query_nll": history[-1]["query_nll"] if history else None,
        "all_finite": all(math.isfinite(item["loss"]) for item in history),
        "stopped_reason": stopped_reason,
    }
    final = evaluate_quartet(
        model, tokenizer, prefix.detach(), quartet, binding_config, device,
        training_metrics=final_training,
    )
    if final["gate"]["passed"] and first_passing_step is None:
        first_passing_step = completed_steps
        final["metrics"]["training"]["first_passing_step"] = completed_steps

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "meta_soft_single_quartet",
        "scope": "contrastive cardinal gradient-memory channel diagnostic; not ARC-AGI-3 capability",
        "source_sha": os.environ.get("ARC_GPT2_SOURCE_SHA", os.environ.get("GITHUB_SHA")),
        "seed": config.seed,
        "initialization": config.initialization,
        "support_objective": "contrastive_cardinal",
        "config": asdict(config),
        "device": str(device),
        "initial": initial,
        "worlds": final["worlds"],
        "metrics": final["metrics"],
        "gate": final["gate"],
        "status": "pass" if final["gate"]["passed"] else "fail",
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.save_model and final["gate"]["passed"]:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        torch.save({"version": 1, "prefix": prefix.detach().cpu(), "config": asdict(config)}, output_dir / "initial_soft_prefix.pt")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default=GPT2_REVISION)
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--output-dir", default="outputs/meta_soft_twin_overfit/pretrained")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--group-seed", type=int, default=314159)
    parser.add_argument("--action", choices=tuple(item.value for item in Action), default="A1")
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--prefix-initialization-std", type=float, default=0.01)
    parser.add_argument("--inner-learning-rate", type=float, default=0.2)
    parser.add_argument("--prefix-learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-evidence-weight", type=float, default=0.25)
    parser.add_argument("--max-outer-steps", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--plateau-patience", type=int, default=60)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=11)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(Config(
        model_name=args.model_name,
        model_revision=args.model_revision,
        initialization=args.initialization,
        output_dir=args.output_dir,
        seed=args.seed,
        group_seed=args.group_seed,
        action=args.action,
        prefix_length=args.prefix_length,
        prefix_initialization_std=args.prefix_initialization_std,
        inner_learning_rate=args.inner_learning_rate,
        prefix_learning_rate=args.prefix_learning_rate,
        model_learning_rate=args.model_learning_rate,
        weight_decay=args.weight_decay,
        no_evidence_weight=args.no_evidence_weight,
        max_outer_steps=args.max_outer_steps,
        evaluation_interval=args.evaluation_interval,
        plateau_patience=args.plateau_patience,
        freeze_first_n_blocks=args.freeze_first_n_blocks,
        save_model=not args.no_save_model,
        require_cuda=args.require_cuda,
    ))


if __name__ == "__main__":
    main()
