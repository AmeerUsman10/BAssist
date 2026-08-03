"""Four-world overfit gate using raw observed-outcome next-token NLL.

This is the strict counterpart to :mod:`meta_soft_twin_overfit`.  The four
worlds still share one grid, action, support prompt, query, and initial soft
prefix, but the inner update is not a four-candidate contrastive classifier.
It is ordinary causal-language-model negative log likelihood of the single
exact outcome observed in that world.  Passing therefore tests the original
claim that raw prediction error can write a causal fact into temporary memory.

The outer query target and uniform pre-evidence target are offline
meta-training supervision.  At adaptation time the only target is the exact
environment outcome.  No auxiliary learned head or model is introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .completion_scorer import score_candidate_completions
from .meta_soft_binding import (
    _DIRECTIONS,
    _DIRECTION_WORD,
    _WORDS,
    encode_text,
    initialize_prefix,
    mapping_query,
    set_seed,
    transition_prompt,
)
from .meta_soft_second_order import outcome_only_target
from .meta_soft_single_binding import (
    Config as BindingConfig,
    SingleBindingEpisode,
    build_model_and_tokenizer,
    corrupted_target,
    query_scores,
)
from .meta_soft_twin_overfit import (
    GPT2_REVISION,
    RESULT_SCHEMA_VERSION,
    _entropy_bits,
    _hash_ids,
    _pairwise_min,
    apply_gate,
    build_quartet,
)
from .phase0_hidden_action import Action, StepRecord


SUPPORT_OBJECTIVE = "raw_outcome_nll"
_CARDINAL_WORDS = frozenset(_WORDS)


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str = GPT2_REVISION
    source_sha: str | None = None
    initialization: str = "pretrained"
    output_dir: str = "outputs/meta_soft_raw_outcome_overfit/pretrained"
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


def resolve_source_sha(config: Config) -> str | None:
    """Prefer an explicit runner SHA, then the two standard execution variables."""

    return (
        config.source_sha
        or os.environ.get("ARC_GPT2_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA")
    )


def raw_support_text(
    prompt_record: StepRecord,
    target_record: StepRecord,
) -> tuple[str, str]:
    """Return the identical intervention prompt and one exact observed target."""

    prompt = transition_prompt(prompt_record)
    target = outcome_only_target(target_record)
    words = set(re.findall(r"[a-z]+", (prompt + "\n" + target).lower()))
    leaked = sorted(words & _CARDINAL_WORDS)
    if leaked:
        raise RuntimeError(
            "raw support text leaked cardinal semantics: " + ", ".join(leaked)
        )
    return prompt, target


def raw_support_token_ids(
    tokenizer: Any,
    prompt_record: StepRecord,
    target_record: StepRecord,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tokenize the raw support pair without inventing counterfactual labels."""

    prompt, target = raw_support_text(prompt_record, target_record)
    return encode_text(tokenizer, prompt), encode_text(tokenizer, target)


def raw_outcome_adapt_prefix(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    prompt_record: StepRecord,
    target_record: StepRecord,
    *,
    inner_learning_rate: float,
    device: torch.device,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Write one exact outcome using its ordinary mean next-token NLL."""

    prompt_ids, target_ids = raw_support_token_ids(
        tokenizer,
        prompt_record,
        target_record,
    )
    log_likelihood = score_candidate_completions(
        model,
        prompt_ids,
        (target_ids,),
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=1,
        reduction="mean",
        soft_prefix=prefix,
    )[0]
    loss = -log_likelihood
    gradient = torch.autograd.grad(
        loss,
        prefix,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    updated = prefix - inner_learning_rate * gradient
    return updated, loss, gradient, len(target_ids)


def raw_episode_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: SingleBindingEpisode,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiate the query through the raw outcome prefix update."""

    updated, support_loss, gradient, target_tokens = raw_outcome_adapt_prefix(
        model,
        tokenizer,
        initial_prefix,
        episode.record,
        episode.record,
        inner_learning_rate=config.inner_learning_rate,
        device=device,
        create_graph=True,
    )
    scores = query_scores(model, tokenizer, updated, episode.action, device)
    truth_index = _DIRECTIONS.index(episode.direction)
    query_loss = F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([truth_index], dtype=torch.long, device=device),
    )
    prior_scores = query_scores(
        model,
        tokenizer,
        initial_prefix,
        episode.action,
        device,
    )
    uniform = torch.full_like(prior_scores, 1.0 / len(_WORDS))
    prior_loss = -(uniform * F.log_softmax(prior_scores, dim=-1)).sum()
    total = query_loss + config.no_evidence_weight * prior_loss
    return total, {
        "query_loss": float(query_loss.detach().item()),
        "prior_loss": float(prior_loss.detach().item()),
        "support_loss": float(support_loss.detach().item()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient.detach()).item()),
        "support_target_tokens": float(target_tokens),
    }


def corrupted_direction_indices(
    quartet: Sequence[SingleBindingEpisode],
) -> tuple[int, ...]:
    """Return exactly one fixed-corruption target index for each quartet world."""

    return tuple(
        _DIRECTIONS.index(corrupted_target(item.record, item.direction)[1])
        for item in quartet
    )


def evaluate_quartet(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    quartet: Sequence[SingleBindingEpisode],
    config: Config,
    device: torch.device,
    *,
    training_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the same functional gate with raw-NLL memory writes."""

    model.eval()
    action = quartet[0].action
    support_pairs = [
        raw_support_token_ids(tokenizer, item.record, item.record)
        for item in quartet
    ]
    prompt_ids = [pair[0] for pair in support_pairs]
    target_ids = [pair[1] for pair in support_pairs]
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
        updated, support_loss, gradient, target_tokens = raw_outcome_adapt_prefix(
            model,
            tokenizer,
            prefix,
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

        corrupt_record, corrupt_direction = corrupted_target(
            episode.record,
            episode.direction,
        )
        corrupt_prefix = initial_prefix.detach().clone().requires_grad_(True)
        corrupt_updated, corrupt_loss, _, corrupt_tokens = raw_outcome_adapt_prefix(
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
                query_scores(model, tokenizer, corrupt_updated, action, device),
                dim=-1,
            )
            repeat = torch.softmax(
                query_scores(model, tokenizer, updated, action, device),
                dim=-1,
            )

        intact_probabilities.append(intact.detach().cpu())
        corrupted_probabilities.append(corrupt.detach().cpu())
        gradients.append(gradient.detach().cpu().flatten())
        updates.append((updated.detach() - prefix.detach()).cpu().flatten())
        repeat_differences.append(float(torch.max(torch.abs(intact - repeat)).item()))
        truth_index = _DIRECTIONS.index(episode.direction)
        corrupt_index = _DIRECTIONS.index(corrupt_direction)
        _, exact_target = raw_support_text(episode.record, episode.record)
        worlds.append(
            {
                "truth_direction": _DIRECTION_WORD[episode.direction],
                "corrupted_direction": _DIRECTION_WORD[corrupt_direction],
                "support_objective": SUPPORT_OBJECTIVE,
                "support_target_sha256": _hash_ids(
                    encode_text(tokenizer, exact_target)
                ),
                "support_target_tokens": target_tokens,
                "corrupted_target_tokens": corrupt_tokens,
                "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
                "update_norm": float(
                    torch.linalg.vector_norm(updated.detach() - prefix.detach()).item()
                ),
                "support_loss": float(support_loss.detach().item()),
                "corrupted_support_loss": float(corrupt_loss.detach().item()),
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
    corrupt_indices = torch.tensor(corrupted_direction_indices(quartet))
    no_adapt_predictions = prior_stack.argmax(dim=-1)
    intact_predictions = intact_stack.argmax(dim=-1)
    corrupt_predictions = corrupt_stack.argmax(dim=-1)
    truth_probs = intact_stack[torch.arange(4), truth_indices]
    prior_truth_probs = prior_stack[torch.arange(4), truth_indices]
    corrupt_target_probs = corrupt_stack[torch.arange(4), corrupt_indices]
    corrupt_truth_probs = corrupt_stack[torch.arange(4), truth_indices]
    max_preupdate_delta = float(
        torch.max(torch.abs(torch.stack(prior_logits) - prior_logits[0])).item()
    )
    prior = prior_stack[0]

    metrics = {
        "attention_impl": getattr(model.config, "_attn_implementation", None),
        "support": {
            "objective": SUPPORT_OBJECTIVE,
            "candidate_count": 1,
            "counterfactuals_used_in_inner_update": False,
            "reduction": "mean",
            "unique_target_count": len(set(target_ids)),
            "min_target_tokens": min(len(item) for item in target_ids),
            "max_target_tokens": max(len(item) for item in target_ids),
        },
        "invariants": {
            "support_prompt_sha256": _hash_ids(prompt_ids[0]),
            "query_prompt_sha256": _hash_ids(query_ids[0]),
            "unique_support_prompt_count": len(set(prompt_ids)),
            "unique_query_prompt_count": len(set(query_ids)),
            "unique_outcome_count": len({item.record.after for item in quartet}),
            "max_preupdate_logit_delta": max_preupdate_delta,
            "deterministic_replay_max_abs_diff": max(repeat_differences),
        },
        "no_evidence": {
            "entropy_bits": _entropy_bits(prior),
            "max_probability": float(prior.max().item()),
            "max_abs_uniform_deviation": float(
                torch.max(torch.abs(prior - 0.25)).item()
            ),
        },
        "intact": {
            "accuracy": float(
                (intact_predictions == truth_indices).float().mean().item()
            ),
            "mean_truth_probability": float(truth_probs.mean().item()),
            "min_truth_probability": float(truth_probs.min().item()),
            "min_correct_vs_other_world_margin": float(
                min(
                    intact_stack[index, index]
                    - torch.max(
                        torch.cat(
                            (
                                intact_stack[index, :index],
                                intact_stack[index, index + 1 :],
                            )
                        )
                    )
                    for index in range(4)
                ).item()
            ),
        },
        "adaptation": {
            "accuracy_gain": float(
                (
                    (intact_predictions == truth_indices).float().mean()
                    - (no_adapt_predictions == truth_indices).float().mean()
                ).item()
            ),
            "mean_truth_probability_gain": float(
                (truth_probs.mean() - prior_truth_probs.mean()).item()
            ),
        },
        "corrupted": {
            "interpretation": "quartet_permutation_consistency_not_independent_evidence",
            "target_accuracy": float(
                (corrupt_predictions == corrupt_indices).float().mean().item()
            ),
            "mean_target_probability": float(corrupt_target_probs.mean().item()),
            "min_target_probability": float(corrupt_target_probs.min().item()),
            "original_truth_accuracy": float(
                (corrupt_predictions == truth_indices).float().mean().item()
            ),
            "mean_original_truth_probability": float(
                corrupt_truth_probs.mean().item()
            ),
            "max_original_truth_probability": float(corrupt_truth_probs.max().item()),
        },
        "gradients": {
            "all_finite": all(bool(torch.isfinite(item).all()) for item in gradients),
            "min_norm": min(
                float(torch.linalg.vector_norm(item).item()) for item in gradients
            ),
            "min_pairwise_relative_distance": _pairwise_min(
                gradients,
                relative=True,
            ),
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
    # eval() removes dropout while retaining full first- and second-order autograd.
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
            {
                "params": [prefix],
                "lr": config.prefix_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": model_parameters,
                "lr": config.model_learning_rate,
                "weight_decay": config.weight_decay,
            },
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
        config,
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
        support_losses: list[float] = []
        for episode in quartet:
            loss, diagnostics = raw_episode_meta_loss(
                model,
                tokenizer,
                prefix,
                episode,
                config,
                device,
            )
            (loss / len(quartet)).backward()
            step_losses.append(float(loss.detach().item()))
            query_losses.append(diagnostics["query_loss"])
            support_losses.append(diagnostics["support_loss"])
        trainable = [prefix, *model_parameters]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        mean_loss = sum(step_losses) / len(step_losses)
        mean_query = sum(query_losses) / len(query_losses)
        mean_support = sum(support_losses) / len(support_losses)
        if initial_query_nll is None:
            initial_query_nll = mean_query
        history.append(
            {
                "step": float(step),
                "loss": mean_loss,
                "query_nll": mean_query,
                "raw_support_nll": mean_support,
            }
        )
        if math.isfinite(mean_loss) and mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            last_improvement = step
        if step == 1 or step % config.evaluation_interval == 0:
            training_metrics = {
                "steps_completed": step,
                "first_passing_step": first_passing_step,
                "initial_query_nll": initial_query_nll,
                "final_query_nll": mean_query,
                "final_raw_support_nll": mean_support,
                "all_finite": all(
                    math.isfinite(item["loss"])
                    and math.isfinite(item["raw_support_nll"])
                    for item in history
                ),
            }
            evaluation = evaluate_quartet(
                model,
                tokenizer,
                prefix.detach(),
                quartet,
                config,
                device,
                training_metrics=training_metrics,
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": mean_loss,
                        "query_nll": mean_query,
                        "raw_support_nll": mean_support,
                        "gate_passed": evaluation["gate"]["passed"],
                        "intact": evaluation["metrics"]["intact"],
                        "no_evidence": evaluation["metrics"]["no_evidence"],
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
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
        "final_raw_support_nll": (
            history[-1]["raw_support_nll"] if history else None
        ),
        "all_finite": all(
            math.isfinite(item["loss"])
            and math.isfinite(item["raw_support_nll"])
            for item in history
        ),
        "stopped_reason": stopped_reason,
    }
    final = evaluate_quartet(
        model,
        tokenizer,
        prefix.detach(),
        quartet,
        config,
        device,
        training_metrics=final_training,
    )
    if final["gate"]["passed"] and first_passing_step is None:
        first_passing_step = completed_steps
        final["metrics"]["training"]["first_passing_step"] = completed_steps

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": "meta_soft_raw_outcome_quartet",
        "scope": (
            "raw exact observed-outcome next-token NLL gradient-memory channel "
            "diagnostic; corruption replays a trained quartet permutation and is "
            "a consistency control, not independent evidence; not ARC-AGI-3 capability"
        ),
        "source_sha": resolve_source_sha(config),
        "seed": config.seed,
        "initialization": config.initialization,
        "support_objective": SUPPORT_OBJECTIVE,
        "counterfactuals_used_in_inner_update": False,
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if config.save_model and final["gate"]["passed"]:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        torch.save(
            {
                "version": 1,
                "prefix": prefix.detach().cpu(),
                "config": asdict(config),
            },
            output_dir / "initial_soft_prefix.pt",
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default=GPT2_REVISION)
    parser.add_argument("--source-sha")
    parser.add_argument(
        "--initialization",
        choices=("pretrained", "random"),
        default="pretrained",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/meta_soft_raw_outcome_overfit/pretrained",
    )
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--group-seed", type=int, default=314159)
    parser.add_argument(
        "--action",
        choices=tuple(item.value for item in Action),
        default="A1",
    )
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
    train(
        Config(
            model_name=args.model_name,
            model_revision=args.model_revision,
            source_sha=args.source_sha,
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
        )
    )


if __name__ == "__main__":
    main()
