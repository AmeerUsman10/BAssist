"""Minimal one-shot gradient-memory gate for the original GPT-2.

This experiment isolates the central claim before asking GPT-2 to solve a full
permutation or an ARC game:

> Can one exact action-outcome observation be written into a temporary soft
> prefix by prediction error and read back by the same GPT-2 when the later
> query contains neither the grid nor the history?

Each episode contains one hidden binding ``action -> cardinal direction``. The
support loss contrasts the four exact counterfactual outcomes. The query asks
only for the observed action's meaning. This is the smallest clean test of
learned gradient memory. It uses one GPT-2 checkpoint, its ordinary LM head, and
one trainable soft prefix; there is no auxiliary learned encoder or classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .completion_scorer import score_candidate_completions
from .meta_soft_binding import (
    _DIRECTIONS,
    _WORDS,
    _DIRECTION_WORD,
    encode_text,
    initialize_prefix,
    make_episode,
    mapping_query,
    set_seed,
    transition_prompt,
)
from .meta_soft_contrastive import counterfactual_probe_records
from .meta_soft_second_order import outcome_only_target
from .natural_protocol import answer_text
from .phase0_hidden_action import Action, Direction, HiddenActionGame, StepRecord


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str | None = None
    initialization: str = "pretrained"
    output_dir: str = "outputs/meta_soft_single/pretrained"
    seed: int = 42
    train_seed_base: int = 910_000
    train_groups: int = 128
    train_mapping_variants: int = 8
    validation_seed_base: int = 920_000
    validation_groups: int = 16
    validation_mapping_variants: int = 8
    prefix_length: int = 8
    prefix_initialization_std: float = 0.01
    inner_learning_rate: float = 0.2
    outer_learning_rate: float = 8e-5
    no_evidence_weight: float = 0.25
    weight_decay: float = 0.01
    max_outer_steps: int = 160
    warmup_steps: int = 12
    freeze_first_n_blocks: int = 11
    train_embeddings: bool = False
    evaluation_groups: int = 8
    save_model: bool = True


@dataclass(frozen=True)
class SingleBindingEpisode:
    group_seed: int
    variant_index: int
    action: Action
    direction: Direction
    record: StepRecord


@dataclass(frozen=True)
class BindingTrace:
    group_seed: int
    variant_index: int
    action: str
    truth_direction: str
    evidence_direction: str | None
    mode: str
    predicted_direction: str
    correct: bool
    truth_probability: float
    entropy_bits: float
    support_loss: float | None
    gradient_norm: float | None


def build_model_and_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            attn_implementation="eager",
            revision=config.model_revision,
        )
    elif config.initialization == "random":
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            revision=config.model_revision,
        )
        model_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError("initialization must be pretrained or random")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("single-binding meta-training requires eager attention")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not 0 <= config.freeze_first_n_blocks <= len(model.transformer.h):
        raise ValueError("freeze_first_n_blocks lies outside GPT-2 depth")
    for block in model.transformer.h[: config.freeze_first_n_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = config.train_embeddings
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = config.train_embeddings
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True
    return model, tokenizer


def make_single_episode(
    group_seed: int,
    variant_index: int,
    variant_count: int,
    action: Action,
) -> SingleBindingEpisode:
    episode = make_episode(group_seed, variant_index, variant_count)
    # Every counterfactual starts from the exact same initial state. Selecting a
    # later record from the sequential probe trajectory would leak the hidden
    # mapping into the BEFORE grid through earlier movements.
    record = HiddenActionGame(episode.spec).step(action)
    if record.status != "ACTIVE" or not record.moved:
        raise RuntimeError("single-binding probe must be a safe cardinal move")
    direction = episode.spec.action_to_direction[action]
    return SingleBindingEpisode(
        group_seed=group_seed,
        variant_index=variant_index,
        action=action,
        direction=direction,
        record=record,
    )


def _candidate_ids(tokenizer: Any, texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    values = tuple(encode_text(tokenizer, text) for text in texts)
    if any(not value for value in values):
        raise ValueError("a candidate tokenized to an empty sequence")
    return values


def support_scores(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    record: StepRecord,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[StepRecord, ...]]:
    candidates = counterfactual_probe_records(record)
    prompt_ids = encode_text(tokenizer, transition_prompt(record))
    targets = _candidate_ids(
        tokenizer,
        tuple(outcome_only_target(candidate) for candidate in candidates),
    )
    scores = score_candidate_completions(
        model,
        prompt_ids,
        targets,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=4,
        reduction="mean",
        soft_prefix=prefix,
    )
    return scores, candidates


def adapt_prefix(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    prompt_record: StepRecord,
    target_record: StepRecord,
    *,
    inner_learning_rate: float,
    device: torch.device,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores, candidates = support_scores(
        model,
        tokenizer,
        prefix,
        prompt_record,
        device,
    )
    matching = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.after == target_record.after
        and candidate.status == target_record.status
    ]
    if not matching:
        raise ValueError("observed outcome is outside the cardinal counterfactual set")
    target = torch.zeros_like(scores)
    target[matching] = 1.0 / len(matching)
    loss = -(target * F.log_softmax(scores, dim=-1)).sum()
    gradient = torch.autograd.grad(
        loss,
        prefix,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    updated = prefix - inner_learning_rate * gradient
    return updated, loss, gradient


def query_scores(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    action: Action,
    device: torch.device,
) -> torch.Tensor:
    prompt = encode_text(tokenizer, mapping_query(action))
    candidates = _candidate_ids(
        tokenizer,
        tuple(answer_text(word) for word in _WORDS),
    )
    return score_candidate_completions(
        model,
        prompt,
        candidates,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=4,
        reduction="sum",
        soft_prefix=prefix,
    )


def query_loss(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    action: Action,
    direction: Direction,
    device: torch.device,
) -> torch.Tensor:
    scores = query_scores(model, tokenizer, prefix, action, device)
    truth_word = _DIRECTION_WORD[direction]
    target_index = _WORDS.index(truth_word)
    return F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([target_index], dtype=torch.long, device=device),
    )


def episode_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: SingleBindingEpisode,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    updated, support_loss, gradient = adapt_prefix(
        model,
        tokenizer,
        initial_prefix,
        episode.record,
        episode.record,
        inner_learning_rate=config.inner_learning_rate,
        device=device,
        create_graph=True,
    )
    loss = query_loss(
        model,
        tokenizer,
        updated,
        episode.action,
        episode.direction,
        device,
    )
    prior_scores = query_scores(
        model,
        tokenizer,
        initial_prefix,
        episode.action,
        device,
    )
    uniform_target = torch.full_like(prior_scores, 1.0 / len(_WORDS))
    prior_loss = -(uniform_target * F.log_softmax(prior_scores, dim=-1)).sum()
    total_loss = loss + config.no_evidence_weight * prior_loss
    return total_loss, {
        "query_loss": float(loss.detach().item()),
        "prior_loss": float(prior_loss.detach().item()),
        "support_loss": float(support_loss.detach().item()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient.detach()).item()),
    }


_DERANGEMENT = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
}


def deranged_direction(direction: Direction) -> Direction:
    """Return a fixed balanced wrong direction for corruption controls."""

    return _DERANGEMENT[direction]


def corrupted_target(record: StepRecord, truth: Direction) -> tuple[StepRecord, Direction]:
    """Inject the fixed deranged outcome and return its semantic direction."""

    candidates = counterfactual_probe_records(record)
    injected = deranged_direction(truth)
    return candidates[_DIRECTIONS.index(injected)], injected


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def evaluate_episode(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: SingleBindingEpisode,
    config: Config,
    device: torch.device,
    *,
    mode: str,
) -> BindingTrace:
    if mode not in {"intact", "no_adaptation", "shuffled_outcome"}:
        raise ValueError("unknown single-binding evaluation mode")
    prefix = initial_prefix.detach().clone().requires_grad_(True)
    support_value: float | None = None
    gradient_norm: float | None = None
    evidence_direction: Direction | None = None
    if mode != "no_adaptation":
        if mode == "intact":
            target = episode.record
            evidence_direction = episode.direction
        else:
            target, evidence_direction = corrupted_target(
                episode.record,
                episode.direction,
            )
        prefix, support_loss, gradient = adapt_prefix(
            model,
            tokenizer,
            prefix,
            episode.record,
            target,
            inner_learning_rate=config.inner_learning_rate,
            device=device,
            create_graph=False,
        )
        prefix = prefix.detach()
        support_value = float(support_loss.detach().item())
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach()).item())

    with torch.no_grad():
        scores = query_scores(
            model,
            tokenizer,
            prefix,
            episode.action,
            device,
        )
        probabilities = torch.softmax(scores, dim=-1)
    truth_word = _DIRECTION_WORD[episode.direction]
    truth_index = _WORDS.index(truth_word)
    predicted_index = int(torch.argmax(probabilities).item())
    return BindingTrace(
        group_seed=episode.group_seed,
        variant_index=episode.variant_index,
        action=episode.action.value,
        truth_direction=truth_word,
        evidence_direction=(
            _DIRECTION_WORD[evidence_direction]
            if evidence_direction is not None
            else None
        ),
        mode=mode,
        predicted_direction=_WORDS[predicted_index],
        correct=predicted_index == truth_index,
        truth_probability=float(probabilities[truth_index].item()),
        entropy_bits=_entropy_bits(probabilities),
        support_loss=support_value,
        gradient_norm=gradient_norm,
    )


def summarize(traces: Sequence[BindingTrace]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for mode in sorted({trace.mode for trace in traces}):
        values = [trace for trace in traces if trace.mode == mode]
        output[mode] = {
            "episodes": float(len(values)),
            "accuracy": sum(trace.correct for trace in values) / len(values),
            "truth_probability": sum(trace.truth_probability for trace in values) / len(values),
            "entropy_bits": sum(trace.entropy_bits for trace in values) / len(values),
            "support_loss": (
                sum(trace.support_loss for trace in values if trace.support_loss is not None)
                / sum(trace.support_loss is not None for trace in values)
                if any(trace.support_loss is not None for trace in values)
                else float("nan")
            ),
            "gradient_norm": (
                sum(trace.gradient_norm for trace in values if trace.gradient_norm is not None)
                / sum(trace.gradient_norm is not None for trace in values)
                if any(trace.gradient_norm is not None for trace in values)
                else float("nan")
            ),
        }
    return output


def evaluate(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    traces: list[BindingTrace] = []
    try:
        for group_offset in range(config.evaluation_groups):
            group_seed = config.validation_seed_base + group_offset
            for variant_index in range(config.validation_mapping_variants):
                for action in Action:
                    episode = make_single_episode(
                        group_seed,
                        variant_index,
                        config.validation_mapping_variants,
                        action,
                    )
                    for mode in ("intact", "no_adaptation", "shuffled_outcome"):
                        traces.append(
                            evaluate_episode(
                                model,
                                tokenizer,
                                initial_prefix,
                                episode,
                                config,
                                device,
                                mode=mode,
                            )
                        )
    finally:
        model.train(was_training)
    return {
        "summary": summarize(traces),
        "traces": [asdict(trace) for trace in traces],
    }


def train(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    initial_prefix = initialize_prefix(
        model,
        prefix_length=config.prefix_length,
        std=config.prefix_initialization_std,
        seed=config.seed,
        device=device,
    )
    model_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable = [*model_trainable, initial_prefix]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.outer_learning_rate,
        weight_decay=config.weight_decay,
    )

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        return max(
            0.05,
            (config.max_outer_steps - step)
            / max(config.max_outer_steps - config.warmup_steps, 1),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_evaluation = evaluate(
        model,
        tokenizer,
        initial_prefix.detach(),
        config,
        device,
    )
    rng = random.Random(config.seed ^ 0x51A61E)
    losses: list[float] = []
    query_losses: list[float] = []
    prior_losses: list[float] = []
    support_losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.time()

    for outer_step in range(1, config.max_outer_steps + 1):
        # eval() disables dropout while preserving autograd. Counterfactual
        # differences must come from evidence, not independent dropout masks.
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        group_seed = config.train_seed_base + rng.randrange(config.train_groups)
        variant_index = rng.randrange(config.train_mapping_variants)
        action = rng.choice(tuple(Action))
        episode = make_single_episode(
            group_seed,
            variant_index,
            config.train_mapping_variants,
            action,
        )
        loss, diagnostics = episode_meta_loss(
            model,
            tokenizer,
            initial_prefix,
            episode,
            config,
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().item()))
        query_losses.append(diagnostics["query_loss"])
        prior_losses.append(diagnostics["prior_loss"])
        support_losses.append(diagnostics["support_loss"])
        gradient_norms.append(diagnostics["gradient_norm"])
        if outer_step == 1 or outer_step % 10 == 0:
            print(
                json.dumps(
                    {
                        "outer_step": outer_step,
                        "max_outer_steps": config.max_outer_steps,
                        "loss": sum(losses[-10:]) / min(10, len(losses)),
                        "query_loss": sum(query_losses[-10:]) / min(10, len(query_losses)),
                        "prior_loss": sum(prior_losses[-10:]) / min(10, len(prior_losses)),
                        "support_loss": sum(support_losses[-10:]) / min(10, len(support_losses)),
                        "gradient_norm": sum(gradient_norms[-10:]) / min(10, len(gradient_norms)),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "device": str(device),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    final_evaluation = evaluate(
        model,
        tokenizer,
        initial_prefix.detach(),
        config,
        device,
    )
    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)
        torch.save(
            {
                "version": 1,
                "prefix": initial_prefix.detach().cpu(),
                "config": asdict(config),
            },
            output_dir / "initial_soft_prefix.pt",
        )

    summary = {
        "status": "completed",
        "scope": (
            "Minimal one-observation gradient-memory binding gate for one GPT-2; "
            "not a full action permutation and not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "trainable_model_parameters": sum(parameter.numel() for parameter in model_trainable),
        "soft_prefix_parameters": initial_prefix.numel(),
        "total_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_total_loss": sum(losses) / len(losses),
        "mean_post_update_query_loss": sum(query_losses) / len(query_losses),
        "mean_no_evidence_loss": sum(prior_losses) / len(prior_losses),
        "mean_support_loss": sum(support_losses) / len(support_losses),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--model-revision")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--output-dir", default="outputs/meta_soft_single/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed-base", type=int, default=910000)
    parser.add_argument("--train-groups", type=int, default=128)
    parser.add_argument("--train-mapping-variants", type=int, default=8)
    parser.add_argument("--validation-seed-base", type=int, default=920000)
    parser.add_argument("--validation-groups", type=int, default=16)
    parser.add_argument("--validation-mapping-variants", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--prefix-initialization-std", type=float, default=0.01)
    parser.add_argument("--inner-learning-rate", type=float, default=0.2)
    parser.add_argument("--outer-learning-rate", type=float, default=8e-5)
    parser.add_argument("--no-evidence-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-outer-steps", type=int, default=160)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=11)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--evaluation-groups", type=int, default=8)
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        Config(
            model_name=args.model_name,
            model_revision=args.model_revision,
            initialization=args.initialization,
            output_dir=args.output_dir,
            seed=args.seed,
            train_seed_base=args.train_seed_base,
            train_groups=args.train_groups,
            train_mapping_variants=args.train_mapping_variants,
            validation_seed_base=args.validation_seed_base,
            validation_groups=args.validation_groups,
            validation_mapping_variants=args.validation_mapping_variants,
            prefix_length=args.prefix_length,
            prefix_initialization_std=args.prefix_initialization_std,
            inner_learning_rate=args.inner_learning_rate,
            outer_learning_rate=args.outer_learning_rate,
            no_evidence_weight=args.no_evidence_weight,
            weight_decay=args.weight_decay,
            max_outer_steps=args.max_outer_steps,
            warmup_steps=args.warmup_steps,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            train_embeddings=args.train_embeddings,
            evaluation_groups=args.evaluation_groups,
            save_model=not args.no_save_model,
        )
    )


if __name__ == "__main__":
    main()
