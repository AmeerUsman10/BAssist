"""Meta-train per-game soft memory inside one GPT-2 checkpoint.

This experiment is intentionally stricter than textual recurrent memory. The
mapping query contains no action-outcome history. Hidden action semantics can
reach the answer only through a small soft prefix updated online from exact
transition-prediction loss.

One GPT-2 checkpoint supplies both the transition likelihood used for adaptation
and the candidate direction scores used for inference. There is no auxiliary
encoder, learned head, teacher model, or independently trained ensemble.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .build_epistemic_counterfactual_dataset import mapping_variants
from .build_epistemic_dataset import allowed_directions, probe_order
from .natural_protocol import answer_text, direction_words, grid_text, transition_text
from .phase0_hidden_action import Action, Direction, HiddenActionGame, StepRecord, generate_game


_ACTIONS = tuple(Action)
_DIRECTIONS = tuple(Direction)
_WORDS = direction_words()
_DIRECTION_WORD = {
    Direction.UP: "north",
    Direction.DOWN: "south",
    Direction.LEFT: "west",
    Direction.RIGHT: "east",
}
_WORD_DIRECTION = {word: direction for direction, word in _DIRECTION_WORD.items()}


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    output_dir: str = "outputs/meta_soft_binding/pretrained"
    seed: int = 42
    train_seed_base: int = 424_242
    train_groups: int = 128
    train_mapping_variants: int = 6
    validation_seed_base: int = 525_252
    validation_groups: int = 16
    validation_mapping_variants: int = 6
    max_length: int = 384
    prefix_length: int = 16
    prefix_initialization_std: float = 0.01
    inner_learning_rate: float = 0.08
    outer_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    episodes_per_outer_step: int = 1
    max_outer_steps: int = 80
    warmup_steps: int = 8
    freeze_first_n_blocks: int = 8
    train_embeddings: bool = False
    query_after_every_probe: bool = True
    evaluation_groups: int = 8
    save_model: bool = True


@dataclass(frozen=True)
class Episode:
    group_seed: int
    variant_index: int
    spec: Any
    records: tuple[StepRecord, ...]


@dataclass(frozen=True)
class EvaluationTrace:
    group_seed: int
    variant_index: int
    mode: str
    probe_count: int
    exact_mapping: bool
    consistent_mass: float
    truth_probability: float
    truth_rank: int
    entropy_bits: float
    per_action_accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def build_model_and_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Soft-memory meta-training differentiates through a gradient update.
    # PyTorch's efficient SDPA backward does not implement the derivative of
    # its own backward pass, so second-order runs must use eager attention.
    if config.initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            attn_implementation="eager",
        )
    elif config.initialization == "random":
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError("initialization must be pretrained or random")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("soft-memory meta-training requires eager attention")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not 0 <= config.freeze_first_n_blocks <= len(model.transformer.h):
        raise ValueError("freeze_first_n_blocks lies outside model depth")
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


def make_episode(group_seed: int, variant_index: int, variant_count: int) -> Episode:
    base = generate_game(group_seed)
    variants = mapping_variants(group_seed, variant_count)
    if not 0 <= variant_index < len(variants):
        raise ValueError("variant_index is out of range")
    mapping = {
        action: direction
        for action, direction in zip(_ACTIONS, variants[variant_index], strict=True)
    }
    spec = replace(base, action_to_direction=mapping)
    game = HiddenActionGame(spec)
    records: list[StepRecord] = []
    for action in probe_order(group_seed):
        record = game.step(action)
        if record.status != "ACTIVE":
            raise RuntimeError("the safe action-binding probe terminated")
        records.append(record)
    return Episode(group_seed, variant_index, spec, tuple(records))


def transition_prompt(record: StepRecord) -> str:
    return "\n\n".join(
        (
            "You are learning the hidden dynamics of one deterministic grid game.",
            "Predict the exact observable result of the stated intervention.",
            "BEFORE ACTION\n" + grid_text(record.before),
            f"INTERVENTION\nApply {record.action.value}.",
            "EXACT RESULT:",
        )
    )


def transition_target(record: StepRecord) -> str:
    return "\n" + transition_text(record)


def mapping_query(action: Action) -> str:
    # No history or current game grid appears here. The temporary prefix is the
    # only channel through which this episode's observations can affect output.
    return "\n\n".join(
        (
            "A temporary memory contains observations from one deterministic grid game.",
            f"In that game, action {action.value} moves the controlled cell",
            "ANSWER:",
        )
    )


def encode_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError("text tokenized to an empty sequence")
    return ids


def truncate_pair(
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    max_length: int,
    prefix_length: int,
    keep_prompt_prefix: int = 64,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    token_budget = max_length - prefix_length
    if len(target_ids) >= token_budget:
        raise ValueError("target is too long for the configured context")
    prompt_budget = token_budget - len(target_ids)
    prompt = list(prompt_ids)
    if len(prompt) > prompt_budget:
        keep_prefix = min(keep_prompt_prefix, prompt_budget // 2)
        prompt = prompt[:keep_prefix] + prompt[-(prompt_budget - keep_prefix) :]
    return tuple(prompt), tuple(target_ids)


def _forward_logits(
    model: Any,
    prefix: torch.Tensor,
    input_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix.ndim != 2:
        raise ValueError("prefix must have shape [prefix_length, hidden_size]")
    longest = max(len(row) for row in input_rows)
    ids = torch.tensor(
        [list(row) + [pad_token_id] * (longest - len(row)) for row in input_rows],
        dtype=torch.long,
        device=device,
    )
    masks = torch.tensor(
        [[1] * len(row) + [0] * (longest - len(row)) for row in input_rows],
        dtype=torch.long,
        device=device,
    )
    token_embeddings = model.get_input_embeddings()(ids)
    prefix_batch = prefix.unsqueeze(0).expand(len(input_rows), -1, -1)
    inputs_embeds = torch.cat((prefix_batch, token_embeddings), dim=1)
    prefix_mask = torch.ones(
        (len(input_rows), prefix.shape[0]), dtype=torch.long, device=device
    )
    attention_mask = torch.cat((prefix_mask, masks), dim=1)
    logits = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    return logits, masks


def sequence_log_likelihood(
    model: Any,
    prefix: torch.Tensor,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    pad_token_id: int,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    prompt, target = truncate_pair(
        prompt_ids,
        target_ids,
        max_length=max_length,
        prefix_length=int(prefix.shape[0]),
    )
    row = (*prompt, *target)
    logits, _ = _forward_logits(
        model,
        prefix,
        [row],
        pad_token_id=pad_token_id,
        device=device,
    )
    start = int(prefix.shape[0]) + len(prompt) - 1
    target_logits = logits[0, start : start + len(target)]
    target_tensor = torch.tensor(target, dtype=torch.long, device=device)
    return F.log_softmax(target_logits, dim=-1).gather(
        1, target_tensor.unsqueeze(1)
    )[:, 0].sum()


def candidate_direction_scores(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    action: Action,
    *,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    prompt_full = encode_text(tokenizer, mapping_query(action))
    targets = tuple(encode_text(tokenizer, answer_text(word)) for word in _WORDS)
    pairs = [
        truncate_pair(
            prompt_full,
            target,
            max_length=max_length,
            prefix_length=int(prefix.shape[0]),
        )
        for target in targets
    ]
    rows = [(*prompt, *target) for prompt, target in pairs]
    logits, _ = _forward_logits(
        model,
        prefix,
        rows,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    scores: list[torch.Tensor] = []
    for row_index, (prompt, target) in enumerate(pairs):
        start = int(prefix.shape[0]) + len(prompt) - 1
        target_logits = logits[row_index, start : start + len(target)]
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        scores.append(
            F.log_softmax(target_logits, dim=-1)
            .gather(1, target_tensor.unsqueeze(1))[:, 0]
            .sum()
        )
    return torch.stack(scores)


def set_target(spec: Any, records: Sequence[StepRecord], action: Action, device) -> torch.Tensor:
    allowed = allowed_directions(spec, records, action)
    probability = 1.0 / len(allowed)
    return torch.tensor(
        [probability if _WORD_DIRECTION[word] in allowed else 0.0 for word in _WORDS],
        dtype=torch.float32,
        device=device,
    )


def set_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(scores, dim=-1)).sum()


def initialize_prefix(
    model: Any,
    *,
    prefix_length: int,
    std: float,
    seed: int,
    device: torch.device,
) -> nn.Parameter:
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        embedding = model.get_input_embeddings().weight.detach()
        center = embedding.mean(dim=0)
        prefix = center.unsqueeze(0).repeat(prefix_length, 1)
        if std > 0.0:
            prefix.add_(
                std
                * torch.randn(
                    prefix.shape,
                    generator=generator,
                    device=device,
                    dtype=prefix.dtype,
                )
            )
    return nn.Parameter(prefix)


def adapt_once(
    model: Any,
    tokenizer: Any,
    fast_prefix: torch.Tensor,
    prompt_record: StepRecord,
    target_record: StepRecord,
    *,
    inner_learning_rate: float,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = encode_text(tokenizer, transition_prompt(prompt_record))
    target_ids = encode_text(tokenizer, transition_target(target_record))
    loss = -sequence_log_likelihood(
        model,
        fast_prefix,
        prompt_ids,
        target_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        max_length=max_length,
    )
    gradient = torch.autograd.grad(
        loss,
        fast_prefix,
        create_graph=False,
        retain_graph=False,
    )[0]
    # First-order meta-learning: the query keeps an identity gradient to the
    # learned initial prefix while the expensive second derivative through the
    # support update is intentionally omitted.
    updated = fast_prefix - inner_learning_rate * gradient.detach()
    return updated, loss.detach()


def episode_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: Episode,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    fast_prefix = initial_prefix
    observed: list[StepRecord] = []
    query_losses: list[torch.Tensor] = []
    support_losses: list[float] = []

    # The no-evidence posterior is part of the curriculum. It must remain broad.
    if config.query_after_every_probe:
        for action in _ACTIONS:
            scores = candidate_direction_scores(
                model,
                tokenizer,
                fast_prefix,
                action,
                device=device,
                max_length=config.max_length,
            )
            query_losses.append(
                set_loss(scores, set_target(episode.spec, observed, action, device))
            )

    for record in episode.records:
        fast_prefix, support_loss = adapt_once(
            model,
            tokenizer,
            fast_prefix,
            record,
            record,
            inner_learning_rate=config.inner_learning_rate,
            device=device,
            max_length=config.max_length,
        )
        support_losses.append(float(support_loss.item()))
        observed.append(record)
        if config.query_after_every_probe or len(observed) == len(episode.records):
            for action in _ACTIONS:
                scores = candidate_direction_scores(
                    model,
                    tokenizer,
                    fast_prefix,
                    action,
                    device=device,
                    max_length=config.max_length,
                )
                query_losses.append(
                    set_loss(scores, set_target(episode.spec, observed, action, device))
                )

    return torch.stack(query_losses).mean(), {
        "support_nll": sum(support_losses) / len(support_losses),
        "query_count": float(len(query_losses)),
    }


def _joint_posterior(score_matrix: torch.Tensor):
    permutations = tuple(itertools.permutations(_DIRECTIONS))
    joint_scores = torch.stack(
        [
            sum(
                score_matrix[action_index, _DIRECTIONS.index(direction)]
                for action_index, direction in enumerate(permutation)
            )
            for permutation in permutations
        ]
    )
    return permutations, torch.softmax(joint_scores, dim=0)


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def evaluate_episode(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    episode: Episode,
    config: Config,
    device: torch.device,
    *,
    mode: str,
) -> list[EvaluationTrace]:
    if mode not in {"intact", "no_adaptation", "shuffled_outcome"}:
        raise ValueError("unknown soft-binding evaluation mode")
    fast_prefix = initial_prefix.detach().clone().requires_grad_(True)
    observed: list[StepRecord] = []
    traces: list[EvaluationTrace] = []
    target_records = (
        episode.records[1:] + episode.records[:1]
        if mode == "shuffled_outcome"
        else episode.records
    )

    def record_trace() -> None:
        score_matrix = torch.stack(
            [
                candidate_direction_scores(
                    model,
                    tokenizer,
                    fast_prefix,
                    action,
                    device=device,
                    max_length=config.max_length,
                )
                for action in _ACTIONS
            ]
        )
        direction_probabilities = torch.softmax(score_matrix, dim=-1)
        permutations, joint = _joint_posterior(score_matrix)
        truth = tuple(episode.spec.action_to_direction[action] for action in _ACTIONS)
        truth_index = permutations.index(truth)
        order = torch.argsort(joint, descending=True).tolist()
        map_index = int(torch.argmax(joint).item())
        observed_mapping = {
            record.action: episode.spec.action_to_direction[record.action]
            for record in observed
        }
        consistent_indices = [
            index
            for index, permutation in enumerate(permutations)
            if all(
                permutation[_ACTIONS.index(action)] == direction
                for action, direction in observed_mapping.items()
            )
        ]
        truth_direction_indices = torch.tensor(
            [_DIRECTIONS.index(direction) for direction in truth],
            device=device,
        )
        traces.append(
            EvaluationTrace(
                group_seed=episode.group_seed,
                variant_index=episode.variant_index,
                mode=mode,
                probe_count=len(observed),
                exact_mapping=permutations[map_index] == truth,
                consistent_mass=float(joint[consistent_indices].sum().item()),
                truth_probability=float(joint[truth_index].item()),
                truth_rank=order.index(truth_index) + 1,
                entropy_bits=_entropy_bits(joint),
                per_action_accuracy=float(
                    (
                        torch.argmax(direction_probabilities, dim=-1)
                        == truth_direction_indices
                    )
                    .float()
                    .mean()
                    .item()
                ),
            )
        )

    model.eval()
    with torch.enable_grad():
        record_trace()
        for prompt_record, target_record in zip(
            episode.records, target_records, strict=True
        ):
            if mode != "no_adaptation":
                fast_prefix, _ = adapt_once(
                    model,
                    tokenizer,
                    fast_prefix,
                    prompt_record,
                    target_record,
                    inner_learning_rate=config.inner_learning_rate,
                    device=device,
                    max_length=config.max_length,
                )
                fast_prefix = fast_prefix.detach().requires_grad_(True)
            observed.append(prompt_record)
            record_trace()
    return traces


def summarize_traces(traces: Sequence[EvaluationTrace]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[EvaluationTrace]] = {}
    for trace in traces:
        grouped.setdefault((trace.mode, trace.probe_count), []).append(trace)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for (mode, probe_count), values in sorted(grouped.items()):
        count = len(values)
        output.setdefault(mode, {})[str(probe_count)] = {
            "episodes": float(count),
            "exact_mapping": sum(value.exact_mapping for value in values) / count,
            "consistent_mass": sum(value.consistent_mass for value in values) / count,
            "truth_probability": sum(value.truth_probability for value in values) / count,
            "truth_rank": sum(value.truth_rank for value in values) / count,
            "entropy_bits": sum(value.entropy_bits for value in values) / count,
            "per_action_accuracy": sum(value.per_action_accuracy for value in values) / count,
        }
    return output


def evaluate(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    traces: list[EvaluationTrace] = []
    for group_offset in range(config.evaluation_groups):
        group_seed = config.validation_seed_base + group_offset
        variants = min(config.validation_mapping_variants, 24)
        for variant_index in range(variants):
            episode = make_episode(group_seed, variant_index, variants)
            for mode in ("intact", "no_adaptation", "shuffled_outcome"):
                traces.extend(
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
    return {
        "summary": summarize_traces(traces),
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
    trainable_model_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable = [*trainable_model_parameters, initial_prefix]
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
        model, tokenizer, initial_prefix.detach(), config, device
    )
    losses: list[float] = []
    support_losses: list[float] = []
    started = time.time()
    rng = random.Random(config.seed ^ 0x5A17)

    for outer_step in range(1, config.max_outer_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        episode_losses: list[torch.Tensor] = []
        for _ in range(config.episodes_per_outer_step):
            group_seed = config.train_seed_base + rng.randrange(config.train_groups)
            variant_index = rng.randrange(config.train_mapping_variants)
            episode = make_episode(
                group_seed,
                variant_index,
                config.train_mapping_variants,
            )
            episode_loss, diagnostics = episode_meta_loss(
                model,
                tokenizer,
                initial_prefix,
                episode,
                config,
                device,
            )
            episode_losses.append(episode_loss)
            support_losses.append(diagnostics["support_nll"])
        loss = torch.stack(episode_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().item()))
        if outer_step == 1 or outer_step % 10 == 0:
            print(
                json.dumps(
                    {
                        "outer_step": outer_step,
                        "max_outer_steps": config.max_outer_steps,
                        "meta_query_loss": sum(losses[-10:]) / min(10, len(losses)),
                        "support_nll": sum(support_losses[-10:]) / min(10, len(support_losses)),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "device": str(device),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    final_evaluation = evaluate(
        model, tokenizer, initial_prefix.detach(), config, device
    )
    if config.save_model:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
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
            "First-order meta-training of a per-game soft prefix for hidden "
            "action semantics; controlled gate, not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in trainable_model_parameters
        ),
        "soft_prefix_parameters": initial_prefix.numel(),
        "total_model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "mean_meta_query_loss": sum(losses) / len(losses),
        "mean_support_nll": sum(support_losses) / len(support_losses),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--output-dir", default="outputs/meta_soft_binding/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed-base", type=int, default=424242)
    parser.add_argument("--train-groups", type=int, default=128)
    parser.add_argument("--train-mapping-variants", type=int, default=6)
    parser.add_argument("--validation-seed-base", type=int, default=525252)
    parser.add_argument("--validation-groups", type=int, default=16)
    parser.add_argument("--validation-mapping-variants", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--prefix-initialization-std", type=float, default=0.01)
    parser.add_argument("--inner-learning-rate", type=float, default=0.08)
    parser.add_argument("--outer-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--episodes-per-outer-step", type=int, default=1)
    parser.add_argument("--max-outer-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--no-query-after-every-probe", action="store_true")
    parser.add_argument("--evaluation-groups", type=int, default=8)
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        Config(
            model_name=args.model_name,
            initialization=args.initialization,
            output_dir=args.output_dir,
            seed=args.seed,
            train_seed_base=args.train_seed_base,
            train_groups=args.train_groups,
            train_mapping_variants=args.train_mapping_variants,
            validation_seed_base=args.validation_seed_base,
            validation_groups=args.validation_groups,
            validation_mapping_variants=args.validation_mapping_variants,
            max_length=args.max_length,
            prefix_length=args.prefix_length,
            prefix_initialization_std=args.prefix_initialization_std,
            inner_learning_rate=args.inner_learning_rate,
            outer_learning_rate=args.outer_learning_rate,
            weight_decay=args.weight_decay,
            episodes_per_outer_step=args.episodes_per_outer_step,
            max_outer_steps=args.max_outer_steps,
            warmup_steps=args.warmup_steps,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            train_embeddings=args.train_embeddings,
            query_after_every_probe=not args.no_query_after_every_probe,
            evaluation_groups=args.evaluation_groups,
            save_model=not args.no_save_model,
        )
    )


if __name__ == "__main__":
    main()
