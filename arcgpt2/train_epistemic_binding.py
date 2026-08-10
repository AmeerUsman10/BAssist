"""Train GPT-2 to preserve and contract an exact action-mapping version space.

The model scores ordinary direction-word completions. Training uses a set-valued
cross entropy: probability is spread uniformly over every direction still
consistent with the partial interaction history. No arbitrary action meaning is
used as a target before evidence identifies it.

The same GPT-2 scores all four variables. Deterministic code only combines those
scores under the declared one-to-one permutation constraint and computes
calibration/evidence controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .natural_protocol import answer_text, direction_words
from .phase0_hidden_action import Action, Direction


_WORD_DIRECTION = {
    "north": Direction.UP,
    "south": Direction.DOWN,
    "west": Direction.LEFT,
    "east": Direction.RIGHT,
}
_DIRECTION_WORD = {value: key for key, value in _WORD_DIRECTION.items()}
_ACTIONS = tuple(Action)
_DIRECTIONS = tuple(Direction)
_WORDS = direction_words()


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/epistemic/data"
    output_dir: str = "outputs/epistemic/pretrained"
    seed: int = 42
    max_length: int = 448
    prefix_keep: int = 96
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 80
    warmup_steps: int = 8
    freeze_first_n_blocks: int = 8
    train_embeddings: bool = False
    evaluation_games: int = 16
    score_batch_size: int = 16
    save_model: bool = True


@dataclass(frozen=True)
class EncodedItem:
    prompt_ids: tuple[int, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    target_probabilities: tuple[float, ...]


class EpistemicDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        prefix_keep: int,
    ) -> None:
        self.items = [
            encode_row(
                row,
                tokenizer,
                context_field="context",
                max_length=max_length,
                prefix_keep=prefix_keep,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedItem:
        return self.items[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "game_seed",
                "probe_count",
                "query_action",
                "context",
                "amnesic_context",
                "shuffled_context",
                "candidate_words",
                "target_distribution",
                "truth_mapping",
                "observed_mapping",
            }
            if not isinstance(row, dict) or not required.issubset(row):
                missing = sorted(required.difference(row if isinstance(row, dict) else {}))
                raise ValueError(f"invalid row at {path}:{line_number}; missing {missing}")
            if tuple(row["candidate_words"]) != _WORDS:
                raise ValueError(f"candidate order drift at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def truncate_prompt(
    ids: Sequence[int],
    *,
    target_length: int,
    max_length: int,
    prefix_keep: int,
) -> list[int]:
    budget = max_length - target_length
    if budget < 32:
        raise ValueError("candidate leaves insufficient prompt budget")
    values = list(ids)
    if len(values) <= budget:
        return values
    keep_prefix = min(prefix_keep, budget // 2, len(values))
    keep_suffix = budget - keep_prefix
    return values[:keep_prefix] + values[-keep_suffix:]


def encode_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    context_field: str,
    max_length: int,
    prefix_keep: int,
) -> EncodedItem:
    prompt_ids_full = tokenizer.encode(str(row[context_field]), add_special_tokens=False)
    candidate_ids = tuple(
        tuple(tokenizer.encode(answer_text(word), add_special_tokens=False))
        for word in _WORDS
    )
    if any(not ids for ids in candidate_ids):
        raise ValueError("a direction completion tokenized to an empty sequence")
    longest_target = max(len(ids) for ids in candidate_ids)
    prompt_ids = truncate_prompt(
        prompt_ids_full,
        target_length=longest_target,
        max_length=max_length,
        prefix_keep=prefix_keep,
    )
    target_distribution = row["target_distribution"]
    probabilities = tuple(float(target_distribution[word]) for word in _WORDS)
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("target probabilities must be non-negative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("target probabilities must sum to one")
    return EncodedItem(tuple(prompt_ids), candidate_ids, probabilities)


def collate_items(batch: Sequence[EncodedItem]) -> list[EncodedItem]:
    return list(batch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def build_model_and_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
    elif config.initialization == "random":
        model_config = AutoConfig.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError("initialization must be pretrained or random")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not 0 <= config.freeze_first_n_blocks <= len(model.transformer.h):
        raise ValueError("freeze_first_n_blocks is outside the model depth")
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


def score_encoded_items(
    model: Any,
    items: Sequence[EncodedItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable candidate scores and target distributions."""

    if not items:
        raise ValueError("score_encoded_items requires at least one item")
    flattened: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for item in items:
        if len(item.candidate_ids) != len(_WORDS):
            raise ValueError("every item must contain four direction candidates")
        flattened.extend((item.prompt_ids, target) for target in item.candidate_ids)

    lengths = [len(prompt) + len(target) for prompt, target in flattened]
    longest = max(lengths)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for (prompt, target), length in zip(flattened, lengths, strict=True):
        padding = longest - length
        input_rows.append([*prompt, *target, *([pad_token_id] * padding)])
        mask_rows.append([1] * length + [0] * padding)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    scores: list[torch.Tensor] = []
    for row_index, (prompt, target) in enumerate(flattened):
        start_logit = len(prompt) - 1
        target_logits = logits[row_index, start_logit : start_logit + len(target)]
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        token_log_probabilities = F.log_softmax(target_logits, dim=-1).gather(
            1, target_tensor.unsqueeze(1)
        )[:, 0]
        scores.append(token_log_probabilities.sum())

    score_tensor = torch.stack(scores).reshape(len(items), len(_WORDS))
    target_tensor = torch.tensor(
        [item.target_probabilities for item in items],
        dtype=score_tensor.dtype,
        device=device,
    )
    return score_tensor, target_tensor


def epistemic_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have identical shapes")
    return -(targets * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


def evaluate_loader(
    model: Any,
    loader: DataLoader,
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    briers: list[float] = []
    with torch.no_grad():
        for items in loader:
            scores, targets = score_encoded_items(
                model,
                items,
                pad_token_id=pad_token_id,
                device=device,
            )
            losses.append(float(epistemic_loss(scores, targets).item()))
            probabilities = torch.softmax(scores, dim=-1)
            briers.extend(
                float(value)
                for value in ((probabilities - targets) ** 2).sum(dim=-1).cpu()
            )
    return {
        "set_cross_entropy": sum(losses) / len(losses),
        "mean_brier": sum(briers) / len(briers),
        "examples": float(len(loader.dataset)),
    }


def _group_test_rows(rows: Sequence[dict[str, Any]]):
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["game_seed"]), int(row["probe_count"]))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: _ACTIONS.index(Action(str(row["query_action"]))))
        if len(group) != len(_ACTIONS):
            raise ValueError(f"group {key} does not contain four action queries")
    return groups


def _entropy(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def _mapping_permutations() -> tuple[tuple[Direction, ...], ...]:
    return tuple(itertools.permutations(_DIRECTIONS))


def _joint_posterior(score_matrix: torch.Tensor):
    permutations = _mapping_permutations()
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


def evaluate_version_space(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: Config,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, Any]:
    context_field = {
        "full": "context",
        "amnesic": "amnesic_context",
        "shuffled": "shuffled_context",
    }.get(mode)
    if context_field is None:
        raise ValueError("mode must be full, amnesic, or shuffled")

    all_groups = _group_test_rows(rows)
    selected_seeds = sorted({seed for seed, _ in all_groups})[: config.evaluation_games]
    groups = {
        key: value for key, value in all_groups.items() if key[0] in selected_seeds
    }
    aggregate: dict[int, dict[str, float]] = {
        probe: {
            "groups": 0.0,
            "consistent_mass": 0.0,
            "truth_probability": 0.0,
            "truth_rank": 0.0,
            "joint_entropy_bits": 0.0,
            "map_exact": 0.0,
            "map_consistent": 0.0,
            "per_action_accuracy": 0.0,
            "set_cross_entropy": 0.0,
            "brier": 0.0,
        }
        for probe in range(5)
    }
    traces: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for (seed, probe_count), group in sorted(groups.items()):
            items = [
                encode_row(
                    row,
                    tokenizer,
                    context_field=context_field,
                    max_length=config.max_length,
                    prefix_keep=config.prefix_keep,
                )
                for row in group
            ]
            scores, targets = score_encoded_items(
                model,
                items,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            direction_probabilities = torch.softmax(scores, dim=-1)
            permutations, joint = _joint_posterior(scores)

            truth_words = group[0]["truth_mapping"]
            truth = tuple(_WORD_DIRECTION[str(truth_words[action.value])] for action in _ACTIONS)
            truth_index = permutations.index(truth)
            order = torch.argsort(joint, descending=True).tolist()
            truth_rank = order.index(truth_index) + 1
            map_index = int(torch.argmax(joint).item())
            map_mapping = permutations[map_index]

            observed_raw = group[0]["observed_mapping"]
            observed = {
                Action(action): Direction(direction)
                for action, direction in observed_raw.items()
            }
            consistent_indices = [
                index
                for index, permutation in enumerate(permutations)
                if all(
                    permutation[_ACTIONS.index(action)] == direction
                    for action, direction in observed.items()
                )
            ]
            consistent_mass = float(joint[consistent_indices].sum().item())
            map_consistent = map_index in consistent_indices
            map_exact = map_mapping == truth

            predicted_directions = torch.argmax(direction_probabilities, dim=-1)
            truth_indices = torch.tensor(
                [_DIRECTIONS.index(truth[index]) for index in range(len(_ACTIONS))],
                device=device,
            )
            per_action_accuracy = float(
                (predicted_directions == truth_indices).float().mean().item()
            )
            set_cross_entropy = float(
                (-(targets * torch.log(direction_probabilities.clamp_min(1e-12))).sum(dim=-1))
                .mean()
                .item()
            )
            brier = float(((direction_probabilities - targets) ** 2).sum(dim=-1).mean().item())

            values = aggregate[probe_count]
            values["groups"] += 1.0
            values["consistent_mass"] += consistent_mass
            values["truth_probability"] += float(joint[truth_index].item())
            values["truth_rank"] += float(truth_rank)
            values["joint_entropy_bits"] += _entropy(joint)
            values["map_exact"] += float(map_exact)
            values["map_consistent"] += float(map_consistent)
            values["per_action_accuracy"] += per_action_accuracy
            values["set_cross_entropy"] += set_cross_entropy
            values["brier"] += brier

            traces.append(
                {
                    "game_seed": seed,
                    "probe_count": probe_count,
                    "consistent_mapping_count": len(consistent_indices),
                    "consistent_mass": consistent_mass,
                    "truth_probability": float(joint[truth_index].item()),
                    "truth_rank": truth_rank,
                    "joint_entropy_bits": _entropy(joint),
                    "map_exact": map_exact,
                    "map_consistent": map_consistent,
                    "per_action_accuracy": per_action_accuracy,
                    "set_cross_entropy": set_cross_entropy,
                    "brier": brier,
                }
            )

    by_probe: dict[str, dict[str, float]] = {}
    for probe, values in aggregate.items():
        count = values["groups"]
        if count == 0:
            continue
        by_probe[str(probe)] = {
            key: value / count
            for key, value in values.items()
            if key != "groups"
        } | {"groups": count}

    return {
        "mode": mode,
        "games": len(selected_seeds),
        "groups": len(groups),
        "by_probe_count": by_probe,
        "traces": traces,
    }


def train(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    test_rows = load_jsonl(data_dir / "test.jsonl")
    model, tokenizer = build_model_and_tokenizer(config)

    train_dataset = EpistemicDataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    validation_dataset = EpistemicDataset(
        validation_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_items,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_items,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        remaining = config.max_optimizer_steps - step
        span = max(config.max_optimizer_steps - config.warmup_steps, 1)
        return max(0.05, remaining / span)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_validation = evaluate_loader(
        model,
        validation_loader,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    initial_full = evaluate_version_space(
        model,
        tokenizer,
        test_rows,
        config,
        device,
        mode="full",
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()
    model.train()

    while optimizer_step < config.max_optimizer_steps:
        for items in train_loader:
            scores, targets = score_encoded_items(
                model,
                items,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            loss = epistemic_loss(scores, targets) / config.gradient_accumulation_steps
            loss.backward()
            losses.append(float(loss.item() * config.gradient_accumulation_steps))
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step == 1 or optimizer_step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "step": optimizer_step,
                                "max_steps": config.max_optimizer_steps,
                                "loss": sum(losses[-10:]) / min(10, len(losses)),
                                "learning_rate": scheduler.get_last_lr()[0],
                                "device": str(device),
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
                if optimizer_step >= config.max_optimizer_steps:
                    break

    final_validation = evaluate_loader(
        model,
        validation_loader,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    controls = {
        mode: evaluate_version_space(
            model,
            tokenizer,
            test_rows,
            config,
            device,
            mode=mode,
        )
        for mode in ("full", "amnesic", "shuffled")
    }

    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    summary: dict[str, Any] = {
        "status": "completed",
        "scope": (
            "Symmetry-aware partial-evidence action binding; controlled "
            "meta-learning gate, not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_rows),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "mean_training_loss": sum(losses) / len(losses),
        "initial_full": initial_full,
        "controls": controls,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--data-dir", default="outputs/epistemic/data")
    parser.add_argument("--output-dir", default="outputs/epistemic/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--prefix-keep", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--evaluation-games", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        model_name=args.model_name,
        initialization=args.initialization,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_length=args.max_length,
        prefix_keep=args.prefix_keep,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_optimizer_steps=args.max_optimizer_steps,
        warmup_steps=args.warmup_steps,
        freeze_first_n_blocks=args.freeze_first_n_blocks,
        train_embeddings=args.train_embeddings,
        evaluation_games=args.evaluation_games,
        score_batch_size=args.score_batch_size,
        save_model=not args.no_save_model,
    )
    train(config)


if __name__ == "__main__":
    main()
