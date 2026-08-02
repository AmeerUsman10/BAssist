"""Train and evaluate the Stage-0.2 pure GPT-2 decomposed controller.

The acting system is one unmodified GPT-2 causal LM queried several times.  No
auxiliary network, planner, retrieval system, learned value function, adapter,
or game-specific rule code is used at inference.  Offline code may generate
labels and score behavior, but it is not consulted when GPT-2 acts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .phase0_hidden_action import Action, HiddenActionGame
from .stage02_decomposed import (
    ACTION_CANDIDATES,
    ACTION_DIGIT,
    ALL_LABELS,
    MAP_CANDIDATES,
    NEED_CANDIDATES,
    Stage02Context,
    compose_prompt,
    decision_for_context,
    direct_prompt,
    format_history,
    infer_mapping_labels,
    load_jsonl,
    make_stage02_spec,
    mapping_prompt,
    need_prompt,
)

LABEL_INDEX = {label: index for index, label in enumerate(ALL_LABELS)}
TASK_ORDER = ("mapping", "need", "compose", "direct")
TASK_INDEX = {task: index for index, task in enumerate(TASK_ORDER)}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    initialization: str
    train_file: str
    evaluation_file: str
    output_dir: str
    seed: int
    max_length: int
    prefix_keep: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_optimizer_steps: int
    warmup_steps: int
    freeze_first_n_blocks: int
    train_eval_seed_start: int
    train_eval_games: int
    heldout_seed_start: int
    heldout_games: int
    max_actions_per_game: int
    history_modes: tuple[str, ...]
    save_model: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def truncate_ids(ids: Sequence[int], *, budget: int, prefix_keep: int) -> list[int]:
    values = list(ids)
    if budget < 1:
        raise ValueError("token budget must be positive")
    if len(values) <= budget:
        return values
    keep_prefix = min(prefix_keep, budget // 2)
    return values[:keep_prefix] + values[-(budget - keep_prefix) :]


class Stage02Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        prefix_keep: int,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for row in rows:
            task = str(row.get("task", ""))
            target = str(row.get("target", ""))
            candidates = tuple(str(value) for value in row.get("candidates", ()))
            if task not in TASK_INDEX:
                raise ValueError(f"unknown task: {task!r}")
            if target not in LABEL_INDEX:
                raise ValueError(f"unknown target label: {target!r}")
            if not candidates or target not in candidates:
                raise ValueError("target must belong to a non-empty candidate set")
            if any(candidate not in LABEL_INDEX for candidate in candidates):
                raise ValueError(f"unknown candidate in {candidates!r}")

            ids = tokenizer.encode(str(row["prompt"]), add_special_tokens=False)
            ids = truncate_ids(ids, budget=max_length, prefix_keep=prefix_keep)
            if not ids:
                raise ValueError("prompt tokenized to an empty sequence")
            valid_mask = [False] * len(ALL_LABELS)
            for candidate in candidates:
                valid_mask[LABEL_INDEX[candidate]] = True

            metadata = row.get("metadata", {})
            phase = str(metadata.get("decision_phase", "unknown"))
            self.items.append(
                {
                    "input_ids": ids,
                    "valid_label_mask": valid_mask,
                    "target_index": LABEL_INDEX[target],
                    "task_id": TASK_INDEX[task],
                    "task": task,
                    "target": target,
                    "phase": phase,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def make_collator(pad_token_id: int):
    def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        valid_masks: list[list[bool]] = []
        target_indices: list[int] = []
        task_ids: list[int] = []
        tasks: list[str] = []
        targets: list[str] = []
        phases: list[str] = []
        for item in batch:
            values = list(item["input_ids"])
            padding = longest - len(values)
            input_ids.append(values + [pad_token_id] * padding)
            attention_masks.append([1] * len(values) + [0] * padding)
            valid_masks.append(list(item["valid_label_mask"]))
            target_indices.append(int(item["target_index"]))
            task_ids.append(int(item["task_id"]))
            tasks.append(str(item["task"]))
            targets.append(str(item["target"]))
            phases.append(str(item["phase"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "valid_label_mask": torch.tensor(valid_masks, dtype=torch.bool),
            "target_index": torch.tensor(target_indices, dtype=torch.long),
            "task_id": torch.tensor(task_ids, dtype=torch.long),
            "tasks": tasks,
            "targets": targets,
            "phases": phases,
        }

    return collate


def balanced_sampling_weights(dataset: Stage02Dataset) -> tuple[list[float], dict[str, Any]]:
    """Equal task mass, then equal target mass within each task."""
    task_target_counts = Counter(
        (str(item["task"]), str(item["target"])) for item in dataset.items
    )
    targets_by_task: dict[str, set[str]] = defaultdict(set)
    for task, target in task_target_counts:
        targets_by_task[task].add(target)
    present_tasks = sorted(targets_by_task)
    task_mass = 1.0 / len(present_tasks)

    weights: list[float] = []
    for item in dataset.items:
        task = str(item["task"])
        target = str(item["target"])
        class_mass = task_mass / len(targets_by_task[task])
        weights.append(class_mass / task_target_counts[(task, target)])

    realized = {
        task: {
            target: sum(
                weight
                for weight, item in zip(weights, dataset.items, strict=True)
                if item["task"] == task and item["target"] == target
            )
            for target in sorted(targets_by_task[task])
        }
        for task in present_tasks
    }
    return weights, {
        "task_target_counts": {
            f"{task}:{target}": count
            for (task, target), count in sorted(task_target_counts.items())
        },
        "expected_mass_by_task_target": realized,
        "total_mass": sum(weights),
    }


def build_model_and_tokenizer(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_token_ids: dict[str, int] = {}
    for label in ALL_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Stage-0.2 label must be one original GPT-2 token: {label!r} -> {ids}")
        label_token_ids[label] = ids[0]
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise ValueError("Stage-0.2 labels did not map to distinct GPT-2 tokens")

    if config.initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
    elif config.initialization == "random":
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(config.model_name)
        )
    else:
        raise ValueError("initialization must be pretrained or random")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    blocks = model.transformer.h
    if not 0 <= config.freeze_first_n_blocks <= len(blocks):
        raise ValueError(f"freeze_first_n_blocks must be in 0..{len(blocks)}")

    # Stage 0.2 intentionally keeps the original tokenizer and embedding table
    # fixed.  Upper transformer blocks must learn to route the existing N/E/S/W,
    # ?, and digit representations rather than relying on newly appended tokens.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in blocks[config.freeze_first_n_blocks :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    return model, tokenizer, label_token_ids


def final_label_logits(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    label_token_ids: Mapping[str, int],
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    final_positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(batch["input_ids"].shape[0], device=batch["input_ids"].device)
    vocabulary_logits = outputs.logits[rows, final_positions]
    candidate_ids = torch.tensor(
        [label_token_ids[label] for label in ALL_LABELS],
        dtype=torch.long,
        device=vocabulary_logits.device,
    )
    return vocabulary_logits.index_select(dim=1, index=candidate_ids)


def masked_label_loss(
    label_logits: torch.Tensor,
    valid_label_mask: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    if label_logits.shape != valid_label_mask.shape:
        raise ValueError("label logits and valid mask shapes differ")
    if not bool(valid_label_mask.any(dim=1).all()):
        raise ValueError("every row needs at least one valid label")
    target_is_valid = valid_label_mask.gather(1, target_index[:, None]).squeeze(1)
    if not bool(target_is_valid.all()):
        raise ValueError("a target lies outside its candidate set")
    masked = label_logits.masked_fill(~valid_label_mask, float("-inf"))
    return F.cross_entropy(masked, target_index)


def _normalized_entropy(counts: Mapping[str, int], labels: Sequence[str]) -> float:
    total = sum(counts.get(label, 0) for label in labels)
    if total <= 0 or len(labels) <= 1:
        return 0.0
    entropy = 0.0
    for label in labels:
        count = counts.get(label, 0)
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(labels))


def classification_metrics(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    label_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    task_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0,
            "predictions": Counter(),
            "targets": Counter(),
            "probability_entropy": [],
        }
    )
    phase_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for raw_batch in loader:
            tensor_batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if isinstance(value, torch.Tensor)
            }
            logits = final_label_logits(model, tensor_batch, label_token_ids)
            loss = masked_label_loss(
                logits,
                tensor_batch["valid_label_mask"],
                tensor_batch["target_index"],
            )
            batch_size = logits.shape[0]
            total_loss += float(loss.item()) * batch_size
            total += batch_size

            masked = logits.masked_fill(
                ~tensor_batch["valid_label_mask"], float("-inf")
            )
            predictions = torch.argmax(masked, dim=1)
            probabilities = F.softmax(masked, dim=1)
            for row in range(batch_size):
                target_index = int(tensor_batch["target_index"][row].item())
                prediction_index = int(predictions[row].item())
                is_correct = target_index == prediction_index
                correct += int(is_correct)
                task = str(raw_batch["tasks"][row])
                target = ALL_LABELS[target_index]
                prediction = ALL_LABELS[prediction_index]
                entry = task_stats[task]
                entry["count"] += 1
                entry["correct"] += int(is_correct)
                entry["predictions"][prediction] += 1
                entry["targets"][target] += 1
                valid_count = int(tensor_batch["valid_label_mask"][row].sum().item())
                row_entropy = 0.0
                if valid_count > 1:
                    for probability in probabilities[row]:
                        value = float(probability.item())
                        if value > 0.0:
                            row_entropy -= value * math.log(value)
                    row_entropy /= math.log(valid_count)
                entry["probability_entropy"].append(row_entropy)
                phase = str(raw_batch["phases"][row])
                phase_stats[phase][0] += int(is_correct)
                phase_stats[phase][1] += 1

    task_output: dict[str, Any] = {}
    for task, entry in sorted(task_stats.items()):
        labels = MAP_CANDIDATES if task in {"mapping", "need"} else ACTION_CANDIDATES
        count = int(entry["count"])
        task_output[task] = {
            "count": count,
            "accuracy": entry["correct"] / max(count, 1),
            "predicted_counts": dict(sorted(entry["predictions"].items())),
            "target_counts": dict(sorted(entry["targets"].items())),
            "prediction_distribution_entropy": _normalized_entropy(
                entry["predictions"], labels
            ),
            "mean_probability_entropy": sum(entry["probability_entropy"])
            / max(len(entry["probability_entropy"]), 1),
            "dominant_prediction_share": max(
                entry["predictions"].values(), default=0
            )
            / max(count, 1),
        }

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "examples": total,
        "by_task": task_output,
        "by_phase": {
            phase: {
                "correct": values[0],
                "count": values[1],
                "accuracy": values[0] / max(values[1], 1),
            }
            for phase, values in sorted(phase_stats.items())
        },
    }


def _encode_prompt_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    prefix_keep: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        truncate_ids(
            tokenizer.encode(prompt, add_special_tokens=False),
            budget=max_length,
            prefix_keep=prefix_keep,
        )
        for prompt in prompts
    ]
    longest = max(len(values) for values in encoded)
    input_ids = []
    attention = []
    for values in encoded:
        padding = longest - len(values)
        input_ids.append(values + [tokenizer.pad_token_id] * padding)
        attention.append([1] * len(values) + [0] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention, dtype=torch.long, device=device),
    )


def score_prompts(
    model: Any,
    tokenizer: Any,
    label_token_ids: Mapping[str, int],
    prompts: Sequence[str],
    candidate_sets: Sequence[Sequence[str]],
    *,
    max_length: int,
    prefix_keep: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if len(prompts) != len(candidate_sets):
        raise ValueError("prompts and candidate sets must have equal length")
    input_ids, attention_mask = _encode_prompt_batch(
        tokenizer,
        prompts,
        max_length=max_length,
        prefix_keep=prefix_keep,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    final_positions = attention_mask.sum(dim=1) - 1
    row_indices = torch.arange(len(prompts), device=device)
    vocabulary_logits = outputs.logits[row_indices, final_positions]

    results: list[dict[str, Any]] = []
    for row, candidates in enumerate(candidate_sets):
        ids = torch.tensor(
            [label_token_ids[label] for label in candidates],
            dtype=torch.long,
            device=device,
        )
        logits = vocabulary_logits[row].index_select(0, ids)
        probabilities = F.softmax(logits, dim=0)
        selected_index = int(torch.argmax(logits).item())
        results.append(
            {
                "selected": str(candidates[selected_index]),
                "probabilities": {
                    str(label): float(probabilities[index].item())
                    for index, label in enumerate(candidates)
                },
            }
        )
    return results


def predict_decision(
    model: Any,
    tokenizer: Any,
    label_token_ids: Mapping[str, int],
    records: Sequence[Any],
    current_grid: Sequence[Sequence[int]],
    *,
    level_index: int,
    history_mode: str,
    max_length: int,
    prefix_keep: int,
    device: torch.device,
) -> dict[str, Any]:
    history = format_history(
        records,
        current_grid,
        level_index=level_index,
        history_mode=history_mode,
    )
    first_prompts = [mapping_prompt(history, action) for action in Action]
    first_prompts.append(need_prompt(history))
    first = score_prompts(
        model,
        tokenizer,
        label_token_ids,
        first_prompts,
        [MAP_CANDIDATES] * 4 + [NEED_CANDIDATES],
        max_length=max_length,
        prefix_keep=prefix_keep,
        device=device,
    )
    predicted_mapping = {
        action: str(first[index]["selected"])
        for index, action in enumerate(Action)
    }
    predicted_need = str(first[-1]["selected"])

    second = score_prompts(
        model,
        tokenizer,
        label_token_ids,
        [
            compose_prompt(predicted_mapping, predicted_need),
            direct_prompt(history),
        ],
        [ACTION_CANDIDATES, ACTION_CANDIDATES],
        max_length=max_length,
        prefix_keep=prefix_keep,
        device=device,
    )
    return {
        "mapping": {ACTION_DIGIT[action]: predicted_mapping[action] for action in Action},
        "mapping_details": {
            ACTION_DIGIT[action]: first[index]
            for index, action in enumerate(Action)
        },
        "needed_direction": predicted_need,
        "need_details": first[-1],
        "composed_action": str(second[0]["selected"]),
        "compose_details": second[0],
        "direct_action": str(second[1]["selected"]),
        "direct_details": second[1],
    }


def evaluate_closed_loop(
    model: Any,
    tokenizer: Any,
    label_token_ids: Mapping[str, int],
    config: TrainConfig,
    device: torch.device,
    *,
    seed_start: int,
    games: int,
    history_mode: str,
) -> dict[str, Any]:
    games_won = 0
    levels_completed = 0
    total_actions = 0
    valid_action_correct = 0
    direct_action_correct = 0
    mapping_slots_correct = 0
    mapping_slots_total = 0
    need_correct = 0
    need_total = 0
    repeated_pairs = 0
    adjacent_pairs = 0
    predicted_actions = Counter()
    traces: list[dict[str, Any]] = []

    for offset in range(games):
        game_seed = seed_start + offset
        spec = make_stage02_spec(game_seed, levels=3)
        game = HiddenActionGame(spec)
        records: list[Any] = []
        action_trace: list[str] = []
        game_trace: list[dict[str, Any]] = []
        completed_this_game = 0

        for action_number in range(1, config.max_actions_per_game + 1):
            context = Stage02Context(
                game_seed=game_seed,
                variant_id="closed-loop",
                records=tuple(records),
                current_grid=game.frame,
                level_index=game.level_index,
                mapping=spec.action_to_direction,
            )
            oracle = decision_for_context(context)
            prediction = predict_decision(
                model,
                tokenizer,
                label_token_ids,
                records,
                game.frame,
                level_index=game.level_index,
                history_mode=history_mode,
                max_length=config.max_length,
                prefix_keep=config.prefix_keep,
                device=device,
            )
            selected_digit = str(prediction["composed_action"])
            selected = next(
                action for action in Action if ACTION_DIGIT[action] == selected_digit
            )
            unknown_digits = {
                ACTION_DIGIT[action]
                for action, label in oracle.mapping_labels.items()
                if label == "?"
            }
            action_is_valid = (
                selected_digit in unknown_digits
                if unknown_digits
                else selected_digit == oracle.target_action
            )
            valid_action_correct += int(action_is_valid)
            direct_action_correct += int(
                str(prediction["direct_action"]) == oracle.target_action
            )
            for action in Action:
                mapping_slots_total += 1
                mapping_slots_correct += int(
                    prediction["mapping"][ACTION_DIGIT[action]]
                    == oracle.mapping_labels[action]
                )
            need_total += 1
            need_correct += int(
                prediction["needed_direction"] == oracle.needed_direction
            )

            if action_trace:
                adjacent_pairs += 1
                repeated_pairs += int(action_trace[-1] == selected_digit)
            action_trace.append(selected_digit)
            predicted_actions[selected_digit] += 1

            record = game.step(selected)
            records.append(record)
            total_actions += 1
            if record.status in {"LEVEL_WIN", "GAME_WIN"}:
                levels_completed += 1
                completed_this_game += 1

            if len(game_trace) < 40:
                game_trace.append(
                    {
                        "action_number": action_number,
                        "level": record.level_index,
                        "oracle_phase": oracle.decision_phase,
                        "oracle_mapping": {
                            ACTION_DIGIT[action]: oracle.mapping_labels[action]
                            for action in Action
                        },
                        "oracle_need": oracle.needed_direction,
                        "oracle_action": oracle.target_action,
                        "unknown_valid_actions": sorted(unknown_digits),
                        "prediction": prediction,
                        "selected_was_valid": action_is_valid,
                        "status": record.status,
                        "moved": record.moved,
                    }
                )
            if game.finished:
                games_won += 1
                break

        traces.append(
            {
                "game_seed": game_seed,
                "won": game.finished,
                "levels_completed": completed_this_game,
                "actions": len(action_trace),
                "action_trace": action_trace,
                "decisions": game_trace,
            }
        )

    return {
        "history_mode": history_mode,
        "games": games,
        "games_won": games_won,
        "game_win_rate": games_won / max(games, 1),
        "levels_completed": levels_completed,
        "level_completion_rate": levels_completed / max(games * 3, 1),
        "total_actions": total_actions,
        "mean_actions_per_game": total_actions / max(games, 1),
        "composed_valid_action_accuracy": valid_action_correct / max(total_actions, 1),
        "direct_oracle_action_accuracy": direct_action_correct / max(total_actions, 1),
        "mapping_slot_accuracy": mapping_slots_correct / max(mapping_slots_total, 1),
        "need_accuracy": need_correct / max(need_total, 1),
        "repeated_action_transition_rate": repeated_pairs / max(adjacent_pairs, 1),
        "predicted_action_counts": dict(sorted(predicted_actions.items())),
        "dominant_action_share": max(predicted_actions.values(), default=0)
        / max(total_actions, 1),
        "prediction_distribution_entropy": _normalized_entropy(
            predicted_actions, ACTION_CANDIDATES
        ),
        "traces": traces,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(config.train_file))
    evaluation_rows = load_jsonl(Path(config.evaluation_file))
    model, tokenizer, label_token_ids = build_model_and_tokenizer(config)
    tokenizer.model_max_length = 1_000_000
    train_dataset = Stage02Dataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    evaluation_dataset = Stage02Dataset(
        evaluation_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    collator = make_collator(tokenizer.pad_token_id)

    weights, sampling_design = balanced_sampling_weights(train_dataset)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        collate_fn=collator,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def lr_multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        remaining = config.max_optimizer_steps - step
        span = max(config.max_optimizer_steps - config.warmup_steps, 1)
        return max(0.05, remaining / span)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    initial_metrics = classification_metrics(
        model, evaluation_loader, device, label_token_ids
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw_losses: list[float] = []
    step_history: list[dict[str, Any]] = []
    realized_task_targets = Counter()
    optimizer_step = 0
    micro_step = 0
    started = time.time()

    while optimizer_step < config.max_optimizer_steps:
        for raw_batch in train_loader:
            for task, target in zip(
                raw_batch["tasks"], raw_batch["targets"], strict=True
            ):
                realized_task_targets[(task, target)] += 1
            tensor_batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if isinstance(value, torch.Tensor)
            }
            logits = final_label_logits(model, tensor_batch, label_token_ids)
            raw_loss = masked_label_loss(
                logits,
                tensor_batch["valid_label_mask"],
                tensor_batch["target_index"],
            )
            (raw_loss / config.gradient_accumulation_steps).backward()
            raw_losses.append(float(raw_loss.detach().item()))
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            record = {
                "optimizer_step": optimizer_step,
                "loss": sum(raw_losses[-config.gradient_accumulation_steps :])
                / config.gradient_accumulation_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            step_history.append(record)
            if optimizer_step == 1 or optimizer_step % 20 == 0:
                print(json.dumps(record), flush=True)
            if optimizer_step >= config.max_optimizer_steps:
                break

    final_metrics = classification_metrics(
        model, evaluation_loader, device, label_token_ids
    )
    closed_loop: dict[str, Any] = {"train": {}, "heldout": {}}
    for mode in config.history_modes:
        closed_loop["train"][mode] = evaluate_closed_loop(
            model,
            tokenizer,
            label_token_ids,
            config,
            device,
            seed_start=config.train_eval_seed_start,
            games=config.train_eval_games,
            history_mode=mode,
        )
        closed_loop["heldout"][mode] = evaluate_closed_loop(
            model,
            tokenizer,
            label_token_ids,
            config,
            device,
            seed_start=config.heldout_seed_start,
            games=config.heldout_games,
            history_mode=mode,
        )

    by_task = final_metrics.get("by_task", {})
    heldout_intact = closed_loop["heldout"].get("intact", {})
    heldout_amnesic = closed_loop["heldout"].get("amnesic", {})
    gates = {
        "mapping_accuracy_at_least_0_90": float(
            by_task.get("mapping", {}).get("accuracy", 0.0)
        )
        >= 0.90,
        "need_accuracy_at_least_0_90": float(
            by_task.get("need", {}).get("accuracy", 0.0)
        )
        >= 0.90,
        "compose_accuracy_at_least_0_95": float(
            by_task.get("compose", {}).get("accuracy", 0.0)
        )
        >= 0.95,
        "direct_accuracy_at_least_0_70": float(
            by_task.get("direct", {}).get("accuracy", 0.0)
        )
        >= 0.70,
        "heldout_intact_level_completion_positive": float(
            heldout_intact.get("level_completion_rate", 0.0)
        )
        > 0.0,
        "heldout_intact_beats_amnesic": float(
            heldout_intact.get("level_completion_rate", 0.0)
        )
        > float(heldout_amnesic.get("level_completion_rate", 0.0)),
        "heldout_no_single_action_collapse": float(
            heldout_intact.get("dominant_action_share", 1.0)
        )
        < 0.80,
    }
    gates["stage02_gate_passed"] = all(gates.values())

    if config.save_model:
        model.save_pretrained(output_dir / "model", safe_serialization=True)
        tokenizer.save_pretrained(output_dir / "model")

    summary = {
        "status": "completed",
        "claim_scope": (
            "Stage-0.2 synthetic fixed-palette no-wall hidden-action curriculum; "
            "not an ARC-AGI-3 public or private score"
        ),
        "purity_contract": (
            "one GPT-2 causal LM and original tokenizer are the only learned "
            "components at inference; the same model performs all four calls"
        ),
        "config": asdict(config),
        "device": str(device),
        "label_token_ids": label_token_ids,
        "train_examples": len(train_dataset),
        "evaluation_examples": len(evaluation_dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sampling_design": sampling_design,
        "realized_task_target_counts": {
            f"{task}:{target}": count
            for (task, target), count in sorted(realized_task_targets.items())
        },
        "mean_training_loss": sum(raw_losses) / max(len(raw_losses), 1),
        "initial_classification": initial_metrics,
        "final_classification": final_metrics,
        "closed_loop": closed_loop,
        "gates": gates,
        "step_history": step_history,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument(
        "--initialization", choices=("pretrained", "random"), default="pretrained"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--evaluation-file", required=True)
    parser.add_argument("--output-dir", default="outputs/stage02/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prefix-keep", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=320)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--train-eval-seed-start", type=int, default=0)
    parser.add_argument("--train-eval-games", type=int, default=2)
    parser.add_argument("--heldout-seed-start", type=int, default=20)
    parser.add_argument("--heldout-games", type=int, default=4)
    parser.add_argument("--max-actions-per-game", type=int, default=32)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=("intact", "amnesic"),
        default=["intact", "amnesic"],
    )
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            model_name=args.model_name,
            initialization=args.initialization,
            train_file=args.train_file,
            evaluation_file=args.evaluation_file,
            output_dir=args.output_dir,
            seed=args.seed,
            max_length=args.max_length,
            prefix_keep=args.prefix_keep,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_optimizer_steps=args.max_optimizer_steps,
            warmup_steps=args.warmup_steps,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            train_eval_seed_start=args.train_eval_seed_start,
            train_eval_games=args.train_eval_games,
            heldout_seed_start=args.heldout_seed_start,
            heldout_games=args.heldout_games,
            max_actions_per_game=args.max_actions_per_game,
            history_modes=tuple(args.history_modes),
            save_model=args.save_model,
        )
    )


if __name__ == "__main__":
    main()
