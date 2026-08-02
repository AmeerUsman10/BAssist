"""Train one GPT-2 checkpoint on multiple set-valued latent-variable tasks.

The model has no auxiliary classifier heads. Every answer is represented as an
ordinary text completion and scored by GPT-2's causal language-model
likelihood. The same weights learn hidden action semantics and terminal goals.
Task-balanced sampling prevents the more numerous action rows from silently
crowding out goal inference.
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
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .completion_scorer import (
    score_candidate_completions,
    score_with_contextual_calibration,
)


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/joint/data"
    output_dir: str = "outputs/joint/pretrained"
    seed: int = 42
    max_length: int = 768
    prefix_keep: int = 128
    gradient_accumulation_steps: int = 2
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 80
    warmup_steps: int = 8
    freeze_first_n_blocks: int = 8
    train_embeddings: bool = False
    candidate_batch_size: int = 2
    contextual_calibration: bool = True
    evaluation_rows_per_task: int = 24
    save_model: bool = True


@dataclass(frozen=True)
class EncodedJointItem:
    task: str
    source_id: str
    information_level: int
    prompt_ids: tuple[int, ...]
    null_prompt_ids: tuple[int, ...]
    control_prompt_ids: Mapping[str, tuple[int, ...]]
    candidate_ids: tuple[tuple[int, ...], ...]
    target_probabilities: tuple[float, ...]
    truth_index: int
    consistent_indices: tuple[int, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "task",
                "source_id",
                "information_level",
                "context",
                "null_context",
                "control_contexts",
                "candidate_texts",
                "target_distribution",
                "truth_index",
                "consistent_indices",
            }
            if not isinstance(row, dict) or not required.issubset(row):
                missing = sorted(required.difference(row if isinstance(row, dict) else {}))
                raise ValueError(f"invalid joint row at {path}:{line_number}; missing {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"joint dataset is empty: {path}")
    return rows


def truncate_prompt(
    ids: Sequence[int],
    *,
    longest_candidate: int,
    max_length: int,
    prefix_keep: int,
) -> tuple[int, ...]:
    budget = max_length - longest_candidate
    if budget < 32:
        raise ValueError("candidate leaves insufficient prompt budget")
    values = list(int(value) for value in ids)
    if len(values) <= budget:
        return tuple(values)
    keep_prefix = min(prefix_keep, budget // 2, len(values))
    keep_suffix = budget - keep_prefix
    return tuple(values[:keep_prefix] + values[-keep_suffix:])


def encode_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    prefix_keep: int,
) -> EncodedJointItem:
    candidates = tuple(
        tuple(tokenizer.encode(str(text), add_special_tokens=False))
        for text in row["candidate_texts"]
    )
    if not candidates or any(not candidate for candidate in candidates):
        raise ValueError("joint row contains an empty candidate completion")
    longest = max(len(candidate) for candidate in candidates)

    def encode_prompt(text: object) -> tuple[int, ...]:
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        return truncate_prompt(
            ids,
            longest_candidate=longest,
            max_length=max_length,
            prefix_keep=prefix_keep,
        )

    targets = tuple(float(value) for value in row["target_distribution"])
    if len(targets) != len(candidates):
        raise ValueError("candidate and target lengths differ")
    if any(value < 0.0 for value in targets) or not math.isclose(
        sum(targets), 1.0, abs_tol=1e-9
    ):
        raise ValueError("target distribution must be non-negative and sum to one")
    consistent = tuple(int(value) for value in row["consistent_indices"])
    if set(consistent) != {
        index for index, probability in enumerate(targets) if probability > 0.0
    }:
        raise ValueError("consistent indices do not match target support")
    truth_index = int(row["truth_index"])
    if truth_index not in consistent:
        raise ValueError("truth was eliminated from the exact target support")

    return EncodedJointItem(
        task=str(row["task"]),
        source_id=str(row["source_id"]),
        information_level=int(row["information_level"]),
        prompt_ids=encode_prompt(row["context"]),
        null_prompt_ids=encode_prompt(row["null_context"]),
        control_prompt_ids={
            str(name): encode_prompt(text)
            for name, text in dict(row["control_contexts"]).items()
        },
        candidate_ids=candidates,
        target_probabilities=targets,
        truth_index=truth_index,
        consistent_indices=consistent,
    )


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
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(config.model_name)
        )
    else:
        raise ValueError("initialization must be pretrained or random")
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


def candidate_scores(
    model: Any,
    item: EncodedJointItem,
    *,
    prompt_ids: Sequence[int],
    pad_token_id: int,
    device: torch.device,
    config: Config,
) -> torch.Tensor:
    if config.contextual_calibration:
        return score_with_contextual_calibration(
            model,
            prompt_ids,
            item.null_prompt_ids,
            item.candidate_ids,
            pad_token_id=pad_token_id,
            device=device,
            candidate_batch_size=config.candidate_batch_size,
            reduction="mean",
        )
    return score_candidate_completions(
        model,
        prompt_ids,
        item.candidate_ids,
        pad_token_id=pad_token_id,
        device=device,
        candidate_batch_size=config.candidate_batch_size,
        reduction="mean",
    )


def set_valued_loss(scores: torch.Tensor, item: EncodedJointItem) -> torch.Tensor:
    targets = torch.tensor(
        item.target_probabilities,
        dtype=scores.dtype,
        device=scores.device,
    )
    return -(targets * F.log_softmax(scores, dim=-1)).sum()


def select_evaluation_items(
    items: Sequence[EncodedJointItem],
    *,
    rows_per_task: int,
    seed: int,
) -> tuple[EncodedJointItem, ...]:
    """Select a deterministic information-level-stratified evaluation slice."""

    if rows_per_task < 1:
        raise ValueError("rows_per_task must be positive")
    grouped: dict[str, dict[int, list[EncodedJointItem]]] = {}
    for item in items:
        grouped.setdefault(item.task, {}).setdefault(item.information_level, []).append(item)
    selected: list[EncodedJointItem] = []
    for task in sorted(grouped):
        rng = random.Random(seed ^ sum(ord(character) for character in task))
        levels = sorted(grouped[task])
        pools: dict[int, list[EncodedJointItem]] = {}
        for level in levels:
            pool = list(grouped[task][level])
            rng.shuffle(pool)
            pools[level] = pool
        cursor = {level: 0 for level in levels}
        while len([item for item in selected if item.task == task]) < rows_per_task:
            progressed = False
            for level in levels:
                index = cursor[level]
                pool = pools[level]
                if index >= len(pool):
                    continue
                selected.append(pool[index])
                cursor[level] += 1
                progressed = True
                if len([item for item in selected if item.task == task]) >= rows_per_task:
                    break
            if not progressed:
                break
    return tuple(selected)


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0.0]
    return float((-(positive * torch.log2(positive))).sum().item())


def evaluate(
    model: Any,
    items: Sequence[EncodedJointItem],
    *,
    mode: str,
    tokenizer: Any,
    device: torch.device,
    config: Config,
) -> dict[str, Any]:
    """Evaluate full evidence or one named information control."""

    del tokenizer
    aggregates: dict[tuple[str, int], dict[str, float]] = {}
    traces: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for item in items:
            if mode == "full":
                prompt = item.prompt_ids
            else:
                prompt = item.control_prompt_ids.get(mode)
                if prompt is None:
                    continue
            scores = candidate_scores(
                model,
                item,
                prompt_ids=prompt,
                pad_token_id=int(model.config.pad_token_id),
                device=device,
                config=config,
            )
            probabilities = torch.softmax(scores, dim=-1)
            targets = torch.tensor(
                item.target_probabilities,
                dtype=probabilities.dtype,
                device=device,
            )
            consistent = list(item.consistent_indices)
            order = torch.argsort(probabilities, descending=True).tolist()
            selected_index = int(torch.argmax(probabilities).item())
            key = (item.task, item.information_level)
            values = aggregates.setdefault(
                key,
                {
                    "rows": 0.0,
                    "consistent_mass": 0.0,
                    "map_consistent": 0.0,
                    "truth_probability": 0.0,
                    "truth_rank": 0.0,
                    "entropy_bits": 0.0,
                    "set_cross_entropy": 0.0,
                    "brier": 0.0,
                },
            )
            values["rows"] += 1.0
            values["consistent_mass"] += float(probabilities[consistent].sum().item())
            values["map_consistent"] += float(selected_index in consistent)
            values["truth_probability"] += float(probabilities[item.truth_index].item())
            values["truth_rank"] += float(order.index(item.truth_index) + 1)
            values["entropy_bits"] += _entropy_bits(probabilities)
            values["set_cross_entropy"] += float(
                (-(targets * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
            )
            values["brier"] += float(((probabilities - targets) ** 2).sum().item())
            traces.append(
                {
                    "task": item.task,
                    "source_id": item.source_id,
                    "information_level": item.information_level,
                    "mode": mode,
                    "consistent_mass": float(probabilities[consistent].sum().item()),
                    "map_consistent": selected_index in consistent,
                    "truth_probability": float(probabilities[item.truth_index].item()),
                    "truth_rank": order.index(item.truth_index) + 1,
                    "entropy_bits": _entropy_bits(probabilities),
                }
            )

    by_task: dict[str, dict[str, Any]] = {}
    for (task, level), values in sorted(aggregates.items()):
        count = values["rows"]
        task_entry = by_task.setdefault(task, {"rows": 0.0, "by_information_level": {}})
        task_entry["rows"] += count
        task_entry["by_information_level"][str(level)] = {
            key: value / count
            for key, value in values.items()
            if key != "rows"
        } | {"rows": count}
    return {"mode": mode, "by_task": by_task, "traces": traces}


def train(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(config)
    train_items = [
        encode_row(
            row,
            tokenizer,
            max_length=config.max_length,
            prefix_keep=config.prefix_keep,
        )
        for row in load_jsonl(data_dir / "train.jsonl")
    ]
    validation_items = [
        encode_row(
            row,
            tokenizer,
            max_length=config.max_length,
            prefix_keep=config.prefix_keep,
        )
        for row in load_jsonl(data_dir / "validation.jsonl")
    ]
    test_items = [
        encode_row(
            row,
            tokenizer,
            max_length=config.max_length,
            prefix_keep=config.prefix_keep,
        )
        for row in load_jsonl(data_dir / "test.jsonl")
    ]
    tasks = sorted({item.task for item in train_items})
    train_by_task = {task: [item for item in train_items if item.task == task] for task in tasks}
    if any(not values for values in train_by_task.values()):
        raise ValueError("every declared task requires training examples")

    evaluation_items = select_evaluation_items(
        test_items,
        rows_per_task=config.evaluation_rows_per_task,
        seed=config.seed + 99,
    )
    validation_selection = select_evaluation_items(
        validation_items,
        rows_per_task=config.evaluation_rows_per_task,
        seed=config.seed + 77,
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
        return max(
            0.05,
            (config.max_optimizer_steps - step)
            / max(config.max_optimizer_steps - config.warmup_steps, 1),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_validation = evaluate(
        model,
        validation_selection,
        mode="full",
        tokenizer=tokenizer,
        device=device,
        config=config,
    )
    initial_test = evaluate(
        model,
        evaluation_items,
        mode="full",
        tokenizer=tokenizer,
        device=device,
        config=config,
    )

    rng = random.Random(config.seed ^ 0xA11CE)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    losses_by_task: dict[str, list[float]] = {task: [] for task in tasks}
    started = time.time()
    model.train()
    while optimizer_step < config.max_optimizer_steps:
        task = tasks[micro_step % len(tasks)]
        item = rng.choice(train_by_task[task])
        scores = candidate_scores(
            model,
            item,
            prompt_ids=item.prompt_ids,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            config=config,
        )
        raw_loss = set_valued_loss(scores, item)
        loss = raw_loss / config.gradient_accumulation_steps
        loss.backward()
        value = float(raw_loss.detach().item())
        losses.append(value)
        losses_by_task[task].append(value)
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
                            "loss": sum(losses[-20:]) / min(20, len(losses)),
                            "task_losses": {
                                task_name: sum(values[-10:]) / min(10, len(values))
                                for task_name, values in losses_by_task.items()
                                if values
                            },
                            "learning_rate": scheduler.get_last_lr()[0],
                            "device": str(device),
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )

    final_validation = evaluate(
        model,
        validation_selection,
        mode="full",
        tokenizer=tokenizer,
        device=device,
        config=config,
    )
    modes = sorted(
        {"full"}.union(
            name
            for item in evaluation_items
            for name in item.control_prompt_ids
        )
    )
    final_controls = {
        mode: evaluate(
            model,
            evaluation_items,
            mode=mode,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )
        for mode in modes
    }

    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    summary = {
        "status": "completed",
        "scope": (
            "One-checkpoint set-valued action-binding and latent-goal training; "
            "controlled gate, not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "tasks": tasks,
        "train_examples": len(train_items),
        "validation_examples": len(validation_items),
        "test_examples": len(test_items),
        "evaluation_examples": len(evaluation_items),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_training_loss": sum(losses) / len(losses),
        "mean_training_loss_by_task": {
            task: sum(values) / len(values) for task, values in losses_by_task.items()
        },
        "initial_validation": initial_validation,
        "initial_test": initial_test,
        "final_validation": final_validation,
        "final_controls": final_controls,
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
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--data-dir", default="outputs/joint/data")
    parser.add_argument("--output-dir", default="outputs/joint/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--prefix-keep", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--no-contextual-calibration", action="store_true")
    parser.add_argument("--evaluation-rows-per-task", type=int, default=24)
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        Config(
            model_name=args.model_name,
            initialization=args.initialization,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            max_length=args.max_length,
            prefix_keep=args.prefix_keep,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_optimizer_steps=args.max_optimizer_steps,
            warmup_steps=args.warmup_steps,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            train_embeddings=args.train_embeddings,
            candidate_batch_size=args.candidate_batch_size,
            contextual_calibration=not args.no_contextual_calibration,
            evaluation_rows_per_task=args.evaluation_rows_per_task,
            save_model=not args.no_save_model,
        )
    )


if __name__ == "__main__":
    main()
