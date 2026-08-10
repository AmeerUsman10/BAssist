"""Train one GPT-2 checkpoint to score a live version space of goal programs.

Every candidate completion is an ordinary-language rendering of a typed
Goal-DSL predicate. Training uses a set-valued objective over all predicates
that exactly replay the observed terminal/non-terminal history. Deterministic
code only executes declared mechanics/goals and calculates probabilities.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .dsl import execute, program_from_phase0_spec
from .goal_dsl import evaluate_goal, parse_goal
from .phase0_hidden_action import Action, generate_game


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/goal_version/data"
    output_dir: str = "outputs/goal_version/pretrained"
    seed: int = 42
    max_length: int = 896
    prefix_keep: int = 160
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 80
    warmup_steps: int = 8
    freeze_first_n_blocks: int = 8
    train_embeddings: bool = False
    evaluation_games: int = 8
    save_model: bool = True


@dataclass(frozen=True)
class EncodedGoalItem:
    prompt_ids: tuple[int, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    target_probabilities: tuple[float, ...]


class GoalVersionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        context_field: str,
        max_length: int,
        prefix_keep: int,
    ) -> None:
        self.items = [
            encode_row(
                row,
                tokenizer,
                context_field=context_field,
                max_length=max_length,
                prefix_keep=prefix_keep,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedGoalItem:
        return self.items[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "game_seed",
        "prefix_length",
        "observed_terminal_count",
        "context",
        "amnesic_context",
        "statusless_context",
        "shuffled_status_context",
        "candidate_texts",
        "candidate_programs",
        "target_distribution",
        "consistent_indices",
        "truth_index",
        "held_out_records",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required.issubset(row):
                missing = sorted(required.difference(row if isinstance(row, dict) else {}))
                raise ValueError(f"invalid row at {path}:{line_number}; missing {missing}")
            if len(row["candidate_texts"]) != len(row["candidate_programs"]):
                raise ValueError(f"candidate field length mismatch at {path}:{line_number}")
            if len(row["candidate_texts"]) != len(row["target_distribution"]):
                raise ValueError(f"target field length mismatch at {path}:{line_number}")
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
) -> tuple[int, ...]:
    budget = max_length - target_length
    if budget < 64:
        raise ValueError("goal candidate leaves insufficient prompt budget")
    values = list(ids)
    if len(values) <= budget:
        return tuple(values)
    keep_prefix = min(prefix_keep, budget // 2, len(values))
    keep_suffix = budget - keep_prefix
    return tuple(values[:keep_prefix] + values[-keep_suffix:])


def encode_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    context_field: str,
    max_length: int,
    prefix_keep: int,
) -> EncodedGoalItem:
    prompt_full = tokenizer.encode(str(row[context_field]), add_special_tokens=False)
    candidate_ids = tuple(
        tuple(tokenizer.encode(str(text), add_special_tokens=False))
        for text in row["candidate_texts"]
    )
    if any(not candidate for candidate in candidate_ids):
        raise ValueError("a goal completion tokenized to an empty sequence")
    prompt_ids = truncate_prompt(
        prompt_full,
        target_length=max(len(candidate) for candidate in candidate_ids),
        max_length=max_length,
        prefix_keep=prefix_keep,
    )
    targets = tuple(float(value) for value in row["target_distribution"])
    if any(value < 0.0 for value in targets) or not math.isclose(
        sum(targets), 1.0, abs_tol=1e-9
    ):
        raise ValueError("goal target distribution must be non-negative and sum to one")
    return EncodedGoalItem(prompt_ids, candidate_ids, targets)


def collate_items(batch: Sequence[EncodedGoalItem]) -> list[EncodedGoalItem]:
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


def score_items(
    model: Any,
    items: Sequence[EncodedGoalItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return one differentiable candidate-score vector per dataset item."""

    if not items:
        raise ValueError("at least one goal item is required")
    flattened: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for item_index, item in enumerate(items):
        flattened.extend(
            (item.prompt_ids, candidate, item_index)
            for candidate in item.candidate_ids
        )
    lengths = [len(prompt) + len(candidate) for prompt, candidate, _ in flattened]
    longest = max(lengths)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for (prompt, candidate, _), length in zip(flattened, lengths, strict=True):
        missing = longest - length
        input_rows.append([*prompt, *candidate, *([pad_token_id] * missing)])
        mask_rows.append([1] * length + [0] * missing)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    grouped_scores: list[list[torch.Tensor]] = [[] for _ in items]
    for row_index, (prompt, candidate, item_index) in enumerate(flattened):
        start = len(prompt) - 1
        target_logits = logits[row_index, start : start + len(candidate)]
        targets = torch.tensor(candidate, dtype=torch.long, device=device)
        log_probability = F.log_softmax(target_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        )[:, 0].sum()
        grouped_scores[item_index].append(log_probability)

    score_tensors = [torch.stack(scores) for scores in grouped_scores]
    target_tensors = [
        torch.tensor(
            item.target_probabilities,
            dtype=score_tensors[index].dtype,
            device=device,
        )
        for index, item in enumerate(items)
    ]
    return score_tensors, target_tensors


def set_valued_loss(
    score_vectors: Sequence[torch.Tensor],
    target_vectors: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(score_vectors) != len(target_vectors) or not score_vectors:
        raise ValueError("score and target batches must be non-empty and aligned")
    losses = [
        -(targets * F.log_softmax(scores, dim=-1)).sum()
        for scores, targets in zip(score_vectors, target_vectors, strict=True)
    ]
    return torch.stack(losses).mean()


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
            scores, targets = score_items(
                model, items, pad_token_id=pad_token_id, device=device
            )
            losses.append(float(set_valued_loss(scores, targets).item()))
            for score, target in zip(scores, targets, strict=True):
                probability = torch.softmax(score, dim=-1)
                briers.append(float(((probability - target) ** 2).sum().item()))
    return {
        "set_cross_entropy": sum(losses) / len(losses),
        "mean_brier": sum(briers) / len(briers),
        "examples": float(len(loader.dataset)),
    }


def held_out_terminal_accuracy(row: Mapping[str, Any], selected_index: int) -> float | None:
    held_out = row["held_out_records"]
    if not held_out:
        return None
    spec = generate_game(int(row["game_seed"]))
    mechanics = program_from_phase0_spec(spec)
    goal = parse_goal(str(row["candidate_programs"][selected_index]))
    correct = 0
    for record in held_out:
        before = tuple(tuple(int(value) for value in grid_row) for grid_row in record["before"])
        action = Action(str(record["action"]))
        execution = execute(mechanics, before, action)
        predicted_terminal = evaluate_goal(goal, execution)
        expected_terminal = str(record["status"]) in {"LEVEL_WIN", "GAME_WIN"}
        correct += int(predicted_terminal == expected_terminal)
    return correct / len(held_out)


def _entropy(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


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
        "statusless": "statusless_context",
        "shuffled_status": "shuffled_status_context",
    }.get(mode)
    if context_field is None:
        raise ValueError("unknown goal evidence mode")

    selected_seeds = sorted({int(row["game_seed"]) for row in rows})[: config.evaluation_games]
    selected_rows = [row for row in rows if int(row["game_seed"]) in selected_seeds]
    aggregates: dict[int, dict[str, float]] = {}
    traces: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for row in selected_rows:
            item = encode_row(
                row,
                tokenizer,
                context_field=context_field,
                max_length=config.max_length,
                prefix_keep=config.prefix_keep,
            )
            score_vectors, target_vectors = score_items(
                model,
                [item],
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            scores = score_vectors[0]
            target = target_vectors[0]
            probabilities = torch.softmax(scores, dim=-1)
            consistent = [int(index) for index in row["consistent_indices"]]
            selected_index = int(torch.argmax(probabilities).item())
            truth_index = int(row["truth_index"])
            truth_order = torch.argsort(probabilities, descending=True).tolist()
            terminal_accuracy = held_out_terminal_accuracy(row, selected_index)
            terminal_count = int(row["observed_terminal_count"])

            values = aggregates.setdefault(
                terminal_count,
                {
                    "rows": 0.0,
                    "consistent_mass": 0.0,
                    "map_consistent": 0.0,
                    "truth_probability": 0.0,
                    "truth_rank": 0.0,
                    "entropy_bits": 0.0,
                    "set_cross_entropy": 0.0,
                    "brier": 0.0,
                    "held_out_rows": 0.0,
                    "held_out_terminal_accuracy": 0.0,
                },
            )
            values["rows"] += 1.0
            values["consistent_mass"] += float(probabilities[consistent].sum().item())
            values["map_consistent"] += float(selected_index in consistent)
            values["truth_probability"] += float(probabilities[truth_index].item())
            values["truth_rank"] += float(truth_order.index(truth_index) + 1)
            values["entropy_bits"] += _entropy(probabilities)
            values["set_cross_entropy"] += float(
                (-(target * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
            )
            values["brier"] += float(((probabilities - target) ** 2).sum().item())
            if terminal_accuracy is not None:
                values["held_out_rows"] += 1.0
                values["held_out_terminal_accuracy"] += terminal_accuracy

            traces.append(
                {
                    "game_seed": int(row["game_seed"]),
                    "prefix_length": int(row["prefix_length"]),
                    "observed_terminal_count": terminal_count,
                    "consistent_goal_count": int(row["consistent_goal_count"]),
                    "consistent_mass": float(probabilities[consistent].sum().item()),
                    "map_consistent": selected_index in consistent,
                    "truth_probability": float(probabilities[truth_index].item()),
                    "truth_rank": truth_order.index(truth_index) + 1,
                    "entropy_bits": _entropy(probabilities),
                    "held_out_terminal_accuracy": terminal_accuracy,
                }
            )

    by_terminal_count: dict[str, dict[str, float]] = {}
    for count, values in sorted(aggregates.items()):
        rows_count = values["rows"]
        held_out_count = values["held_out_rows"]
        by_terminal_count[str(count)] = {
            "rows": rows_count,
            "consistent_mass": values["consistent_mass"] / rows_count,
            "map_consistent": values["map_consistent"] / rows_count,
            "truth_probability": values["truth_probability"] / rows_count,
            "truth_rank": values["truth_rank"] / rows_count,
            "entropy_bits": values["entropy_bits"] / rows_count,
            "set_cross_entropy": values["set_cross_entropy"] / rows_count,
            "brier": values["brier"] / rows_count,
            "held_out_rows": held_out_count,
            "held_out_terminal_accuracy": (
                values["held_out_terminal_accuracy"] / held_out_count
                if held_out_count
                else float("nan")
            ),
        }
    return {
        "mode": mode,
        "games": len(selected_seeds),
        "rows": len(selected_rows),
        "by_observed_terminal_count": by_terminal_count,
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

    train_dataset = GoalVersionDataset(
        train_rows,
        tokenizer,
        context_field="context",
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    validation_dataset = GoalVersionDataset(
        validation_rows,
        tokenizer,
        context_field="context",
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
        return max(
            0.05,
            (config.max_optimizer_steps - step)
            / max(config.max_optimizer_steps - config.warmup_steps, 1),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_validation = evaluate_loader(
        model,
        validation_loader,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    initial_full = evaluate_version_space(
        model, tokenizer, test_rows, config, device, mode="full"
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()
    model.train()
    while optimizer_step < config.max_optimizer_steps:
        for items in train_loader:
            score_vectors, target_vectors = score_items(
                model,
                items,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            loss = set_valued_loss(score_vectors, target_vectors) / config.gradient_accumulation_steps
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
            model, tokenizer, test_rows, config, device, mode=mode
        )
        for mode in ("full", "amnesic", "statusless", "shuffled_status")
    }

    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    summary = {
        "status": "completed",
        "scope": (
            "Set-valued atomic Goal-DSL inference under known mechanics; "
            "controlled gate, not ARC-AGI-3 evaluation."
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
    parser.add_argument("--data-dir", default="outputs/goal_version/data")
    parser.add_argument("--output-dir", default="outputs/goal_version/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=896)
    parser.add_argument("--prefix-keep", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--evaluation-games", type=int, default=8)
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
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_optimizer_steps=args.max_optimizer_steps,
            warmup_steps=args.warmup_steps,
            freeze_first_n_blocks=args.freeze_first_n_blocks,
            train_embeddings=args.train_embeddings,
            evaluation_games=args.evaluation_games,
            save_model=not args.no_save_model,
        )
    )


if __name__ == "__main__":
    main()
