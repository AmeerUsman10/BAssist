"""Train and evaluate GPT-2 on the Phase-0 hidden-action task.

The model receives exact serialized interaction history and is trained only to
predict the next action token. Closed-loop evaluation measures whether the
model uses history to infer a per-game action permutation.
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
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .codec import tokens_to_text
from .phase0_hidden_action import (
    ACTION_TOKEN,
    Action,
    HiddenActionGame,
    append_record_tokens,
    generate_game,
    initial_transcript,
    simulate_source_history,
)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    initialization: str
    data_dir: str
    output_dir: str
    seed: int
    max_length: int
    prefix_keep: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_optimizer_steps: int
    warmup_steps: int
    freeze_first_n_blocks: int
    evaluation_games: int
    evaluation_seed_start: int
    max_actions_per_game: int


class ActionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        prefix_keep: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            context_ids = tokenizer.encode(str(row["context"]), add_special_tokens=False)
            target_ids = tokenizer.encode(str(row["target"]), add_special_tokens=False)
            if len(target_ids) != 1:
                raise ValueError(f"action target must be one token: {row['target']!r} -> {target_ids}")
            context_ids = truncate_ids(
                context_ids,
                budget=max_length - 1,
                prefix_keep=prefix_keep,
            )
            input_ids = context_ids + target_ids
            labels = [-100] * len(context_ids) + target_ids
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def truncate_ids(ids: Sequence[int], *, budget: int, prefix_keep: int) -> list[int]:
    if budget < 1:
        raise ValueError("token budget must be positive")
    values = list(ids)
    if len(values) <= budget:
        return values
    keep_prefix = min(prefix_keep, budget // 2, len(values))
    keep_suffix = budget - keep_prefix
    return values[:keep_prefix] + values[-keep_suffix:]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or "context" not in item or "target" not in item:
                raise ValueError(f"invalid row at {path}:{line_number}")
            rows.append(item)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def make_collator(pad_token_id: int):
    def collate(batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for item in batch:
            padding = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_masks.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

    return collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def build_model_and_tokenizer(config: TrainConfig, special_tokens: Sequence[str]):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens)})

    if config.initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
    elif config.initialization == "random":
        model_config = AutoConfig.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError("initialization must be 'pretrained' or 'random'")

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    blocks = model.transformer.h
    if config.freeze_first_n_blocks < 0 or config.freeze_first_n_blocks > len(blocks):
        raise ValueError(f"freeze_first_n_blocks must be in 0..{len(blocks)}")
    for block in blocks[: config.freeze_first_n_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    # New ARC tokens live in the shared word embedding / LM head, so that
    # matrix must remain trainable. Position embeddings and final layer norm
    # also remain trainable.
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    action_token_ids: dict[Action, int] = {}
    for action, token in ACTION_TOKEN.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"action token was not atomic: {token} -> {ids}")
        action_token_ids[action] = ids[0]
    return model, tokenizer, action_token_ids


def classification_metrics(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    action_ids: set[int],
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    action_id_list = sorted(action_ids)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
            labels = batch["labels"]
            for row_index in range(labels.shape[0]):
                target_positions = torch.nonzero(labels[row_index] != -100, as_tuple=False)
                if target_positions.numel() != 1:
                    raise RuntimeError("each example must contain one target position")
                position = int(target_positions.item())
                prediction_logits = outputs.logits[row_index, position - 1, action_id_list]
                predicted_id = action_id_list[int(torch.argmax(prediction_logits).item())]
                target_id = int(labels[row_index, position].item())
                correct += int(predicted_id == target_id)
                total += 1
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "accuracy": correct / max(total, 1),
        "examples": float(total),
    }


def inference_context_tokens(
    transcript: Sequence[str],
    *,
    mode: str,
    game: HiddenActionGame,
) -> list[str]:
    if mode == "intact":
        return [*transcript, "<DECIDE>"]

    if mode == "amnesic":
        return [
            "<GAME_START>",
            f"<LEVEL_{game.level_index}>",
            "<FRAME>",
            *(__import__("arcgpt2.codec", fromlist=["encode_frame"]).encode_frame(game.frame)),
            "</FRAME>",
            "<AVAILABLE>",
            *(ACTION_TOKEN[action] for action in Action),
            "</AVAILABLE>",
            "<DECIDE>",
        ]

    if mode == "shuffled":
        rotation = {
            "<A1>": "<A2>",
            "<A2>": "<A3>",
            "<A3>": "<A4>",
            "<A4>": "<A1>",
        }
        corrupted: list[str] = []
        inside_historical_action = False
        for token in transcript:
            if token == "<ACTION>":
                inside_historical_action = True
                corrupted.append(token)
                continue
            if token == "</ACTION>":
                inside_historical_action = False
                corrupted.append(token)
                continue
            corrupted.append(rotation.get(token, token) if inside_historical_action else token)
        return [*corrupted, "<DECIDE>"]

    raise ValueError(f"unknown history mode: {mode}")


def choose_action(
    model: Any,
    tokenizer: Any,
    action_token_ids: dict[Action, int],
    transcript: Sequence[str],
    game: HiddenActionGame,
    *,
    history_mode: str,
    max_length: int,
    prefix_keep: int,
    device: torch.device,
) -> Action:
    tokens = inference_context_tokens(transcript, mode=history_mode, game=game)
    ids = tokenizer.encode(tokens_to_text(tokens), add_special_tokens=False)
    ids = truncate_ids(ids, budget=max_length, prefix_keep=prefix_keep)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
    actions = list(Action)
    candidate_ids = torch.tensor([action_token_ids[action] for action in actions], device=device)
    selected = int(torch.argmax(logits[candidate_ids]).item())
    return actions[selected]


def evaluate_closed_loop(
    model: Any,
    tokenizer: Any,
    action_token_ids: dict[Action, int],
    config: TrainConfig,
    device: torch.device,
    *,
    history_mode: str,
) -> dict[str, Any]:
    games_won = 0
    levels_completed = 0
    total_actions = 0
    source_actions = 0
    level_actions: dict[int, list[int]] = {}
    traces: list[dict[str, Any]] = []

    for offset in range(config.evaluation_games):
        game_seed = config.evaluation_seed_start + offset
        spec = generate_game(game_seed)
        game = HiddenActionGame(spec)
        transcript = initial_transcript(spec)
        source_count = len(simulate_source_history(spec))
        source_actions += source_count
        actions_this_level = 0
        completed_this_game = 0
        action_trace: list[str] = []

        for _ in range(config.max_actions_per_game):
            action = choose_action(
                model,
                tokenizer,
                action_token_ids,
                transcript,
                game,
                history_mode=history_mode,
                max_length=config.max_length,
                prefix_keep=config.prefix_keep,
                device=device,
            )
            record = game.step(action)
            total_actions += 1
            actions_this_level += 1
            action_trace.append(action.value)
            next_frame = game.frame if record.status == "LEVEL_WIN" else None
            append_record_tokens(transcript, record, next_frame=next_frame)

            if record.status in {"LEVEL_WIN", "GAME_WIN"}:
                completed_this_game += 1
                levels_completed += 1
                level_actions.setdefault(record.level_index, []).append(actions_this_level)
                actions_this_level = 0
            if record.status == "GAME_WIN":
                games_won += 1
                break

        traces.append(
            {
                "game_seed": game_seed,
                "won": game.finished,
                "levels_completed": completed_this_game,
                "actions": len(action_trace),
                "source_actions": source_count,
                "trace": action_trace,
            }
        )

    per_level_mean = {
        str(level): sum(values) / len(values)
        for level, values in sorted(level_actions.items())
        if values
    }
    return {
        "history_mode": history_mode,
        "games": config.evaluation_games,
        "games_won": games_won,
        "game_win_rate": games_won / config.evaluation_games,
        "levels_completed": levels_completed,
        "level_completion_rate": levels_completed / (config.evaluation_games * 3),
        "total_actions": total_actions,
        "mean_actions_per_game": total_actions / config.evaluation_games,
        "source_actions": source_actions,
        "source_efficiency_ratio": source_actions / max(total_actions, 1),
        "mean_actions_by_completed_level": per_level_mean,
        "traces": traces,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = json.loads((data_dir / "special_tokens.json").read_text(encoding="utf-8"))
    train_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    test_rows = load_jsonl(data_dir / "test.jsonl")

    model, tokenizer, action_token_ids = build_model_and_tokenizer(config, special_tokens)
    train_dataset = ActionDataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    validation_dataset = ActionDataset(
        validation_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    test_dataset = ActionDataset(
        test_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )

    collator = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
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

    def learning_rate_multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        remaining = config.max_optimizer_steps - step
        decay_span = max(config.max_optimizer_steps - config.warmup_steps, 1)
        return max(0.05, remaining / decay_span)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    initial_validation = classification_metrics(
        model,
        validation_loader,
        device,
        set(action_token_ids.values()),
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()

    while optimizer_step < config.max_optimizer_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
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
                                "loss": round(sum(losses[-10:]) / min(len(losses), 10), 6),
                                "learning_rate": scheduler.get_last_lr()[0],
                                "elapsed_seconds": round(time.time() - started, 2),
                                "device": str(device),
                            }
                        ),
                        flush=True,
                    )
                if optimizer_step >= config.max_optimizer_steps:
                    break

    final_validation = classification_metrics(
        model,
        validation_loader,
        device,
        set(action_token_ids.values()),
    )
    test_classification = classification_metrics(
        model,
        test_loader,
        device,
        set(action_token_ids.values()),
    )

    closed_loop = {
        mode: evaluate_closed_loop(
            model,
            tokenizer,
            action_token_ids,
            config,
            device,
            history_mode=mode,
        )
        for mode in ("intact", "amnesic", "shuffled")
    }

    model.save_pretrained(output_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "model")
    summary: dict[str, Any] = {
        "status": "completed",
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_training_loss": sum(losses) / len(losses),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "test_classification": test_classification,
        "closed_loop": closed_loop,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--data-dir", default="outputs/phase0/data")
    parser.add_argument("--output-dir", default="outputs/phase0/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--prefix-keep", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=32)
    parser.add_argument("--evaluation-seed-start", type=int, default=900_000)
    parser.add_argument("--max-actions-per-game", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
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
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        max_actions_per_game=args.max_actions_per_game,
    )
    train(config)


if __name__ == "__main__":
    main()
