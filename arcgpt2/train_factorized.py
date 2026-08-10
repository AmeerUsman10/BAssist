"""Train GPT-2 to infer hidden action semantics through natural language.

For each game GPT-2 answers four ordinary-language questions, one per action.
The deterministic evaluator combines the 4x4 likelihood matrix under the known
Phase-0 permutation constraint, expands the selected mapping into an executable
ARC-DSL program, and measures downstream completion.

The only learned model is one original GPT-2-family checkpoint.
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
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .build_factorized_dataset import build_control_prompt
from .dsl import program_from_phase0_spec
from .mapping_target import expand_mapping
from .natural_protocol import answer_text, direction_words
from .phase0_hidden_action import Action, Direction, generate_game
from .train_program_induction import execute_selected_program


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/factorized/data"
    output_dir: str = "outputs/factorized/pretrained"
    seed: int = 42
    max_length: int = 512
    prefix_keep: int = 96
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 300
    warmup_steps: int = 20
    freeze_first_n_blocks: int = 8
    evaluation_games: int = 64
    score_batch_size: int = 16
    max_plan_depth: int = 64
    save_model: bool = True


class CompletionDataset(Dataset):
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
            if not target_ids:
                raise ValueError("direction target tokenized to an empty sequence")
            context_ids = truncate_context(
                context_ids,
                target_length=len(target_ids),
                max_length=max_length,
                prefix_keep=prefix_keep,
            )
            self.items.append(
                {
                    "input_ids": context_ids + target_ids,
                    "labels": [-100] * len(context_ids) + target_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def truncate_context(
    ids: Sequence[int],
    *,
    target_length: int,
    max_length: int,
    prefix_keep: int,
) -> list[int]:
    budget = max_length - target_length
    if budget < 16:
        raise ValueError("target leaves insufficient context budget")
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
            row = json.loads(line)
            if not isinstance(row, dict) or "context" not in row or "target" not in row:
                raise ValueError(f"invalid row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def make_collator(pad_token_id: int):
    def collate(batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            padding = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * padding)
            attention_masks.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


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

    # Keep identical trainable parameter groups in pretrained and random runs.
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True
    return model, tokenizer


def mean_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    return sum(losses) / len(losses)


def score_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int,
    prefix_keep: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have equal length")

    encoded: list[tuple[list[int], list[int]]] = []
    for prompt, completion in zip(prompts, completions, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(completion, add_special_tokens=False)
        prompt_ids = truncate_context(
            prompt_ids,
            target_length=len(target_ids),
            max_length=max_length,
            prefix_keep=prefix_keep,
        )
        encoded.append((prompt_ids, target_ids))

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            lengths = [len(prompt_ids) + len(target_ids) for prompt_ids, target_ids in chunk]
            longest = max(lengths)
            rows: list[list[int]] = []
            masks: list[list[int]] = []
            for (prompt_ids, target_ids), length in zip(chunk, lengths, strict=True):
                padding = longest - length
                rows.append(prompt_ids + target_ids + [tokenizer.pad_token_id] * padding)
                masks.append([1] * length + [0] * padding)

            input_ids = torch.tensor(rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            for row_index, (prompt_ids, target_ids) in enumerate(chunk):
                start_logit = len(prompt_ids) - 1
                target_logits = logits[
                    row_index,
                    start_logit : start_logit + len(target_ids),
                ]
                target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
                token_log_probs = F.log_softmax(target_logits, dim=-1).gather(
                    1, target_tensor.unsqueeze(1)
                )[:, 0]
                scores.append(float(token_log_probs.sum().item()))
    return scores


def select_mapping(score_matrix: Sequence[Sequence[float]]) -> tuple[dict[Action, Direction], dict[str, Any]]:
    actions = tuple(Action)
    directions = tuple(Direction)
    if len(score_matrix) != len(actions) or any(
        len(row) != len(directions) for row in score_matrix
    ):
        raise ValueError("score matrix must be 4x4")

    candidates: list[tuple[float, tuple[Direction, ...]]] = []
    for permutation in itertools.permutations(directions):
        score = sum(
            score_matrix[action_index][directions.index(direction)]
            for action_index, direction in enumerate(permutation)
        )
        candidates.append((score, permutation))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    logits = torch.tensor([score for score, _ in candidates], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=0)
    mapping = {action: direction for action, direction in zip(actions, best, strict=True)}
    return mapping, {
        "candidate_scores": [score for score, _ in candidates],
        "candidate_probabilities": [float(value) for value in probabilities],
        "best_score": best_score,
        "ordered_permutations": [
            [direction.value for direction in permutation]
            for _, permutation in candidates
        ],
    }


def evaluate(
    model: Any,
    tokenizer: Any,
    game_seeds: Sequence[int],
    config: Config,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, Any]:
    directions = tuple(Direction)
    exact = 0
    action_correct = 0
    truth_ranks: list[int] = []
    truth_probabilities: list[float] = []
    games_won = 0
    levels_completed = 0
    traces: list[dict[str, Any]] = []

    for seed in game_seeds[: config.evaluation_games]:
        spec = generate_game(seed)
        prompts: list[str] = []
        completions: list[str] = []
        for action in Action:
            prompt = build_control_prompt(seed, action, mode)
            for direction in directions:
                prompts.append(prompt)
                completions.append(answer_text(direction.value.lower()))
        flat_scores = score_completions(
            model,
            tokenizer,
            prompts,
            completions,
            max_length=config.max_length,
            prefix_keep=config.prefix_keep,
            batch_size=config.score_batch_size,
            device=device,
        )
        score_matrix = [
            flat_scores[index * len(directions) : (index + 1) * len(directions)]
            for index in range(len(Action))
        ]
        selected, details = select_mapping(score_matrix)
        truth = dict(spec.action_to_direction)
        is_exact = selected == truth
        exact += int(is_exact)
        per_action = sum(selected[action] == truth[action] for action in Action)
        action_correct += per_action

        truth_permutation = [truth[action].value for action in Action]
        ordered = details["ordered_permutations"]
        truth_rank = ordered.index(truth_permutation) + 1
        truth_probability = details["candidate_probabilities"][truth_rank - 1]
        truth_ranks.append(truth_rank)
        truth_probabilities.append(truth_probability)

        truth_program = program_from_phase0_spec(spec)
        selected_program = expand_mapping(selected, truth_program)
        downstream = execute_selected_program(spec, selected_program, config.max_plan_depth)
        games_won += int(downstream["game_won"])
        levels_completed += int(downstream["levels_completed"])
        traces.append(
            {
                "game_seed": seed,
                "exact_mapping": is_exact,
                "correct_actions": per_action,
                "truth_rank": truth_rank,
                "truth_probability": truth_probability,
                "selected_mapping": {
                    action.value: selected[action].value for action in Action
                },
                "truth_mapping": {
                    action.value: truth[action].value for action in Action
                },
                "score_matrix": score_matrix,
                "downstream": downstream,
            }
        )

    games = len(game_seeds[: config.evaluation_games])
    return {
        "mode": mode,
        "games": games,
        "exact_mappings": exact,
        "exact_mapping_accuracy": exact / games,
        "per_action_accuracy": action_correct / (games * len(Action)),
        "mean_truth_rank": sum(truth_ranks) / games,
        "mean_truth_probability": sum(truth_probabilities) / games,
        "downstream_games_won": games_won,
        "downstream_game_win_rate": games_won / games,
        "downstream_levels_completed": levels_completed,
        "downstream_level_completion_rate": levels_completed / (games * 3),
        "downstream_note": (
            "Mapping-only upper bound: non-mapping DSL fields come from the "
            "known Phase-0 generator so this isolates action-semantic induction."
        ),
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
    test_seeds = sorted({int(row["game_seed"]) for row in test_rows})

    model, tokenizer = build_model_and_tokenizer(config)
    train_dataset = CompletionDataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    validation_dataset = CompletionDataset(
        validation_rows,
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
    initial_validation_loss = mean_loss(model, validation_loader, device)
    initial_full = evaluate(model, tokenizer, test_seeds, config, device, mode="full")

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()
    model.train()

    while optimizer_step < config.max_optimizer_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / config.gradient_accumulation_steps
            loss.backward()
            losses.append(float(loss.item() * config.gradient_accumulation_steps))
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step == 1 or optimizer_step % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "step": optimizer_step,
                                "max_steps": config.max_optimizer_steps,
                                "loss": sum(losses[-20:]) / min(20, len(losses)),
                                "learning_rate": scheduler.get_last_lr()[0],
                                "device": str(device),
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
                if optimizer_step >= config.max_optimizer_steps:
                    break

    final_validation_loss = mean_loss(model, validation_loader, device)
    controls = {
        mode: evaluate(model, tokenizer, test_seeds, config, device, mode=mode)
        for mode in ("full", "amnesic", "shuffled")
    }

    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    summary = {
        "status": "completed",
        "scope": (
            "Natural-language factorized Phase-0 action semantics; mapping-only "
            "downstream upper bound, not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_games": len(test_seeds),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "initial_validation_perplexity": math.exp(min(initial_validation_loss, 20)),
        "final_validation_perplexity": math.exp(min(final_validation_loss, 20)),
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
    parser.add_argument("--data-dir", default="outputs/factorized/data")
    parser.add_argument("--output-dir", default="outputs/factorized/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prefix-keep", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--evaluation-games", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-plan-depth", type=int, default=64)
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
        evaluation_games=args.evaluation_games,
        score_batch_size=args.score_batch_size,
        max_plan_depth=args.max_plan_depth,
        save_model=not args.no_save_model,
    )
    train(config)


if __name__ == "__main__":
    main()
