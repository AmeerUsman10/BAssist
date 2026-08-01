"""Fine-tune GPT-2 to infer an executable ARC-DSL program from interaction.

Evaluation ranks all 24 valid Phase-0 movement programs by GPT-2 conditional
log-likelihood, then executes the selected program in a deterministic planner.
The finite candidate enumeration is an explicit controlled-gate oracle; later
ARC stages require grammar-constrained program generation.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .build_program_dataset import build_control_context
from .dsl import (
    Program,
    canonical_program,
    enumerate_phase0_programs,
    program_from_phase0_spec,
    shortest_plan,
)
from .phase0_hidden_action import Action, HiddenActionGame, generate_game


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/program_induction/data"
    output_dir: str = "outputs/program_induction/pretrained"
    seed: int = 42
    max_length: int = 896
    prefix_keep: int = 192
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 120
    warmup_steps: int = 12
    freeze_first_n_blocks: int = 8
    evaluation_games: int = 64
    candidate_batch_size: int = 6
    max_plan_depth: int = 64
    save_model: bool = True


class ProgramDataset(Dataset):
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
                raise ValueError("program target tokenized to an empty sequence")
            context_ids = truncate_context(
                context_ids,
                target_length=len(target_ids),
                max_length=max_length,
                prefix_keep=prefix_keep,
            )
            input_ids = context_ids + target_ids
            labels = [-100] * len(context_ids) + target_ids
            self.items.append({"input_ids": input_ids, "labels": labels})

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
            missing = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * missing)
            attention_masks.append([1] * len(item["input_ids"]) + [0] * missing)
            labels.append(item["labels"] + [-100] * missing)
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


def build_model_and_tokenizer(config: Config, special_tokens: Sequence[str]):
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
        raise ValueError("initialization must be pretrained or random")

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    blocks = model.transformer.h
    if not 0 <= config.freeze_first_n_blocks <= len(blocks):
        raise ValueError(f"freeze_first_n_blocks must be in 0..{len(blocks)}")
    for block in blocks[: config.freeze_first_n_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    # ARC tokens are represented in the tied embedding/LM-head matrix, which
    # must remain trainable even when early transformer blocks are frozen.
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True
    return model, tokenizer


def mean_loader_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    return sum(losses) / len(losses)


def _candidate_target(program: Program) -> str:
    return canonical_program(program) + "\n</PROGRAM>"


def score_candidates(
    model: Any,
    tokenizer: Any,
    context: str,
    candidates: Sequence[Program],
    *,
    max_length: int,
    prefix_keep: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Return mean target-token log likelihood for every candidate program."""

    context_ids_full = tokenizer.encode(context, add_special_tokens=False)
    encoded: list[tuple[list[int], list[int]]] = []
    for program in candidates:
        target_ids = tokenizer.encode(_candidate_target(program), add_special_tokens=False)
        context_ids = truncate_context(
            context_ids_full,
            target_length=len(target_ids),
            max_length=max_length,
            prefix_keep=prefix_keep,
        )
        encoded.append((context_ids, target_ids))

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch_items = encoded[start : start + batch_size]
            lengths = [len(context_ids) + len(target_ids) for context_ids, target_ids in batch_items]
            longest = max(lengths)
            input_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            for (context_ids, target_ids), length in zip(batch_items, lengths, strict=True):
                missing = longest - length
                input_rows.append(context_ids + target_ids + [tokenizer.pad_token_id] * missing)
                attention_rows.append([1] * length + [0] * missing)

            input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
            attention_tensor = torch.tensor(attention_rows, dtype=torch.long, device=device)
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits

            for row_index, (context_ids, target_ids) in enumerate(batch_items):
                context_length = len(context_ids)
                target_length = len(target_ids)
                start_logit = context_length - 1
                end_logit = start_logit + target_length
                target_logits = logits[row_index, start_logit:end_logit]
                targets = torch.tensor(target_ids, dtype=torch.long, device=device)
                token_log_probs = F.log_softmax(target_logits, dim=-1).gather(
                    1, targets.unsqueeze(1)
                )[:, 0]
                scores.append(float(token_log_probs.mean().item()))
    return scores


def mapping_accuracy(predicted: Program, truth: Program) -> float:
    predicted_rules = predicted.by_action
    truth_rules = truth.by_action
    correct = 0
    for action in Action:
        left = predicted_rules[action]
        right = truth_rules[action]
        correct += int((left.dy, left.dx) == (right.dy, right.dx))
    return correct / len(Action)


def execute_selected_program(spec, selected: Program, max_plan_depth: int) -> dict[str, Any]:
    game = HiddenActionGame(spec)
    actions_taken = 0

    # Reproduce the four evidence probes represented in the induction prompt.
    for action in Action:
        game.step(action)
        actions_taken += 1

    levels_completed = 0
    plans: list[list[str]] = []
    while not game.finished:
        level_before = game.level_index
        plan = shortest_plan(selected, game.frame, max_depth=max_plan_depth)
        if plan is None:
            break
        plans.append([action.value for action in plan])
        for action in plan:
            game.step(action)
            actions_taken += 1
            if game.finished or game.level_index != level_before:
                break
        if game.finished:
            levels_completed += 1
            break
        if game.level_index != level_before:
            levels_completed += 1
            continue
        # The selected program predicted a terminal path that did not complete
        # the real level. Stop rather than adding an unrelated recovery policy.
        break

    return {
        "game_won": game.finished,
        "levels_completed": levels_completed,
        "actions_taken_including_probes": actions_taken,
        "plans": plans,
    }


def evaluate_program_selection(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    config: Config,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, Any]:
    selected_rows = list(rows[: config.evaluation_games])
    exact = 0
    mapping_scores: list[float] = []
    posterior_truth: list[float] = []
    ranks: list[int] = []
    games_won = 0
    levels_completed = 0
    traces: list[dict[str, Any]] = []

    for row in selected_rows:
        seed = int(row["game_seed"])
        spec = generate_game(seed)
        truth = program_from_phase0_spec(spec)
        candidates = enumerate_phase0_programs(spec)
        context = build_control_context(seed, mode)
        scores = score_candidates(
            model,
            tokenizer,
            context,
            candidates,
            max_length=config.max_length,
            prefix_keep=config.prefix_keep,
            batch_size=config.candidate_batch_size,
            device=device,
        )
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        selected_index = order[0]
        selected = candidates[selected_index]
        truth_index = candidates.index(truth)
        rank = order.index(truth_index) + 1
        probabilities = torch.softmax(torch.tensor(scores), dim=0)

        is_exact = selected == truth
        exact += int(is_exact)
        mapping_scores.append(mapping_accuracy(selected, truth))
        posterior_truth.append(float(probabilities[truth_index].item()))
        ranks.append(rank)

        downstream = execute_selected_program(spec, selected, config.max_plan_depth)
        games_won += int(downstream["game_won"])
        levels_completed += int(downstream["levels_completed"])
        traces.append(
            {
                "game_seed": seed,
                "selected_program_sha256": selected.sha256,
                "truth_program_sha256": truth.sha256,
                "exact": is_exact,
                "mapping_accuracy": mapping_scores[-1],
                "truth_rank": rank,
                "truth_probability": posterior_truth[-1],
                "downstream": downstream,
            }
        )

    games = len(selected_rows)
    return {
        "mode": mode,
        "games": games,
        "exact_programs": exact,
        "exact_program_accuracy": exact / games,
        "mean_action_mapping_accuracy": sum(mapping_scores) / games,
        "mean_truth_rank": sum(ranks) / games,
        "mean_truth_probability": sum(posterior_truth) / games,
        "downstream_games_won": games_won,
        "downstream_game_win_rate": games_won / games,
        "downstream_levels_completed": levels_completed,
        "downstream_level_completion_rate": levels_completed / (games * 3),
        "traces": traces,
    }


def train(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = json.loads((data_dir / "special_tokens.json").read_text(encoding="utf-8"))
    train_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    test_rows = load_jsonl(data_dir / "test.jsonl")

    model, tokenizer = build_model_and_tokenizer(config, special_tokens)
    train_dataset = ProgramDataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    validation_dataset = ProgramDataset(
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
        decay = (config.max_optimizer_steps - step) / max(
            config.max_optimizer_steps - config.warmup_steps, 1
        )
        return max(0.05, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_validation_loss = mean_loader_loss(model, validation_loader, device)
    initial_selection = evaluate_program_selection(
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

    final_validation_loss = mean_loader_loss(model, validation_loader, device)
    controls = {
        mode: evaluate_program_selection(
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

    summary = {
        "status": "completed",
        "scope": "Finite-family Phase-0 ARC-DSL induction; not ARC-AGI-3 evaluation.",
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_rows),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "mean_training_loss": sum(losses) / len(losses),
        "initial_full_selection": initial_selection,
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
    parser.add_argument("--data-dir", default="outputs/program_induction/data")
    parser.add_argument("--output-dir", default="outputs/program_induction/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=896)
    parser.add_argument("--prefix-keep", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=8)
    parser.add_argument("--evaluation-games", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=6)
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
        candidate_batch_size=args.candidate_batch_size,
        max_plan_depth=args.max_plan_depth,
        save_model=not args.no_save_model,
    )
    train(config)


if __name__ == "__main__":
    main()
