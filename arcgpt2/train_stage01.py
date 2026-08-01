"""Train and evaluate GPT-2 with Stage-0.1 set-valued supervision.

The learned actor remains one standard GPT-2 causal LM. Offline source code is
used only to construct supervision and score held-out behavior. At inference the
policy receives an exact transcript and chooses among four legal action tokens
using its own next-token logits.
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
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .codec import encode_frame, tokens_to_text
from .phase0_hidden_action import (
    ACTION_TOKEN,
    Action,
    HiddenActionGame,
    SourceLearner,
    append_record_tokens,
    generate_game,
    initial_transcript,
)
from .stage01_hidden_action import valid_actions


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    initialization: str
    data_dir: str
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
    evaluation_games: int
    evaluation_seed_start: int
    max_actions_per_game: int
    history_modes: tuple[str, ...]
    save_model: bool


ACTION_LIST = tuple(Action)
ACTION_INDEX = {ACTION_TOKEN[action]: index for index, action in enumerate(ACTION_LIST)}
PHASE_INDEX = {"probe": 0, "navigate": 1}
PHASE_NAME = {value: key for key, value in PHASE_INDEX.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


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
            required = {
                "context",
                "target",
                "valid_targets",
                "decision_phase",
                "level_index",
            }
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"invalid Stage-0.1 row at {path}:{line_number}")
            rows.append(item)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


class SetValuedActionDataset(Dataset):
    """Contexts paired with a mask over every equally valid next action."""

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
            context_ids = tokenizer.encode(str(row["context"]), add_special_tokens=False)
            context_ids = truncate_ids(
                context_ids,
                budget=max_length,
                prefix_keep=prefix_keep,
            )
            if not context_ids:
                raise ValueError("context tokenized to an empty sequence")

            target = str(row["target"])
            if target not in ACTION_INDEX:
                raise ValueError(f"unknown canonical target: {target}")
            valid_targets = tuple(str(value) for value in row["valid_targets"])
            if not valid_targets or any(value not in ACTION_INDEX for value in valid_targets):
                raise ValueError(f"invalid valid target set: {valid_targets}")
            if target not in valid_targets:
                raise ValueError("canonical target must belong to valid target set")

            phase = str(row["decision_phase"])
            if phase not in PHASE_INDEX:
                raise ValueError(f"unknown decision phase: {phase}")
            valid_mask = [0] * len(ACTION_LIST)
            for value in valid_targets:
                valid_mask[ACTION_INDEX[value]] = 1

            self.items.append(
                {
                    "input_ids": context_ids,
                    "valid_action_mask": valid_mask,
                    "canonical_target_index": ACTION_INDEX[target],
                    "phase_id": PHASE_INDEX[phase],
                    "level_index": int(row["level_index"]),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def make_collator(pad_token_id: int):
    def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        valid_masks: list[list[int]] = []
        canonical_targets: list[int] = []
        phases: list[int] = []
        levels: list[int] = []

        for item in batch:
            values = list(item["input_ids"])
            padding = longest - len(values)
            input_ids.append(values + [pad_token_id] * padding)
            attention_masks.append([1] * len(values) + [0] * padding)
            valid_masks.append(list(item["valid_action_mask"]))
            canonical_targets.append(int(item["canonical_target_index"]))
            phases.append(int(item["phase_id"]))
            levels.append(int(item["level_index"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "valid_action_mask": torch.tensor(valid_masks, dtype=torch.bool),
            "canonical_target_index": torch.tensor(canonical_targets, dtype=torch.long),
            "phase_id": torch.tensor(phases, dtype=torch.long),
            "level_index": torch.tensor(levels, dtype=torch.long),
        }

    return collate


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

    # The LM head is tied to the token embedding matrix. It must remain
    # trainable because the ARC action and codec tokens were newly appended.
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
            raise ValueError(f"action token is not atomic: {token} -> {ids}")
        action_token_ids[action] = ids[0]
    return model, tokenizer, action_token_ids


def next_action_logits(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    action_token_ids: Mapping[Action, int],
) -> torch.Tensor:
    """Return Bx4 logits for the token immediately after each context."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    final_positions = batch["attention_mask"].sum(dim=1) - 1
    row_indices = torch.arange(
        batch["input_ids"].shape[0],
        device=batch["input_ids"].device,
    )
    vocabulary_logits = outputs.logits[row_indices, final_positions]
    candidate_ids = torch.tensor(
        [action_token_ids[action] for action in ACTION_LIST],
        dtype=torch.long,
        device=vocabulary_logits.device,
    )
    return vocabulary_logits.index_select(dim=1, index=candidate_ids)


def set_valued_action_loss(
    action_logits: torch.Tensor,
    valid_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability mass assigned to the complete valid set."""
    if action_logits.ndim != 2 or action_logits.shape[1] != len(ACTION_LIST):
        raise ValueError("action_logits must have shape [batch, 4]")
    if valid_action_mask.shape != action_logits.shape:
        raise ValueError("valid_action_mask shape must match action_logits")
    if valid_action_mask.dtype is not torch.bool:
        valid_action_mask = valid_action_mask.bool()
    if not bool(valid_action_mask.any(dim=1).all()):
        raise ValueError("every example requires at least one valid action")

    all_log_mass = torch.logsumexp(action_logits, dim=1)
    valid_logits = action_logits.masked_fill(~valid_action_mask, float("-inf"))
    valid_log_mass = torch.logsumexp(valid_logits, dim=1)
    return (all_log_mass - valid_log_mass).mean()


def _normalized_count_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def classification_metrics(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    action_token_ids: Mapping[Action, int],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    set_correct = 0
    canonical_correct = 0
    predicted_counts = [0] * len(ACTION_LIST)
    canonical_counts = [0] * len(ACTION_LIST)
    valid_support_counts = [0] * len(ACTION_LIST)
    confusion = [[0] * len(ACTION_LIST) for _ in ACTION_LIST]
    navigation_confusion = [[0] * len(ACTION_LIST) for _ in ACTION_LIST]
    probability_entropies: list[float] = []
    phase_stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "set_correct": 0,
            "canonical_correct": 0,
            "predicted_counts": [0] * len(ACTION_LIST),
            "valid_cardinality_total": 0,
        }
    )
    level_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for raw_batch in loader:
            batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
            }
            logits = next_action_logits(model, batch, action_token_ids)
            loss = set_valued_action_loss(logits, batch["valid_action_mask"])
            batch_size = logits.shape[0]
            total_loss += float(loss.item()) * batch_size
            total += batch_size

            probabilities = F.softmax(logits, dim=1)
            log_probabilities = F.log_softmax(logits, dim=1)
            entropies = -(
                probabilities * log_probabilities
            ).sum(dim=1) / math.log(len(ACTION_LIST))
            probability_entropies.extend(float(value) for value in entropies.cpu())

            predictions = torch.argmax(logits, dim=1)
            valid = batch["valid_action_mask"]
            canonical = batch["canonical_target_index"]
            phases = batch["phase_id"]
            levels = batch["level_index"]

            for row in range(batch_size):
                prediction = int(predictions[row].item())
                target = int(canonical[row].item())
                phase = int(phases[row].item())
                level = int(levels[row].item())
                valid_row = valid[row]
                is_valid = bool(valid_row[prediction].item())
                is_canonical = prediction == target

                set_correct += int(is_valid)
                canonical_correct += int(is_canonical)
                predicted_counts[prediction] += 1
                canonical_counts[target] += 1
                confusion[target][prediction] += 1
                for action_index in range(len(ACTION_LIST)):
                    valid_support_counts[action_index] += int(
                        valid_row[action_index].item()
                    )

                phase_entry = phase_stats[phase]
                phase_entry["count"] += 1
                phase_entry["set_correct"] += int(is_valid)
                phase_entry["canonical_correct"] += int(is_canonical)
                phase_entry["predicted_counts"][prediction] += 1
                phase_entry["valid_cardinality_total"] += int(valid_row.sum().item())

                level_stats[level][0] += int(is_valid)
                level_stats[level][1] += 1
                if phase == PHASE_INDEX["navigate"]:
                    navigation_confusion[target][prediction] += 1

    action_names = [action.value for action in ACTION_LIST]
    phase_output: dict[str, Any] = {}
    for phase_id, values in sorted(phase_stats.items()):
        count = int(values["count"])
        phase_output[PHASE_NAME[phase_id]] = {
            "count": count,
            "set_accuracy": values["set_correct"] / max(count, 1),
            "canonical_accuracy": values["canonical_correct"] / max(count, 1),
            "predicted_action_counts": dict(
                zip(action_names, values["predicted_counts"], strict=True)
            ),
            "prediction_entropy": _normalized_count_entropy(
                values["predicted_counts"]
            ),
            "mean_valid_set_size": values["valid_cardinality_total"] / max(count, 1),
        }

    return {
        "set_mass_loss": total_loss / max(total, 1),
        "set_accuracy": set_correct / max(total, 1),
        "canonical_accuracy": canonical_correct / max(total, 1),
        "examples": total,
        "predicted_action_counts": dict(
            zip(action_names, predicted_counts, strict=True)
        ),
        "canonical_target_counts": dict(
            zip(action_names, canonical_counts, strict=True)
        ),
        "valid_target_support_counts": dict(
            zip(action_names, valid_support_counts, strict=True)
        ),
        "prediction_distribution_entropy": _normalized_count_entropy(
            predicted_counts
        ),
        "mean_probability_entropy": (
            sum(probability_entropies) / max(len(probability_entropies), 1)
        ),
        "dominant_action_share": max(predicted_counts) / max(total, 1),
        "confusion_matrix_rows_target_columns_prediction": {
            action_names[row]: dict(
                zip(action_names, confusion[row], strict=True)
            )
            for row in range(len(ACTION_LIST))
        },
        "navigation_confusion_matrix": {
            action_names[row]: dict(
                zip(action_names, navigation_confusion[row], strict=True)
            )
            for row in range(len(ACTION_LIST))
        },
        "by_phase": phase_output,
        "by_level": {
            str(level): {
                "correct": values[0],
                "count": values[1],
                "set_accuracy": values[0] / max(values[1], 1),
            }
            for level, values in sorted(level_stats.items())
        },
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
            *encode_frame(game.frame),
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
            corrupted.append(
                rotation.get(token, token) if inside_historical_action else token
            )
        return [*corrupted, "<DECIDE>"]
    raise ValueError(f"unknown history mode: {mode}")


def choose_action(
    model: Any,
    tokenizer: Any,
    action_token_ids: Mapping[Action, int],
    transcript: Sequence[str],
    game: HiddenActionGame,
    *,
    history_mode: str,
    max_length: int,
    prefix_keep: int,
    device: torch.device,
) -> tuple[Action, dict[str, float]]:
    tokens = inference_context_tokens(transcript, mode=history_mode, game=game)
    ids = tokenizer.encode(tokens_to_text(tokens), add_special_tokens=False)
    ids = truncate_ids(ids, budget=max_length, prefix_keep=prefix_keep)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        vocabulary_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[0, -1]
    candidate_ids = torch.tensor(
        [action_token_ids[action] for action in ACTION_LIST],
        device=device,
    )
    candidate_logits = vocabulary_logits[candidate_ids]
    probabilities = F.softmax(candidate_logits, dim=0)
    selected_index = int(torch.argmax(candidate_logits).item())
    return ACTION_LIST[selected_index], {
        action.value: float(probabilities[index].item())
        for index, action in enumerate(ACTION_LIST)
    }


def _longest_run(values: Sequence[str]) -> int:
    longest = 0
    current = 0
    previous: str | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        longest = max(longest, current)
    return longest


def evaluate_closed_loop(
    model: Any,
    tokenizer: Any,
    action_token_ids: Mapping[Action, int],
    config: TrainConfig,
    device: torch.device,
    *,
    history_mode: str,
) -> dict[str, Any]:
    games_won = 0
    levels_completed = 0
    total_actions = 0
    phase_correct = Counter()
    phase_total = Counter()
    predicted_counts = Counter()
    mapping_completed_games = 0
    mapping_action_counts: list[int] = []
    repeated_transitions = 0
    total_adjacent_pairs = 0
    longest_repeated_run = 0
    level_action_counts: dict[int, list[int]] = defaultdict(list)
    traces: list[dict[str, Any]] = []

    for offset in range(config.evaluation_games):
        game_seed = config.evaluation_seed_start + offset
        spec = generate_game(game_seed)
        game = HiddenActionGame(spec)
        metric_learner = SourceLearner(spec)
        transcript = initial_transcript(spec)
        action_trace: list[str] = []
        detailed_trace: list[dict[str, Any]] = []
        completed_this_game = 0
        actions_this_level = 0
        mapping_completed_at: int | None = None

        for action_number in range(1, config.max_actions_per_game + 1):
            valid_set, phase = valid_actions(metric_learner, game)
            selected, probabilities = choose_action(
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
            is_valid = selected in valid_set
            phase_total[phase] += 1
            phase_correct[phase] += int(is_valid)
            predicted_counts[selected.value] += 1

            record = game.step(selected)
            metric_learner.observe(record)
            if (
                mapping_completed_at is None
                and len(metric_learner.direction_to_action) == len(Action)
            ):
                mapping_completed_at = action_number
                mapping_completed_games += 1
                mapping_action_counts.append(action_number)

            total_actions += 1
            actions_this_level += 1
            if action_trace:
                total_adjacent_pairs += 1
                repeated_transitions += int(action_trace[-1] == selected.value)
            action_trace.append(selected.value)

            next_frame = game.frame if record.status == "LEVEL_WIN" else None
            append_record_tokens(transcript, record, next_frame=next_frame)
            detailed_trace.append(
                {
                    "action_number": action_number,
                    "level_index": record.level_index,
                    "phase": phase,
                    "selected": selected.value,
                    "valid": [action.value for action in valid_set],
                    "selection_was_valid": is_valid,
                    "moved": record.moved,
                    "status": record.status,
                    "probabilities": probabilities,
                    "mapping_known": len(metric_learner.direction_to_action),
                }
            )

            if record.status in {"LEVEL_WIN", "GAME_WIN"}:
                completed_this_game += 1
                levels_completed += 1
                level_action_counts[record.level_index].append(actions_this_level)
                actions_this_level = 0
            if record.status == "GAME_WIN":
                games_won += 1
                break

        longest_repeated_run = max(longest_repeated_run, _longest_run(action_trace))
        traces.append(
            {
                "game_seed": game_seed,
                "won": game.finished,
                "levels_completed": completed_this_game,
                "actions": len(action_trace),
                "mapping_completed_at_action": mapping_completed_at,
                "action_trace": action_trace,
                "decisions": detailed_trace,
            }
        )

    action_names = [action.value for action in ACTION_LIST]
    counts = [predicted_counts[name] for name in action_names]
    return {
        "history_mode": history_mode,
        "games": config.evaluation_games,
        "games_won": games_won,
        "game_win_rate": games_won / config.evaluation_games,
        "levels_completed": levels_completed,
        "level_completion_rate": levels_completed / (config.evaluation_games * 3),
        "total_actions": total_actions,
        "mean_actions_per_game": total_actions / config.evaluation_games,
        "mapping_completed_games": mapping_completed_games,
        "mapping_completion_rate": mapping_completed_games / config.evaluation_games,
        "mean_actions_to_mapping": (
            sum(mapping_action_counts) / len(mapping_action_counts)
            if mapping_action_counts
            else None
        ),
        "probe_valid_action_accuracy": (
            phase_correct["probe"] / max(phase_total["probe"], 1)
        ),
        "navigation_oracle_action_accuracy": (
            phase_correct["navigate"] / max(phase_total["navigate"], 1)
        ),
        "phase_counts": {
            phase: {
                "correct": phase_correct[phase],
                "count": phase_total[phase],
                "accuracy": phase_correct[phase] / max(phase_total[phase], 1),
            }
            for phase in ("probe", "navigate")
        },
        "predicted_action_counts": dict(
            zip(action_names, counts, strict=True)
        ),
        "prediction_distribution_entropy": _normalized_count_entropy(counts),
        "dominant_action_share": max(counts) / max(sum(counts), 1),
        "repeated_action_transition_rate": (
            repeated_transitions / max(total_adjacent_pairs, 1)
        ),
        "longest_repeated_action_run": longest_repeated_run,
        "mean_actions_by_completed_level": {
            str(level): sum(values) / len(values)
            for level, values in sorted(level_action_counts.items())
            if values
        },
        "traces": traces,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = json.loads(
        (data_dir / "special_tokens.json").read_text(encoding="utf-8")
    )
    train_rows = load_jsonl(Path(config.train_file))
    evaluation_rows = load_jsonl(Path(config.evaluation_file))

    model, tokenizer, action_token_ids = build_model_and_tokenizer(
        config,
        special_tokens,
    )
    train_dataset = SetValuedActionDataset(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    evaluation_dataset = SetValuedActionDataset(
        evaluation_rows,
        tokenizer,
        max_length=config.max_length,
        prefix_keep=config.prefix_keep,
    )
    collator = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
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

    def learning_rate_multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        remaining = config.max_optimizer_steps - step
        decay_span = max(config.max_optimizer_steps - config.warmup_steps, 1)
        return max(0.05, remaining / decay_span)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )

    initial_metrics = classification_metrics(
        model,
        evaluation_loader,
        device,
        action_token_ids,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    raw_losses: list[float] = []
    step_history: list[dict[str, Any]] = []
    started = time.time()

    while optimizer_step < config.max_optimizer_steps:
        for raw_batch in train_loader:
            batch = {
                key: value.to(device)
                for key, value in raw_batch.items()
            }
            logits = next_action_logits(model, batch, action_token_ids)
            raw_loss = set_valued_action_loss(
                logits,
                batch["valid_action_mask"],
            )
            loss = raw_loss / config.gradient_accumulation_steps
            loss.backward()
            raw_losses.append(float(raw_loss.detach().item()))
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            record = {
                "optimizer_step": optimizer_step,
                "loss": sum(raw_losses[-config.gradient_accumulation_steps :])
                / min(config.gradient_accumulation_steps, len(raw_losses)),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            step_history.append(record)
            if optimizer_step == 1 or optimizer_step % 10 == 0:
                print(json.dumps(record), flush=True)
            if optimizer_step >= config.max_optimizer_steps:
                break

    final_metrics = classification_metrics(
        model,
        evaluation_loader,
        device,
        action_token_ids,
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
        for mode in config.history_modes
    }

    intact = closed_loop.get("intact", {})
    navigation_metrics = final_metrics.get("by_phase", {}).get("navigate", {})
    gates = {
        "teacher_set_accuracy_at_least_0_98": (
            float(final_metrics["set_accuracy"]) >= 0.98
        ),
        "teacher_navigation_accuracy_at_least_0_95": (
            float(navigation_metrics.get("set_accuracy", 0.0)) >= 0.95
        ),
        "intact_game_win_rate_at_least_0_90": (
            float(intact.get("game_win_rate", 0.0)) >= 0.90
        ),
        "intact_mapping_completion_at_least_0_90": (
            float(intact.get("mapping_completion_rate", 0.0)) >= 0.90
        ),
        "intact_probe_accuracy_at_least_0_95": (
            float(intact.get("probe_valid_action_accuracy", 0.0)) >= 0.95
        ),
        "no_single_action_collapse": (
            float(intact.get("dominant_action_share", 1.0)) < 0.80
        ),
    }
    gates["overfit_sanity_passed"] = all(gates.values())

    if config.save_model:
        model.save_pretrained(output_dir / "model", safe_serialization=True)
        tokenizer.save_pretrained(output_dir / "model")

    summary: dict[str, Any] = {
        "status": "completed",
        "claim_scope": (
            "Stage-0.1 synthetic hidden-action overfit/transfer gate; not an "
            "ARC-AGI-3 public or private score"
        ),
        "objective": (
            "negative log probability mass on every valid action; singleton "
            "navigation targets"
        ),
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "evaluation_examples": len(evaluation_dataset),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "mean_training_loss": sum(raw_losses) / max(len(raw_losses), 1),
        "initial_classification": initial_metrics,
        "final_classification": final_metrics,
        "closed_loop": closed_loop,
        "gates": gates,
        "step_history": step_history,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument(
        "--initialization",
        choices=("pretrained", "random"),
        default="pretrained",
    )
    parser.add_argument("--data-dir", default="outputs/stage01/data")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--evaluation-file", required=True)
    parser.add_argument("--output-dir", default="outputs/stage01/pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--prefix-keep", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-optimizer-steps", type=int, default=160)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=8)
    parser.add_argument("--evaluation-seed-start", type=int, default=1729)
    parser.add_argument("--max-actions-per-game", type=int, default=64)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=("intact", "amnesic", "shuffled"),
        default=["intact"],
    )
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        initialization=args.initialization,
        data_dir=args.data_dir,
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
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        max_actions_per_game=args.max_actions_per_game,
        history_modes=tuple(args.history_modes),
        save_model=args.save_model,
    )
    train(config)


if __name__ == "__main__":
    main()
