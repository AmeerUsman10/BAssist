"""GPT-2-only training and inference utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .curriculum import MotionEpisode
from .protocol import ACTION_ORDER, action_prompt, format_mapping, memory_prompt, parse_mapping


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load GPT-2's original tokenizer without adding learned tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_gpt2(
    model_name_or_path: str,
    *,
    random_init: bool = False,
    device: torch.device | None = None,
) -> PreTrainedModel:
    """Load pretrained GPT-2 or the identical randomly initialized architecture."""
    if random_init:
        config = AutoConfig.from_pretrained(model_name_or_path)
        model: PreTrainedModel = GPT2LMHeadModel(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if device is not None:
        model.to(device)
    return model


class JsonlCausalDataset(Dataset[dict[str, list[int]]]):
    """Mask prompt tokens so loss is applied only to the intended completion."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        tasks: set[str] | None = None,
    ) -> None:
        if max_length <= 8:
            raise ValueError("max_length is too small")
        self.items: list[dict[str, list[int]]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task = str(record.get("task", ""))
                if tasks is not None and task not in tasks:
                    continue
                prompt = str(record["prompt"])
                target = str(record["target"])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                if not target_ids:
                    raise ValueError(f"empty target at line {line_number}")
                # Preserve the supervised target and left-truncate old context.
                if len(target_ids) >= max_length:
                    target_ids = target_ids[: max_length - 1]
                prompt_budget = max_length - len(target_ids)
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                if not any(label != -100 for label in labels):
                    raise ValueError(f"no supervised tokens at line {line_number}")
                self.items.append({"input_ids": input_ids, "labels": labels})
        if not self.items:
            raise ValueError(f"no records loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def make_collator(pad_token_id: int):
    def collate(batch: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_rows: list[list[int]] = []
        masks: list[list[int]] = []
        label_rows: list[list[int]] = []
        for item in batch:
            input_ids = list(item["input_ids"])
            labels = list(item["labels"])
            padding = longest - len(input_ids)
            input_rows.append(input_ids + [pad_token_id] * padding)
            masks.append([1] * len(input_ids) + [0] * padding)
            label_rows.append(labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }

    return collate


def configure_trainable_parameters(
    model: PreTrainedModel, freeze_first_n_blocks: int
) -> dict[str, int]:
    """Freeze old embeddings and early GPT-2 blocks; train the final causal stack."""
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise TypeError("the model is not a standard GPT-2 architecture")
    blocks = model.transformer.h
    if not 0 <= freeze_first_n_blocks <= len(blocks):
        raise ValueError(
            f"freeze_first_n_blocks must be in [0, {len(blocks)}], got "
            f"{freeze_first_n_blocks}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = True
    if freeze_first_n_blocks > 0:
        for parameter in model.transformer.wte.parameters():
            parameter.requires_grad = False
        for parameter in model.transformer.wpe.parameters():
            parameter.requires_grad = False
        for block in blocks[:freeze_first_n_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise ValueError("no trainable parameters remain")
    return {"trainable_parameters": trainable, "total_parameters": total}


def _context_limit(model: PreTrainedModel) -> int:
    return int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", 1024)
    )


def _prepare_ids(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[-max_tokens:]


def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 48,
    device: torch.device | None = None,
) -> str:
    """Greedily continue a protocol prompt with the same GPT-2 model."""
    if device is None:
        device = next(model.parameters()).device
    context_limit = _context_limit(model)
    prompt_ids = _prepare_ids(
        tokenizer, prompt, max_tokens=max(1, context_limit - max_new_tokens)
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if was_training:
        model.train()
    new_ids = generated[0, len(prompt_ids) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def score_candidate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    candidate: str,
    *,
    device: torch.device | None = None,
) -> float:
    """Return mean conditional log-probability of a candidate continuation."""
    if device is None:
        device = next(model.parameters()).device
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        raise ValueError("candidate must tokenize to at least one token")
    context_limit = _context_limit(model)
    prompt_ids = _prepare_ids(
        tokenizer,
        prompt,
        max_tokens=max(1, context_limit - len(candidate_ids)),
    )
    all_ids = prompt_ids + candidate_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)
    if was_training:
        model.train()
    start = len(prompt_ids)
    token_scores: list[torch.Tensor] = []
    for offset, token_id in enumerate(candidate_ids):
        position = start + offset - 1
        if position < 0:
            raise ValueError("prompt must provide at least one conditioning token")
        token_scores.append(log_probs[0, position, token_id])
    return float(torch.stack(token_scores).mean().item())


def choose_action_with_gpt2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    available_actions: Iterable[str] = ACTION_ORDER,
    *,
    device: torch.device | None = None,
) -> tuple[str, dict[str, float]]:
    """Constrain the final legal choice using only GPT-2 continuation scores."""
    normalized = [str(action).strip().upper() for action in available_actions]
    if not normalized:
        raise ValueError("at least one action must be available")
    scores = {
        action: score_candidate(
            model,
            tokenizer,
            prompt,
            candidate=f" {action}",
            device=device,
        )
        for action in normalized
    }
    selected = max(normalized, key=lambda action: (scores[action], -normalized.index(action)))
    return selected, scores


def generated_memory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    episode: MotionEpisode,
    *,
    device: torch.device | None = None,
) -> tuple[str, dict[str, str], str]:
    """Ask GPT-2 to write its recurrent memory and normalize parseable fields."""
    completion = generate_completion(
        model,
        tokenizer,
        memory_prompt(episode.transitions, episode.current_grid),
        max_new_tokens=48,
        device=device,
    )
    before_close = completion.split("[[/MEMORY]]", maxsplit=1)[0]
    parsed = parse_mapping(before_close)
    normalized = format_mapping(parsed)
    return normalized, parsed, completion


def load_action_episodes(path: str | Path, limit: int | None = None) -> list[MotionEpisode]:
    episodes: list[MotionEpisode] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task") != "action":
                continue
            episode = MotionEpisode.from_dict(record["episode"])
            if episode.episode_id in seen:
                continue
            seen.add(episode.episode_id)
            episodes.append(episode)
            if limit is not None and len(episodes) >= limit:
                break
    if not episodes:
        raise ValueError(f"no action episodes found in {path}")
    return episodes


def evaluate_end_to_end(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    episodes: Sequence[MotionEpisode],
    *,
    device: torch.device | None = None,
    collect_examples: int = 8,
) -> dict[str, Any]:
    """Evaluate action choice after GPT-2 generates its own mapping memory."""
    if device is None:
        device = next(model.parameters()).device
    action_correct = 0
    oracle_memory_action_correct = 0
    exact_memory = 0
    field_correct = 0
    field_total = 0
    by_probe: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    examples: list[dict[str, Any]] = []

    for episode in episodes:
        memory_text, parsed_memory, raw_completion = generated_memory(
            model, tokenizer, episode, device=device
        )
        expected_memory = format_mapping(episode.known_mapping)
        if memory_text == expected_memory:
            exact_memory += 1
        for action in ACTION_ORDER:
            expected = episode.known_mapping.get(action, "?")
            actual = parsed_memory.get(action, "?")
            field_correct += int(actual == expected)
            field_total += 1

        pure_prompt = action_prompt(
            episode.transitions,
            episode.current_grid,
            memory=memory_text,
        )
        selected, scores = choose_action_with_gpt2(
            model, tokenizer, pure_prompt, device=device
        )
        correct = int(selected == episode.target_action)
        action_correct += correct
        by_probe[len(episode.transitions)][0] += correct
        by_probe[len(episode.transitions)][1] += 1
        by_kind[episode.decision_kind][0] += correct
        by_kind[episode.decision_kind][1] += 1

        oracle_prompt = action_prompt(
            episode.transitions,
            episode.current_grid,
            memory=expected_memory,
        )
        oracle_selected, _ = choose_action_with_gpt2(
            model, tokenizer, oracle_prompt, device=device
        )
        oracle_memory_action_correct += int(oracle_selected == episode.target_action)

        if len(examples) < collect_examples:
            examples.append(
                {
                    "episode_id": episode.episode_id,
                    "probe_count": len(episode.transitions),
                    "decision_kind": episode.decision_kind,
                    "expected_memory": expected_memory,
                    "generated_memory": memory_text,
                    "raw_memory_completion": raw_completion,
                    "target_action": episode.target_action,
                    "selected_action": selected,
                    "oracle_memory_selected_action": oracle_selected,
                    "scores": scores,
                }
            )

    count = len(episodes)
    return {
        "episodes": count,
        "random_action_accuracy": 0.25,
        "end_to_end_action_accuracy": action_correct / count,
        "oracle_memory_action_accuracy": oracle_memory_action_correct / count,
        "memory_exact_accuracy": exact_memory / count,
        "memory_field_accuracy": field_correct / field_total,
        "by_probe_count": {
            str(key): {
                "correct": values[0],
                "count": values[1],
                "accuracy": values[0] / values[1],
            }
            for key, values in sorted(by_probe.items())
        },
        "by_decision_kind": {
            key: {
                "correct": values[0],
                "count": values[1],
                "accuracy": values[0] / values[1],
            }
            for key, values in sorted(by_kind.items())
        },
        "examples": examples,
    }


def evaluate_loss(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


def train_steps(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    log_every: int = 5,
) -> list[dict[str, float | int]]:
    """Run a bounded causal-LM optimization loop and return exact step logs."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    recent_losses: list[float] = []

    while optimizer_step < max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                raw_loss = model(**batch).loss
                loss = raw_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            micro_step += 1
            recent_losses.append(float(raw_loss.detach().item()))

            if micro_step % gradient_accumulation_steps != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            mean_recent = sum(recent_losses[-gradient_accumulation_steps:]) / min(
                gradient_accumulation_steps, len(recent_losses)
            )
            record = {
                "optimizer_step": optimizer_step,
                "loss": mean_recent,
                "learning_rate": learning_rate,
            }
            history.append(record)
            if optimizer_step == 1 or optimizer_step % log_every == 0:
                print(json.dumps(record), flush=True)
            if optimizer_step >= max_steps:
                break
    return history
