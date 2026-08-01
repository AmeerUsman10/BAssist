from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .protocol import SPECIAL_TOKENS
except ImportError:  # direct script execution
    from protocol import SPECIAL_TOKENS


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    data_dir: str
    output_dir: str
    seed: int = 42
    max_length: int = 768
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 80
    warmup_steps: int = 5
    freeze_first_n_blocks: int = 8
    freeze_embeddings: bool = True
    add_protocol_tokens: bool = False
    max_train_records: int | None = None
    max_validation_records: int | None = None
    gradient_clip: float = 1.0

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class CompletionDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        for index, record in enumerate(records):
            prompt = str(record["prompt"])
            completion = str(record["completion"]) + tokenizer.eos_token
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            if len(prompt_ids) + len(completion_ids) > max_length:
                raise ValueError(
                    f"Record {index} requires {len(prompt_ids) + len(completion_ids)} "
                    f"tokens, exceeding max_length={max_length}"
                )
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            self.items.append({"input_ids": input_ids, "labels": labels})
        if not self.items:
            raise ValueError("Dataset is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or "prompt" not in item or "completion" not in item:
                raise ValueError(f"Malformed record at {path}:{line_number}")
            records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def make_collator(pad_token_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
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


def evaluate_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    if not losses:
        raise ValueError("Validation loader is empty")
    return sum(losses) / len(losses)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one GPT-2 on learning histories")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    set_seed(config.seed)
    started = time.time()

    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(
        data_dir / "train.jsonl", limit=config.max_train_records
    )
    validation_records = load_jsonl(
        data_dir / "validation.jsonl", limit=config.max_validation_records
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.add_protocol_tokens:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    token for token in SPECIAL_TOKENS if token != "<PAD>"
                ],
                "pad_token": "<PAD>",
            }
        )
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CompletionDataset(
        train_records, tokenizer=tokenizer, max_length=config.max_length
    )
    validation_dataset = CompletionDataset(
        validation_records, tokenizer=tokenizer, max_length=config.max_length
    )
    collator = make_collator(tokenizer.pad_token_id)
    loader_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if config.add_protocol_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    if not 0 <= config.freeze_first_n_blocks <= len(model.transformer.h):
        raise ValueError("freeze_first_n_blocks exceeds GPT-2 depth")
    if config.freeze_embeddings:
        for parameter in model.transformer.wte.parameters():
            parameter.requires_grad = False
        for parameter in model.transformer.wpe.parameters():
            parameter.requires_grad = False
    for block in model.transformer.h[: config.freeze_first_n_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def learning_rate_multiplier(step: int) -> float:
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        remaining = max(1, config.max_optimizer_steps - config.warmup_steps)
        progress = (step - config.warmup_steps) / remaining
        return max(0.05, 1.0 - progress)

    initial_validation_loss = evaluate_loss(model, validation_loader, device)

    optimizer_step = 0
    micro_step = 0
    training_losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    while optimizer_step < config.max_optimizer_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                loss = model(**batch).loss
                scaled_loss = loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            training_losses.append(float(loss.item()))
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip
                )
                multiplier = learning_rate_multiplier(optimizer_step)
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * multiplier
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step == 1 or optimizer_step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "optimizer_step": optimizer_step,
                                "max_steps": config.max_optimizer_steps,
                                "loss": round(training_losses[-1], 6),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "device": str(device),
                            }
                        ),
                        flush=True,
                    )
                if optimizer_step >= config.max_optimizer_steps:
                    break
        model.train()

    final_validation_loss = evaluate_loss(model, validation_loader, device)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "status": "completed",
        "model_name": config.model_name,
        "device": str(device),
        "config": config.__dict__,
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "optimizer_steps": optimizer_step,
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "initial_validation_perplexity": math.exp(min(initial_validation_loss, 20)),
        "final_validation_perplexity": math.exp(min(final_validation_loss, 20)),
        "mean_training_loss": sum(training_losses) / len(training_losses),
        "elapsed_seconds": time.time() - started,
        "source_commit": os.getenv("GITHUB_SHA"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
