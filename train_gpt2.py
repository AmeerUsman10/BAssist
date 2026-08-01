from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    data_path: str = "data/train.jsonl"
    output_dir: str = "outputs/gpt2-tuned"
    seed: int = 42
    block_size: int = 128
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_optimizer_steps: int = 12
    repeat_training_records: int = 8
    validation_records: int = 2
    freeze_first_n_blocks: int = 10


CFG = Config()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def read_jsonl(path: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} is not an object")
            prompt = str(item.get("prompt", "")).strip()
            response = str(item.get("response", "")).strip()
            if not prompt or not response:
                raise ValueError(f"Line {line_number} needs prompt and response")
            records.append({"prompt": prompt, "response": response})
    if len(records) < 4:
        raise ValueError("At least four records are required")
    return records


def prompt_text(prompt: str) -> str:
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


class PromptResponseDataset(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def encode_records(
    records: list[dict[str, str]], tokenizer: Any, block_size: int
) -> list[dict[str, list[int]]]:
    encoded: list[dict[str, list[int]]] = []
    for record in records:
        prefix = prompt_text(record["prompt"])
        suffix = record["response"] + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        input_ids = (prefix_ids + suffix_ids)[:block_size]
        labels = ([-100] * len(prefix_ids) + suffix_ids)[:block_size]
        if input_ids and any(label != -100 for label in labels):
            encoded.append({"input_ids": input_ids, "labels": labels})
    if not encoded:
        raise ValueError("No records survived tokenization")
    return encoded


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


def evaluate(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    if not losses:
        raise ValueError("Validation loader is empty")
    return sum(losses) / len(losses)


def generate(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> str:
    model.eval()
    inputs = tokenizer(prompt_text(prompt), return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    started = time.time()
    set_determinism(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = read_jsonl(CFG.data_path)
    validation_records = records[-CFG.validation_records :]
    training_records = records[: -CFG.validation_records]
    training_records = training_records * CFG.repeat_training_records

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = encode_records(training_records, tokenizer, CFG.block_size)
    validation_rows = encode_records(validation_records, tokenizer, CFG.block_size)
    collate = make_collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        PromptResponseDataset(train_rows),
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        PromptResponseDataset(validation_rows),
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(CFG.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    base_sample = generate(
        model,
        tokenizer,
        "How should an uncertain result be reported?",
        device,
    )
    initial_validation_loss = evaluate(model, validation_loader, device)

    if CFG.freeze_first_n_blocks > len(model.transformer.h):
        raise ValueError("freeze_first_n_blocks exceeds model depth")
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = False
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = False
    for block in model.transformer.h[: CFG.freeze_first_n_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []

    while optimizer_step < CFG.max_optimizer_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / CFG.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            losses.append(float(loss.item() * CFG.gradient_accumulation_steps))

            if micro_step % CFG.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                print(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_step,
                            "loss": round(losses[-1], 6),
                            "device": str(device),
                        }
                    ),
                    flush=True,
                )
                if optimizer_step >= CFG.max_optimizer_steps:
                    break

    final_validation_loss = evaluate(model, validation_loader, device)
    tuned_sample = generate(
        model,
        tokenizer,
        "How should an uncertain result be reported?",
        device,
    )

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "status": "completed",
        "device": str(device),
        "base_model": CFG.model_name,
        "config": asdict(CFG),
        "source_records": len(records),
        "training_examples_after_repeat": len(train_rows),
        "validation_examples": len(validation_rows),
        "optimizer_steps": optimizer_step,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "initial_perplexity": math.exp(min(initial_validation_loss, 20)),
        "final_perplexity": math.exp(min(final_validation_loss, 20)),
        "mean_training_loss": sum(losses) / len(losses),
        "elapsed_seconds": time.time() - started,
        "base_sample": base_sample,
        "tuned_sample": tuned_sample,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "sample_comparison.txt").write_text(
        "BASE MODEL\n==========\n"
        + base_sample
        + "\n\nTUNED MODEL\n===========\n"
        + tuned_sample
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
