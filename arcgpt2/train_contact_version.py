"""Train one GPT-2 checkpoint to infer hidden contact primitives.

Each candidate is an ordinary-language rendering of a typed executable
ContactMode. Training uses a set-valued objective over every mode retained by
exact replay. Contextual calibration subtracts each completion's general
language prior, reducing preference for shorter or more familiar descriptions.
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

from .completion_scorer import score_with_contextual_calibration


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    initialization: str = "pretrained"
    data_dir: str = "outputs/contact/data"
    output_dir: str = "outputs/contact/pretrained"
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
    candidate_batch_size: int = 5
    evaluation_groups: int = 12
    save_model: bool = True


@dataclass(frozen=True)
class EncodedContactItem:
    group_seed: int
    mode_variant: str
    prefix_length: int
    prompt_ids: tuple[int, ...]
    null_prompt_ids: tuple[int, ...]
    precontact_prompt_ids: tuple[int, ...]
    shuffled_prompt_ids: tuple[int, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    target_probabilities: tuple[float, ...]
    truth_index: int
    consistent_indices: tuple[int, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "counterfactual_group_seed",
        "mode_variant",
        "prefix_length",
        "context",
        "amnesic_context",
        "precontact_context",
        "shuffled_contact_context",
        "candidate_texts",
        "target_distribution",
        "truth_index",
        "consistent_indices",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required.issubset(row):
                missing = sorted(required.difference(row if isinstance(row, dict) else {}))
                raise ValueError(f"invalid contact row at {path}:{line_number}; missing {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"contact dataset is empty: {path}")
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
        raise ValueError("contact candidate leaves insufficient prompt budget")
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
) -> EncodedContactItem:
    candidates = tuple(
        tuple(tokenizer.encode(str(text), add_special_tokens=False))
        for text in row["candidate_texts"]
    )
    if len(candidates) != 5 or any(not candidate for candidate in candidates):
        raise ValueError("contact rows require five non-empty candidate completions")
    longest = max(len(candidate) for candidate in candidates)

    def prompt(field: str) -> tuple[int, ...]:
        ids = tokenizer.encode(str(row[field]), add_special_tokens=False)
        return truncate_prompt(
            ids,
            longest_candidate=longest,
            max_length=max_length,
            prefix_keep=prefix_keep,
        )

    target = tuple(float(value) for value in row["target_distribution"])
    if len(target) != len(candidates) or any(value < 0.0 for value in target):
        raise ValueError("invalid contact target distribution")
    if not math.isclose(sum(target), 1.0, abs_tol=1e-9):
        raise ValueError("contact target distribution must sum to one")
    consistent = tuple(int(value) for value in row["consistent_indices"])
    if set(consistent) != {
        index for index, probability in enumerate(target) if probability > 0.0
    }:
        raise ValueError("contact support does not match non-zero target mass")
    truth_index = int(row["truth_index"])
    if truth_index not in consistent:
        raise ValueError("contact truth was eliminated by exact replay")
    return EncodedContactItem(
        group_seed=int(row["counterfactual_group_seed"]),
        mode_variant=str(row["mode_variant"]),
        prefix_length=int(row["prefix_length"]),
        prompt_ids=prompt("context"),
        null_prompt_ids=prompt("amnesic_context"),
        precontact_prompt_ids=prompt("precontact_context"),
        shuffled_prompt_ids=prompt("shuffled_contact_context"),
        candidate_ids=candidates,
        target_probabilities=target,
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
    item: EncodedContactItem,
    prompt_ids: Sequence[int],
    *,
    device: torch.device,
    config: Config,
) -> torch.Tensor:
    return score_with_contextual_calibration(
        model,
        prompt_ids,
        item.null_prompt_ids,
        item.candidate_ids,
        pad_token_id=int(model.config.pad_token_id),
        device=device,
        candidate_batch_size=config.candidate_batch_size,
        reduction="mean",
    )


def set_loss(scores: torch.Tensor, item: EncodedContactItem) -> torch.Tensor:
    target = torch.tensor(
        item.target_probabilities,
        dtype=scores.dtype,
        device=scores.device,
    )
    return -(target * F.log_softmax(scores, dim=-1)).sum()


def select_groups(
    items: Sequence[EncodedContactItem],
    *,
    group_limit: int,
) -> tuple[EncodedContactItem, ...]:
    seeds = sorted({item.group_seed for item in items})[:group_limit]
    return tuple(item for item in items if item.group_seed in seeds)


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0.0]
    return float((-(positive * torch.log2(positive))).sum().item())


def evaluate(
    model: Any,
    items: Sequence[EncodedContactItem],
    *,
    mode: str,
    device: torch.device,
    config: Config,
) -> dict[str, Any]:
    prompt_attribute = {
        "full": "prompt_ids",
        "amnesic": "null_prompt_ids",
        "precontact": "precontact_prompt_ids",
        "shuffled_contact": "shuffled_prompt_ids",
    }.get(mode)
    if prompt_attribute is None:
        raise ValueError("unknown contact evidence mode")

    aggregates: dict[int, dict[str, float]] = {}
    traces: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for item in items:
            scores = candidate_scores(
                model,
                item,
                getattr(item, prompt_attribute),
                device=device,
                config=config,
            )
            probabilities = torch.softmax(scores, dim=-1)
            target = torch.tensor(
                item.target_probabilities,
                dtype=probabilities.dtype,
                device=device,
            )
            order = torch.argsort(probabilities, descending=True).tolist()
            selected = int(torch.argmax(probabilities).item())
            values = aggregates.setdefault(
                item.prefix_length,
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
            values["consistent_mass"] += float(
                probabilities[list(item.consistent_indices)].sum().item()
            )
            values["map_consistent"] += float(selected in item.consistent_indices)
            values["truth_probability"] += float(probabilities[item.truth_index].item())
            values["truth_rank"] += float(order.index(item.truth_index) + 1)
            values["entropy_bits"] += _entropy_bits(probabilities)
            values["set_cross_entropy"] += float(
                (-(target * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
            )
            values["brier"] += float(((probabilities - target) ** 2).sum().item())
            traces.append(
                {
                    "group_seed": item.group_seed,
                    "mode_variant": item.mode_variant,
                    "prefix_length": item.prefix_length,
                    "evidence_mode": mode,
                    "consistent_mass": float(
                        probabilities[list(item.consistent_indices)].sum().item()
                    ),
                    "map_consistent": selected in item.consistent_indices,
                    "truth_probability": float(probabilities[item.truth_index].item()),
                    "truth_rank": order.index(item.truth_index) + 1,
                    "entropy_bits": _entropy_bits(probabilities),
                }
            )

    by_prefix = {
        str(prefix): {
            key: value / values["rows"]
            for key, value in values.items()
            if key != "rows"
        }
        | {"rows": values["rows"]}
        for prefix, values in sorted(aggregates.items())
    }
    return {"mode": mode, "by_prefix_length": by_prefix, "traces": traces}


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
    evaluation_items = select_groups(
        test_items,
        group_limit=config.evaluation_groups,
    )
    validation_selection = select_groups(
        validation_items,
        group_limit=config.evaluation_groups,
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
        device=device,
        config=config,
    )
    initial_test = evaluate(
        model,
        evaluation_items,
        mode="full",
        device=device,
        config=config,
    )

    rng = random.Random(config.seed ^ 0xC07A_C7)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()
    model.train()
    while optimizer_step < config.max_optimizer_steps:
        item = rng.choice(train_items)
        scores = candidate_scores(
            model,
            item,
            item.prompt_ids,
            device=device,
            config=config,
        )
        raw_loss = set_loss(scores, item)
        (raw_loss / config.gradient_accumulation_steps).backward()
        losses.append(float(raw_loss.detach().item()))
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
        device=device,
        config=config,
    )
    controls = {
        mode: evaluate(
            model,
            evaluation_items,
            mode=mode,
            device=device,
            config=config,
        )
        for mode in ("full", "amnesic", "precontact", "shuffled_contact")
    }

    if config.save_model:
        model_path = output_dir / "model"
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)

    summary = {
        "status": "completed",
        "scope": (
            "Set-valued hidden contact-mechanics inference with exact replay "
            "targets; controlled Gate C, not ARC-AGI-3 evaluation."
        ),
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_items),
        "validation_examples": len(validation_items),
        "test_examples": len(test_items),
        "evaluation_examples": len(evaluation_items),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_training_loss": sum(losses) / len(losses),
        "initial_validation": initial_validation,
        "initial_test": initial_test,
        "final_validation": final_validation,
        "controls": controls,
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
    parser.add_argument("--data-dir", default="outputs/contact/data")
    parser.add_argument("--output-dir", default="outputs/contact/pretrained")
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
    parser.add_argument("--candidate-batch-size", type=int, default=5)
    parser.add_argument("--evaluation-groups", type=int, default=12)
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
            evaluation_groups=args.evaluation_groups,
            save_model=not args.no_save_model,
        )
    )


if __name__ == "__main__":
    main()
