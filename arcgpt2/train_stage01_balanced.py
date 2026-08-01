"""Class-balanced Stage-0.1 training entry point.

The set-valued objective fixes arbitrary probe labels, but the navigation traces
can still contain many more examples for one action than another. This module
keeps the same single-GPT-2 inference policy while balancing the *offline*
training stream:

* one half of expected sampling mass goes to probe decisions;
* one half goes to navigation decisions;
* navigation mass is divided equally among A1..A4;
* examples are sampled with replacement, deterministically from the run seed.

No weighting or sampler is present at inference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .train_stage01 import (
    ACTION_LIST,
    PHASE_INDEX,
    SetValuedActionDataset,
    TrainConfig,
    build_model_and_tokenizer,
    classification_metrics,
    evaluate_closed_loop,
    load_jsonl,
    make_collator,
    next_action_logits,
    set_seed,
    set_valued_action_loss,
)


def balanced_sampling_weights(
    dataset: SetValuedActionDataset,
) -> tuple[list[float], dict[str, Any]]:
    """Assign equal expected mass to probe and each navigation action class."""
    probe_count = 0
    navigation_counts = Counter()
    for item in dataset.items:
        if int(item["phase_id"]) == PHASE_INDEX["probe"]:
            probe_count += 1
        else:
            navigation_counts[int(item["canonical_target_index"])] += 1

    if probe_count == 0:
        raise ValueError("balanced Stage-0.1 training requires probe examples")
    if any(navigation_counts[index] == 0 for index in range(len(ACTION_LIST))):
        raise ValueError("every navigation action class requires at least one example")

    probe_mass = 0.5
    navigation_mass_per_action = 0.5 / len(ACTION_LIST)
    weights: list[float] = []
    for item in dataset.items:
        if int(item["phase_id"]) == PHASE_INDEX["probe"]:
            weights.append(probe_mass / probe_count)
        else:
            target = int(item["canonical_target_index"])
            weights.append(
                navigation_mass_per_action / navigation_counts[target]
            )

    expected_probe_mass = sum(
        weight
        for weight, item in zip(weights, dataset.items, strict=True)
        if int(item["phase_id"]) == PHASE_INDEX["probe"]
    )
    expected_navigation_mass = {
        ACTION_LIST[index].value: sum(
            weight
            for weight, item in zip(weights, dataset.items, strict=True)
            if int(item["phase_id"]) == PHASE_INDEX["navigate"]
            and int(item["canonical_target_index"]) == index
        )
        for index in range(len(ACTION_LIST))
    }
    summary = {
        "probe_examples": probe_count,
        "navigation_examples_by_action": {
            ACTION_LIST[index].value: navigation_counts[index]
            for index in range(len(ACTION_LIST))
        },
        "expected_probe_sampling_mass": expected_probe_mass,
        "expected_navigation_sampling_mass_by_action": expected_navigation_mass,
        "total_sampling_mass": sum(weights),
    }
    return weights, summary


def train_balanced(config: TrainConfig) -> dict[str, Any]:
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
    # Encoding can legitimately inspect a transcript longer than GPT-2's final
    # context because deterministic truncation happens immediately afterwards.
    tokenizer.model_max_length = 1_000_000

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
    weights, sampling_summary = balanced_sampling_weights(train_dataset)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        collate_fn=collator,
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
    sampled_phase_counts = Counter()
    sampled_navigation_counts = Counter()
    step_history: list[dict[str, Any]] = []
    started = time.time()

    while optimizer_step < config.max_optimizer_steps:
        for raw_batch in train_loader:
            phase_ids = raw_batch["phase_id"]
            canonical = raw_batch["canonical_target_index"]
            for row in range(len(phase_ids)):
                phase = int(phase_ids[row].item())
                sampled_phase_counts[phase] += 1
                if phase == PHASE_INDEX["navigate"]:
                    sampled_navigation_counts[int(canonical[row].item())] += 1

            batch = {key: value.to(device) for key, value in raw_batch.items()}
            logits = next_action_logits(model, batch, action_token_ids)
            raw_loss = set_valued_action_loss(
                logits,
                batch["valid_action_mask"],
            )
            (raw_loss / config.gradient_accumulation_steps).backward()
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

    sampled_total = sum(sampled_phase_counts.values())
    summary: dict[str, Any] = {
        "status": "completed",
        "claim_scope": (
            "Stage-0.1 synthetic hidden-action overfit gate; not an ARC-AGI-3 "
            "public or private score"
        ),
        "objective": (
            "set-valued action probability mass with phase/action-balanced "
            "offline sampling"
        ),
        "config": asdict(config),
        "device": str(device),
        "train_examples": len(train_dataset),
        "evaluation_examples": len(evaluation_dataset),
        "sampling_design": sampling_summary,
        "realized_sample_counts": {
            "probe": sampled_phase_counts[PHASE_INDEX["probe"]],
            "navigate": sampled_phase_counts[PHASE_INDEX["navigate"]],
            "navigation_by_action": {
                ACTION_LIST[index].value: sampled_navigation_counts[index]
                for index in range(len(ACTION_LIST))
            },
            "total": sampled_total,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
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
    parser.add_argument("--max-optimizer-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=8)
    parser.add_argument("--evaluation-seed-start", type=int, default=1729)
    parser.add_argument("--max-actions-per-game", type=int, default=48)
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
    train_balanced(config)


if __name__ == "__main__":
    main()
