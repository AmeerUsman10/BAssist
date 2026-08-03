"""Bounded, evidence-first QLoRA adaptation for Instella action semantics.

This module is prepared but must not be launched until the corrected frozen
replication has completed. It adapts only ``q_proj`` and ``o_proj`` LoRA matrices
on the DPO checkpoint, keeps the router and every expert frozen, and evaluates
whole held-out seeds before and after training with the mode-correct calibrated
replication metric.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
from typing import Any, Iterable, Sequence

from .action_replication import run_replication
from .action_training_data import (
    ActionTrainingExample,
    build_action_training_examples,
    manifest as data_manifest,
)
from .backend import TransformersBackend
from .catalog import CHECKPOINTS
from .kaggle_runner import gpu_inventory, load_with_fallbacks


ALLOWED_LORA_SUFFIXES = ("q_proj", "o_proj")
EXPECTED_RANK8_TRAINABLE_PARAMETERS = 1_769_472


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-50:],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_token_ids(backend: TransformersBackend) -> tuple[int, ...]:
    from arcgpt2.natural_protocol import answer_text

    ids: list[int] = []
    for word in ("north", "south", "west", "east"):
        encoded = backend.tokenizer.encode(
            answer_text(word), add_special_tokens=False
        )
        if len(encoded) != 1:
            raise RuntimeError(
                f"candidate {word!r} must be one tokenizer token, got {encoded}"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 4:
        raise RuntimeError("direction candidates must map to distinct tokens")
    return tuple(ids)


def prompt_ids(
    backend: TransformersBackend,
    example: ActionTrainingExample,
    *,
    max_context_tokens: int,
) -> list[int]:
    rendered = backend.render(example.prompt)
    ids = backend.tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) > max_context_tokens:
        prefix = min(256, max_context_tokens // 4)
        ids = ids[:prefix] + ids[-(max_context_tokens - prefix) :]
    if len(ids) < 2:
        raise RuntimeError("training prompt tokenized to fewer than two tokens")
    return ids


def trainable_inventory(model: Any) -> dict[str, Any]:
    names: list[str] = []
    parameters = 0
    invalid: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        parameters += int(parameter.numel())
        lowered = name.lower()
        if "lora_" not in lowered or not any(
            suffix in lowered for suffix in ALLOWED_LORA_SUFFIXES
        ):
            invalid.append(name)
    return {
        "trainable_parameters": parameters,
        "trainable_tensor_count": len(names),
        "trainable_names": names,
        "invalid_trainable_names": invalid,
        "router_or_expert_trainable": any(
            "router" in name.lower()
            or ".gate." in name.lower()
            or ".experts." in name.lower()
            or "shared_expert" in name.lower()
            for name in names
        ),
    }


def prepare_qlora_model(
    backend: TransformersBackend,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> dict[str, Any]:
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError("PEFT is required for QLoRA adaptation") from exc

    model = backend.model
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(ALLOWED_LORA_SUFFIXES),
    )
    model = get_peft_model(model, config)
    model.train()
    backend.model = model
    inventory = trainable_inventory(model)
    if inventory["invalid_trainable_names"]:
        raise RuntimeError(
            "unexpected trainable parameters escaped the q/o LoRA boundary: "
            + ", ".join(inventory["invalid_trainable_names"][:20])
        )
    if inventory["router_or_expert_trainable"]:
        raise RuntimeError("router or expert parameters became trainable")
    if rank == 8 and inventory["trainable_parameters"] != EXPECTED_RANK8_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "rank-8 trainable parameter count changed: "
            f"expected {EXPECTED_RANK8_TRAINABLE_PARAMETERS}, "
            f"got {inventory['trainable_parameters']}"
        )
    return {
        "lora_config": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": list(ALLOWED_LORA_SUFFIXES),
        },
        "inventory": inventory,
    }


def example_loss(
    backend: TransformersBackend,
    example: ActionTrainingExample,
    *,
    candidate_ids: Sequence[int],
    max_context_tokens: int,
):
    import torch
    from torch.nn import functional as F

    ids = prompt_ids(
        backend,
        example,
        max_context_tokens=max_context_tokens,
    )
    device = backend.input_device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = backend.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = output.logits[0, -1]
    candidate_tensor = torch.tensor(
        list(candidate_ids), dtype=torch.long, device=logits.device
    )
    selected = logits.index_select(0, candidate_tensor)
    log_probabilities = F.log_softmax(selected.float(), dim=0)
    target = torch.tensor(
        example.target_distribution,
        dtype=log_probabilities.dtype,
        device=log_probabilities.device,
    )
    loss = -(target * log_probabilities).sum()
    probabilities = log_probabilities.detach().exp().cpu().tolist()
    return loss, tuple(float(value) for value in probabilities), len(ids)


def evaluate_examples(
    backend: TransformersBackend,
    examples: Sequence[ActionTrainingExample],
    *,
    candidate_ids: Sequence[int],
    max_context_tokens: int,
) -> dict[str, Any]:
    import torch

    backend.model.eval()
    losses: list[float] = []
    allowed_mass: list[float] = []
    top1_consistent: list[float] = []
    prompt_tokens: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for example in examples:
            loss, probabilities, token_count = example_loss(
                backend,
                example,
                candidate_ids=candidate_ids,
                max_context_tokens=max_context_tokens,
            )
            losses.append(float(loss.item()))
            allowed_indices = [
                example.candidate_words.index(word)
                for word in example.allowed_words
            ]
            allowed_mass.append(
                sum(probabilities[index] for index in allowed_indices)
            )
            top1 = max(range(len(probabilities)), key=probabilities.__getitem__)
            top1_consistent.append(1.0 if top1 in allowed_indices else 0.0)
            prompt_tokens.append(token_count)
    elapsed = time.perf_counter() - started
    backend.model.train()
    return {
        "examples": len(examples),
        "mean_loss": statistics.fmean(losses),
        "mean_allowed_mass": statistics.fmean(allowed_mass),
        "top1_consistent_accuracy": statistics.fmean(top1_consistent),
        "mean_prompt_tokens": statistics.fmean(prompt_tokens),
        "max_prompt_tokens": max(prompt_tokens),
        "elapsed_seconds": elapsed,
    }


def train(
    backend: TransformersBackend,
    examples: Sequence[ActionTrainingExample],
    *,
    candidate_ids: Sequence[int],
    epochs: int,
    learning_rate: float,
    gradient_accumulation: int,
    warmup_ratio: float,
    max_grad_norm: float,
    max_context_tokens: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from transformers import get_linear_schedule_with_warmup

    trainable = [
        parameter
        for parameter in backend.model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("QLoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(examples) / gradient_accumulation
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    generator = random.Random(seed)
    history: list[dict[str, Any]] = []
    global_example_step = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    for epoch in range(epochs):
        order = list(range(len(examples)))
        generator.shuffle(order)
        epoch_losses: list[float] = []
        epoch_tokens: list[int] = []
        for position, index in enumerate(order, start=1):
            example = examples[index]
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                loss, _, token_count = example_loss(
                    backend,
                    example,
                    candidate_ids=candidate_ids,
                    max_context_tokens=max_context_tokens,
                )
                scaled_loss = loss / gradient_accumulation
            scaler.scale(scaled_loss).backward()
            epoch_losses.append(float(loss.detach().item()))
            epoch_tokens.append(token_count)
            global_example_step += 1

            should_step = (
                position % gradient_accumulation == 0
                or position == len(order)
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = float(
                    clip_grad_norm_(trainable, max_grad_norm).item()
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                if optimizer_step % 10 == 0 or optimizer_step == total_optimizer_steps:
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "optimizer_step": optimizer_step,
                            "global_example_step": global_example_step,
                            "recent_mean_loss": statistics.fmean(
                                epoch_losses[-min(64, len(epoch_losses)) :]
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "gradient_norm": gradient_norm,
                        }
                    )

        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "global_example_step": global_example_step,
                "epoch_mean_loss": statistics.fmean(epoch_losses),
                "epoch_mean_prompt_tokens": statistics.fmean(epoch_tokens),
            }
        )

    return {
        "epochs": epochs,
        "examples": len(examples),
        "gradient_accumulation": gradient_accumulation,
        "optimizer_steps": optimizer_step,
        "total_optimizer_steps_planned": total_optimizer_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def delta_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_metrics = before["summary"]["metrics"]
    after_metrics = after["summary"]["metrics"]
    keys = (
        "calibrated_intact_truth_probability",
        "calibrated_shuffled_mode_truth_probability",
        "calibrated_corruption_preference_margin",
        "calibrated_intact_top1_accuracy",
        "calibrated_shuffled_follows_corruption_top1_accuracy",
    )
    return {
        key: (
            after_metrics[key]["mean"] - before_metrics[key]["mean"]
        )
        for key in keys
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=("dpo", "base"), default="dpo")
    parser.add_argument("--train-seed-base", type=int, default=960_000)
    parser.add_argument("--train-games", type=int, default=32)
    parser.add_argument("--validation-games", type=int, default=8)
    parser.add_argument("--test-games", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "qlora_status.json"
    report_path = args.output_dir / "qlora_report.json"
    adapter_dir = args.output_dir / "adapter"
    spec = CHECKPOINTS[args.checkpoint]
    status: dict[str, Any] = {
        "schema": "instella_arc.action_qlora_status.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checkpoint": {
            "repository_id": spec.repository_id,
            "revision": spec.revision,
        },
        "gpu_inventory": gpu_inventory(),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        seed_everything(args.seed)
        train_start = args.train_seed_base
        validation_start = train_start + args.train_games
        test_start = validation_start + args.validation_games
        train_examples = build_action_training_examples(
            range(train_start, validation_start)
        )
        validation_examples = build_action_training_examples(
            range(validation_start, test_start)
        )
        test_seeds = range(test_start, test_start + args.test_games)

        backend, load_attempts = load_with_fallbacks(
            checkpoint=args.checkpoint,
            quantizations=["int4", "int8"],
            max_context_tokens=args.max_context_tokens,
            allow_fp16_offload=False,
        )
        status["load_attempts"] = load_attempts
        tokens = candidate_token_ids(backend)

        before_test = run_replication(
            backend,
            seed_base=test_start,
            games=args.test_games,
        )
        before_validation = evaluate_examples(
            backend,
            validation_examples,
            candidate_ids=tokens,
            max_context_tokens=args.max_context_tokens,
        )

        qlora_setup = prepare_qlora_model(
            backend,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        training = train(
            backend,
            train_examples,
            candidate_ids=tokens,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation=args.gradient_accumulation,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            max_context_tokens=args.max_context_tokens,
            seed=args.seed,
        )
        after_validation = evaluate_examples(
            backend,
            validation_examples,
            candidate_ids=tokens,
            max_context_tokens=args.max_context_tokens,
        )
        after_test = run_replication(
            backend,
            seed_base=test_start,
            games=args.test_games,
        )

        adapter_dir.mkdir(parents=True, exist_ok=True)
        backend.model.save_pretrained(adapter_dir, safe_serialization=True)
        backend.tokenizer.save_pretrained(adapter_dir)
        adapter_files = sorted(
            path for path in adapter_dir.rglob("*") if path.is_file()
        )
        adapter_hashes = {
            str(path.relative_to(adapter_dir)): _sha256(path)
            for path in adapter_files
        }
        deltas = delta_summary(before_test, after_test)
        post_gates = after_test["summary"]["gates"]
        promotion = {
            "held_out_action_gate_passed": bool(
                post_gates["promote_to_frozen_goal_and_mechanics"]
            ),
            "intact_probability_improved": (
                deltas["calibrated_intact_truth_probability"] > 0.0
            ),
            "corruption_following_improved": (
                deltas["calibrated_corruption_preference_margin"] > 0.0
            ),
        }
        promotion["promote_adapter_to_goal_and_mechanics"] = all(
            promotion.values()
        )
        promotion["authorize_arc_submission"] = False

        report = {
            "schema": "instella_arc.action_qlora.v1",
            "checkpoint": {
                "repository_id": spec.repository_id,
                "revision": spec.revision,
                "quantization": backend.metadata.get("quantization"),
            },
            "split": {
                "train_seed_start": train_start,
                "train_games": args.train_games,
                "validation_seed_start": validation_start,
                "validation_games": args.validation_games,
                "test_seed_start": test_start,
                "test_games": args.test_games,
                "whole_seed_disjoint": True,
            },
            "training_data": data_manifest(train_examples),
            "validation_data": data_manifest(validation_examples),
            "qlora_setup": qlora_setup,
            "candidate_token_ids": list(tokens),
            "before_validation": before_validation,
            "after_validation": after_validation,
            "training": training,
            "before_test": before_test,
            "after_test": after_test,
            "held_out_deltas": deltas,
            "promotion": promotion,
            "adapter_files": [
                str(path.relative_to(adapter_dir)) for path in adapter_files
            ],
            "adapter_sha256": adapter_hashes,
            "scope": (
                "Bounded synthetic hidden-action QLoRA gate. Not ARC-AGI-3 "
                "evaluation and not authority to submit externally."
            ),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        status.update(
            status="success",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            report=str(report_path),
            report_sha256=_sha256(report_path),
            promotion=promotion,
        )
    except Exception as exc:
        status.update(
            status="failure",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=_error(exc),
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
