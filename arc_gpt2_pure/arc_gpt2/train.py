"""Train and evaluate the pure GPT-2 ARC curriculum."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .modeling import (
    JsonlCausalDataset,
    configure_trainable_parameters,
    evaluate_end_to_end,
    evaluate_loss,
    load_action_episodes,
    load_gpt2,
    load_tokenizer,
    make_collator,
    set_seed,
    sha256_file,
    train_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-first-n-blocks", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["memory", "action", "prediction"],
        choices=["memory", "action", "prediction"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model)
    train_dataset = JsonlCausalDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        tasks=set(args.tasks),
    )
    eval_dataset = JsonlCausalDataset(
        args.eval_file,
        tokenizer,
        max_length=args.max_length,
        tasks=set(args.tasks),
    )
    generator = torch.Generator().manual_seed(args.seed)
    collator = make_collator(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    episodes = load_action_episodes(args.eval_file, limit=args.eval_episodes)

    model = load_gpt2(args.model, random_init=args.random_init, device=device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    print(
        json.dumps(
            {
                "event": "loaded",
                "device": str(device),
                "model": args.model,
                "random_init": args.random_init,
                "train_records": len(train_dataset),
                "eval_records": len(eval_dataset),
                "eval_episodes": len(episodes),
            }
        ),
        flush=True,
    )

    initial_loss = evaluate_loss(model, eval_loader, device)
    initial_behavior = evaluate_end_to_end(
        model,
        tokenizer,
        episodes,
        device=device,
    )

    parameter_summary = configure_trainable_parameters(
        model, args.freeze_first_n_blocks
    )
    model.config.use_cache = False
    training_history = train_steps(
        model,
        train_loader,
        device=device,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model.config.use_cache = True
    final_loss = evaluate_loss(model, eval_loader, device)
    final_behavior = evaluate_end_to_end(
        model,
        tokenizer,
        episodes,
        device=device,
    )

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    initial_accuracy = float(initial_behavior["end_to_end_action_accuracy"])
    final_accuracy = float(final_behavior["end_to_end_action_accuracy"])
    final_memory_accuracy = float(final_behavior["memory_field_accuracy"])
    gates = {
        "infrastructure_completed": True,
        "beats_random_action": final_accuracy > 0.25,
        "beats_untouched_gpt2": final_accuracy > initial_accuracy,
        "memory_fields_above_random": final_memory_accuracy > 0.25,
        "loss_improved": float(final_loss["loss"]) < float(initial_loss["loss"]),
    }

    summary: dict[str, Any] = {
        "status": "completed",
        "research_claim": (
            "This is a controlled remapped-action curriculum result, not an "
            "ARC-AGI-3 solve or public-game score."
        ),
        "model": args.model,
        "random_init": args.random_init,
        "device": str(device),
        "seed": args.seed,
        "configuration": vars(args),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "data": {
            "train_file": str(Path(args.train_file)),
            "train_sha256": sha256_file(args.train_file),
            "train_records": len(train_dataset),
            "eval_file": str(Path(args.eval_file)),
            "eval_sha256": sha256_file(args.eval_file),
            "eval_records": len(eval_dataset),
            "behavior_eval_episodes": len(episodes),
        },
        "parameters": parameter_summary,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_behavior": initial_behavior,
        "final_behavior": final_behavior,
        "gates": gates,
        "training_history": training_history,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
