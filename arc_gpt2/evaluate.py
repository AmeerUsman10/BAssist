from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

try:
    from .protocol import (
        build_prompt,
        encode_delta,
        extract_action,
        extract_memory,
        initial_memory,
    )
    from .world import HiddenRuleGrid, generate_episode
except ImportError:  # direct script execution
    from protocol import build_prompt, encode_delta, extract_action, extract_memory, initial_memory
    from world import HiddenRuleGrid, generate_episode


class StopAfterAction(StoppingCriteria):
    def __init__(self, tokenizer: Any, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = start_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        generated = self.tokenizer.decode(
            input_ids[0, self.start_length :], skip_special_tokens=False
        )
        return "</ACT>" in generated or "<END>" in generated


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def sequence_log_probability(
    model: Any,
    tokenizer: Any,
    context: str,
    candidate: str,
    device: torch.device,
) -> float:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        return float("-inf")
    max_positions = int(getattr(model.config, "n_positions", 1024))
    available_context = max_positions - len(candidate_ids)
    context_ids = context_ids[-available_context:]
    input_ids = torch.tensor([context_ids + candidate_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    start = len(context_ids) - 1
    total = 0.0
    for offset, token_id in enumerate(candidate_ids):
        position = start + offset
        if position < 0:
            return float("-inf")
        total += float(log_probs[position, token_id].item())
    return total


def choose_by_probability(
    model: Any,
    tokenizer: Any,
    context: str,
    legal_actions: Sequence[int],
    device: torch.device,
) -> int:
    scored = [
        (
            sequence_log_probability(
                model, tokenizer, context, f"<A{action}>", device
            ),
            action,
        )
        for action in legal_actions
    ]
    scored.sort(reverse=True)
    return scored[0][1]


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    max_positions = int(getattr(model.config, "n_positions", 1024))
    if input_ids.shape[1] + max_new_tokens > max_positions:
        keep = max_positions - max_new_tokens
        input_ids = input_ids[:, -keep:]
        attention_mask = attention_mask[:, -keep:]
    stopping = StoppingCriteriaList(
        [StopAfterAction(tokenizer, start_length=input_ids.shape[1])]
    )
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    return tokenizer.decode(
        generated[0, input_ids.shape[1] :], skip_special_tokens=False
    )


def teacher_forced_action_accuracy(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    correct = 0
    predictions: list[dict[str, Any]] = []
    for record in records:
        completion = str(record["completion"])
        before_action, separator, _ = completion.partition("<ACT>")
        if not separator:
            continue
        context = str(record["prompt"]) + before_action + "<ACT>"
        target = int(record["metadata"]["action"])
        predicted = choose_by_probability(
            model, tokenizer, context, (1, 2, 3, 4), device
        )
        correct += int(predicted == target)
        predictions.append(
            {
                "seed": record["metadata"]["seed"],
                "step": record["metadata"]["step"],
                "target": target,
                "predicted": predicted,
            }
        )
    total = len(predictions)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "sample": predictions[:12],
    }


def play_model_episode(
    model: Any,
    tokenizer: Any,
    seed: int,
    device: torch.device,
    max_steps: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    environment = HiddenRuleGrid.generate(seed)
    memory = initial_memory(environment.legal_actions)
    previous_action: int | None = None
    previous_grid: list[list[int]] | None = None
    trace: list[dict[str, Any]] = []
    valid_memory_count = 0
    parsed_action_count = 0

    for step_index in range(max_steps):
        grid = environment.render()
        delta = encode_delta(previous_grid, grid) if previous_grid is not None else None
        prompt = build_prompt(
            memory=memory,
            grid=grid,
            legal_actions=environment.legal_actions,
            state="RUN",
            previous_action=previous_action,
            previous_delta=delta,
        )
        completion = generate_completion(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        generated_memory = extract_memory(completion)
        if generated_memory is not None:
            memory = generated_memory
            valid_memory_count += 1
        action = extract_action(completion, environment.legal_actions)
        if action is not None:
            parsed_action_count += 1
        else:
            fallback_context = prompt + completion + "<ACT>"
            action = choose_by_probability(
                model,
                tokenizer,
                fallback_context,
                environment.legal_actions,
                device,
            )

        before_grid = grid
        transition = environment.step(action)
        trace.append(
            {
                "step": step_index,
                "action": action,
                "moved": transition.moved,
                "won": transition.won,
                "completion": completion[:1200],
                "memory": memory,
            }
        )
        previous_action = action
        previous_grid = before_grid
        if transition.won:
            break

    return {
        "seed": seed,
        "won": environment.won,
        "actions": len(trace),
        "valid_memory_rate": valid_memory_count / len(trace) if trace else 0.0,
        "parsed_action_rate": parsed_action_count / len(trace) if trace else 0.0,
        "trace": trace,
    }


def play_random_episode(seed: int, max_steps: int) -> dict[str, Any]:
    environment = HiddenRuleGrid.generate(seed)
    rng = random.Random(seed ^ 0x55AA55AA)
    actions = 0
    while actions < max_steps and not environment.won:
        environment.step(rng.choice(environment.legal_actions))
        actions += 1
    return {"seed": seed, "won": environment.won, "actions": actions}


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [episode for episode in episodes if episode["won"]]
    return {
        "episodes": len(episodes),
        "wins": len(wins),
        "win_rate": len(wins) / len(episodes) if episodes else 0.0,
        "mean_actions_all": (
            sum(int(episode["actions"]) for episode in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "mean_actions_on_wins": (
            sum(int(episode["actions"]) for episode in wins) / len(wins)
            if wins
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pure GPT-2 Stage-0 agent")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-forced-records", type=int, default=48)
    parser.add_argument("--closed-loop-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    test_records = load_jsonl(
        args.data_dir / "test.jsonl", limit=args.teacher_forced_records
    )
    teacher_forced = teacher_forced_action_accuracy(
        model, tokenizer, test_records, device
    )

    manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    start_seed = int(manifest["splits"]["test"]["start_seed"])
    seeds = list(range(start_seed, start_seed + args.closed_loop_episodes))

    model_episodes = [
        play_model_episode(
            model,
            tokenizer,
            seed,
            device,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
        for seed in seeds
    ]
    random_episodes = [play_random_episode(seed, args.max_steps) for seed in seeds]
    oracle_episodes = []
    for seed in seeds:
        records = generate_episode(seed, max_steps=args.max_steps)
        completed = bool(records and records[-1]["metadata"]["completed"])
        oracle_episodes.append(
            {"seed": seed, "won": completed, "actions": len(records)}
        )

    report = {
        "status": "completed",
        "device": str(device),
        "model_dir": str(args.model_dir),
        "teacher_forced_action": teacher_forced,
        "closed_loop_model": summarize_episodes(model_episodes),
        "closed_loop_random": summarize_episodes(random_episodes),
        "oracle_trace": summarize_episodes(oracle_episodes),
        "model_episodes": model_episodes,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
