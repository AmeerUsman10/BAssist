"""Benchmark the Gate-C executable posterior on hidden contact mechanics."""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean
from typing import Mapping, Sequence

from .contact_world_agent import (
    UniformContactScores,
    run_contact_agent,
    summarize_contact_result,
)
from .mechanics_v2 import ContactMode
from .phase0_hidden_action import Action, Direction
from .primitive_contact_game import (
    ContactGameSpec,
    ContactStepRecord,
    PrimitiveContactGame,
    generate_contact_game,
)


class AdversarialContactScores(UniformContactScores):
    _PREFERRED = {
        Action.A1: Direction.UP,
        Action.A2: Direction.RIGHT,
        Action.A3: Direction.DOWN,
        Action.A4: Direction.LEFT,
    }

    def direction_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec, history, current_grid
        preferred = self._PREFERRED[action]
        return {
            direction: 10.0 if direction is preferred else 0.0
            for direction in Direction
        }

    def contact_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        current_grid,
    ) -> Mapping[ContactMode, float]:
        del spec, history, current_grid
        return {
            mode: 10.0 if mode is ContactMode.BLOCK else 0.0
            for mode in ContactMode
        }


def run_random(spec: ContactGameSpec, *, max_actions: int, seed: int):
    game = PrimitiveContactGame(spec)
    rng = random.Random(seed)
    levels = 0
    trace: list[str] = []
    for action_number in range(1, max_actions + 1):
        action = rng.choice(tuple(Action))
        record = game.step(action)
        trace.append(action.value)
        if record.status in {"LEVEL_WIN", "GAME_WIN"}:
            levels += 1
        if game.finished:
            return {
                "won": True,
                "levels_completed": levels,
                "actions": action_number,
                "action_trace": trace,
            }
    return {
        "won": False,
        "levels_completed": levels,
        "actions": max_actions,
        "action_trace": trace,
    }


def aggregate(results):
    return {
        "games": len(results),
        "games_won": sum(bool(result["won"]) for result in results),
        "win_rate": sum(bool(result["won"]) for result in results) / len(results),
        "levels_completed": sum(int(result["levels_completed"]) for result in results),
        "mean_actions": mean(int(result["actions"]) for result in results),
        "max_actions": max(int(result["actions"]) for result in results),
    }


def benchmark(*, games: int, seed_base: int, max_actions: int):
    if games < 1 or max_actions < 1:
        raise ValueError("games and max_actions must be positive")
    uniform_results = []
    adversarial_results = []
    random_results = []
    by_mode = {
        mode.value: {
            "games": 0,
            "uniform_wins": 0,
            "adversarial_wins": 0,
            "random_wins": 0,
            "uniform_actions": [],
        }
        for mode in ContactMode
    }
    traces = []

    for offset in range(games):
        seed = seed_base + offset
        spec = generate_contact_game(seed)
        uniform = summarize_contact_result(
            run_contact_agent(
                spec,
                UniformContactScores(),
                max_actions=max_actions,
                planner_depth=2,
            )
        )
        adversarial = summarize_contact_result(
            run_contact_agent(
                spec,
                AdversarialContactScores(),
                max_actions=max_actions,
                planner_depth=2,
            )
        )
        random_result = run_random(
            spec,
            max_actions=max_actions,
            seed=seed ^ 0xC07AC7,
        )
        uniform_results.append(uniform)
        adversarial_results.append(adversarial)
        random_results.append(random_result)

        bucket = by_mode[spec.contact_mode.value]
        bucket["games"] += 1
        bucket["uniform_wins"] += int(uniform["won"])
        bucket["adversarial_wins"] += int(adversarial["won"])
        bucket["random_wins"] += int(random_result["won"])
        bucket["uniform_actions"].append(int(uniform["actions"]))
        if offset < 8:
            traces.append(
                {
                    "game_seed": seed,
                    "contact_mode": spec.contact_mode.value,
                    "uniform": uniform,
                    "adversarial": adversarial,
                    "random": random_result,
                }
            )

    mode_summary = {
        mode: {
            "games": values["games"],
            "uniform_win_rate": values["uniform_wins"] / values["games"],
            "adversarial_win_rate": values["adversarial_wins"] / values["games"],
            "random_win_rate": values["random_wins"] / values["games"],
            "uniform_mean_actions": mean(values["uniform_actions"]),
        }
        for mode, values in by_mode.items()
        if values["games"]
    }
    return {
        "status": "completed",
        "scope": (
            "Synthetic hidden-action plus hidden-contact executable-posterior "
            "integration benchmark; not GPT-2 capability and not ARC-AGI-3 evaluation."
        ),
        "config": {
            "games": games,
            "seed_base": seed_base,
            "max_actions": max_actions,
        },
        "uniform_prior_agent": aggregate(uniform_results),
        "adversarial_prior_agent": aggregate(adversarial_results),
        "random_action_baseline": aggregate(random_results),
        "by_contact_mode": mode_summary,
        "traces": traces,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed-base", type=int, default=830_000)
    parser.add_argument("--max-actions", type=int, default=192)
    parser.add_argument("--output", default="outputs/contact_world_agent/benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        games=args.games,
        seed_base=args.seed_base,
        max_actions=args.max_actions,
    )
    from pathlib import Path

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
