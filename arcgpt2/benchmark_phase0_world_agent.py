"""Benchmark the executable-posterior shell on unseen hidden-action worlds.

This does not claim GPT-2 capability. It answers a prerequisite question: once
GPT-2 supplies uncertain priors over action meanings, can exact replay,
information-seeking intervention, and generic planning convert them into a
reliable closed-loop policy without game-specific action code?
"""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean
from typing import Mapping, Sequence

from .phase0_hidden_action import (
    Action,
    Direction,
    GameSpec,
    HiddenActionGame,
    StepRecord,
    generate_game,
)
from .phase0_world_agent import UniformMappingScores, run_agent, summarize_result


class FixedBiasedScores:
    """A deliberately overconfident, usually wrong prior used as a verifier test."""

    _PREFERRED = {
        Action.A1: Direction.UP,
        Action.A2: Direction.RIGHT,
        Action.A3: Direction.DOWN,
        Action.A4: Direction.LEFT,
    }

    def __init__(self, strength: float = 8.0) -> None:
        self.strength = float(strength)

    def score(
        self,
        spec: GameSpec,
        history: Sequence[StepRecord],
        query_action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec, history, current_grid
        preferred = self._PREFERRED[query_action]
        return {
            direction: self.strength if direction is preferred else 0.0
            for direction in Direction
        }


def run_random(spec: GameSpec, *, max_actions: int, seed: int) -> dict[str, object]:
    game = HiddenActionGame(spec)
    rng = random.Random(seed)
    levels = 0
    action_trace: list[str] = []
    for action_number in range(1, max_actions + 1):
        action = rng.choice(tuple(Action))
        record = game.step(action)
        action_trace.append(action.value)
        if record.status in {"LEVEL_WIN", "GAME_WIN"}:
            levels += 1
        if game.finished:
            return {
                "won": True,
                "levels_completed": levels,
                "actions": action_number,
                "action_trace": action_trace,
            }
    return {
        "won": False,
        "levels_completed": levels,
        "actions": max_actions,
        "action_trace": action_trace,
    }


def aggregate(results: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "games": len(results),
        "games_won": sum(bool(result["won"]) for result in results),
        "win_rate": sum(bool(result["won"]) for result in results) / len(results),
        "levels_completed": sum(int(result["levels_completed"]) for result in results),
        "mean_actions": mean(int(result["actions"]) for result in results),
        "max_actions": max(int(result["actions"]) for result in results),
    }


def benchmark(*, games: int, seed_base: int, max_actions: int) -> dict[str, object]:
    if games < 1 or max_actions < 1:
        raise ValueError("games and max_actions must be positive")
    uniform_results: list[dict[str, object]] = []
    biased_results: list[dict[str, object]] = []
    random_results: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    for offset in range(games):
        seed = seed_base + offset
        spec = generate_game(seed)
        uniform = run_agent(spec, UniformMappingScores(), max_actions=max_actions)
        biased = run_agent(spec, FixedBiasedScores(), max_actions=max_actions)
        random_result = run_random(
            spec,
            max_actions=max_actions,
            seed=seed ^ 0xBADC0DE,
        )
        uniform_summary = summarize_result(uniform)
        biased_summary = summarize_result(biased)
        uniform_results.append(uniform_summary)
        biased_results.append(biased_summary)
        random_results.append(random_result)
        if offset < 8:
            traces.append(
                {
                    "game_seed": seed,
                    "uniform": uniform_summary,
                    "biased": biased_summary,
                    "random": random_result,
                }
            )

    return {
        "status": "completed",
        "scope": (
            "Synthetic hidden-action executable-posterior integration benchmark; "
            "not GPT-2 capability and not ARC-AGI-3 evaluation."
        ),
        "config": {
            "games": games,
            "seed_base": seed_base,
            "max_actions": max_actions,
        },
        "uniform_prior_agent": aggregate(uniform_results),
        "adversarial_biased_prior_agent": aggregate(biased_results),
        "random_action_baseline": aggregate(random_results),
        "traces": traces,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=730_000)
    parser.add_argument("--max-actions", type=int, default=96)
    parser.add_argument("--output", default="outputs/phase0_world_agent/benchmark.json")
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
