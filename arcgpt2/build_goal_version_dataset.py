"""Build a set-valued latent-goal curriculum for one GPT-2 checkpoint.

The transition mechanics are held fixed and exactly executable for this gate.
The unknown variable is the terminal predicate. At every evidence prefix, the
target distribution is uniform over all bounded Goal-DSL candidates that agree
with every observed terminal and non-terminal transition.

This separates goal induction from dynamics induction before the two are joined.
It is a controlled curriculum, not an ARC-AGI-3 score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .dsl import program_from_phase0_spec
from .goal_dsl import (
    GoalProgram,
    enumerate_simple_goals,
    phase0_goal,
    replay_goal,
)
from .goal_protocol import (
    goal_candidate_text,
    goal_prompt,
    rotate_terminal_reports,
)
from .phase0_hidden_action import StepRecord, generate_game, simulate_source_history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def evidence_prefixes(records: Sequence[StepRecord]) -> tuple[int, ...]:
    """Choose uncertainty, pre-terminal, post-terminal, and full-history views."""

    if not records:
        return (0,)
    indices = {0, 1, min(4, len(records)), len(records)}
    for index, record in enumerate(records, start=1):
        if record.status in {"LEVEL_WIN", "GAME_WIN"}:
            indices.add(max(0, index - 1))
            indices.add(index)
    return tuple(sorted(index for index in indices if 0 <= index <= len(records)))


def candidate_goals(spec) -> tuple[GoalProgram, ...]:
    colors = (
        spec.palette.background,
        spec.palette.wall,
        spec.palette.agent,
        spec.palette.goal,
    )
    return enumerate_simple_goals(colors)


def consistent_goal_indices(
    mechanics,
    goals: Sequence[GoalProgram],
    records: Sequence[StepRecord],
) -> tuple[int, ...]:
    if not records:
        return tuple(range(len(goals)))
    return tuple(
        index
        for index, goal in enumerate(goals)
        if replay_goal(mechanics, goal, records).consistent
    )


def build_examples(seed: int) -> list[dict[str, object]]:
    spec = generate_game(seed)
    records = simulate_source_history(spec)
    mechanics = program_from_phase0_spec(spec)
    truth = phase0_goal(spec)
    goals = candidate_goals(spec)
    truth_index = goals.index(truth)
    initial_grid = records[0].before

    candidate_texts = [goal_candidate_text(goal) for goal in goals]
    candidate_programs = [goal.canonical_text() for goal in goals]
    candidate_hashes = [goal.sha256 for goal in goals]
    examples: list[dict[str, object]] = []

    for prefix_length in evidence_prefixes(records):
        prefix = tuple(records[:prefix_length])
        consistent = consistent_goal_indices(mechanics, goals, prefix)
        if truth_index not in consistent:
            raise RuntimeError("the ground-truth goal was eliminated by its own history")
        if not consistent:
            raise RuntimeError("goal version space became empty")
        probability = 1.0 / len(consistent)
        target_distribution = [
            probability if index in consistent else 0.0
            for index in range(len(goals))
        ]
        observed_terminals = sum(
            record.status in {"LEVEL_WIN", "GAME_WIN"} for record in prefix
        )
        examples.append(
            {
                "schema": "arcgpt2.goal_version_space.v1",
                "game_seed": seed,
                "prefix_length": prefix_length,
                "total_records": len(records),
                "observed_terminal_count": observed_terminals,
                "remaining_records": len(records) - prefix_length,
                "context": goal_prompt(initial_grid, prefix),
                "amnesic_context": goal_prompt(initial_grid, ()),
                "statusless_context": goal_prompt(
                    initial_grid,
                    prefix,
                    include_terminal_reports=False,
                ),
                "shuffled_status_context": goal_prompt(
                    initial_grid,
                    prefix,
                    displayed_terminal_reports=rotate_terminal_reports(prefix),
                ),
                "candidate_texts": candidate_texts,
                "candidate_programs": candidate_programs,
                "candidate_hashes": candidate_hashes,
                "target_distribution": target_distribution,
                "consistent_indices": list(consistent),
                "consistent_goal_count": len(consistent),
                "truth_index": truth_index,
                "truth_program": truth.canonical_text(),
                "truth_sha256": truth.sha256,
                "mechanics_sha256": mechanics.sha256,
                "held_out_records": [
                    {
                        "action": record.action.value,
                        "status": record.status,
                        "before": [list(row) for row in record.before],
                    }
                    for record in records[prefix_length:]
                ],
            }
        )
    return examples


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    data_path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        data_path,
        (example for seed in seeds for example in build_examples(seed)),
    )
    return {
        "games": len(seeds),
        "examples": count,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "file": data_path.name,
        "sha256": _sha256(data_path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_games: int,
    validation_games: int,
    test_games: int,
    seed_base: int,
) -> dict[str, object]:
    if min(train_games, validation_games, test_games) < 1:
        raise ValueError("each split requires at least one game")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = range(seed_base, seed_base + train_games)
    validation = range(train.stop, train.stop + validation_games)
    test = range(validation.stop, validation.stop + test_games)
    manifest = {
        "schema": "arcgpt2.goal_version_space.v1",
        "scope": (
            "Set-valued inference over bounded atomic Goal-DSL candidates with "
            "known transition mechanics; controlled gate, not ARC-AGI-3 evaluation."
        ),
        "representation": (
            "Ordinary GPT-2 vocabulary, exact pre-action grids, exact changed "
            "cells, exact actions, and exact terminal reports."
        ),
        "target": (
            "Uniform distribution over every Goal-DSL candidate consistent with "
            "all observed terminal and non-terminal records."
        ),
        "seed_base": seed_base,
        "splits": {
            "train": _build_split(output_dir, "train", train),
            "validation": _build_split(output_dir, "validation", validation),
            "test": _build_split(output_dir, "test", test),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/goal_version/data"))
    parser.add_argument("--train-games", type=int, default=128)
    parser.add_argument("--validation-games", type=int, default=16)
    parser.add_argument("--test-games", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=161_803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_games=args.train_games,
        validation_games=args.validation_games,
        test_games=args.test_games,
        seed_base=args.seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
