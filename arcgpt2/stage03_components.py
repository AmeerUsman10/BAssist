"""Stage 0.3 component curriculum for a single pure GPT-2 agent.

Stage 0.2 asked one GPT-2 call to find a relevant transition inside a long
history, decode its spatial displacement, bind that displacement to an action,
read the current goal direction, and compose the result.  This module isolates
those operations without adding another learned component:

* ``mapping`` means raw-transition perception: before/after grids -> direction;
* ``need`` means current-grid perception: mover/goal grid -> useful direction;
* ``compose`` means natural action map + useful direction -> action;
* ``direct`` combines an explicit GPT-2-readable memory with the current grid.

At eventual closed-loop inference the same GPT-2 performs each operation.  The
harness may copy GPT-2's own direction word next to the action that produced the
transition, but it may not compute the direction, infer a goal, plan, or choose
an action itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .phase0_hidden_action import Action, Direction, DIRECTION_DELTA
from .stage02_decomposed import (
    ACTION_CANDIDATES,
    ACTION_DIGIT,
    MAP_CANDIDATES,
    NEED_CANDIDATES,
    all_action_mappings,
    normalize_grid,
)
from .stage02_natural import ACTION_WORD, DIRECTION_WORD, mapping_summary
from .stage02_sparse import sparse_grid_text

Grid = tuple[tuple[int, ...], ...]

DIRECTION_CANONICAL = {
    Direction.UP: "N",
    Direction.RIGHT: "E",
    Direction.DOWN: "S",
    Direction.LEFT: "W",
}
CANONICAL_DIRECTION = {value: key for key, value in DIRECTION_CANONICAL.items()}

COMPONENT_HEADER = (
    "Exact grid. Format HxW;bV;cells means every unlisted cell has value V; "
    "listed cells are exact rROWcCOLUMN=VALUE entries. Values: 0 empty, "
    "1 wall, 2 mover, 3 goal. Row numbers increase south and column numbers "
    "increase east."
)


def render_grid(
    height: int,
    width: int,
    mover: tuple[int, int],
    goal: tuple[int, int],
) -> Grid:
    if mover == goal:
        raise ValueError("mover and goal must differ")
    values = [[0 for _ in range(width)] for _ in range(height)]
    values[goal[0]][goal[1]] = 3
    values[mover[0]][mover[1]] = 2
    return normalize_grid(values)


def transition_prompt(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    action: Action,
) -> str:
    digit = ACTION_DIGIT[action]
    return (
        COMPONENT_HEADER
        + f"\nBefore: {sparse_grid_text(before)}."
        + f"\nAction: {ACTION_WORD[digit]}."
        + f"\nAfter: {sparse_grid_text(after)}."
        + "\nQuestion: In which direction did the mover change position? "
          "Answer exactly north, east, south, or west.\nAnswer:"
    )


def need_prompt(grid: Sequence[Sequence[int]]) -> str:
    return (
        COMPONENT_HEADER
        + f"\nCurrent: {sparse_grid_text(grid)}."
        + "\nQuestion: Which direction reduces the larger absolute row or "
          "column distance from mover to goal? Use vertical when the absolute "
          "distances tie. Answer exactly north, east, south, or west.\nAnswer:"
    )


def compose_prompt_natural(
    mapping: Mapping[Action, Direction],
    needed_direction: Direction,
) -> str:
    canonical = {
        action: DIRECTION_CANONICAL[direction]
        for action, direction in mapping.items()
    }
    return (
        "Action meanings: "
        + mapping_summary(canonical)
        + f". Needed direction: {DIRECTION_WORD[DIRECTION_CANONICAL[needed_direction]]}. "
          "Choose the action whose meaning equals the needed direction. Answer "
          "exactly one, two, three, or four.\nAnswer:"
    )


def direct_prompt_natural(
    mapping: Mapping[Action, Direction],
    grid: Sequence[Sequence[int]],
) -> str:
    canonical = {
        action: DIRECTION_CANONICAL[direction]
        for action, direction in mapping.items()
    }
    return (
        COMPONENT_HEADER
        + "\nMemory: "
        + mapping_summary(canonical)
        + f".\nCurrent: {sparse_grid_text(grid)}."
        + "\nQuestion: First determine the direction that reduces the larger "
          "absolute row or column distance from mover to goal, using vertical "
          "on a tie. Then choose the action with that meaning. Answer exactly "
          "one, two, three, or four.\nAnswer:"
    )


def needed_direction(
    mover: tuple[int, int], goal: tuple[int, int]
) -> Direction:
    row_delta = goal[0] - mover[0]
    column_delta = goal[1] - mover[1]
    if row_delta == 0 and column_delta == 0:
        raise ValueError("mover already occupies goal")
    if row_delta != 0 and abs(row_delta) >= abs(column_delta):
        return Direction.DOWN if row_delta > 0 else Direction.UP
    return Direction.RIGHT if column_delta > 0 else Direction.LEFT


def action_for_direction(
    mapping: Mapping[Action, Direction], direction: Direction
) -> Action:
    return next(action for action, value in mapping.items() if value == direction)


def _safe_transition_positions(
    rng: random.Random,
    direction: Direction,
    *,
    height: int,
    width: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    row_delta, column_delta = DIRECTION_DELTA[direction]
    while True:
        before = (rng.randrange(1, height - 1), rng.randrange(1, width - 1))
        after = (before[0] + row_delta, before[1] + column_delta)
        goal = (rng.randrange(height), rng.randrange(width))
        if goal not in {before, after}:
            return before, after, goal


def _goal_offset(variant: int, seed: int) -> tuple[int, int]:
    offsets = (
        (-3, 0), (3, 0), (0, -3), (0, 3),
        (-3, 1), (3, -1), (-1, -3), (1, 3),
        (-2, 2), (2, 2), (2, -2), (-2, -2),
    )
    return offsets[(seed * 5 + variant) % len(offsets)]


def rows_for_game(game_seed: int, *, variants: int) -> list[dict[str, Any]]:
    mappings = all_action_mappings()
    mapping = mappings[game_seed % len(mappings)]
    rows: list[dict[str, Any]] = []
    for variant in range(variants):
        rng = random.Random(game_seed * 100_003 + variant * 997 + 17)
        metadata_base = {
            "game_seed": game_seed,
            "variant": variant,
            "mapping": {
                ACTION_DIGIT[action]: DIRECTION_CANONICAL[direction]
                for action, direction in mapping.items()
            },
        }

        for action in Action:
            direction = mapping[action]
            before_position, after_position, distractor_goal = _safe_transition_positions(
                rng, direction, height=9, width=9
            )
            before = render_grid(9, 9, before_position, distractor_goal)
            after = render_grid(9, 9, after_position, distractor_goal)
            rows.append(
                {
                    "task": "mapping",
                    "task_detail": f"perceive_{ACTION_DIGIT[action]}",
                    "prompt": transition_prompt(before, after, action),
                    "target": DIRECTION_CANONICAL[direction],
                    "candidates": ["N", "E", "S", "W"],
                    "metadata": {
                        **metadata_base,
                        "decision_phase": "perception",
                        "before_mover": list(before_position),
                        "after_mover": list(after_position),
                    },
                }
            )

        center = (4, 4)
        offset = _goal_offset(variant, game_seed)
        goal = (center[0] + offset[0], center[1] + offset[1])
        current = render_grid(9, 9, center, goal)
        need = needed_direction(center, goal)
        target_action = action_for_direction(mapping, need)
        rows.append(
            {
                "task": "need",
                "task_detail": "spatial_need",
                "prompt": need_prompt(current),
                "target": DIRECTION_CANONICAL[need],
                "candidates": ["N", "E", "S", "W"],
                "metadata": {
                    **metadata_base,
                    "decision_phase": "need",
                    "mover": list(center),
                    "goal": list(goal),
                },
            }
        )
        rows.append(
            {
                "task": "compose",
                "task_detail": "text_compose",
                "prompt": compose_prompt_natural(mapping, need),
                "target": ACTION_DIGIT[target_action],
                "candidates": list(ACTION_CANDIDATES),
                "metadata": {
                    **metadata_base,
                    "decision_phase": "compose",
                    "needed_direction": DIRECTION_CANONICAL[need],
                },
            }
        )
        rows.append(
            {
                "task": "direct",
                "task_detail": "memory_plus_grid",
                "prompt": direct_prompt_natural(mapping, current),
                "target": ACTION_DIGIT[target_action],
                "candidates": list(ACTION_CANDIDATES),
                "metadata": {
                    **metadata_base,
                    "decision_phase": "direct",
                    "needed_direction": DIRECTION_CANONICAL[need],
                },
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    path: Path,
    seeds: range,
    *,
    variants: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(rows_for_game(seed, variants=variants))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    task_counts = Counter(str(row["task"]) for row in rows)
    target_counts = Counter(
        (str(row["task"]), str(row["target"])) for row in rows
    )
    return {
        "games": len(seeds),
        "variants_per_game": variants,
        "rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "task_target_counts": {
            f"{task}:{target}": count
            for (task, target), count in sorted(target_counts.items())
        },
        "file": path.name,
        "sha256": sha256_file(path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_seed_start: int,
    train_games: int,
    validation_seed_start: int,
    validation_games: int,
    test_seed_start: int,
    test_games: int,
    variants: int,
) -> dict[str, Any]:
    if min(train_games, validation_games, test_games, variants) < 1:
        raise ValueError("all counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "arcgpt2.stage03.components.v1",
        "scope": (
            "synthetic component diagnostics for one GPT-2; fixed palette, no "
            "walls, explicit component questions; not an ARC-AGI-3 score"
        ),
        "purity": (
            "offline labels only; at inference the same GPT-2 must perceive raw "
            "transitions, read goal direction, and compose actions"
        ),
        "splits": {
            "train": write_split(
                output_dir / "train.jsonl",
                range(train_seed_start, train_seed_start + train_games),
                variants=variants,
            ),
            "validation": write_split(
                output_dir / "validation.jsonl",
                range(validation_seed_start, validation_seed_start + validation_games),
                variants=variants,
            ),
            "test": write_split(
                output_dir / "test.jsonl",
                range(test_seed_start, test_seed_start + test_games),
                variants=variants,
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage03-components/data"))
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-games", type=int, default=24)
    parser.add_argument("--validation-seed-start", type=int, default=24)
    parser.add_argument("--validation-games", type=int, default=8)
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-games", type=int, default=8)
    parser.add_argument("--variants", type=int, default=8)
    args = parser.parse_args()
    result = build_dataset(
        args.output_dir,
        train_seed_start=args.train_seed_start,
        train_games=args.train_games,
        validation_seed_start=args.validation_seed_start,
        validation_games=args.validation_games,
        test_seed_start=args.test_seed_start,
        test_games=args.test_games,
        variants=args.variants,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
