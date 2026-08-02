"""Stage 0.2: decomposed in-context control using one ordinary GPT-2.

This stage is deliberately narrower than ARC-AGI-3.  It isolates a necessary
capability that Stage 0.1 failed to learn: infer a per-game permutation of four
actions from literal transitions, retain it in the context, read a raw grid,
and select the next action without an external planner.

Purity contract at inference:

* one ``GPT2LMHeadModel`` and its original tokenizer are the only learned state;
* the same GPT-2 is queried for mapping, direction, composition, and a direct
  action ablation;
* deterministic code only serializes exact grids, copies model answers into the
  next prompt, masks output to legal single-character labels, and executes the
  chosen action;
* shortest paths and true mappings are used only to generate offline labels and
  score experiments, never by the acting policy.

The synthetic world uses a fixed, explicitly documented palette and no walls.
That is intentional: visual identity, hidden action semantics, memory, and
control composition must work before adding palette changes, obstacles, object
interactions, delayed effects, or hidden goals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .phase0_hidden_action import (
    ACTION_TOKEN,
    DIRECTION_DELTA,
    Action,
    Direction,
    GameSpec,
    HiddenActionGame,
    LevelSpec,
    Palette,
    StepRecord,
)

Grid = tuple[tuple[int, ...], ...]

FIXED_PALETTE = Palette(background=0, wall=1, agent=2, goal=3)
ACTION_DIGIT: dict[Action, str] = {
    Action.A1: "1",
    Action.A2: "2",
    Action.A3: "3",
    Action.A4: "4",
}
DIGIT_ACTION = {value: key for key, value in ACTION_DIGIT.items()}
DIRECTION_LETTER: dict[Direction, str] = {
    Direction.UP: "N",
    Direction.RIGHT: "E",
    Direction.DOWN: "S",
    Direction.LEFT: "W",
}
LETTER_DIRECTION = {value: key for key, value in DIRECTION_LETTER.items()}
DIRECTION_CYCLE = (
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
)
MAP_CANDIDATES = ("N", "E", "S", "W", "?")
NEED_CANDIDATES = MAP_CANDIDATES
ACTION_CANDIDATES = ("1", "2", "3", "4")
ALL_LABELS = MAP_CANDIDATES + ACTION_CANDIDATES

WORLD_HEADER = (
    "This is an exact grid-control record. Grid cells are digits: "
    "0 empty, 1 wall, 2 mover, 3 goal. Rows run top to bottom and columns "
    "left to right. Actions 1, 2, 3, and 4 are a different permutation of "
    "north, east, south, and west in every game. Infer the permutation only "
    "from observed cell changes."
)


@dataclass(frozen=True)
class Stage02Context:
    """One literal history and the current state used for all GPT-2 calls."""

    game_seed: int
    variant_id: str
    records: tuple[StepRecord, ...]
    current_grid: Grid
    level_index: int
    mapping: Mapping[Action, Direction]


@dataclass(frozen=True)
class Stage02Decision:
    mapping_labels: Mapping[Action, str]
    needed_direction: str
    target_action: str
    decision_phase: str


def normalize_grid(grid: Sequence[Sequence[int]]) -> Grid:
    rows = tuple(tuple(int(value) for value in row) for row in grid)
    if not rows or not rows[0]:
        raise ValueError("grid must be non-empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("grid must be rectangular")
    if len(rows) > 64 or width > 64:
        raise ValueError("grid exceeds ARC's 64x64 limit")
    if any(value < 0 or value > 15 for row in rows for value in row):
        raise ValueError("grid values must be in 0..15")
    return rows


def grid_text(grid: Sequence[Sequence[int]]) -> str:
    """Compact, reversible, semantic-free row serialization."""
    rows = normalize_grid(grid)
    return f"{len(rows)}x{len(rows[0])}:" + "/".join(
        "".join(format(value, "x") for value in row) for row in rows
    )


def parse_grid_text(value: str) -> Grid:
    try:
        dimensions, body = value.split(":", 1)
        raw_height, raw_width = dimensions.split("x", 1)
        height = int(raw_height)
        width = int(raw_width)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid grid serialization: {value!r}") from exc
    raw_rows = body.split("/")
    if len(raw_rows) != height or any(len(row) != width for row in raw_rows):
        raise ValueError("serialized grid dimensions do not match its rows")
    try:
        return normalize_grid(
            [[int(character, 16) for character in row] for row in raw_rows]
        )
    except ValueError as exc:
        raise ValueError(f"invalid grid cell in {value!r}") from exc


def delta_text(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> str:
    left = normalize_grid(before)
    right = normalize_grid(after)
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        return "full=" + grid_text(right)
    changes = [
        f"r{row}c{column}:{left[row][column]}>{right[row][column]}"
        for row in range(len(left))
        for column in range(len(left[0]))
        if left[row][column] != right[row][column]
    ]
    return ",".join(changes) if changes else "none"


def _find_color(grid: Sequence[Sequence[int]], color: int) -> tuple[int, int]:
    positions = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if int(value) == color
    ]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one cell of color {color}, got {positions}")
    return positions[0]


def observed_direction(record: StepRecord) -> Direction | None:
    """Decode only a literal one-cell mover displacement from a transition."""
    try:
        before = _find_color(record.before, FIXED_PALETTE.agent)
        after = _find_color(record.after, FIXED_PALETTE.agent)
    except ValueError:
        return None
    displacement = (after[0] - before[0], after[1] - before[1])
    for direction, delta in DIRECTION_DELTA.items():
        if displacement == delta:
            return direction
    return None


def infer_mapping_labels(records: Sequence[StepRecord]) -> dict[Action, str]:
    labels = {action: "?" for action in Action}
    for record in records:
        direction = observed_direction(record)
        if direction is not None:
            labels[record.action] = DIRECTION_LETTER[direction]
    return labels


def needed_direction_from_grid(
    grid: Sequence[Sequence[int]], *, mapping_complete: bool
) -> str:
    """Offline label for the no-wall curriculum.

    Before all four meanings are known the policy should gather information, so
    the desired direction is deliberately ``?``.  Once known, choose the axis
    with the larger absolute remaining distance; vertical wins ties.  This gives
    one deterministic efficient move even when a failed/repeated probe leaves
    the mover diagonally offset from the goal.
    """
    if not mapping_complete:
        return "?"
    mover_row, mover_column = _find_color(grid, FIXED_PALETTE.agent)
    goal_row, goal_column = _find_color(grid, FIXED_PALETTE.goal)
    row_delta = goal_row - mover_row
    column_delta = goal_column - mover_column
    if row_delta == 0 and column_delta == 0:
        raise ValueError("no decision is required after reaching the goal")
    if row_delta != 0 and abs(row_delta) >= abs(column_delta):
        return "S" if row_delta > 0 else "N"
    return "E" if column_delta > 0 else "W"


def decision_for_context(context: Stage02Context) -> Stage02Decision:
    labels = infer_mapping_labels(context.records)
    unknown = [action for action in Action if labels[action] == "?"]
    if unknown:
        target_action = min(ACTION_DIGIT[action] for action in unknown)
        needed = "?"
        phase = "probe"
    else:
        needed = needed_direction_from_grid(
            context.current_grid,
            mapping_complete=True,
        )
        target_action = next(
            ACTION_DIGIT[action]
            for action in Action
            if labels[action] == needed
        )
        phase = "navigate"
    return Stage02Decision(
        mapping_labels=labels,
        needed_direction=needed,
        target_action=target_action,
        decision_phase=phase,
    )


def format_history(
    records: Sequence[StepRecord],
    current_grid: Sequence[Sequence[int]],
    *,
    level_index: int,
    history_mode: str = "intact",
) -> str:
    """Render exact observations without identifying objects or solving paths."""
    current = normalize_grid(current_grid)
    lines = [WORLD_HEADER]
    if history_mode not in {"intact", "amnesic"}:
        raise ValueError("history_mode must be intact or amnesic")
    if history_mode == "intact" and records:
        lines.append("Observed history:")
        previous_level: int | None = None
        for index, record in enumerate(records):
            if record.level_index != previous_level:
                lines.append(
                    f"Level {record.level_index} began with grid {grid_text(record.before)}."
                )
                previous_level = record.level_index
            lines.append(
                "Step "
                f"{index}: action {ACTION_DIGIT[record.action]}; "
                f"change {delta_text(record.before, record.after)}; "
                f"status {record.status}."
            )
    else:
        lines.append("No earlier action outcomes are available.")
    lines.append(f"Current level {level_index} grid {grid_text(current)}.")
    return "\n".join(lines)


def mapping_prompt(history: str, action: Action) -> str:
    return (
        history
        + "\nQuestion: What direction does action "
        + ACTION_DIGIT[action]
        + " move? If its meaning cannot be inferred from the observations, "
          "answer ?. Answer exactly one character: N, E, S, W, or ?.\nAnswer:"
    )


def need_prompt(history: str) -> str:
    return (
        history
        + "\nQuestion: If every action meaning has been observed, which direction "
          "should the mover take now to reduce the larger row/column distance "
          "to the goal, using vertical on a tie? If any action meaning remains "
          "unknown, answer ?. Answer exactly N, E, S, W, or ?.\nAnswer:"
    )


def mapping_summary(labels: Mapping[Action, str]) -> str:
    return " ".join(
        f"{ACTION_DIGIT[action]}={labels.get(action, '?')}" for action in Action
    )


def compose_prompt(labels: Mapping[Action, str], needed_direction: str) -> str:
    return (
        "Action meanings: "
        + mapping_summary(labels)
        + f". Needed direction: {needed_direction}. "
          "Decision rule: if any action meaning is ?, choose the smallest-numbered "
          "unknown action. Otherwise choose the action whose direction equals the "
          "needed direction. Answer exactly 1, 2, 3, or 4.\nAnswer:"
    )


def direct_prompt(history: str) -> str:
    return (
        history
        + "\nQuestion: Choose the next action. First infer which action numbers "
          "have known directions. If any are unknown, choose the smallest-numbered "
          "unknown action. If all are known, choose the action that moves toward "
          "the goal by reducing the larger row/column distance, using vertical on "
          "a tie. Answer exactly 1, 2, 3, or 4.\nAnswer:"
    )


def all_action_mappings() -> tuple[dict[Action, Direction], ...]:
    return tuple(
        dict(zip(tuple(Action), order, strict=True))
        for order in permutations(tuple(Direction))
    )


def _goal_position(
    size: int,
    start: tuple[int, int],
    direction: Direction,
    distance: int,
) -> tuple[int, int]:
    row_delta, column_delta = DIRECTION_DELTA[direction]
    return (
        start[0] + row_delta * distance,
        start[1] + column_delta * distance,
    )


def make_stage02_spec(
    game_seed: int,
    *,
    first_goal_direction: Direction | None = None,
    levels: int = 3,
) -> GameSpec:
    if levels < 1:
        raise ValueError("levels must be positive")
    mapping = all_action_mappings()[game_seed % len(all_action_mappings())]
    level_specs: list[LevelSpec] = []

    size = 9
    center = (size // 2, size // 2)
    first_direction = first_goal_direction or DIRECTION_CYCLE[(game_seed // 24) % 4]
    level_specs.append(
        LevelSpec(
            height=size,
            width=size,
            walls=frozenset(),
            start=center,
            goal=_goal_position(size, center, first_direction, 3),
        )
    )

    for level_index in range(1, levels):
        size = 7 + 2 * (level_index % 2)
        center = (size // 2, size // 2)
        direction = DIRECTION_CYCLE[(game_seed + level_index) % 4]
        distance = 2 + ((game_seed + level_index) % 2)
        level_specs.append(
            LevelSpec(
                height=size,
                width=size,
                walls=frozenset(),
                start=center,
                goal=_goal_position(size, center, direction, distance),
            )
        )

    return GameSpec(
        game_seed=game_seed,
        action_to_direction=mapping,
        palette=FIXED_PALETTE,
        levels=tuple(level_specs),
        probe_order=tuple(Action),
    )


def ordered_distinct_sequences() -> tuple[tuple[Action, ...], ...]:
    sequences: list[tuple[Action, ...]] = [tuple()]
    actions = tuple(Action)
    for length in range(1, len(actions) + 1):
        sequences.extend(tuple(value) for value in permutations(actions, length))
    return tuple(sequences)


def training_sequences() -> tuple[tuple[Action, ...], ...]:
    """All distinct prefixes plus one deterministic repeat-recovery variant."""
    values = list(ordered_distinct_sequences())
    for sequence in ordered_distinct_sequences():
        if sequence:
            values.append((*sequence, sequence[0]))
    # Stable de-duplication.
    return tuple(dict.fromkeys(values))


def simulate_context(
    game_seed: int,
    sequence: Sequence[Action],
    *,
    goal_direction: Direction,
    variant_id: str,
) -> Stage02Context:
    spec = make_stage02_spec(
        game_seed,
        first_goal_direction=goal_direction,
        levels=1,
    )
    game = HiddenActionGame(spec)
    records: list[StepRecord] = []
    for action in sequence:
        if game.finished:
            raise RuntimeError("training sequence reached the goal unexpectedly")
        record = game.step(action)
        if record.status != "ACTIVE":
            raise RuntimeError("probe curriculum must not terminate during history")
        records.append(record)
    return Stage02Context(
        game_seed=game_seed,
        variant_id=variant_id,
        records=tuple(records),
        current_grid=game.frame,
        level_index=game.level_index,
        mapping=spec.action_to_direction,
    )


def task_rows(context: Stage02Context) -> list[dict[str, Any]]:
    history = format_history(
        context.records,
        context.current_grid,
        level_index=context.level_index,
    )
    decision = decision_for_context(context)
    metadata = {
        "game_seed": context.game_seed,
        "variant_id": context.variant_id,
        "history_length": len(context.records),
        "decision_phase": decision.decision_phase,
        "mapping": {
            ACTION_DIGIT[action]: DIRECTION_LETTER[direction]
            for action, direction in context.mapping.items()
        },
    }
    rows: list[dict[str, Any]] = []
    for action in Action:
        rows.append(
            {
                "task": "mapping",
                "task_detail": f"map_{ACTION_DIGIT[action]}",
                "prompt": mapping_prompt(history, action),
                "target": decision.mapping_labels[action],
                "candidates": list(MAP_CANDIDATES),
                "metadata": metadata,
            }
        )
    rows.append(
        {
            "task": "need",
            "task_detail": "need_direction",
            "prompt": need_prompt(history),
            "target": decision.needed_direction,
            "candidates": list(NEED_CANDIDATES),
            "metadata": metadata,
        }
    )
    rows.append(
        {
            "task": "compose",
            "task_detail": "compose_action",
            "prompt": compose_prompt(
                decision.mapping_labels,
                decision.needed_direction,
            ),
            "target": decision.target_action,
            "candidates": list(ACTION_CANDIDATES),
            "metadata": metadata,
        }
    )
    rows.append(
        {
            "task": "direct",
            "task_detail": "direct_action",
            "prompt": direct_prompt(history),
            "target": decision.target_action,
            "candidates": list(ACTION_CANDIDATES),
            "metadata": metadata,
        }
    )
    return rows


def build_rows(game_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sequence in enumerate(training_sequences()):
        direction = DIRECTION_CYCLE[(game_seed + index) % len(DIRECTION_CYCLE)]
        variant_id = "-".join(ACTION_DIGIT[action] for action in sequence) or "empty"
        context = simulate_context(
            game_seed,
            sequence,
            goal_direction=direction,
            variant_id=f"{index:03d}-{variant_id}",
        )
        rows.extend(task_rows(context))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(path: Path, seeds: range) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(build_rows(seed))
    write_jsonl(path, rows)
    task_counts = Counter(str(row["task"]) for row in rows)
    task_target_counts = Counter(
        (str(row["task"]), str(row["target"])) for row in rows
    )
    phase_counts = Counter(
        str(row["metadata"]["decision_phase"]) for row in rows if row["task"] == "direct"
    )
    return {
        "games": len(seeds),
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "rows": len(rows),
        "contexts": len(rows) // 7,
        "task_counts": dict(sorted(task_counts.items())),
        "task_target_counts": {
            f"{task}:{target}": count
            for (task, target), count in sorted(task_target_counts.items())
        },
        "direct_decision_phase_counts": dict(sorted(phase_counts.items())),
        "file": path.name,
        "sha256": sha256_file(path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_games: int,
    validation_games: int,
    test_games: int,
    train_seed_start: int,
    validation_seed_start: int,
    test_seed_start: int,
) -> dict[str, Any]:
    if min(train_games, validation_games, test_games) < 1:
        raise ValueError("all dataset splits require at least one game")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": _split_summary(
            output_dir / "train.jsonl",
            range(train_seed_start, train_seed_start + train_games),
        ),
        "validation": _split_summary(
            output_dir / "validation.jsonl",
            range(validation_seed_start, validation_seed_start + validation_games),
        ),
        "test": _split_summary(
            output_dir / "test.jsonl",
            range(test_seed_start, test_seed_start + test_games),
        ),
    }
    manifest = {
        "schema": "arcgpt2.stage02.decomposed-natural.v1",
        "claim_scope": (
            "synthetic fixed-palette no-wall hidden-action curriculum; not an "
            "ARC-AGI-3 public or private score"
        ),
        "purity": (
            "one standard GPT-2 is queried for mapping, need, composition, and "
            "direct action; deterministic code only serializes, masks labels, "
            "copies answers, executes actions, and scores offline truth"
        ),
        "labels": list(ALL_LABELS),
        "contexts_per_game": len(training_sequences()),
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage02/data"))
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=2)
    parser.add_argument("--test-games", type=int, default=4)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--validation-seed-start", type=int, default=8)
    parser.add_argument("--test-seed-start", type=int, default=20)
    args = parser.parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_games=args.train_games,
        validation_games=args.validation_games,
        test_games=args.test_games,
        train_seed_start=args.train_seed_start,
        validation_seed_start=args.validation_seed_start,
        test_seed_start=args.test_seed_start,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
