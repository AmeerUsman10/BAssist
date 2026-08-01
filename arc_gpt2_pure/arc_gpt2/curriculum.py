"""Procedural curricula for teaching one GPT-2 to learn from trajectories.

Version 0 isolates a necessary ARC-AGI-3 capability: action semantics are
randomly remapped in every episode. The model must infer the mapping from
literal transitions, preserve it in its own text memory, and either navigate
toward a goal or perform an information-seeking action when the required
control is still unknown.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .codec import Grid, encode_delta
from .protocol import (
    ACTION_ORDER,
    DIRECTION_ORDER,
    Transition,
    action_prompt,
    format_mapping,
    memory_prompt,
    prediction_prompt,
)

DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}

EMPTY = 0
WALL = 1
AGENT = 2
GOAL = 3


@dataclass(frozen=True)
class MotionEpisode:
    """One hidden-action navigation problem and its observed learning history."""

    episode_id: str
    mapping: dict[str, str]
    known_mapping: dict[str, str]
    transitions: list[Transition]
    current_grid: Grid
    target_action: str
    target_delta: str
    desired_direction: str
    decision_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "mapping": dict(self.mapping),
            "known_mapping": dict(self.known_mapping),
            "transitions": [
                {
                    "before": transition.before,
                    "action": transition.action,
                    "after": transition.after,
                }
                for transition in self.transitions
            ],
            "current_grid": self.current_grid,
            "target_action": self.target_action,
            "target_delta": self.target_delta,
            "desired_direction": self.desired_direction,
            "decision_kind": self.decision_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionEpisode":
        return cls(
            episode_id=str(value["episode_id"]),
            mapping={str(k): str(v) for k, v in dict(value["mapping"]).items()},
            known_mapping={
                str(k): str(v) for k, v in dict(value["known_mapping"]).items()
            },
            transitions=[
                Transition(
                    before=[[int(cell) for cell in row] for row in item["before"]],
                    action=str(item["action"]),
                    after=[[int(cell) for cell in row] for row in item["after"]],
                )
                for item in value["transitions"]
            ],
            current_grid=[
                [int(cell) for cell in row] for row in value["current_grid"]
            ],
            target_action=str(value["target_action"]),
            target_delta=str(value["target_delta"]),
            desired_direction=str(value["desired_direction"]),
            decision_kind=str(value["decision_kind"]),
        )


def find_value(grid: Sequence[Sequence[int]], value: int) -> tuple[int, int]:
    locations = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if int(cell) == value
    ]
    if len(locations) != 1:
        raise ValueError(f"expected exactly one cell with value {value}")
    return locations[0]


def move_agent(grid: Sequence[Sequence[int]], direction: str) -> Grid:
    """Apply one cardinal move; walls and bounds produce a literal no-op."""
    direction = direction.upper()
    if direction not in DIRECTION_DELTAS:
        raise ValueError(f"unknown direction: {direction}")
    result = [[int(cell) for cell in row] for row in grid]
    row, column = find_value(result, AGENT)
    row_delta, column_delta = DIRECTION_DELTAS[direction]
    next_row = row + row_delta
    next_column = column + column_delta
    if not (0 <= next_row < len(result) and 0 <= next_column < len(result[0])):
        return result
    destination = result[next_row][next_column]
    if destination == WALL:
        return result
    result[row][column] = EMPTY
    result[next_row][next_column] = AGENT
    # The synthetic goal remains represented after entry so the target location
    # can still be inferred from the current frame. Completion is evaluated by
    # position in generated metadata, not by erasing the goal cell.
    if destination == GOAL:
        result[row][column] = EMPTY
    return result


def _walkable_neighbors(
    grid: Sequence[Sequence[int]], row: int, column: int
) -> Iterable[tuple[str, int, int]]:
    for direction in DIRECTION_ORDER:
        row_delta, column_delta = DIRECTION_DELTAS[direction]
        next_row = row + row_delta
        next_column = column + column_delta
        if not (0 <= next_row < len(grid) and 0 <= next_column < len(grid[0])):
            continue
        if int(grid[next_row][next_column]) != WALL:
            yield direction, next_row, next_column


def shortest_direction(grid: Sequence[Sequence[int]]) -> str:
    """Return the first direction on a deterministic shortest path to value 3."""
    start = find_value(grid, AGENT)
    goal = find_value(grid, GOAL)
    queue: deque[tuple[tuple[int, int], str | None]] = deque([(start, None)])
    visited = {start}
    while queue:
        (row, column), first_direction = queue.popleft()
        if (row, column) == goal:
            if first_direction is None:
                raise ValueError("agent already occupies the goal")
            return first_direction
        for direction, next_row, next_column in _walkable_neighbors(
            grid, row, column
        ):
            position = (next_row, next_column)
            if position in visited:
                continue
            visited.add(position)
            queue.append((position, first_direction or direction))
    raise ValueError("goal is unreachable")


def _probe_grid() -> Grid:
    grid = [[EMPTY for _ in range(5)] for _ in range(5)]
    grid[0][0] = GOAL
    grid[2][2] = AGENT
    return grid


def _mapping(rng: random.Random) -> dict[str, str]:
    directions = list(DIRECTION_ORDER)
    rng.shuffle(directions)
    return dict(zip(ACTION_ORDER, directions, strict=True))


def _generate_navigation_grid(rng: random.Random) -> Grid:
    for _ in range(500):
        height = rng.randint(5, 8)
        width = rng.randint(5, 8)
        grid = [[EMPTY for _ in range(width)] for _ in range(height)]

        # Keep the agent away from the border and clear all four adjacent cells,
        # so an information-seeking action reveals its direction instead of an
        # ambiguous wall collision.
        agent_row = rng.randint(1, height - 2)
        agent_column = rng.randint(1, width - 2)
        protected = {
            (agent_row, agent_column),
            (agent_row - 1, agent_column),
            (agent_row + 1, agent_column),
            (agent_row, agent_column - 1),
            (agent_row, agent_column + 1),
        }

        candidates = [
            (row, column)
            for row in range(height)
            for column in range(width)
            if (row, column) not in protected
        ]
        goal_row, goal_column = rng.choice(candidates)
        protected.add((goal_row, goal_column))

        for row in range(height):
            for column in range(width):
                if (row, column) in protected:
                    continue
                if rng.random() < 0.15:
                    grid[row][column] = WALL

        grid[agent_row][agent_column] = AGENT
        grid[goal_row][goal_column] = GOAL
        try:
            shortest_direction(grid)
        except ValueError:
            continue
        return grid
    raise RuntimeError("failed to generate a connected navigation grid")


def generate_episode(seed: int, probe_count: int | None = None) -> MotionEpisode:
    """Generate one deterministic hidden-action episode from ``seed``."""
    rng = random.Random(seed)
    mapping = _mapping(rng)
    if probe_count is None:
        # Bias toward partial histories; full mappings remain common enough for
        # stable early training.
        probe_count = rng.choices([0, 1, 2, 3, 4], weights=[1, 2, 3, 3, 4], k=1)[0]
    if not 0 <= probe_count <= 4:
        raise ValueError("probe_count must be between 0 and 4")

    observed_actions = rng.sample(list(ACTION_ORDER), k=probe_count)
    transitions: list[Transition] = []
    for action in observed_actions:
        before = _probe_grid()
        after = move_agent(before, mapping[action])
        transitions.append(Transition(before=before, action=action, after=after))

    known_mapping = {
        action: mapping[action] if action in observed_actions else "?"
        for action in ACTION_ORDER
    }
    current_grid = _generate_navigation_grid(rng)
    desired = shortest_direction(current_grid)
    desired_action = next(
        action for action, direction in mapping.items() if direction == desired
    )

    if known_mapping[desired_action] == desired:
        target_action = desired_action
        decision_kind = "navigate"
    else:
        target_action = next(
            action for action in ACTION_ORDER if known_mapping[action] == "?"
        )
        decision_kind = "probe"

    after = move_agent(current_grid, mapping[target_action])
    return MotionEpisode(
        episode_id=f"motion-{seed:010d}",
        mapping=mapping,
        known_mapping=known_mapping,
        transitions=transitions,
        current_grid=current_grid,
        target_action=target_action,
        target_delta=encode_delta(current_grid, after),
        desired_direction=desired,
        decision_kind=decision_kind,
    )


def episode_examples(episode: MotionEpisode) -> list[dict[str, Any]]:
    """Create masked-causal-LM examples for memory, action, and prediction."""
    memory = format_mapping(episode.known_mapping)
    episode_dict = episode.to_dict()
    common_metadata = {
        "episode_id": episode.episode_id,
        "decision_kind": episode.decision_kind,
        "probe_count": len(episode.transitions),
        "target_action": episode.target_action,
        "desired_direction": episode.desired_direction,
    }
    return [
        {
            "task": "memory",
            "prompt": memory_prompt(episode.transitions, episode.current_grid),
            "target": f" {memory}\n[[/MEMORY]]",
            "metadata": common_metadata,
            "episode": episode_dict,
        },
        {
            "task": "action",
            "prompt": action_prompt(
                episode.transitions,
                episode.current_grid,
                memory=memory,
            ),
            "target": f" {episode.target_action}\n[[/ACTION]]",
            "metadata": common_metadata,
            "episode": episode_dict,
        },
        {
            "task": "prediction",
            "prompt": prediction_prompt(
                episode.transitions,
                episode.current_grid,
                action=episode.target_action,
                memory=memory,
            ),
            "target": f" {episode.target_delta}\n[[/NEXT]]",
            "metadata": common_metadata,
            "episode": episode_dict,
        },
    ]


def generate_records(episodes: int, seed: int) -> list[dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records: list[dict[str, Any]] = []
    for offset in range(episodes):
        records.extend(episode_examples(generate_episode(seed + offset)))
    return records


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def build_datasets(
    train_out: str | Path,
    eval_out: str | Path,
    train_episodes: int = 512,
    eval_episodes: int = 128,
    train_seed: int = 100_000,
    eval_seed: int = 900_000,
) -> dict[str, int]:
    """Generate non-overlapping deterministic training and held-out files."""
    train_records = generate_records(train_episodes, train_seed)
    eval_records = generate_records(eval_episodes, eval_seed)
    write_jsonl(train_out, train_records)
    write_jsonl(eval_out, eval_records)
    return {
        "train_episodes": train_episodes,
        "train_records": len(train_records),
        "eval_episodes": eval_episodes,
        "eval_records": len(eval_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--eval-out", default="data/eval.jsonl")
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=100_000)
    parser.add_argument("--eval-seed", type=int, default=900_000)
    args = parser.parse_args()
    summary = build_datasets(
        train_out=args.train_out,
        eval_out=args.eval_out,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
