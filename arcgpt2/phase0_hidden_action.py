"""Phase-0 hidden-action multi-level environment.

The environment is intentionally small but causally decisive. A static frame
cannot determine which action means which direction because the mapping is
randomized for every game. The mapping persists across levels, so a sequence
model can improve from interaction history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterator, Mapping

from .codec import Grid, TransitionEncoding, encode_frame, encode_transition, normalize_grid, tokens_to_text


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Action(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


DIRECTION_DELTA: dict[Direction, tuple[int, int]] = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

ACTION_TOKEN: dict[Action, str] = {
    Action.A1: "<A1>",
    Action.A2: "<A2>",
    Action.A3: "<A3>",
    Action.A4: "<A4>",
}


@dataclass(frozen=True)
class Palette:
    background: int
    wall: int
    agent: int
    goal: int


@dataclass(frozen=True)
class LevelSpec:
    height: int
    width: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]


@dataclass(frozen=True)
class GameSpec:
    game_seed: int
    action_to_direction: Mapping[Action, Direction]
    palette: Palette
    levels: tuple[LevelSpec, ...]
    probe_order: tuple[Action, ...]


@dataclass(frozen=True)
class StepRecord:
    level_index: int
    step_index: int
    before: Grid
    action: Action
    after: Grid
    status: str
    moved: bool
    transition: TransitionEncoding


@dataclass(frozen=True)
class DecisionExample:
    game_seed: int
    level_index: int
    step_index: int
    context: str
    target: str


def _neighbors(position: tuple[int, int]) -> Iterator[tuple[Direction, tuple[int, int]]]:
    y, x = position
    for direction, (dy, dx) in DIRECTION_DELTA.items():
        yield direction, (y + dy, x + dx)


def _shortest_directions(level: LevelSpec, start: tuple[int, int]) -> list[Direction] | None:
    if start == level.goal:
        return []
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], Direction] | None] = {start: None}

    while queue:
        current = queue.popleft()
        for direction, nxt in _neighbors(current):
            y, x = nxt
            if y < 0 or y >= level.height or x < 0 or x >= level.width:
                continue
            if nxt in level.walls or nxt in parent:
                continue
            parent[nxt] = (current, direction)
            if nxt == level.goal:
                path: list[Direction] = []
                cursor = nxt
                while parent[cursor] is not None:
                    previous, step_direction = parent[cursor]  # type: ignore[misc]
                    path.append(step_direction)
                    cursor = previous
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _render(level: LevelSpec, palette: Palette, agent: tuple[int, int]) -> Grid:
    grid = [[palette.background for _ in range(level.width)] for _ in range(level.height)]
    for y, x in level.walls:
        grid[y][x] = palette.wall
    gy, gx = level.goal
    grid[gy][gx] = palette.goal
    ay, ax = agent
    grid[ay][ax] = palette.agent
    return normalize_grid(grid)


def _generate_level(rng: random.Random, *, level_index: int) -> LevelSpec:
    size = 5 + min(level_index, 2)
    height = size
    width = size
    center = (height // 2, width // 2)

    # The first level is a controlled identification arena. Starting from the
    # center, any permutation of the four unique directions remains in bounds
    # and returns to the center. A distant goal prevents a probe from ending the
    # level before all action meanings are observable.
    if level_index == 0:
        candidates = [
            (y, x)
            for y in range(height)
            for x in range(width)
            if abs(y - center[0]) + abs(x - center[1]) >= 3
        ]
        return LevelSpec(
            height=height,
            width=width,
            walls=frozenset(),
            start=center,
            goal=rng.choice(candidates),
        )

    for _ in range(1_000):
        start = (rng.randrange(height), rng.randrange(width))
        goal = (rng.randrange(height), rng.randrange(width))
        if goal == start:
            continue

        protected = {start, goal}
        density = 0.08 + 0.05 * level_index
        walls = {
            (y, x)
            for y in range(height)
            for x in range(width)
            if (y, x) not in protected and rng.random() < density
        }
        level = LevelSpec(height, width, frozenset(walls), start, goal)
        if _shortest_directions(level, start) is not None:
            return level

    raise RuntimeError("failed to generate a connected level")


def generate_game(game_seed: int, *, levels: int = 3) -> GameSpec:
    if levels < 2:
        raise ValueError("phase-0 games require at least two levels")
    rng = random.Random(game_seed)
    directions = list(Direction)
    rng.shuffle(directions)
    action_to_direction = {action: direction for action, direction in zip(Action, directions, strict=True)}

    colors = list(range(16))
    rng.shuffle(colors)
    palette = Palette(
        background=colors[0],
        wall=colors[1],
        agent=colors[2],
        goal=colors[3],
    )
    probe_order = list(Action)
    rng.shuffle(probe_order)
    level_specs = tuple(_generate_level(rng, level_index=index) for index in range(levels))
    return GameSpec(
        game_seed=game_seed,
        action_to_direction=action_to_direction,
        palette=palette,
        levels=level_specs,
        probe_order=tuple(probe_order),
    )


class HiddenActionGame:
    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.level_index = 0
        self.step_index = 0
        self.agent = spec.levels[0].start
        self.finished = False

    @property
    def level(self) -> LevelSpec:
        return self.spec.levels[self.level_index]

    @property
    def frame(self) -> Grid:
        return _render(self.level, self.spec.palette, self.agent)

    def step(self, action: Action) -> StepRecord:
        if self.finished:
            raise RuntimeError("cannot act after the game is finished")
        before = self.frame
        direction = self.spec.action_to_direction[action]
        dy, dx = DIRECTION_DELTA[direction]
        candidate = (self.agent[0] + dy, self.agent[1] + dx)
        y, x = candidate
        moved = (
            0 <= y < self.level.height
            and 0 <= x < self.level.width
            and candidate not in self.level.walls
        )
        if moved:
            self.agent = candidate

        status = "ACTIVE"
        reached_goal = self.agent == self.level.goal
        if reached_goal:
            if self.level_index + 1 == len(self.spec.levels):
                status = "GAME_WIN"
                self.finished = True
            else:
                status = "LEVEL_WIN"

        after = self.frame
        record = StepRecord(
            level_index=self.level_index,
            step_index=self.step_index,
            before=before,
            action=action,
            after=after,
            status=status,
            moved=moved,
            transition=encode_transition(before, after),
        )
        self.step_index += 1

        if status == "LEVEL_WIN":
            self.level_index += 1
            self.agent = self.level.start
            self.step_index = 0

        return record


class SourceLearner:
    """A generic history-generating learner used only offline.

    It discovers unknown action semantics by observation, then follows a shortest
    path. The action mapping is never exposed in the transcript.
    """

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.direction_to_action: dict[Direction, Action] = {}

    def observe(self, record: StepRecord) -> None:
        if not record.moved:
            return
        before_agent = self._find_agent(record.before)
        after_agent = self._find_agent(record.after)
        dy = after_agent[0] - before_agent[0]
        dx = after_agent[1] - before_agent[1]
        for direction, delta in DIRECTION_DELTA.items():
            if delta == (dy, dx):
                self.direction_to_action[direction] = record.action
                break

    def choose(self, game: HiddenActionGame) -> Action:
        unknown_actions = [
            action
            for action in self.spec.probe_order
            if action not in self.direction_to_action.values()
        ]
        if unknown_actions:
            return unknown_actions[0]

        path = _shortest_directions(game.level, game.agent)
        if path is None:
            raise RuntimeError("generated level unexpectedly became unsolvable")
        if not path:
            # The environment advances levels immediately after reaching a goal,
            # so this branch should be unreachable during a decision.
            return self.spec.probe_order[0]
        action = self.direction_to_action.get(path[0])
        if action is None:
            raise RuntimeError("source learner did not identify the full action mapping")
        return action

    def _find_agent(self, frame: Grid) -> tuple[int, int]:
        color = self.spec.palette.agent
        positions = [
            (y, x)
            for y, row in enumerate(frame)
            for x, value in enumerate(row)
            if value == color
        ]
        if len(positions) != 1:
            raise RuntimeError("phase-0 frame must contain exactly one agent cell")
        return positions[0]


def simulate_source_history(spec: GameSpec, *, max_actions: int = 128) -> list[StepRecord]:
    game = HiddenActionGame(spec)
    learner = SourceLearner(spec)
    records: list[StepRecord] = []
    for _ in range(max_actions):
        action = learner.choose(game)
        record = game.step(action)
        records.append(record)
        learner.observe(record)
        if game.finished:
            return records
    raise RuntimeError("source learner exceeded action budget")


def phase0_special_tokens() -> list[str]:
    tokens = [
        "<GAME_START>",
        "<GAME_END>",
        "<FRAME>",
        "</FRAME>",
        "<ACTION>",
        "</ACTION>",
        "<OUTCOME>",
        "</OUTCOME>",
        "<AVAILABLE>",
        "</AVAILABLE>",
        "<STATUS_ACTIVE>",
        "<STATUS_LEVEL_WIN>",
        "<STATUS_GAME_WIN>",
        "<MOVED_0>",
        "<MOVED_1>",
        "<TRANSITION_DELTA>",
        "<TRANSITION_FULL>",
        "<DECIDE>",
    ]
    tokens.extend(ACTION_TOKEN.values())
    tokens.extend(f"<LEVEL_{index}>" for index in range(16))
    tokens.extend(f"<STEP_{index}>" for index in range(256))
    return tokens


def _status_token(status: str) -> str:
    mapping = {
        "ACTIVE": "<STATUS_ACTIVE>",
        "LEVEL_WIN": "<STATUS_LEVEL_WIN>",
        "GAME_WIN": "<STATUS_GAME_WIN>",
    }
    return mapping[status]


def initial_transcript(spec: GameSpec) -> list[str]:
    game = HiddenActionGame(spec)
    return [
        "<GAME_START>",
        "<LEVEL_0>",
        "<FRAME>",
        *encode_frame(game.frame),
        "</FRAME>",
        "<AVAILABLE>",
        *(ACTION_TOKEN[action] for action in Action),
        "</AVAILABLE>",
    ]


def append_record_tokens(transcript: list[str], record: StepRecord, *, next_frame: Grid | None) -> None:
    transcript.extend(
        (
            f"<STEP_{record.step_index}>",
            "<ACTION>",
            ACTION_TOKEN[record.action],
            "</ACTION>",
            "<OUTCOME>",
            "<MOVED_1>" if record.moved else "<MOVED_0>",
            _status_token(record.status),
            "<TRANSITION_DELTA>" if record.transition.kind == "delta" else "<TRANSITION_FULL>",
            *record.transition.tokens,
            "</OUTCOME>",
        )
    )
    if record.status == "LEVEL_WIN":
        if next_frame is None:
            raise ValueError("next_frame is required after a level win")
        transcript.extend(
            (
                f"<LEVEL_{record.level_index + 1}>",
                "<FRAME>",
                *encode_frame(next_frame),
                "</FRAME>",
                "<AVAILABLE>",
                *(ACTION_TOKEN[action] for action in Action),
                "</AVAILABLE>",
            )
        )


def build_decision_examples(spec: GameSpec) -> list[DecisionExample]:
    records = simulate_source_history(spec)
    game = HiddenActionGame(spec)
    learner = SourceLearner(spec)
    transcript = initial_transcript(spec)
    examples: list[DecisionExample] = []

    for expected_record in records:
        action = learner.choose(game)
        if action != expected_record.action:
            raise RuntimeError("source history replay diverged")
        context_tokens = [*transcript, "<DECIDE>"]
        examples.append(
            DecisionExample(
                game_seed=spec.game_seed,
                level_index=game.level_index,
                step_index=game.step_index,
                context=tokens_to_text(context_tokens),
                target=ACTION_TOKEN[action],
            )
        )
        actual_record = game.step(action)
        learner.observe(actual_record)
        next_frame = game.frame if actual_record.status == "LEVEL_WIN" else None
        append_record_tokens(transcript, actual_record, next_frame=next_frame)

    transcript.append("<GAME_END>")
    return examples


def game_summary(spec: GameSpec) -> dict[str, object]:
    records = simulate_source_history(spec)
    per_level: dict[int, int] = {}
    for record in records:
        per_level[record.level_index] = per_level.get(record.level_index, 0) + 1
    return {
        "game_seed": spec.game_seed,
        "levels": len(spec.levels),
        "actions": len(records),
        "actions_per_level": per_level,
        "mapping": {action.value: direction.value for action, direction in spec.action_to_direction.items()},
        "palette": {
            "background": spec.palette.background,
            "wall": spec.palette.wall,
            "agent": spec.palette.agent,
            "goal": spec.palette.goal,
        },
    }
