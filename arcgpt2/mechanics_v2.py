"""Typed executable mechanics kernel beyond simple movement.

Version 1 of the ARC-GPT2 DSL represented only cardinal movement and blocking.
This module adds a deliberately small family of contact mechanics while keeping
execution exact, bounded, and inspectable. It is the first Gate-C language for
learning *what objects do*, not merely which action means which direction.

The deterministic interpreter contains no game-selection intelligence. A
program must explicitly declare the moving color, background, blockers, goal
colors, action displacements, the special contact color, and its effect.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import math
from typing import Iterable, Mapping, Sequence

from .codec import Grid, normalize_grid
from .phase0_hidden_action import Action, DIRECTION_DELTA, Direction


class MechanicsV2Error(ValueError):
    """Raised when a mechanics program is invalid or cannot be interpreted."""


class ContactMode(str, Enum):
    """Bounded one-cell contact primitives."""

    BLOCK = "BLOCK"
    COLLECT = "COLLECT"
    PUSH = "PUSH"
    ERASE = "ERASE"
    SWAP = "SWAP"


class MechanicsStatus(str, Enum):
    ACTIVE = "ACTIVE"
    WIN = "WIN"


@dataclass(frozen=True)
class ActionMove:
    action: Action
    dy: int
    dx: int

    def __post_init__(self) -> None:
        if (self.dy, self.dx) not in set(DIRECTION_DELTA.values()):
            raise MechanicsV2Error(
                "Gate-C action moves must be one-cell cardinal displacements"
            )

    def canonical_line(self) -> str:
        return f"MOVE {self.action.value} DY {self.dy} DX {self.dx}"


@dataclass(frozen=True)
class MechanicsProgramV2:
    moving_color: int
    background_color: int
    blocking_colors: tuple[int, ...]
    goal_colors: tuple[int, ...]
    interaction_color: int
    contact_mode: ContactMode
    moves: tuple[ActionMove, ...]
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise MechanicsV2Error("only mechanics program version 2 is supported")
        colors = (
            self.moving_color,
            self.background_color,
            self.interaction_color,
            *self.blocking_colors,
            *self.goal_colors,
        )
        if any(color < 0 or color > 15 for color in colors):
            raise MechanicsV2Error("all colors must lie in 0..15")
        if self.moving_color == self.background_color:
            raise MechanicsV2Error("moving and background colors must differ")
        if self.interaction_color in {
            self.moving_color,
            self.background_color,
        }:
            raise MechanicsV2Error(
                "interaction color must differ from moving and background colors"
            )
        if set(self.blocking_colors).intersection(self.goal_colors):
            raise MechanicsV2Error("a color cannot be both a blocker and a goal")
        if len(self.blocking_colors) != len(set(self.blocking_colors)):
            raise MechanicsV2Error("blocking colors may not contain duplicates")
        if len(self.goal_colors) != len(set(self.goal_colors)):
            raise MechanicsV2Error("goal colors may not contain duplicates")
        actions = [move.action for move in self.moves]
        if set(actions) != set(Action) or len(actions) != len(set(actions)):
            raise MechanicsV2Error(
                "Gate-C programs must define exactly one move for A1..A4"
            )
        displacements = [(move.dy, move.dx) for move in self.moves]
        if len(displacements) != len(set(displacements)):
            raise MechanicsV2Error(
                "Gate-C action semantics must form a cardinal permutation"
            )

    @property
    def by_action(self) -> Mapping[Action, ActionMove]:
        return {move.action: move for move in self.moves}

    def canonical_text(self) -> str:
        blockers = ",".join(f"C{value}" for value in sorted(self.blocking_colors)) or "-"
        goals = ",".join(f"C{value}" for value in sorted(self.goal_colors)) or "-"
        lines = [
            "ARC-MECHANICS 2",
            f"ENTITY C{self.moving_color}",
            f"BACKGROUND C{self.background_color}",
            f"BLOCKERS {blockers}",
            f"GOALS {goals}",
            f"INTERACTION C{self.interaction_color} MODE {self.contact_mode.value}",
        ]
        lines.extend(
            move.canonical_line()
            for move in sorted(self.moves, key=lambda item: item.action.value)
        )
        lines.append("END")
        return "\n".join(lines)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MechanicsExecutionV2:
    before: Grid
    action: Action
    after: Grid
    status: MechanicsStatus
    changed: bool
    blocked: bool
    moved: bool
    contacted_color: int | None
    contact_mode: ContactMode | None


@dataclass(frozen=True)
class MechanicsObservationV2:
    before: Grid
    action: Action
    after: Grid
    terminal: bool


@dataclass(frozen=True)
class MechanicsReplayMismatchV2:
    index: int
    action: Action
    expected_after: Grid
    predicted_after: Grid
    expected_terminal: bool
    predicted_terminal: bool
    differing_cells: int


@dataclass(frozen=True)
class MechanicsReplayResultV2:
    consistent: bool
    checked: int
    mismatch: MechanicsReplayMismatchV2 | None


def _find_unique_color(grid: Grid, color: int) -> tuple[int, int] | None:
    positions = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == color
    ]
    if len(positions) != 1:
        return None
    return positions[0]


def execute_v2(
    program: MechanicsProgramV2,
    grid: Sequence[Sequence[int]],
    action: Action,
) -> MechanicsExecutionV2:
    before = normalize_grid(grid)
    move = program.by_action.get(action)
    source = _find_unique_color(before, program.moving_color)
    if move is None or source is None:
        return MechanicsExecutionV2(
            before,
            action,
            before,
            MechanicsStatus.ACTIVE,
            False,
            True,
            False,
            None,
            None,
        )

    height = len(before)
    width = len(before[0])
    destination = (source[0] + move.dy, source[1] + move.dx)
    if not (0 <= destination[0] < height and 0 <= destination[1] < width):
        return MechanicsExecutionV2(
            before,
            action,
            before,
            MechanicsStatus.ACTIVE,
            False,
            True,
            False,
            None,
            None,
        )

    destination_color = before[destination[0]][destination[1]]
    if destination_color in program.blocking_colors:
        return MechanicsExecutionV2(
            before,
            action,
            before,
            MechanicsStatus.ACTIVE,
            False,
            True,
            False,
            destination_color,
            ContactMode.BLOCK,
        )

    canvas = [list(row) for row in before]
    status = (
        MechanicsStatus.WIN
        if destination_color in program.goal_colors
        else MechanicsStatus.ACTIVE
    )
    contacted_mode: ContactMode | None = None
    blocked = False
    moved = False

    if destination_color == program.background_color or destination_color in program.goal_colors:
        canvas[source[0]][source[1]] = program.background_color
        canvas[destination[0]][destination[1]] = program.moving_color
        moved = True
    elif destination_color == program.interaction_color:
        contacted_mode = program.contact_mode
        if program.contact_mode is ContactMode.BLOCK:
            blocked = True
        elif program.contact_mode is ContactMode.COLLECT:
            canvas[source[0]][source[1]] = program.background_color
            canvas[destination[0]][destination[1]] = program.moving_color
            moved = True
        elif program.contact_mode is ContactMode.ERASE:
            canvas[destination[0]][destination[1]] = program.background_color
        elif program.contact_mode is ContactMode.SWAP:
            canvas[source[0]][source[1]] = program.interaction_color
            canvas[destination[0]][destination[1]] = program.moving_color
            moved = True
        elif program.contact_mode is ContactMode.PUSH:
            pushed_to = (destination[0] + move.dy, destination[1] + move.dx)
            can_push = (
                0 <= pushed_to[0] < height
                and 0 <= pushed_to[1] < width
                and before[pushed_to[0]][pushed_to[1]] == program.background_color
            )
            if can_push:
                canvas[source[0]][source[1]] = program.background_color
                canvas[destination[0]][destination[1]] = program.moving_color
                canvas[pushed_to[0]][pushed_to[1]] = program.interaction_color
                moved = True
            else:
                blocked = True
        else:  # pragma: no cover - exhaustive Enum protection
            raise MechanicsV2Error(f"unsupported contact mode: {program.contact_mode}")
    else:
        # Undeclared colors are conservative blockers. This makes omissions
        # falsifiable and prevents the interpreter from supplying semantics.
        blocked = True

    after = normalize_grid(canvas)
    return MechanicsExecutionV2(
        before=before,
        action=action,
        after=after,
        status=status,
        changed=after != before,
        blocked=blocked,
        moved=moved,
        contacted_color=(
            destination_color
            if destination_color != program.background_color
            else None
        ),
        contact_mode=contacted_mode,
    )


def _differing_cells(left: Grid, right: Grid) -> int:
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        return max(len(left) * len(left[0]), len(right) * len(right[0]))
    return sum(
        left[row][column] != right[row][column]
        for row in range(len(left))
        for column in range(len(left[0]))
    )


def replay_v2(
    program: MechanicsProgramV2,
    records: Sequence[MechanicsObservationV2],
) -> MechanicsReplayResultV2:
    for index, record in enumerate(records):
        prediction = execute_v2(program, record.before, record.action)
        predicted_terminal = prediction.status is MechanicsStatus.WIN
        if prediction.after != record.after or predicted_terminal != record.terminal:
            return MechanicsReplayResultV2(
                consistent=False,
                checked=index + 1,
                mismatch=MechanicsReplayMismatchV2(
                    index=index,
                    action=record.action,
                    expected_after=record.after,
                    predicted_after=prediction.after,
                    expected_terminal=record.terminal,
                    predicted_terminal=predicted_terminal,
                    differing_cells=_differing_cells(record.after, prediction.after),
                ),
            )
    return MechanicsReplayResultV2(True, len(records), None)


def mapping_from_program(program: MechanicsProgramV2) -> Mapping[Action, Direction]:
    reverse = {delta: direction for direction, delta in DIRECTION_DELTA.items()}
    return {
        action: reverse[(move.dy, move.dx)]
        for action, move in program.by_action.items()
    }


def enumerate_candidate_programs_v2(
    *,
    moving_color: int,
    background_color: int,
    blocking_colors: Sequence[int],
    goal_colors: Sequence[int],
    interaction_color: int,
    contact_modes: Sequence[ContactMode] = tuple(ContactMode),
) -> tuple[MechanicsProgramV2, ...]:
    """Enumerate a bounded exact action-mapping × contact-mode version space."""

    programs: list[MechanicsProgramV2] = []
    for permutation in itertools.permutations(tuple(Direction)):
        moves = tuple(
            ActionMove(action, *DIRECTION_DELTA[direction])
            for action, direction in zip(tuple(Action), permutation, strict=True)
        )
        for mode in contact_modes:
            programs.append(
                MechanicsProgramV2(
                    moving_color=moving_color,
                    background_color=background_color,
                    blocking_colors=tuple(sorted(set(blocking_colors))),
                    goal_colors=tuple(sorted(set(goal_colors))),
                    interaction_color=interaction_color,
                    contact_mode=mode,
                    moves=moves,
                )
            )
    return tuple(programs)


def filter_consistent_v2(
    programs: Iterable[MechanicsProgramV2],
    records: Sequence[MechanicsObservationV2],
) -> tuple[MechanicsProgramV2, ...]:
    return tuple(program for program in programs if replay_v2(program, records).consistent)


def disagreement_partition_v2(
    programs: Sequence[MechanicsProgramV2],
    grid: Sequence[Sequence[int]],
    action: Action,
) -> Mapping[tuple[Grid, MechanicsStatus], tuple[MechanicsProgramV2, ...]]:
    groups: dict[
        tuple[Grid, MechanicsStatus], list[MechanicsProgramV2]
    ] = defaultdict(list)
    for program in programs:
        result = execute_v2(program, grid, action)
        groups[(result.after, result.status)].append(program)
    return {key: tuple(value) for key, value in groups.items()}


def partition_entropy_v2(
    groups: Mapping[object, Sequence[MechanicsProgramV2]],
) -> float:
    total = sum(len(group) for group in groups.values())
    if total == 0:
        return 0.0
    return -sum(
        (len(group) / total) * math.log2(len(group) / total)
        for group in groups.values()
        if group
    )


def most_informative_action_v2(
    programs: Sequence[MechanicsProgramV2],
    grid: Sequence[Sequence[int]],
) -> tuple[Action, float]:
    scored = [
        (
            action,
            partition_entropy_v2(disagreement_partition_v2(programs, grid, action)),
        )
        for action in Action
    ]
    return max(scored, key=lambda item: (item[1], -list(Action).index(item[0])))


def shortest_plan_v2(
    program: MechanicsProgramV2,
    start: Sequence[Sequence[int]],
    *,
    max_depth: int = 96,
    actions: Sequence[Action] = tuple(Action),
) -> tuple[Action, ...] | None:
    """Generic BFS through states predicted by one version-2 program."""

    initial = normalize_grid(start)
    queue: deque[tuple[Grid, tuple[Action, ...]]] = deque([(initial, ())])
    visited = {initial}
    while queue:
        state, prefix = queue.popleft()
        if len(prefix) >= max_depth:
            continue
        for action in actions:
            result = execute_v2(program, state, action)
            candidate = (*prefix, action)
            if result.status is MechanicsStatus.WIN:
                return candidate
            if result.after == state or result.after in visited:
                continue
            visited.add(result.after)
            queue.append((result.after, candidate))
    return None
