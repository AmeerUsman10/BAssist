"""A small, typed, executable ARC-DSL kernel.

This first kernel intentionally supports only deterministic color-region motion.
It is sufficient to express the Phase-0 hidden-action games and to test the
program-induction / exact-replay architecture before the language is expanded.

The interpreter is semantic-free. A program must explicitly state which color
moves, which colors block it, which colors are terminal targets, and how each
action transforms the grid.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import math
import re
from typing import Iterable, Mapping, Sequence

from .codec import Grid, normalize_grid
from .phase0_hidden_action import Action, Direction, DIRECTION_DELTA, GameSpec, StepRecord


class DSLError(ValueError):
    """Raised when a program is syntactically invalid or cannot execute."""


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    WIN = "WIN"


@dataclass(frozen=True)
class MoveColorRule:
    action: Action
    moving_color: int
    dy: int
    dx: int
    background_color: int
    blocking_colors: tuple[int, ...] = ()
    win_on_colors: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("moving_color", self.moving_color),
            ("background_color", self.background_color),
        ):
            if value < 0 or value > 15:
                raise DSLError(f"{name} must be in 0..15")
        for color in (*self.blocking_colors, *self.win_on_colors):
            if color < 0 or color > 15:
                raise DSLError("rule colors must be in 0..15")
        if self.dy == 0 and self.dx == 0:
            raise DSLError("MOVE requires a non-zero displacement")
        if abs(self.dy) > 64 or abs(self.dx) > 64:
            raise DSLError("MOVE displacement is outside the bounded DSL")
        if self.moving_color == self.background_color:
            raise DSLError("moving and background colors must differ")

    def canonical_line(self) -> str:
        block = ",".join(f"C{color}" for color in sorted(set(self.blocking_colors))) or "-"
        win = ",".join(f"C{color}" for color in sorted(set(self.win_on_colors))) or "-"
        return (
            f"RULE {self.action.value} MOVE C{self.moving_color} "
            f"DY {self.dy} DX {self.dx} BG C{self.background_color} "
            f"BLOCK {block} WIN {win}"
        )


@dataclass(frozen=True)
class Program:
    rules: tuple[MoveColorRule, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise DSLError(f"unsupported ARC-DSL version: {self.version}")
        if not self.rules:
            raise DSLError("program must contain at least one rule")
        actions = [rule.action for rule in self.rules]
        if len(actions) != len(set(actions)):
            raise DSLError("a program may define at most one rule per action")

    @property
    def by_action(self) -> Mapping[Action, MoveColorRule]:
        return {rule.action: rule for rule in self.rules}

    def canonical_text(self) -> str:
        lines = [f"ARC-DSL {self.version}"]
        lines.extend(rule.canonical_line() for rule in sorted(self.rules, key=lambda item: item.action.value))
        lines.append("END")
        return "\n".join(lines)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Execution:
    before: Grid
    action: Action
    after: Grid
    status: Status
    changed: bool
    blocked: bool
    contacted_colors: tuple[int, ...]


@dataclass(frozen=True)
class ReplayMismatch:
    index: int
    action: Action
    expected_after: Grid
    predicted_after: Grid
    expected_terminal: bool
    predicted_terminal: bool
    differing_cells: int


@dataclass(frozen=True)
class ReplayResult:
    consistent: bool
    checked: int
    mismatch: ReplayMismatch | None


_COLOR_RE = re.compile(r"^C([0-9]|1[0-5])$")
_INT_RE = re.compile(r"^-?[0-9]+$")


def _parse_color(token: str) -> int:
    match = _COLOR_RE.match(token)
    if match is None:
        raise DSLError(f"expected color token C0..C15, received {token!r}")
    return int(match.group(1))


def _parse_color_set(token: str) -> tuple[int, ...]:
    if token == "-":
        return ()
    values = tuple(_parse_color(part) for part in token.split(","))
    if len(values) != len(set(values)):
        raise DSLError("color sets may not contain duplicates")
    return tuple(sorted(values))


def _parse_int(token: str, label: str) -> int:
    if _INT_RE.match(token) is None:
        raise DSLError(f"expected integer after {label}, received {token!r}")
    return int(token)


def parse_program(text: str) -> Program:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise DSLError("program is incomplete")
    if lines[0] != "ARC-DSL 1":
        raise DSLError("program must begin with 'ARC-DSL 1'")
    if lines[-1] != "END":
        raise DSLError("program must end with 'END'")

    rules: list[MoveColorRule] = []
    for line_number, line in enumerate(lines[1:-1], start=2):
        parts = line.split()
        if len(parts) != 15:
            raise DSLError(f"line {line_number}: expected 15 fields, received {len(parts)}")
        if parts[0] != "RULE" or parts[2] != "MOVE":
            raise DSLError(f"line {line_number}: only RULE ... MOVE is supported")
        try:
            action = Action(parts[1])
        except ValueError as exc:
            raise DSLError(f"line {line_number}: invalid action {parts[1]!r}") from exc
        expected_labels = {4: "DY", 6: "DX", 8: "BG", 10: "BLOCK", 12: "WIN"}
        for index, label in expected_labels.items():
            if parts[index] != label:
                raise DSLError(f"line {line_number}: expected {label} at field {index + 1}")
        if parts[3].startswith("C") is False:
            raise DSLError(f"line {line_number}: MOVE requires a color")
        if parts[9].startswith("C") is False:
            raise DSLError(f"line {line_number}: BG requires a color")
        # The grammar has 14 semantic fields plus an optional future-proof
        # terminator. Version 1 requires a literal semicolon as field 15.
        if parts[14] != ";":
            raise DSLError(f"line {line_number}: rule must end with ';'")
        rules.append(
            MoveColorRule(
                action=action,
                moving_color=_parse_color(parts[3]),
                dy=_parse_int(parts[5], "DY"),
                dx=_parse_int(parts[7], "DX"),
                background_color=_parse_color(parts[9]),
                blocking_colors=_parse_color_set(parts[11]),
                win_on_colors=_parse_color_set(parts[13]),
            )
        )
    return Program(tuple(rules))


def canonical_program(program: Program) -> str:
    """Serialize a program in its unique canonical form."""

    # Add the grammar terminator during textual serialization. Keeping it out
    # of the dataclass representation makes equality structural.
    lines = [f"ARC-DSL {program.version}"]
    for rule in sorted(program.rules, key=lambda item: item.action.value):
        lines.append(rule.canonical_line() + " ;")
    lines.append("END")
    return "\n".join(lines)


def roundtrip_program(program: Program) -> Program:
    return parse_program(canonical_program(program))


def execute(program: Program, grid: Sequence[Sequence[int]], action: Action) -> Execution:
    before = normalize_grid(grid)
    rule = program.by_action.get(action)
    if rule is None:
        return Execution(before, action, before, Status.ACTIVE, False, True, ())

    moving_cells = [
        (y, x)
        for y, row in enumerate(before)
        for x, value in enumerate(row)
        if value == rule.moving_color
    ]
    if not moving_cells:
        return Execution(before, action, before, Status.ACTIVE, False, True, ())

    height = len(before)
    width = len(before[0])
    moving_set = set(moving_cells)
    destinations: list[tuple[int, int]] = []
    contacted: set[int] = set()

    for y, x in moving_cells:
        ny = y + rule.dy
        nx = x + rule.dx
        if ny < 0 or ny >= height or nx < 0 or nx >= width:
            return Execution(before, action, before, Status.ACTIVE, False, True, ())
        destination_color = before[ny][nx]
        if (ny, nx) not in moving_set and destination_color in rule.blocking_colors:
            return Execution(before, action, before, Status.ACTIVE, False, True, ())
        if (ny, nx) not in moving_set:
            contacted.add(destination_color)
        destinations.append((ny, nx))

    canvas = [list(row) for row in before]
    for y, x in moving_cells:
        canvas[y][x] = rule.background_color
    for y, x in destinations:
        canvas[y][x] = rule.moving_color

    after = normalize_grid(canvas)
    status = Status.WIN if contacted.intersection(rule.win_on_colors) else Status.ACTIVE
    return Execution(
        before=before,
        action=action,
        after=after,
        status=status,
        changed=after != before,
        blocked=False,
        contacted_colors=tuple(sorted(contacted)),
    )


def _differing_cells(left: Grid, right: Grid) -> int:
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        return max(len(left) * len(left[0]), len(right) * len(right[0]))
    return sum(
        left[y][x] != right[y][x]
        for y in range(len(left))
        for x in range(len(left[0]))
    )


def replay(program: Program, records: Sequence[StepRecord]) -> ReplayResult:
    for index, record in enumerate(records):
        prediction = execute(program, record.before, record.action)
        expected_terminal = record.status in {"LEVEL_WIN", "GAME_WIN"}
        predicted_terminal = prediction.status is Status.WIN
        if prediction.after != record.after or expected_terminal != predicted_terminal:
            return ReplayResult(
                consistent=False,
                checked=index + 1,
                mismatch=ReplayMismatch(
                    index=index,
                    action=record.action,
                    expected_after=record.after,
                    predicted_after=prediction.after,
                    expected_terminal=expected_terminal,
                    predicted_terminal=predicted_terminal,
                    differing_cells=_differing_cells(record.after, prediction.after),
                ),
            )
    return ReplayResult(consistent=True, checked=len(records), mismatch=None)


def program_from_phase0_spec(spec: GameSpec) -> Program:
    rules: list[MoveColorRule] = []
    for action, direction in spec.action_to_direction.items():
        dy, dx = DIRECTION_DELTA[direction]
        rules.append(
            MoveColorRule(
                action=action,
                moving_color=spec.palette.agent,
                dy=dy,
                dx=dx,
                background_color=spec.palette.background,
                blocking_colors=(spec.palette.wall,),
                win_on_colors=(spec.palette.goal,),
            )
        )
    return Program(tuple(rules))


def enumerate_phase0_programs(spec: GameSpec) -> tuple[Program, ...]:
    """Enumerate the 24 possible hidden action mappings for a Phase-0 game.

    This is an exact small-world oracle used for data generation and regression,
    not the final private-game hypothesis generator.
    """

    programs: list[Program] = []
    directions = tuple(Direction)
    for permutation in itertools.permutations(directions):
        mapping = {action: direction for action, direction in zip(Action, permutation, strict=True)}
        candidate = GameSpec(
            game_seed=spec.game_seed,
            action_to_direction=mapping,
            palette=spec.palette,
            levels=spec.levels,
            probe_order=spec.probe_order,
        )
        programs.append(program_from_phase0_spec(candidate))
    return tuple(programs)


def filter_consistent(programs: Iterable[Program], records: Sequence[StepRecord]) -> tuple[Program, ...]:
    return tuple(program for program in programs if replay(program, records).consistent)


def disagreement_partition(
    programs: Sequence[Program],
    grid: Sequence[Sequence[int]],
    action: Action,
) -> Mapping[tuple[Grid, Status], tuple[Program, ...]]:
    groups: dict[tuple[Grid, Status], list[Program]] = defaultdict(list)
    for program in programs:
        result = execute(program, grid, action)
        groups[(result.after, result.status)].append(program)
    return {key: tuple(value) for key, value in groups.items()}


def entropy_of_partition(groups: Mapping[object, Sequence[Program]]) -> float:
    total = sum(len(group) for group in groups.values())
    if total == 0:
        return 0.0
    return -sum(
        (len(group) / total) * math.log2(len(group) / total)
        for group in groups.values()
        if group
    )


def most_informative_action(
    programs: Sequence[Program],
    grid: Sequence[Sequence[int]],
    actions: Iterable[Action] = tuple(Action),
) -> tuple[Action, float]:
    scored = [
        (action, entropy_of_partition(disagreement_partition(programs, grid, action)))
        for action in actions
    ]
    return max(scored, key=lambda item: (item[1], -list(Action).index(item[0])))


def shortest_plan(
    program: Program,
    start: Sequence[Sequence[int]],
    *,
    max_depth: int = 64,
    actions: Sequence[Action] = tuple(Action),
) -> tuple[Action, ...] | None:
    """Generic breadth-first search through states predicted by one program."""

    initial = normalize_grid(start)
    queue: deque[tuple[Grid, tuple[Action, ...]]] = deque([(initial, ())])
    visited = {initial}

    while queue:
        state, prefix = queue.popleft()
        if len(prefix) >= max_depth:
            continue
        for action in actions:
            result = execute(program, state, action)
            candidate_prefix = (*prefix, action)
            if result.status is Status.WIN:
                return candidate_prefix
            if result.after == state or result.after in visited:
                continue
            visited.add(result.after)
            queue.append((result.after, candidate_prefix))
    return None
