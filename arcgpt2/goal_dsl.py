"""Typed terminal-goal predicates for ARC-GPT2.

Mechanics and goals are separated. GPT-2 may propose or score a transition
program and a terminal predicate independently; deterministic replay checks both
against observed state changes and terminal signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable, Protocol, Sequence

from .codec import Grid, normalize_grid
from .dsl import Execution, MoveColorRule, Program, execute
from .phase0_hidden_action import Action, GameSpec, StepRecord


class GoalError(ValueError):
    """Raised when a goal predicate is malformed or cannot be evaluated."""


@dataclass(frozen=True)
class GoalContext:
    before: Grid
    after: Grid
    action: Action
    execution: Execution


class GoalPredicate(Protocol):
    def evaluate(self, context: GoalContext) -> bool: ...

    def canonical(self) -> str: ...


@dataclass(frozen=True)
class ContactedColor:
    color: int

    def __post_init__(self) -> None:
        _validate_color(self.color)

    def evaluate(self, context: GoalContext) -> bool:
        return self.color in context.execution.contacted_colors

    def canonical(self) -> str:
        return f"CONTACT C{self.color}"


@dataclass(frozen=True)
class ColorAbsent:
    color: int

    def __post_init__(self) -> None:
        _validate_color(self.color)

    def evaluate(self, context: GoalContext) -> bool:
        return all(value != self.color for row in context.after for value in row)

    def canonical(self) -> str:
        return f"ABSENT C{self.color}"


class Comparison(str, Enum):
    EQ = "EQ"
    LE = "LE"
    GE = "GE"


@dataclass(frozen=True)
class ColorCount:
    color: int
    comparison: Comparison
    value: int

    def __post_init__(self) -> None:
        _validate_color(self.color)
        if self.value < 0 or self.value > 4096:
            raise GoalError("color-count target must be in 0..4096")

    def evaluate(self, context: GoalContext) -> bool:
        count = sum(value == self.color for row in context.after for value in row)
        if self.comparison is Comparison.EQ:
            return count == self.value
        if self.comparison is Comparison.LE:
            return count <= self.value
        if self.comparison is Comparison.GE:
            return count >= self.value
        raise GoalError(f"unsupported comparison: {self.comparison}")

    def canonical(self) -> str:
        return f"COUNT C{self.color} {self.comparison.value} {self.value}"


class Connectivity(str, Enum):
    FOUR = "FOUR"
    EIGHT = "EIGHT"


@dataclass(frozen=True)
class ColorsTouch:
    left: int
    right: int
    connectivity: Connectivity = Connectivity.FOUR

    def __post_init__(self) -> None:
        _validate_color(self.left)
        _validate_color(self.right)
        if self.left == self.right:
            raise GoalError("ColorsTouch requires two different colors")

    def evaluate(self, context: GoalContext) -> bool:
        frame = context.after
        height = len(frame)
        width = len(frame[0])
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.connectivity is Connectivity.EIGHT:
            offsets.extend(((1, 1), (1, -1), (-1, 1), (-1, -1)))
        for row in range(height):
            for column in range(width):
                if frame[row][column] != self.left:
                    continue
                for dy, dx in offsets:
                    ny = row + dy
                    nx = column + dx
                    if 0 <= ny < height and 0 <= nx < width and frame[ny][nx] == self.right:
                        return True
        return False

    def canonical(self) -> str:
        return f"TOUCH C{self.left} C{self.right} {self.connectivity.value}"


@dataclass(frozen=True)
class NotGoal:
    child: GoalPredicate

    def evaluate(self, context: GoalContext) -> bool:
        return not self.child.evaluate(context)

    def canonical(self) -> str:
        return f"(NOT {self.child.canonical()})"


@dataclass(frozen=True)
class AllGoals:
    children: tuple[GoalPredicate, ...]

    def __post_init__(self) -> None:
        if len(self.children) < 2:
            raise GoalError("AND requires at least two children")

    def evaluate(self, context: GoalContext) -> bool:
        return all(child.evaluate(context) for child in self.children)

    def canonical(self) -> str:
        ordered = sorted(child.canonical() for child in self.children)
        return "(AND " + " ".join(ordered) + ")"


@dataclass(frozen=True)
class AnyGoal:
    children: tuple[GoalPredicate, ...]

    def __post_init__(self) -> None:
        if len(self.children) < 2:
            raise GoalError("OR requires at least two children")

    def evaluate(self, context: GoalContext) -> bool:
        return any(child.evaluate(context) for child in self.children)

    def canonical(self) -> str:
        ordered = sorted(child.canonical() for child in self.children)
        return "(OR " + " ".join(ordered) + ")"


@dataclass(frozen=True)
class GoalProgram:
    predicate: GoalPredicate
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise GoalError(f"unsupported goal DSL version: {self.version}")

    def canonical_text(self) -> str:
        return f"ARC-GOAL {self.version}\n{self.predicate.canonical()}\nEND"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()


def _validate_color(color: int) -> None:
    if color < 0 or color > 15:
        raise GoalError("colors must lie in 0..15")


def strip_terminal(program: Program) -> Program:
    """Remove embedded win colors while preserving transition mechanics."""

    return Program(
        tuple(
            MoveColorRule(
                action=rule.action,
                moving_color=rule.moving_color,
                dy=rule.dy,
                dx=rule.dx,
                background_color=rule.background_color,
                blocking_colors=rule.blocking_colors,
                win_on_colors=(),
            )
            for rule in program.rules
        ),
        version=program.version,
    )


def phase0_goal(spec: GameSpec) -> GoalProgram:
    return GoalProgram(ContactedColor(spec.palette.goal))


def evaluate_goal(
    mechanics: Program,
    goal: GoalProgram,
    before: Sequence[Sequence[int]],
    action: Action,
) -> tuple[bool, GoalContext]:
    state = normalize_grid(before)
    execution = execute(strip_terminal(mechanics), state, action)
    context = GoalContext(
        before=state,
        after=execution.after,
        action=action,
        execution=execution,
    )
    return goal.predicate.evaluate(context), context


@dataclass(frozen=True)
class GoalReplayMismatch:
    index: int
    action: Action
    expected_terminal: bool
    predicted_terminal: bool


@dataclass(frozen=True)
class GoalReplayResult:
    consistent: bool
    checked: int
    mismatch: GoalReplayMismatch | None


def replay_goal(
    mechanics: Program,
    goal: GoalProgram,
    records: Sequence[StepRecord],
) -> GoalReplayResult:
    for index, record in enumerate(records):
        predicted, _ = evaluate_goal(mechanics, goal, record.before, record.action)
        expected = record.status in {"LEVEL_WIN", "GAME_WIN"}
        if predicted != expected:
            return GoalReplayResult(
                consistent=False,
                checked=index + 1,
                mismatch=GoalReplayMismatch(
                    index=index,
                    action=record.action,
                    expected_terminal=expected,
                    predicted_terminal=predicted,
                ),
            )
    return GoalReplayResult(consistent=True, checked=len(records), mismatch=None)


def enumerate_simple_goals(colors: Iterable[int]) -> tuple[GoalProgram, ...]:
    """Generate a bounded generic goal family from mechanically observed colors."""

    palette = tuple(sorted(set(int(color) for color in colors)))
    for color in palette:
        _validate_color(color)
    candidates: list[GoalProgram] = []
    for color in palette:
        candidates.append(GoalProgram(ContactedColor(color)))
        candidates.append(GoalProgram(ColorAbsent(color)))
        candidates.append(GoalProgram(ColorCount(color, Comparison.EQ, 0)))
        candidates.append(GoalProgram(ColorCount(color, Comparison.EQ, 1)))
    for left_index, left in enumerate(palette):
        for right in palette[left_index + 1 :]:
            candidates.append(GoalProgram(ColorsTouch(left, right, Connectivity.FOUR)))
            candidates.append(GoalProgram(ColorsTouch(left, right, Connectivity.EIGHT)))
    unique = {candidate.sha256: candidate for candidate in candidates}
    return tuple(unique[key] for key in sorted(unique))


_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def parse_goal(text: str) -> GoalProgram:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 3 or lines[0] != "ARC-GOAL 1" or lines[-1] != "END":
        raise GoalError("goal program must be ARC-GOAL 1, one expression, and END")
    tokens = _TOKEN_RE.findall(lines[1])
    predicate, cursor = _parse_expression(tokens, 0)
    if cursor != len(tokens):
        raise GoalError("unexpected trailing goal tokens")
    return GoalProgram(predicate)


def _parse_expression(tokens: Sequence[str], cursor: int) -> tuple[GoalPredicate, int]:
    if cursor >= len(tokens):
        raise GoalError("truncated goal expression")
    token = tokens[cursor]
    if token != "(":
        return _parse_atom(tokens, cursor)

    if cursor + 1 >= len(tokens):
        raise GoalError("truncated compound goal")
    operator = tokens[cursor + 1]
    cursor += 2
    children: list[GoalPredicate] = []
    while cursor < len(tokens) and tokens[cursor] != ")":
        child, cursor = _parse_expression(tokens, cursor)
        children.append(child)
    if cursor >= len(tokens) or tokens[cursor] != ")":
        raise GoalError("compound goal is missing ')'")
    cursor += 1
    if operator == "NOT":
        if len(children) != 1:
            raise GoalError("NOT requires exactly one child")
        return NotGoal(children[0]), cursor
    if operator == "AND":
        return AllGoals(tuple(children)), cursor
    if operator == "OR":
        return AnyGoal(tuple(children)), cursor
    raise GoalError(f"unknown compound goal operator: {operator}")


def _parse_atom(tokens: Sequence[str], cursor: int) -> tuple[GoalPredicate, int]:
    operator = tokens[cursor]
    if operator in {"CONTACT", "ABSENT"}:
        if cursor + 1 >= len(tokens):
            raise GoalError(f"{operator} requires a color")
        color = _parse_color(tokens[cursor + 1])
        predicate: GoalPredicate = ContactedColor(color) if operator == "CONTACT" else ColorAbsent(color)
        return predicate, cursor + 2
    if operator == "COUNT":
        if cursor + 3 >= len(tokens):
            raise GoalError("COUNT requires color, comparison, and value")
        color = _parse_color(tokens[cursor + 1])
        try:
            comparison = Comparison(tokens[cursor + 2])
            value = int(tokens[cursor + 3])
        except (ValueError, TypeError) as exc:
            raise GoalError("invalid COUNT expression") from exc
        return ColorCount(color, comparison, value), cursor + 4
    if operator == "TOUCH":
        if cursor + 3 >= len(tokens):
            raise GoalError("TOUCH requires two colors and connectivity")
        left = _parse_color(tokens[cursor + 1])
        right = _parse_color(tokens[cursor + 2])
        try:
            connectivity = Connectivity(tokens[cursor + 3])
        except ValueError as exc:
            raise GoalError("invalid TOUCH connectivity") from exc
        return ColorsTouch(left, right, connectivity), cursor + 4
    raise GoalError(f"unknown goal atom: {operator}")


def _parse_color(token: str) -> int:
    if not token.startswith("C"):
        raise GoalError(f"expected color token, received {token!r}")
    try:
        color = int(token[1:])
    except ValueError as exc:
        raise GoalError(f"invalid color token: {token!r}") from exc
    _validate_color(color)
    return color
