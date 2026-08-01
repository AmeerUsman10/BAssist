"""Generic exact posterior over GPT-2-scored typed hypothesis variables.

GPT-2 supplies ordinary-language log scores for candidate answers. This module
combines them under explicit structural constraints, normalizes the resulting
assignments, and exposes marginals. It contains no learned model or hidden
semantic heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Generic, Mapping, Sequence, TypeVar


Value = TypeVar("Value")


class ConstraintError(ValueError):
    """Raised when a constrained posterior is malformed or empty."""


@dataclass(frozen=True)
class Variable(Generic[Value]):
    name: str
    values: tuple[Value, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ConstraintError("variable name may not be empty")
        if not self.values:
            raise ConstraintError(f"variable {self.name!r} has no candidate values")
        if len(self.values) != len(set(self.values)):
            raise ConstraintError(f"variable {self.name!r} contains duplicate values")


PartialAssignment = Mapping[str, object]
Constraint = Callable[[PartialAssignment], bool]


@dataclass(frozen=True)
class ScoredAssignment(Generic[Value]):
    assignment: Mapping[str, Value]
    log_score: float
    probability: float


@dataclass(frozen=True)
class ExactPosterior(Generic[Value]):
    variables: tuple[Variable[Value], ...]
    assignments: tuple[ScoredAssignment[Value], ...]

    def map_assignment(self) -> ScoredAssignment[Value]:
        return self.assignments[0]

    def marginal(self, variable_name: str) -> Mapping[Value, float]:
        variable = next(
            (candidate for candidate in self.variables if candidate.name == variable_name),
            None,
        )
        if variable is None:
            raise ConstraintError(f"unknown variable: {variable_name}")
        probabilities = {value: 0.0 for value in variable.values}
        for item in self.assignments:
            probabilities[item.assignment[variable_name]] += item.probability
        return probabilities

    def entropy_bits(self) -> float:
        return -sum(
            item.probability * math.log2(item.probability)
            for item in self.assignments
            if item.probability > 0.0
        )


def all_different(*variable_names: str) -> Constraint:
    names = tuple(variable_names)
    if len(names) < 2:
        raise ConstraintError("all_different requires at least two variables")

    def constraint(assignment: PartialAssignment) -> bool:
        values = [assignment[name] for name in names if name in assignment]
        return len(values) == len(set(values))

    return constraint


def equal(left: str, right: str) -> Constraint:
    def constraint(assignment: PartialAssignment) -> bool:
        if left not in assignment or right not in assignment:
            return True
        return assignment[left] == assignment[right]

    return constraint


def not_equal(left: str, right: str) -> Constraint:
    def constraint(assignment: PartialAssignment) -> bool:
        if left not in assignment or right not in assignment:
            return True
        return assignment[left] != assignment[right]

    return constraint


def allowed_pairs(left: str, right: str, pairs: Sequence[tuple[object, object]]) -> Constraint:
    allowed = frozenset(pairs)
    if not allowed:
        raise ConstraintError("allowed_pairs may not be empty")

    def constraint(assignment: PartialAssignment) -> bool:
        if left not in assignment or right not in assignment:
            return True
        return (assignment[left], assignment[right]) in allowed

    return constraint


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ConstraintError("cannot normalize an empty score set")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def solve_exact(
    variables: Sequence[Variable[Value]],
    unary_log_scores: Mapping[str, Mapping[Value, float]],
    constraints: Sequence[Constraint] = (),
    *,
    top_k: int | None = None,
) -> ExactPosterior[Value]:
    """Enumerate all structurally valid assignments and normalize their scores."""

    ordered = tuple(variables)
    if not ordered:
        raise ConstraintError("solve_exact requires at least one variable")
    names = [variable.name for variable in ordered]
    if len(names) != len(set(names)):
        raise ConstraintError("variable names must be unique")

    for variable in ordered:
        score_map = unary_log_scores.get(variable.name)
        if score_map is None:
            raise ConstraintError(f"missing score map for {variable.name}")
        missing = [value for value in variable.values if value not in score_map]
        if missing:
            raise ConstraintError(
                f"score map for {variable.name} lacks values: {missing!r}"
            )

    raw: list[tuple[dict[str, Value], float]] = []

    def visit(index: int, assignment: dict[str, Value], score: float) -> None:
        if index == len(ordered):
            raw.append((dict(assignment), score))
            return
        variable = ordered[index]
        for value in variable.values:
            assignment[variable.name] = value
            if all(constraint(assignment) for constraint in constraints):
                visit(
                    index + 1,
                    assignment,
                    score + float(unary_log_scores[variable.name][value]),
                )
            assignment.pop(variable.name)

    visit(0, {}, 0.0)
    if not raw:
        raise ConstraintError("no assignment satisfies the declared constraints")
    raw.sort(
        key=lambda item: (
            -item[1],
            tuple(repr(item[0][name]) for name in names),
        )
    )
    if top_k is not None:
        if top_k < 1:
            raise ConstraintError("top_k must be positive")
        raw = raw[:top_k]

    normalizer = _logsumexp([score for _, score in raw])
    assignments = tuple(
        ScoredAssignment(
            assignment=assignment,
            log_score=score,
            probability=math.exp(score - normalizer),
        )
        for assignment, score in raw
    )
    if not math.isclose(sum(item.probability for item in assignments), 1.0, abs_tol=1e-9):
        raise ConstraintError("assignment posterior did not normalize")
    return ExactPosterior(ordered, assignments)


def solve_beam(
    variables: Sequence[Variable[Value]],
    unary_log_scores: Mapping[str, Mapping[Value, float]],
    constraints: Sequence[Constraint] = (),
    *,
    beam_size: int = 128,
) -> ExactPosterior[Value]:
    """Approximate large assignment spaces while preserving explicit constraints.

    Probabilities are normalized only within the retained beam and must be
    reported as approximate.
    """

    if beam_size < 1:
        raise ConstraintError("beam_size must be positive")
    ordered = tuple(variables)
    if not ordered:
        raise ConstraintError("solve_beam requires variables")

    beam: list[tuple[dict[str, Value], float]] = [({}, 0.0)]
    for variable in ordered:
        score_map = unary_log_scores.get(variable.name)
        if score_map is None:
            raise ConstraintError(f"missing score map for {variable.name}")
        expanded: list[tuple[dict[str, Value], float]] = []
        for assignment, score in beam:
            for value in variable.values:
                candidate = dict(assignment)
                candidate[variable.name] = value
                if all(constraint(candidate) for constraint in constraints):
                    expanded.append((candidate, score + float(score_map[value])))
        if not expanded:
            raise ConstraintError("beam became empty under the declared constraints")
        expanded.sort(
            key=lambda item: (
                -item[1],
                tuple(repr(item[0].get(v.name)) for v in ordered),
            )
        )
        beam = expanded[:beam_size]

    normalizer = _logsumexp([score for _, score in beam])
    assignments = tuple(
        ScoredAssignment(
            assignment=assignment,
            log_score=score,
            probability=math.exp(score - normalizer),
        )
        for assignment, score in beam
    )
    return ExactPosterior(ordered, assignments)
