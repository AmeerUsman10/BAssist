from __future__ import annotations

import math

import pytest

from arcgpt2.constraint_posterior import (
    ConstraintError,
    Variable,
    all_different,
    solve_beam,
    solve_exact,
)
from arcgpt2.phase0_hidden_action import Action, Direction


def test_phase0_permutation_posterior_has_24_assignments_under_uniform_scores() -> None:
    variables = tuple(
        Variable(action.value, tuple(Direction))
        for action in Action
    )
    scores = {
        action.value: {direction: 0.0 for direction in Direction}
        for action in Action
    }
    posterior = solve_exact(
        variables,
        scores,
        (all_different(*(action.value for action in Action)),),
    )
    assert len(posterior.assignments) == 24
    assert posterior.entropy_bits() == pytest.approx(math.log2(24))
    for action in Action:
        marginal = posterior.marginal(action.value)
        assert set(marginal) == set(Direction)
        assert list(marginal.values()) == pytest.approx([0.25] * 4)


def test_unary_scores_and_global_constraint_select_a_consistent_mapping() -> None:
    variables = tuple(Variable(action.value, tuple(Direction)) for action in Action)
    truth = {
        Action.A1: Direction.UP,
        Action.A2: Direction.RIGHT,
        Action.A3: Direction.DOWN,
        Action.A4: Direction.LEFT,
    }
    scores = {
        action.value: {
            direction: (5.0 if direction is truth[action] else 0.0)
            for direction in Direction
        }
        for action in Action
    }
    posterior = solve_exact(
        variables,
        scores,
        (all_different(*(action.value for action in Action)),),
    )
    assert posterior.map_assignment().assignment == {
        action.value: truth[action] for action in Action
    }
    assert posterior.map_assignment().probability > 0.95


def test_global_assignment_repairs_conflicting_independent_argmaxes() -> None:
    variables = (
        Variable("A1", (Direction.UP, Direction.DOWN)),
        Variable("A2", (Direction.UP, Direction.DOWN)),
    )
    scores = {
        "A1": {Direction.UP: 3.0, Direction.DOWN: 2.0},
        "A2": {Direction.UP: 4.0, Direction.DOWN: 0.0},
    }
    posterior = solve_exact(variables, scores, (all_different("A1", "A2"),))
    assert posterior.map_assignment().assignment == {
        "A1": Direction.DOWN,
        "A2": Direction.UP,
    }


def test_beam_matches_exact_when_the_beam_covers_the_space() -> None:
    variables = (
        Variable("x", (0, 1, 2)),
        Variable("y", (0, 1, 2)),
    )
    scores = {
        "x": {0: 0.0, 1: 1.0, 2: 2.0},
        "y": {0: 2.0, 1: 1.0, 2: 0.0},
    }
    constraints = (all_different("x", "y"),)
    exact = solve_exact(variables, scores, constraints)
    beam = solve_beam(variables, scores, constraints, beam_size=9)
    assert beam.map_assignment().assignment == exact.map_assignment().assignment
    assert beam.map_assignment().probability == pytest.approx(
        exact.map_assignment().probability
    )


def test_invalid_constraints_and_scores_fail_closed() -> None:
    with pytest.raises(ConstraintError):
        Variable("", (1,))
    with pytest.raises(ConstraintError):
        Variable("x", ())
    with pytest.raises(ConstraintError):
        all_different("x")
    with pytest.raises(ConstraintError):
        solve_exact((Variable("x", (1, 2)),), {"x": {1: 0.0}})
