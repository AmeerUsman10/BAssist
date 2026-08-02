from __future__ import annotations

from arcgpt2.phase0_hidden_action import Action, Direction
from arcgpt2 import stage02_decomposed as base
from arcgpt2.stage02_sparse import (
    parse_sparse_grid_text,
    sparse_grid_text,
)


def test_sparse_grid_round_trip_with_nonzero_baseline() -> None:
    grid = (
        (5, 5, 5, 5),
        (5, 2, 5, 3),
        (5, 5, 5, 5),
    )
    encoded = sparse_grid_text(grid)
    assert encoded.startswith("3x4;b5;")
    assert parse_sparse_grid_text(encoded) == grid


def test_sparse_grid_is_exact_for_empty_exception_set() -> None:
    grid = ((0, 0), (0, 0))
    assert sparse_grid_text(grid) == "2x2;b0;none"
    assert parse_sparse_grid_text("2x2;b0;none") == grid


def test_import_installs_sparse_protocol_without_changing_decision_logic() -> None:
    assert base.grid_text is sparse_grid_text
    context = base.simulate_context(
        0,
        (Action.A1, Action.A2),
        goal_direction=Direction.UP,
        variant_id="sparse",
    )
    history = base.format_history(
        context.records,
        context.current_grid,
        level_index=context.level_index,
    )
    assert ";b" in history
    decision = base.decision_for_context(context)
    assert decision.target_action == "3"
