from __future__ import annotations

from instella_arc.action_belief import ActionBeliefState, EffectSignature
from instella_arc.navigation_stride import (
    cardinal_direction,
    generate_stride_navigation_plans,
    infer_stride_movement_bindings,
)
from instella_arc.receipts import RichActionObservation
from instella_arc.world_state import grid_facts


def _movement(action: str, vector, *, color=2):
    return RichActionObservation(
        action=action,
        coordinate=None,
        before_sha256="before-" + action,
        after_sha256="after-" + action,
        effect=EffectSignature(
            unchanged=False,
            changed_cell_count=2,
            translation_vectors=(vector,),
            colors_added=(),
            colors_removed=(),
            level_progress=0,
            terminal_state="IN_PROGRESS",
        ),
        metadata={
            "observation_before_sha256": "obs-before-" + action,
            "observation_after_sha256": "obs-after-" + action,
            "moved_components": [
                {
                    "color": color,
                    "normalized_shape": [[0, 0]],
                    "delta_row": vector[0],
                    "delta_column": vector[1],
                }
            ],
        },
        receipt_sha256="receipt-" + action,
    )


def _belief():
    belief = ActionBeliefState()
    for action, vector in (
        ("ACTION1", (-5, 0)),
        ("ACTION2", (5, 0)),
        ("ACTION3", (0, -5)),
        ("ACTION4", (0, 5)),
    ):
        belief.add_transition(_movement(action, vector))
    return belief


def test_cardinal_direction_accepts_non_unit_pure_axis_stride() -> None:
    assert cardinal_direction((-5, 0)) == "north"
    assert cardinal_direction((10, 0)) == "south"
    assert cardinal_direction((0, -3)) == "west"
    assert cardinal_direction((0, 7)) == "east"
    assert cardinal_direction((1, 1)) is None
    assert cardinal_direction((0, 0)) is None


def test_stride_bindings_preserve_observed_rendered_delta() -> None:
    bindings = infer_stride_movement_bindings(_belief())
    by_direction = {binding.direction: binding for binding in bindings}
    assert by_direction["north"].action == "ACTION1"
    assert by_direction["north"].stride == 5
    assert by_direction["south"].delta_row == 5
    assert by_direction["west"].delta_column == -5
    assert by_direction["east"].delta_column == 5


def test_stride_navigation_generates_executable_plan_on_five_cell_lattice() -> None:
    grid = tuple(
        tuple(
            2 if (row, column) == (5, 5)
            else 3 if (row, column) == (5, 20)
            else 0
            for column in range(26)
        )
        for row in range(11)
    )
    plans = generate_stride_navigation_plans(
        grid_facts(grid),
        _belief(),
        max_plans=64,
    )
    target = next(
        plan
        for plan in plans
        if plan.agent.color == 2
        and plan.target.color == 3
        and plan.target.relation == "overlap"
    )
    assert target.action_sequence == ("ACTION4", "ACTION4", "ACTION4")
    assert target.position_sequence == ((5, 5), (5, 10), (5, 15), (5, 20))
