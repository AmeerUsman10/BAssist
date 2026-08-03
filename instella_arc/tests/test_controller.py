from __future__ import annotations

from arcgpt2.official_observation import OfficialFrameSequence

from instella_arc.action_belief import ActionBeliefState, EffectSignature
from instella_arc.controller import ClosedLoopController, PlannedAction
from instella_arc.receipts import RichActionObservation


def _sequence(grid, *, levels=0, actions=None, state="IN_PROGRESS"):
    return OfficialFrameSequence(
        game_id="test",
        state=state,
        levels_completed=levels,
        win_levels=3,
        full_reset=False,
        available_actions=tuple(
            actions
            or ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4")
        ),
        rendered_frames=(tuple(tuple(row) for row in grid),),
    )


def _movement_observation(action: str, vector, color=2):
    effect = EffectSignature(
        unchanged=False,
        changed_cell_count=2,
        translation_vectors=(vector,),
        colors_added=(),
        colors_removed=(),
        level_progress=0,
        terminal_state="IN_PROGRESS",
    )
    return RichActionObservation(
        action=action,
        coordinate=None,
        before_sha256="before-" + action,
        after_sha256="after-" + action,
        effect=effect,
        metadata={
            "moved_components": [
                {
                    "color": color,
                    "normalized_shape": [[0, 0]],
                    "delta_row": vector[0],
                    "delta_column": vector[1],
                }
            ]
        },
        receipt_sha256="receipt-" + action,
    )


def _controller_with_navigation_knowledge(grid):
    controller = ClosedLoopController()
    controller.initialize(_sequence(grid))
    for action, vector in (
        ("ACTION1", (-1, 0)),
        ("ACTION2", (1, 0)),
        ("ACTION3", (0, -1)),
        ("ACTION4", (0, 1)),
    ):
        controller.state.belief.add_transition(
            _movement_observation(action, vector)
        )
    return controller


def test_uninformed_controller_probes_and_avoids_reset() -> None:
    controller = ClosedLoopController()
    controller.initialize(_sequence(((0, 2, 0), (0, 0, 0))))
    decision = controller.choose_action()
    assert decision.source == "probe"
    assert decision.action != "RESET"
    assert controller.state.last_action == decision


def test_informed_controller_starts_executable_navigation_plan() -> None:
    grid = (
        (0, 0, 0, 0, 0),
        (0, 2, 0, 0, 3),
        (0, 0, 0, 0, 0),
    )
    controller = _controller_with_navigation_knowledge(grid)
    decision = controller.choose_action()
    assert decision.source == "navigation"
    assert decision.plan_id is not None
    assert decision.expected_delta in {
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    }
    assert controller.state.active_navigation is not None


def test_exact_no_change_contradiction_invalidates_navigation_plan() -> None:
    grid = (
        (0, 0, 0, 0, 0),
        (0, 2, 0, 0, 3),
        (0, 0, 0, 0, 0),
    )
    controller = _controller_with_navigation_knowledge(grid)
    decision = controller.choose_action()
    plan_id = decision.plan_id
    assert plan_id is not None
    receipt = controller.observe(_sequence(grid))
    assert receipt is not None
    assert receipt.effect.unchanged
    assert controller.state.active_navigation is None
    assert plan_id in controller.state.failed_plan_ids


def test_level_progress_clears_local_plan_but_preserves_game_action_beliefs() -> None:
    grid = ((0, 2, 0, 3),)
    controller = _controller_with_navigation_knowledge(grid)
    controller.state.failed_plan_ids.add("old-plan")
    profile_trials_before = controller.state.belief.profile("ACTION4").trials
    controller.state.last_action = PlannedAction(
        action="ACTION4",
        coordinate=None,
        source="test",
        purpose="trigger level completion",
    )
    next_grid = ((0, 0, 2, 3),)
    receipt = controller.observe(_sequence(next_grid, levels=1))
    assert receipt is not None
    assert receipt.effect.level_progress == 1
    assert controller.state.failed_plan_ids == set()
    assert controller.state.actions_this_level == 0
    assert controller.state.level_epoch == 1
    assert controller.state.belief.profile("ACTION4").trials == profile_trials_before + 1


def test_controller_requires_observation_between_actions() -> None:
    controller = ClosedLoopController()
    controller.initialize(_sequence(((0, 2), (0, 0))))
    controller.choose_action()
    try:
        controller.choose_action()
    except RuntimeError:
        pass
    else:
        raise AssertionError("controller must observe the last result before acting again")
