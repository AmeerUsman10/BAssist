from __future__ import annotations

from arcgpt2.mechanics_v2 import (
    ActionMove,
    ContactMode,
    MechanicsObservationV2,
    MechanicsProgramV2,
    MechanicsStatus,
    enumerate_candidate_programs_v2,
    execute_v2,
    filter_consistent_v2,
    mapping_from_program,
    replay_v2,
    shortest_plan_v2,
)
from arcgpt2.phase0_hidden_action import Action, Direction


def _program(mode: ContactMode) -> MechanicsProgramV2:
    return MechanicsProgramV2(
        moving_color=1,
        background_color=0,
        blocking_colors=(4,),
        goal_colors=(3,),
        interaction_color=2,
        contact_mode=mode,
        moves=(
            ActionMove(Action.A1, -1, 0),
            ActionMove(Action.A2, 1, 0),
            ActionMove(Action.A3, 0, -1),
            ActionMove(Action.A4, 0, 1),
        ),
    )


def _contact_grid():
    return (
        (0, 0, 3, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 1, 2, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )


def test_all_contact_primitives_have_distinct_exact_effects() -> None:
    grid = _contact_grid()
    results = {
        mode: execute_v2(_program(mode), grid, Action.A4)
        for mode in ContactMode
    }
    assert len({result.after for result in results.values()}) == len(ContactMode)

    assert results[ContactMode.BLOCK].after == grid
    assert results[ContactMode.BLOCK].blocked
    assert not results[ContactMode.BLOCK].moved

    collect = results[ContactMode.COLLECT]
    assert collect.after[2][2] == 0
    assert collect.after[2][3] == 1
    assert collect.moved

    push = results[ContactMode.PUSH]
    assert push.after[2][2] == 0
    assert push.after[2][3] == 1
    assert push.after[2][4] == 2
    assert push.moved

    erase = results[ContactMode.ERASE]
    assert erase.after[2][2] == 1
    assert erase.after[2][3] == 0
    assert not erase.moved

    swap = results[ContactMode.SWAP]
    assert swap.after[2][2] == 2
    assert swap.after[2][3] == 1
    assert swap.moved


def test_push_blocks_when_the_cell_behind_is_not_background() -> None:
    grid = (
        (0, 0, 3, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 1, 2, 4),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    result = execute_v2(_program(ContactMode.PUSH), grid, Action.A4)
    assert result.after == grid
    assert result.blocked
    assert not result.moved


def test_goal_contact_is_terminal_and_moves_the_entity() -> None:
    grid = (
        (0, 0, 3, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    result = execute_v2(_program(ContactMode.BLOCK), grid, Action.A1)
    assert result.status is MechanicsStatus.WIN
    assert result.after[0][2] == 1
    assert result.after[1][2] == 0


def test_candidate_version_space_contains_all_mappings_and_modes() -> None:
    programs = enumerate_candidate_programs_v2(
        moving_color=1,
        background_color=0,
        blocking_colors=(4,),
        goal_colors=(3,),
        interaction_color=2,
    )
    assert len(programs) == 24 * len(ContactMode)
    assert len({program.sha256 for program in programs}) == len(programs)
    mappings = {
        tuple(sorted(mapping_from_program(program).items(), key=lambda item: item[0].value))
        for program in programs
    }
    assert len(mappings) == 24


def test_exact_replay_eliminates_wrong_contact_modes() -> None:
    truth = _program(ContactMode.PUSH)
    grid = _contact_grid()
    execution = execute_v2(truth, grid, Action.A4)
    observation = MechanicsObservationV2(
        before=grid,
        action=Action.A4,
        after=execution.after,
        terminal=False,
    )
    candidates = tuple(_program(mode) for mode in ContactMode)
    surviving = filter_consistent_v2(candidates, (observation,))
    assert surviving == (truth,)
    assert replay_v2(truth, (observation,)).consistent
    mismatch = replay_v2(_program(ContactMode.COLLECT), (observation,))
    assert not mismatch.consistent
    assert mismatch.mismatch is not None
    assert mismatch.mismatch.differing_cells > 0


def test_generic_search_can_plan_through_collectable_object() -> None:
    # The interaction object lies in the straight path; COLLECT makes the level
    # solvable, whereas BLOCK would require a detour.
    grid = (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (1, 2, 0, 0, 3),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    plan = shortest_plan_v2(_program(ContactMode.COLLECT), grid, max_depth=12)
    assert plan is not None
    state = grid
    terminal = False
    for action in plan:
        result = execute_v2(_program(ContactMode.COLLECT), state, action)
        state = result.after
        terminal = result.status is MechanicsStatus.WIN
    assert terminal
    assert plan == (Action.A4, Action.A4, Action.A4, Action.A4)


def test_mapping_roundtrip_is_the_declared_cardinal_semantics() -> None:
    mapping = mapping_from_program(_program(ContactMode.ERASE))
    assert mapping == {
        Action.A1: Direction.UP,
        Action.A2: Direction.DOWN,
        Action.A3: Direction.LEFT,
        Action.A4: Direction.RIGHT,
    }
