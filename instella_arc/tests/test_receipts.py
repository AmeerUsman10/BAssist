from __future__ import annotations

from arcgpt2.official_observation import OfficialFrameSequence

from instella_arc.receipts import rich_action_observation


def _sequence(grid, *, levels=0, state="IN_PROGRESS"):
    return OfficialFrameSequence(
        game_id="test",
        state=state,
        levels_completed=levels,
        win_levels=3,
        full_reset=False,
        available_actions=("ACTION1", "ACTION2"),
        rendered_frames=(tuple(tuple(row) for row in grid),),
    )


def test_rich_receipt_preserves_component_motion_and_progress() -> None:
    previous = _sequence(((0, 2, 0), (0, 0, 0)), levels=0)
    current = _sequence(((0, 0, 2), (0, 0, 0)), levels=1)
    receipt = rich_action_observation(
        action="ACTION1",
        coordinate=None,
        previous=previous,
        current=current,
    )
    assert receipt.effect.level_progress == 1
    assert receipt.effect.translation_vectors == ((0, 1),)
    moved = receipt.metadata["moved_components"]
    assert len(moved) >= 1
    target = next(item for item in moved if item["color"] == 2)
    assert target["normalized_shape"] == [[0, 0]]
    assert (target["delta_row"], target["delta_column"]) == (0, 1)
    assert len(receipt.receipt_sha256) == 64


def test_rich_receipt_is_deterministic() -> None:
    previous = _sequence(((0, 2), (0, 0)))
    current = _sequence(((0, 0), (0, 2)))
    first = rich_action_observation(
        action="ACTION2", coordinate=(1, 1), previous=previous, current=current
    )
    second = rich_action_observation(
        action="ACTION2", coordinate=(1, 1), previous=previous, current=current
    )
    assert first == second
