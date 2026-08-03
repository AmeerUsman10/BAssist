from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from arcgpt2.codec_text import decode_delta, decode_grid
from arcgpt2.official_observation import (
    OfficialFrameSequence,
    OfficialObservationError,
    action_transition,
)


class State(Enum):
    NOT_PLAYED = 0
    IN_PROGRESS = 1


class Action(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2


def _frame_data(frames, *, state=State.IN_PROGRESS, actions=None):
    return SimpleNamespace(
        game_id="test01",
        state=state,
        levels_completed=1,
        win_levels=3,
        full_reset=False,
        available_actions=actions or [Action.RESET, Action.ACTION1],
        frame=frames,
    )


def test_frame_sequence_preserves_every_animation_frame_and_final_state() -> None:
    frames = [
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]
    sequence = OfficialFrameSequence.from_frame_data(_frame_data(frames))
    assert sequence.rendered_frames == tuple(
        tuple(tuple(value for value in row) for row in frame) for frame in frames
    )
    assert sequence.final_grid == ((1, 0), (0, 0))
    assert sequence.shape == (2, 2)
    assert sequence.available_actions == ("ACTION1", "RESET")
    assert len(sequence.animation_deltas) == 2
    assert decode_delta(sequence.rendered_frames[0], sequence.animation_deltas[0]) == sequence.rendered_frames[1]
    assert decode_delta(sequence.rendered_frames[1], sequence.animation_deltas[1]) == sequence.rendered_frames[2]


def test_single_grid_form_is_accepted_without_inventing_an_animation_axis() -> None:
    sequence = OfficialFrameSequence.from_frame_data(
        _frame_data([[0, 1, 0], [0, 0, 0]])
    )
    assert len(sequence.rendered_frames) == 1
    assert sequence.final_grid == ((0, 1, 0), (0, 0, 0))
    assert sequence.animation_deltas == ()


def test_canonical_text_is_stable_and_reversible_at_each_grid_boundary() -> None:
    sequence = OfficialFrameSequence.from_frame_data(
        _frame_data([[[0, 2], [0, 0]], [[0, 0], [2, 0]]])
    )
    text = sequence.canonical_text()
    assert text.startswith("OFFICIAL_FRAME_SEQUENCE")
    assert text.endswith("END_OFFICIAL_FRAME_SEQUENCE")
    frame_lines = [line for line in text.splitlines() if line.startswith("FRAME ")]
    assert len(frame_lines) == 2
    decoded = [decode_grid(line.split(" ", 2)[2]) for line in frame_lines]
    assert tuple(decoded) == sequence.rendered_frames
    assert sequence.sha256 == OfficialFrameSequence.from_frame_data(
        _frame_data([[[0, 2], [0, 0]], [[0, 0], [2, 0]]])
    ).sha256


def test_action_transition_separates_animation_from_persistent_delta() -> None:
    previous = OfficialFrameSequence.from_frame_data(
        _frame_data([[[1, 0], [0, 0]]])
    )
    current_data = _frame_data(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]],
        ]
    )
    transition = action_transition(Action.ACTION1, previous, current_data)
    assert transition.action == "ACTION1"
    assert transition.before_final == ((1, 0), (0, 0))
    assert transition.after_final == ((0, 0), (0, 1))
    assert transition.persistent_delta is not None
    assert decode_delta(transition.before_final, transition.persistent_delta) == transition.after_final
    assert "ANIMATION_DELTA 0->1" in transition.canonical_text()
    assert "PERSISTENT_DELTA" in transition.canonical_text()


def test_empty_frame_sequence_is_metadata_only() -> None:
    sequence = OfficialFrameSequence.from_frame_data(_frame_data([]))
    assert sequence.final_grid is None
    assert sequence.shape is None
    assert sequence.animation_deltas == ()
    assert "FRAMES=0" in sequence.canonical_text()


def test_malformed_or_shape_varying_frames_fail_loudly() -> None:
    with pytest.raises(OfficialObservationError):
        OfficialFrameSequence.from_frame_data(_frame_data([[[0, 0]], [[0], [0]]]))
    with pytest.raises(OfficialObservationError):
        OfficialFrameSequence.from_frame_data(_frame_data("not-a-frame"))
