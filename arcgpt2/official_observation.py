"""Lossless ARC-AGI-3 observation adapter with explicit animation handling.

Official ``FrameData`` observations may contain several rendered frames for one
action. Treating every animation frame as a new environment state was an early
failure mode in the project. This adapter therefore keeps three objects
separate without introducing learned perception:

1. the full exact rendered sequence returned by the toolkit;
2. exact deltas between consecutive rendered frames;
3. the final rendered grid used as the persistent state for the next action.

The adapter is intentionally duck-typed so unit tests do not need the official
package and the core research code remains usable in Python 3.11 environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

from .codec import Grid, normalize_grid
from .codec_text import encode_delta_text, encode_grid


class OfficialObservationError(ValueError):
    """Raised when an official observation cannot be represented losslessly."""


def _enum_text(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _action_text(value: Any) -> str:
    """Canonicalize toolkit action IDs and enum values to ``ACTIONN``/``RESET``.

    Depending on the serialization layer, ``available_actions`` may contain
    ``GameAction`` enums, integer IDs, digit strings, or strings such as
    ``GameAction.ACTION4``. The environment's executable action space uses enum
    names, so the observation adapter must not leak the transport representation
    into the controller.
    """

    name = getattr(value, "name", None)
    if name is not None:
        text = str(name).upper()
    elif isinstance(value, bool):
        raise OfficialObservationError("boolean is not a valid game action")
    elif isinstance(value, int):
        text = str(value)
    else:
        text = str(value).strip().upper()

    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text == "RESET":
        return "RESET"
    if text.isdigit():
        action_id = int(text)
        if action_id == 0:
            return "RESET"
        if 1 <= action_id <= 7:
            return f"ACTION{action_id}"
        raise OfficialObservationError(f"action id must be in 0..7: {action_id}")
    match = re.fullmatch(r"A(?:CTION)?([0-7])", text)
    if match:
        action_id = int(match.group(1))
        return "RESET" if action_id == 0 else f"ACTION{action_id}"
    raise OfficialObservationError(f"unrecognized game action: {value!r}")


def _to_nested_lists(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _normalize_frame_stack(value: Any) -> tuple[Grid, ...]:
    raw = _to_nested_lists(value)
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise OfficialObservationError("FrameData.frame must be an iterable")
    if not raw:
        return ()

    # One 2-D grid and a list of 2-D grids are both accepted. The official
    # toolkit currently returns a sequence, but accepting the single-grid form
    # makes the contract robust to serialization and test fixtures.
    first = _to_nested_lists(raw[0])
    if isinstance(first, (list, tuple)) and first:
        first_inner = _to_nested_lists(first[0])
    else:
        first_inner = None
    looks_like_single_grid = not isinstance(first_inner, (list, tuple))
    layers = (raw,) if looks_like_single_grid else raw

    grids: list[Grid] = []
    shape: tuple[int, int] | None = None
    for index, layer in enumerate(layers):
        try:
            grid = normalize_grid(_to_nested_lists(layer))
        except Exception as exc:
            raise OfficialObservationError(
                f"rendered frame {index} is not a valid integer grid: {exc}"
            ) from exc
        current_shape = (len(grid), len(grid[0]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise OfficialObservationError(
                "all rendered frames for one action must share a shape"
            )
        grids.append(grid)
    return tuple(grids)


def _available_actions(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = (value,)
    return tuple(sorted({_action_text(item) for item in items}))


@dataclass(frozen=True)
class OfficialFrameSequence:
    game_id: str | None
    state: str
    levels_completed: int | None
    win_levels: int | None
    full_reset: bool | None
    available_actions: tuple[str, ...]
    rendered_frames: tuple[Grid, ...]

    @classmethod
    def from_frame_data(cls, frame_data: Any) -> "OfficialFrameSequence":
        if frame_data is None:
            raise OfficialObservationError("frame_data may not be None")
        rendered = _normalize_frame_stack(getattr(frame_data, "frame", None))
        return cls(
            game_id=(
                str(getattr(frame_data, "game_id"))
                if getattr(frame_data, "game_id", None) is not None
                else None
            ),
            state=_enum_text(getattr(frame_data, "state", None)),
            levels_completed=(
                int(getattr(frame_data, "levels_completed"))
                if getattr(frame_data, "levels_completed", None) is not None
                else None
            ),
            win_levels=(
                int(getattr(frame_data, "win_levels"))
                if getattr(frame_data, "win_levels", None) is not None
                else None
            ),
            full_reset=(
                bool(getattr(frame_data, "full_reset"))
                if getattr(frame_data, "full_reset", None) is not None
                else None
            ),
            available_actions=_available_actions(
                getattr(frame_data, "available_actions", None)
            ),
            rendered_frames=rendered,
        )

    @property
    def final_grid(self) -> Grid | None:
        return self.rendered_frames[-1] if self.rendered_frames else None

    @property
    def shape(self) -> tuple[int, int] | None:
        grid = self.final_grid
        return (len(grid), len(grid[0])) if grid is not None else None

    @property
    def animation_deltas(self) -> tuple[str, ...]:
        return tuple(
            encode_delta_text(before, after)
            for before, after in zip(
                self.rendered_frames,
                self.rendered_frames[1:],
                strict=False,
            )
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()

    def canonical_text(self) -> str:
        """Return a reversible, stable textual record for model input or logs."""

        actions = ",".join(self.available_actions) or "-"
        metadata = (
            f"OFFICIAL_FRAME_SEQUENCE GAME={self.game_id or '-'} "
            f"STATE={self.state} LEVELS={self.levels_completed} "
            f"WIN_LEVELS={self.win_levels} FULL_RESET={self.full_reset} "
            f"ACTIONS={actions} FRAMES={len(self.rendered_frames)}"
        )
        lines = [metadata]
        for index, grid in enumerate(self.rendered_frames):
            lines.append(f"FRAME {index} {encode_grid(grid)}")
            if index > 0:
                lines.append(
                    f"ANIMATION_DELTA {index - 1}->{index} "
                    f"{encode_delta_text(self.rendered_frames[index - 1], grid)}"
                )
        lines.append("END_OFFICIAL_FRAME_SEQUENCE")
        return "\n".join(lines)


@dataclass(frozen=True)
class OfficialActionTransition:
    action: str
    before_final: Grid | None
    observation: OfficialFrameSequence

    @property
    def after_final(self) -> Grid | None:
        return self.observation.final_grid

    @property
    def persistent_delta(self) -> str | None:
        if self.before_final is None or self.after_final is None:
            return None
        return encode_delta_text(self.before_final, self.after_final)

    def canonical_text(self) -> str:
        lines = [f"OFFICIAL_ACTION {self.action}"]
        if self.before_final is None:
            lines.append("BEFORE_FINAL -")
        else:
            lines.append(f"BEFORE_FINAL {encode_grid(self.before_final)}")
        lines.append(self.observation.canonical_text())
        lines.append(
            "PERSISTENT_DELTA "
            + (self.persistent_delta if self.persistent_delta is not None else "-")
        )
        lines.append("END_OFFICIAL_ACTION")
        return "\n".join(lines)


def action_transition(
    action: Any,
    previous: OfficialFrameSequence | None,
    current_frame_data: Any,
) -> OfficialActionTransition:
    """Construct an exact action receipt from two official observations."""

    current = OfficialFrameSequence.from_frame_data(current_frame_data)
    return OfficialActionTransition(
        action=_action_text(action),
        before_final=previous.final_grid if previous is not None else None,
        observation=current,
    )
