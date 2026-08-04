"""Durable rich action receipts linking official frames to online beliefs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from arcgpt2.official_observation import OfficialFrameSequence

from .action_belief import EffectSignature
from .world_state import ComponentTranslation, transition_facts


def _normalized_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    top = min(row for row, _ in cells)
    left = min(column for _, column in cells)
    return tuple(sorted((row - top, column - left) for row, column in cells))


def moved_component_payload(
    translation: ComponentTranslation,
) -> dict[str, Any]:
    return {
        "color": translation.color,
        "before_cells": [list(cell) for cell in translation.before_cells],
        "after_cells": [list(cell) for cell in translation.after_cells],
        "normalized_shape": [
            list(cell) for cell in _normalized_shape(translation.before_cells)
        ],
        "delta_row": translation.delta_row,
        "delta_column": translation.delta_column,
    }


@dataclass(frozen=True)
class RichActionObservation:
    action: str
    coordinate: tuple[int, int] | None
    before_sha256: str
    after_sha256: str
    effect: EffectSignature
    metadata: Mapping[str, Any]
    receipt_sha256: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "coordinate": list(self.coordinate) if self.coordinate is not None else None,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "effect": asdict(self.effect),
            "metadata": dict(self.metadata),
        }


def rich_action_observation(
    *,
    action: str,
    coordinate: tuple[int, int] | None,
    previous: OfficialFrameSequence,
    current: OfficialFrameSequence,
) -> RichActionObservation:
    if previous.final_grid is None or current.final_grid is None:
        raise ValueError("rich action receipt requires persistent before and after grids")
    facts = transition_facts(previous.final_grid, current.final_grid)
    previous_levels = previous.levels_completed or 0
    current_levels = current.levels_completed or 0
    effect = EffectSignature.from_transition(
        facts,
        level_progress=current_levels - previous_levels,
        terminal_state=current.state,
    )
    metadata = {
        "game_id": current.game_id,
        "levels_before": previous.levels_completed,
        "levels_after": current.levels_completed,
        "win_levels": current.win_levels,
        "full_reset": current.full_reset,
        "available_actions_after": list(current.available_actions),
        # Full observation hashes preserve temporary animation evidence that may
        # disappear from the final persistent frame in click/timing games.
        "observation_before_sha256": previous.sha256,
        "observation_after_sha256": current.sha256,
        "rendered_frame_count": len(current.rendered_frames),
        "animation_deltas": list(current.animation_deltas),
        "rendered_frame_sha256": [
            hashlib.sha256(
                json.dumps(frame, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for frame in current.rendered_frames
        ],
        "moved_components": [
            moved_component_payload(translation)
            for translation in facts.translations
        ],
        "changed_cells": [
            {
                "row": change.row,
                "column": change.column,
                "before": change.before,
                "after": change.after,
            }
            for change in facts.changed_cells
        ],
    }
    payload = {
        "action": str(action),
        "coordinate": list(coordinate) if coordinate is not None else None,
        "before_sha256": facts.before_sha256,
        "after_sha256": facts.after_sha256,
        "effect": asdict(effect),
        "metadata": metadata,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RichActionObservation(
        action=str(action),
        coordinate=coordinate,
        before_sha256=facts.before_sha256,
        after_sha256=facts.after_sha256,
        effect=effect,
        metadata=metadata,
        receipt_sha256=digest,
    )
