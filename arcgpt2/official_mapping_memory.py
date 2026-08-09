"""Exact-gradient GPT-2 action memory for official public environments.

The validated meta-training protocol learned a *single-observation* write into
an eight-token soft prefix.  This adapter keeps one independent temporary
prefix per simple action so public play does not silently change that contract.
Only small two-cell exact transitions are eligible; larger or ambiguous
official animations remain available to the transition-frontier policy but do
not update the learned memory.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .codec import Grid, encode_transition, normalize_grid
from .phase0_hidden_action import Action, StepRecord


_OFFICIAL_TO_INTERNAL = {
    "ACTION1": Action.A1,
    "ACTION2": Action.A2,
    "ACTION3": Action.A3,
    "ACTION4": Action.A4,
}


def _changed_cells(before: Grid, after: Grid) -> list[tuple[int, int]]:
    if (len(before), len(before[0])) != (len(after), len(after[0])):
        return []
    return [
        (row, column)
        for row in range(len(before))
        for column in range(len(before[0]))
        if before[row][column] != after[row][column]
    ]


def _exact_crop(
    before: Grid,
    after: Grid,
    changed: Sequence[tuple[int, int]],
    *,
    margin: int = 1,
) -> tuple[Grid, Grid]:
    rows = [row for row, _ in changed]
    columns = [column for _, column in changed]
    row_start = max(0, min(rows) - margin)
    row_stop = min(len(before), max(rows) + margin + 1)
    column_start = max(0, min(columns) - margin)
    column_stop = min(len(before[0]), max(columns) + margin + 1)
    return (
        tuple(tuple(row[column_start:column_stop]) for row in before[row_start:row_stop]),
        tuple(tuple(row[column_start:column_stop]) for row in after[row_start:row_stop]),
    )


def _jsonable_receipt(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_receipt(item) for key, item in value.items()}
    return str(value)


class OfficialGPT2MappingMemory:
    """One exact-gradient, one-observation soft prefix for each simple action."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        initial_prefix: Any,
        *,
        device: Any = None,
        inner_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if inner_factory is None:
            from .adaptive_mapping_scores import AdaptiveGPT2MappingScores

            def inner_factory(_: str) -> Any:
                return AdaptiveGPT2MappingScores(
                    model,
                    tokenizer,
                    initial_prefix,
                    inner_learning_rate=0.2,
                    max_length=512,
                    gradient_clip=1.0e9,
                    device=device,
                )

        self._memories = {
            action_name: inner_factory(action_name)
            for action_name in _OFFICIAL_TO_INTERNAL
        }
        self._observed: set[str] = set()
        self._receipts: list[dict[str, Any]] = []

    def reset(self) -> None:
        for memory in self._memories.values():
            reset = getattr(memory, "reset", None)
            if callable(reset):
                reset()
        self._observed.clear()
        self._receipts.clear()

    @property
    def receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._receipts)

    def observe(
        self,
        action_name: str,
        before_grid: Sequence[Sequence[int]],
        after_grid: Sequence[Sequence[int]],
        status: str,
    ) -> dict[str, Any]:
        """Write the first eligible exact local consequence for one action."""

        action_name = str(action_name).upper()
        base = {
            "action": action_name,
            "eligible": False,
            "adapted": False,
            "reason": None,
        }
        if action_name not in _OFFICIAL_TO_INTERNAL:
            base["reason"] = "not_a_simple_cardinal_action"
            return base
        if action_name in self._observed:
            base["reason"] = "single_observation_contract_already_used"
            return base
        before = normalize_grid(before_grid)
        after = normalize_grid(after_grid)
        changed = _changed_cells(before, after)
        if len(changed) != 2:
            base["reason"] = "requires_exactly_two_persistent_cell_changes"
            base["changed_cells"] = len(changed)
            return base
        cropped_before, cropped_after = _exact_crop(before, after, changed)
        crop_shape = (len(cropped_before), len(cropped_before[0]))
        if crop_shape[0] > 8 or crop_shape[1] > 8:
            base["reason"] = "local_transition_crop_exceeds_8x8"
            base["crop_shape"] = list(crop_shape)
            return base

        internal_action = _OFFICIAL_TO_INTERNAL[action_name]
        terminal_status = "GAME_WIN" if str(status).upper() == "WIN" else "ACTIVE"
        record = StepRecord(
            level_index=0,
            step_index=0,
            before=cropped_before,
            action=internal_action,
            after=cropped_after,
            status=terminal_status,
            moved=True,
            transition=encode_transition(cropped_before, cropped_after),
        )
        inner = self._memories[action_name].observe(record)
        digest_payload = json.dumps(
            {"before": cropped_before, "after": cropped_after},
            separators=(",", ":"),
        ).encode("utf-8")
        receipt = {
            **base,
            "eligible": True,
            "adapted": True,
            "reason": "first_eligible_exact_transition",
            "changed_cells": 2,
            "crop_shape": list(crop_shape),
            "crop_sha256": hashlib.sha256(digest_payload).hexdigest(),
            "inner": _jsonable_receipt(inner),
        }
        self._observed.add(action_name)
        self._receipts.append(receipt)
        return receipt

    def direction_scores(self, action_name: str) -> Mapping[str, float]:
        action_name = str(action_name).upper()
        if action_name not in _OFFICIAL_TO_INTERNAL or action_name not in self._observed:
            return {}
        internal_action = _OFFICIAL_TO_INTERNAL[action_name]
        scores = self._memories[action_name].score(
            None,
            (),
            internal_action,
            (),
        )
        return {
            direction.value if hasattr(direction, "value") else str(direction): float(value)
            for direction, value in scores.items()
        }
