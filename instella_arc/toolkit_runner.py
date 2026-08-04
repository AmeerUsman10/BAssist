"""ARC-AGI-3 toolkit runner for the closed-loop Instella controller.

The runner is duck-typed around ``EnvironmentWrapper`` so its core behavior is
unit-testable without an API key. In real use it receives an environment created
exactly once by ``arc_agi.Arcade.make``. It never opens scorecards, resets games,
or submits externally on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from arcgpt2.official_observation import OfficialFrameSequence

from .controller import ClosedLoopController, PlannedAction


@dataclass(frozen=True)
class ToolkitActionReceipt:
    step: int
    action: str
    coordinate_row_column: tuple[int, int] | None
    api_data: Mapping[str, int]
    source: str
    purpose: str
    plan_id: str | None
    expected_delta: tuple[int, int] | None
    levels_before: int | None
    levels_after: int | None
    state_after: str
    observation_before_sha256: str
    observation_after_sha256: str
    persistent_unchanged: bool | None
    changed_cell_count: int | None
    translation_vectors: tuple[tuple[int, int], ...]
    animation_frame_count: int
    animation_delta_count: int
    moved_components: tuple[Mapping[str, Any], ...]
    changed_cells: tuple[Mapping[str, Any], ...]
    action_receipt_sha256: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class ToolkitRunResult:
    schema: str
    started_at_utc: str
    finished_at_utc: str
    game_id: str | None
    terminal_state: str
    levels_completed: int | None
    win_levels: int | None
    actions: int
    max_actions: int
    initial_observation_sha256: str
    final_observation_sha256: str
    initial_final_grid: tuple[tuple[int, ...], ...] | None
    final_final_grid: tuple[tuple[int, ...], ...] | None
    controller_summary: Mapping[str, Any]
    trace: tuple[ToolkitActionReceipt, ...]


def _action_name(action: Any) -> str:
    name = getattr(action, "name", None)
    return str(name) if name is not None else str(action)


def _is_complex(action: Any) -> bool:
    method = getattr(action, "is_complex", None)
    return bool(method()) if callable(method) else False


def _enum_action_map(actions: Any) -> dict[str, Any]:
    return {_action_name(action): action for action in tuple(actions or ())}


def _api_data(
    decision: PlannedAction,
    *,
    action_object: Any,
) -> dict[str, int]:
    if not _is_complex(action_object):
        return {}
    if decision.coordinate is None:
        raise ValueError(
            f"complex action {decision.action} requires a row/column coordinate"
        )
    row, column = decision.coordinate
    # ARC toolkit complex action data follows x=column, y=row.
    return {"x": int(column), "y": int(row)}


def _state_is_final(sequence: OfficialFrameSequence) -> bool:
    state = sequence.state.upper()
    if state == "WIN":
        return True
    if (
        sequence.win_levels is not None
        and sequence.levels_completed is not None
        and sequence.levels_completed >= sequence.win_levels
    ):
        return True
    return False


@dataclass
class ToolkitControllerRunner:
    environment: Any
    controller: ClosedLoopController
    max_actions: int = 80

    def _initial_observation(self) -> OfficialFrameSequence:
        raw = getattr(self.environment, "observation_space", None)
        if raw is None:
            reset = getattr(self.environment, "reset", None)
            if not callable(reset):
                raise RuntimeError("environment has no initial observation or reset method")
            raw = reset()
        return OfficialFrameSequence.from_frame_data(raw)

    def run(self) -> ToolkitRunResult:
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        current = self._initial_observation()
        initial = current
        self.controller.initialize(current)
        trace: list[ToolkitActionReceipt] = []

        for step in range(self.max_actions):
            if _state_is_final(current):
                break
            action_map = _enum_action_map(getattr(self.environment, "action_space", ()))
            if not action_map:
                raise RuntimeError("environment exposed no legal actions")
            complexity = {
                name: _is_complex(action) for name, action in action_map.items()
            }
            decision = self.controller.choose_action(
                action_complexity=complexity
            )
            try:
                action_object = action_map[decision.action]
            except KeyError as exc:
                raise RuntimeError(
                    f"controller selected non-legal action {decision.action!r}; "
                    f"available={tuple(action_map)}"
                ) from exc
            data = _api_data(decision, action_object=action_object)
            levels_before = current.levels_completed
            observation_before_sha = current.sha256
            raw_after = self.environment.step(
                action_object,
                data=data,
                reasoning=decision.reasoning,
            )
            after = OfficialFrameSequence.from_frame_data(raw_after)
            receipt = self.controller.observe(after)
            metadata = receipt.metadata if receipt is not None else {}
            trace.append(
                ToolkitActionReceipt(
                    step=step,
                    action=decision.action,
                    coordinate_row_column=decision.coordinate,
                    api_data=data,
                    source=decision.source,
                    purpose=decision.purpose,
                    plan_id=decision.plan_id,
                    expected_delta=decision.expected_delta,
                    levels_before=levels_before,
                    levels_after=after.levels_completed,
                    state_after=after.state,
                    observation_before_sha256=observation_before_sha,
                    observation_after_sha256=after.sha256,
                    persistent_unchanged=(
                        receipt.effect.unchanged if receipt is not None else None
                    ),
                    changed_cell_count=(
                        receipt.effect.changed_cell_count
                        if receipt is not None
                        else None
                    ),
                    translation_vectors=(
                        receipt.effect.translation_vectors
                        if receipt is not None
                        else ()
                    ),
                    animation_frame_count=len(after.rendered_frames),
                    animation_delta_count=len(after.animation_deltas),
                    moved_components=tuple(
                        metadata.get("moved_components", ())
                    ),
                    changed_cells=tuple(
                        metadata.get("changed_cells", ())[:96]
                    ),
                    action_receipt_sha256=(
                        receipt.receipt_sha256 if receipt is not None else None
                    ),
                    elapsed_seconds=time.perf_counter() - started_clock,
                )
            )
            current = after

        finished_at = datetime.now(timezone.utc)
        return ToolkitRunResult(
            schema="instella_arc.toolkit_run.v2",
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            game_id=current.game_id,
            terminal_state=current.state,
            levels_completed=current.levels_completed,
            win_levels=current.win_levels,
            actions=len(trace),
            max_actions=self.max_actions,
            initial_observation_sha256=initial.sha256,
            final_observation_sha256=current.sha256,
            initial_final_grid=initial.final_grid,
            final_final_grid=current.final_grid,
            controller_summary=self.controller.summary(),
            trace=tuple(trace),
        )


def write_run_result(result: ToolkitRunResult, path: Path) -> str:
    payload = asdict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
