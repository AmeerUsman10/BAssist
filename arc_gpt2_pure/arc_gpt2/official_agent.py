"""Adapter exposing the same pure GPT-2 policy through the official agent API.

This module is not imported by the package root, so the core curriculum remains
usable without installing the ARC SDK. Copy or import ``PureGPT2`` inside an
ARC-AGI-3-Agents checkout after installing its official dependencies.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .codec import Grid, extract_last_grid
from .modeling import (
    choose_action_with_gpt2,
    generate_completion,
    load_gpt2,
    load_tokenizer,
)
from .protocol import (
    ACTION_ORDER,
    Transition,
    action_prompt,
    coordinate_prompt,
    format_mapping,
    memory_prompt,
    parse_coordinate,
    parse_mapping,
)

try:
    from arcengine import FrameData, GameAction, GameState
    from agents.agent import Agent
except ImportError as exc:  # pragma: no cover - exercised in the official repo
    raise ImportError(
        "official_agent.py must run inside the official ARC-AGI-3-Agents "
        "environment with arcengine and agents.agent installed"
    ) from exc


def _action_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 7 else None
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            number = int(digits)
            return number if 0 <= number <= 7 else None
        return None
    if isinstance(value, Mapping):
        for key in ("id", "action_id", "value", "name"):
            if key in value:
                found = _action_id(value[key])
                if found is not None:
                    return found
        return None
    for attribute in ("id", "action_id", "value", "name"):
        if hasattr(value, attribute):
            found = _action_id(getattr(value, attribute))
            if found is not None:
                return found
    return None


def available_action_names(raw: object) -> list[str]:
    """Losslessly normalize common SDK action-list payloads to A1...A7."""
    candidates: list[object]
    if raw is None:
        candidates = list(range(1, 8))
    elif isinstance(raw, Mapping):
        candidates = list(raw.keys()) + list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        candidates = list(raw)
    else:
        candidates = [raw]
    values = sorted(
        {
            action_id
            for candidate in candidates
            if (action_id := _action_id(candidate)) is not None and action_id != 0
        }
    )
    return [f"A{value}" for value in values] or list(ACTION_ORDER)


class PureGPT2(Agent):
    """One GPT-2 checkpoint acting from its own generated trajectory memory."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        model_path = os.environ.get("ARC_GPT2_MODEL", "openai-community/gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_path)
        self.model = load_gpt2(model_path, device=self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.trajectory: list[Transition] = []
        self.previous_grid: Grid | None = None
        self.previous_action: str | None = None
        self.memory = format_mapping({})

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _observe_pending_transition(self, current_grid: Grid) -> None:
        if self.previous_grid is None or self.previous_action is None:
            return
        self.trajectory.append(
            Transition(
                before=self.previous_grid,
                action=self.previous_action,
                after=current_grid,
            )
        )
        # Preserve the finite GPT-2 context while retaining recent literal evidence.
        self.trajectory = self.trajectory[-12:]
        self.previous_grid = None
        self.previous_action = None

    def _write_memory(self, grid: Grid, available: Sequence[str]) -> tuple[str, str]:
        prompt = memory_prompt(
            self.trajectory,
            grid,
            available_actions=available,
            history_limit=8,
        )
        completion = generate_completion(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=48,
            device=self.device,
        )
        memory_body = completion.split("[[/MEMORY]]", maxsplit=1)[0]
        self.memory = format_mapping(parse_mapping(memory_body))
        return self.memory, completion

    def _coordinate(self, grid: Grid, memory: str) -> tuple[int, int, str]:
        prompt = coordinate_prompt(self.trajectory, grid, memory=memory)
        completion = generate_completion(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=24,
            device=self.device,
        )
        coordinate = parse_coordinate(
            completion,
            width=len(grid[0]),
            height=len(grid),
        )
        # A transport fallback is necessary for malformed text. It contains no
        # game-specific semantics and does not choose among meaningful objects.
        if coordinate is None:
            coordinate = (0, 0)
        return coordinate[0], coordinate[1], completion

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        grid = extract_last_grid(latest_frame.frame)
        self._observe_pending_transition(grid)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            self.previous_grid = grid
            self.previous_action = "A0"
            action.reasoning = {
                "model": "pure-gpt2",
                "reason": "environment requires reset",
            }
            return action

        available = available_action_names(latest_frame.available_actions)
        memory, memory_completion = self._write_memory(grid, available)
        prompt = action_prompt(
            self.trajectory,
            grid,
            memory=memory,
            available_actions=available,
            history_limit=8,
        )
        selected, scores = choose_action_with_gpt2(
            self.model,
            self.tokenizer,
            prompt,
            available_actions=available,
            device=self.device,
        )
        action = GameAction.from_id(int(selected[1:]))
        coordinate_completion: str | None = None
        if action.is_complex():
            x, y, coordinate_completion = self._coordinate(grid, memory)
            action.set_data({"x": x, "y": y})

        action.reasoning = {
            "model": "pure-gpt2",
            "memory": memory,
            "memory_raw": memory_completion,
            "selected": selected,
            "candidate_log_probabilities": scores,
            "coordinate_raw": coordinate_completion,
            "trajectory_transitions": len(self.trajectory),
        }
        self.previous_grid = grid
        self.previous_action = selected
        return action
