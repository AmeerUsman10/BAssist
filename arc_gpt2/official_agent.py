"""Adapter for copying the pure GPT-2 policy into ARC-AGI-3-Agents.

This file intentionally contains no learned component other than one GPT-2 model.
The current Stage-0 checkpoint is trained only on ACTION1-ACTION4 movement worlds; the
adapter is included now so the interface remains fixed while the curriculum expands.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from arcengine import FrameData, GameAction, GameState
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from agents.agent import Agent

from arc_gpt2.protocol import (
    build_prompt,
    encode_delta,
    extract_action,
    extract_memory,
    initial_memory,
    parse_coordinates,
)


class _StopAfterAction(StoppingCriteria):
    def __init__(self, tokenizer: Any, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = start_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        generated = self.tokenizer.decode(
            input_ids[0, self.start_length :], skip_special_tokens=False
        )
        return "</ACT>" in generated or "<END>" in generated


class PureGPT2(Agent):
    """One GPT-2 checkpoint with self-written recurrent memory."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        model_path = os.environ.get("ARC_GPT2_MODEL", "outputs/arc-gpt2-stage0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.memory = initial_memory((1, 2, 3, 4))
        self.previous_grid: list[list[int]] | None = None
        self.previous_action: int | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _legal_actions(self) -> list[int]:
        mapping = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        available = set(self.arc_env.action_space)
        return [index for index, action in mapping.items() if action in available]

    def _generate(self, prompt: str, max_new_tokens: int = 192) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        max_positions = int(getattr(self.model.config, "n_positions", 1024))
        if input_ids.shape[1] + max_new_tokens > max_positions:
            keep = max_positions - max_new_tokens
            input_ids = input_ids[:, -keep:]
            attention_mask = attention_mask[:, -keep:]
        stopping = StoppingCriteriaList(
            [_StopAfterAction(self.tokenizer, input_ids.shape[1])]
        )
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        return self.tokenizer.decode(
            output[0, input_ids.shape[1] :], skip_special_tokens=False
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in {GameState.NOT_PLAYED, GameState.GAME_OVER}:
            self.memory = initial_memory((1, 2, 3, 4))
            self.previous_grid = None
            self.previous_action = None
            return GameAction.RESET

        legal = self._legal_actions()
        if not legal:
            return GameAction.RESET

        grid = [list(row) for row in latest_frame.frame]
        delta = (
            encode_delta(self.previous_grid, grid)
            if self.previous_grid is not None
            else None
        )
        prompt = build_prompt(
            memory=self.memory,
            grid=grid,
            legal_actions=legal,
            state="RUN",
            previous_action=self.previous_action,
            previous_delta=delta,
        )
        completion = self._generate(prompt)
        generated_memory = extract_memory(completion)
        if generated_memory is not None:
            self.memory = generated_memory

        selected = extract_action(completion, legal)
        if selected is None:
            # This is only a legality fallback: the chosen value still comes from
            # the same GPT-2 output distribution, represented by the first legal
            # action token that the model emitted anywhere in its completion.
            selected = legal[0]

        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        action = action_map[selected]
        if action is GameAction.ACTION6:
            coordinates = parse_coordinates(completion) or (0, 0)
            action.set_data({"x": coordinates[0], "y": coordinates[1]})
        action.reasoning = {
            "model": "pure-gpt2",
            "memory": self.memory,
            "completion": completion,
        }
        self.previous_grid = grid
        self.previous_action = selected
        return action
