"""Online hidden-action memory for one meta-trained GPT-2 checkpoint.

The provider exposes the same mapping-score interface used by the executable
posterior agent, but all game-specific information is carried in a temporary
soft prefix. After each exact transition, prediction error updates only that
prefix. GPT-2 weights remain shared and frozen during play.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import meta_soft_binding as meta
from .meta_soft_second_order import outcome_only_target
from .phase0_hidden_action import Action, Direction, GameSpec, StepRecord


@dataclass(frozen=True)
class AdaptationReceipt:
    observation_number: int
    loss_before_update: float
    gradient_norm: float
    prefix_norm: float


class AdaptiveGPT2MappingScores:
    """Read and write a per-game soft prefix using one GPT-2 model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        initial_prefix: torch.Tensor,
        *,
        inner_learning_rate: float = 0.2,
        max_length: int = 256,
        gradient_clip: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        if initial_prefix.ndim != 2 or initial_prefix.shape[0] < 1:
            raise ValueError("initial_prefix must have shape [tokens, hidden_size]")
        if inner_learning_rate <= 0.0:
            raise ValueError("inner_learning_rate must be positive")
        if max_length <= initial_prefix.shape[0] + 32:
            raise ValueError("max_length leaves too little token context")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model.config.pad_token_id = tokenizer.pad_token_id

        hidden_size = int(self.model.get_input_embeddings().embedding_dim)
        if int(initial_prefix.shape[1]) != hidden_size:
            raise ValueError("initial_prefix hidden size does not match GPT-2")
        self._initial_prefix = initial_prefix.detach().to(self.device).clone()
        self.prefix = self._initial_prefix.clone().requires_grad_(True)
        self.inner_learning_rate = float(inner_learning_rate)
        self.max_length = int(max_length)
        self.gradient_clip = float(gradient_clip)
        self.observation_count = 0
        self.receipts: list[AdaptationReceipt] = []

    @classmethod
    def from_checkpoint(
        cls,
        model: Any,
        tokenizer: Any,
        prefix_path: str | Path,
        **kwargs,
    ) -> "AdaptiveGPT2MappingScores":
        payload = torch.load(prefix_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "prefix" not in payload:
            raise ValueError("soft-prefix checkpoint lacks a 'prefix' tensor")
        return cls(model, tokenizer, payload["prefix"], **kwargs)

    def reset(self) -> None:
        self.prefix = self._initial_prefix.clone().requires_grad_(True)
        self.observation_count = 0
        self.receipts.clear()

    def score(
        self,
        spec: GameSpec,
        history: Sequence[StepRecord],
        query_action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec, history, current_grid
        scores = meta.candidate_direction_scores(
            self.model,
            self.tokenizer,
            self.prefix,
            query_action,
            device=self.device,
            max_length=self.max_length,
        )
        word_direction = {
            "north": Direction.UP,
            "south": Direction.DOWN,
            "west": Direction.LEFT,
            "east": Direction.RIGHT,
        }
        return {
            word_direction[word]: float(scores[index].detach().item())
            for index, word in enumerate(meta._WORDS)
        }

    def observe(self, record: StepRecord) -> AdaptationReceipt:
        """Write one exact action consequence into the temporary prefix."""

        prompt_ids = meta.encode_text(self.tokenizer, meta.transition_prompt(record))
        target_ids = meta.encode_text(self.tokenizer, outcome_only_target(record))
        summed = meta.sequence_log_likelihood(
            self.model,
            self.prefix,
            prompt_ids,
            target_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
            device=self.device,
            max_length=self.max_length,
        )
        loss = -summed / len(target_ids)
        gradient = torch.autograd.grad(loss, self.prefix, create_graph=False)[0]
        norm = float(torch.linalg.vector_norm(gradient).item())
        if norm > self.gradient_clip:
            gradient = gradient * (self.gradient_clip / max(norm, 1e-12))
        with torch.no_grad():
            updated = self.prefix - self.inner_learning_rate * gradient
        self.prefix = updated.detach().requires_grad_(True)
        self.observation_count += 1
        receipt = AdaptationReceipt(
            observation_number=self.observation_count,
            loss_before_update=float(loss.detach().item()),
            gradient_norm=norm,
            prefix_norm=float(torch.linalg.vector_norm(self.prefix).item()),
        )
        self.receipts.append(receipt)
        return receipt
