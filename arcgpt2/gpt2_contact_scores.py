"""One-checkpoint GPT-2 priors for the hidden-contact posterior agent.

The provider scores both action semantics and contact primitives with the same
causal language model. Exact replay remains authoritative: these scores order
valid hypotheses but cannot preserve a program contradicted by reality.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .completion_scorer import score_with_contextual_calibration
from .contact_protocol import contact_completion, contact_prompt
from .mechanics_v2 import ContactMode
from .natural_protocol import answer_text, direction_words, mapping_prompt
from .phase0_hidden_action import Action, Direction
from .primitive_contact_game import ContactGameSpec, ContactStepRecord


class GPT2ContactScores:
    """Score all Gate-C latent values using one GPT-2 checkpoint."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str | None = None,
        candidate_batch_size: int = 5,
        soft_prefix: torch.Tensor | None = None,
    ) -> None:
        if candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.candidate_batch_size = candidate_batch_size
        self.soft_prefix = (
            soft_prefix.detach().to(self.device).clone()
            if soft_prefix is not None
            else None
        )
        if self.soft_prefix is not None:
            if self.soft_prefix.ndim != 2:
                raise ValueError("soft_prefix must have shape [tokens, hidden_size]")
            hidden_size = int(self.model.get_input_embeddings().embedding_dim)
            if int(self.soft_prefix.shape[1]) != hidden_size:
                raise ValueError("soft_prefix hidden size does not match GPT-2")

    def _score(
        self,
        context: str,
        null_context: str,
        candidates: Sequence[str],
    ) -> torch.Tensor:
        prompt_ids = self.tokenizer.encode(context, add_special_tokens=False)
        null_ids = self.tokenizer.encode(null_context, add_special_tokens=False)
        candidate_ids = tuple(
            tuple(self.tokenizer.encode(candidate, add_special_tokens=False))
            for candidate in candidates
        )
        with torch.no_grad():
            return score_with_contextual_calibration(
                self.model,
                prompt_ids,
                null_ids,
                candidate_ids,
                pad_token_id=int(self.tokenizer.pad_token_id),
                device=self.device,
                candidate_batch_size=self.candidate_batch_size,
                reduction="mean",
                soft_prefix=self.soft_prefix,
            ).detach().cpu()

    def direction_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec
        context = mapping_prompt(current_grid, history, action)
        null = mapping_prompt(
            current_grid,
            (),
            action,
            include_evidence=False,
        )
        words = direction_words()
        scores = self._score(
            context,
            null,
            tuple(answer_text(word) for word in words),
        )
        word_direction = {
            "north": Direction.UP,
            "south": Direction.DOWN,
            "west": Direction.LEFT,
            "east": Direction.RIGHT,
        }
        return {
            word_direction[word]: float(scores[index].item())
            for index, word in enumerate(words)
        }

    def contact_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        current_grid,
    ) -> Mapping[ContactMode, float]:
        context = contact_prompt(
            current_grid,
            history,
            spec.palette.interaction,
        )
        null = contact_prompt(
            current_grid,
            (),
            spec.palette.interaction,
        )
        modes = tuple(ContactMode)
        scores = self._score(
            context,
            null,
            tuple(
                contact_completion(mode, spec.palette.interaction)
                for mode in modes
            ),
        )
        return {
            mode: float(scores[index].item())
            for index, mode in enumerate(modes)
        }
