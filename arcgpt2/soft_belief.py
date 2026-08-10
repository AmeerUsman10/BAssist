"""Per-game soft belief particles for one frozen GPT-2 checkpoint.

A particle is a small continuous prefix prepended to ordinary GPT-2 token
embeddings. All particles share exactly the same GPT-2 weights. They represent
alternative per-game latent states, not an ensemble of independently trained
models.

The runtime update is deliberately self-supervised:

1. score the actually observed transition under every particle;
2. update particle weights by that predictive likelihood;
3. take a few gradient steps on each prefix using only the same transition;
4. resample prefixes if the posterior degenerates.

No external model, hidden mechanics label, reward model, or semantic encoder is
used. This module is a later-stage component: it should be promoted only after
the discrete version-space controls demonstrate that interaction history is
being used correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Sequence

import torch
from torch.nn import functional as F

from .belief_math import (
    bayes_update,
    effective_sample_size,
    entropy_bits,
    mixture_log_probability,
    normalize_log_weights,
    should_resample,
    systematic_resample,
)


class SoftBeliefError(ValueError):
    """Raised when a soft-belief operation is malformed."""


@dataclass(frozen=True)
class ParticleDiagnostics:
    predictive_log_likelihoods: tuple[float, ...]
    posterior_probabilities: tuple[float, ...]
    entropy_bits: float
    effective_sample_size: float
    resampled: bool
    adaptation_losses: tuple[float, ...]


class GPT2SoftBelief:
    """A particle posterior over trainable soft prompts for one GPT-2 model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        num_particles: int = 8,
        prefix_length: int = 16,
        initialization_std: float = 0.02,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if num_particles < 1:
            raise SoftBeliefError("num_particles must be positive")
        if prefix_length < 1:
            raise SoftBeliefError("prefix_length must be positive")
        if initialization_std < 0.0:
            raise SoftBeliefError("initialization_std must be non-negative")

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

        embedding = self.model.get_input_embeddings()
        hidden_size = int(embedding.embedding_dim)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        with torch.no_grad():
            # Initializing around the mean existing token embedding keeps the
            # prefix in the scale/distribution GPT-2 already expects.
            center = embedding.weight.detach().mean(dim=0)
            particles = center.view(1, 1, hidden_size).repeat(
                num_particles, prefix_length, 1
            )
            if initialization_std > 0.0:
                noise = torch.randn(
                    particles.shape,
                    generator=generator,
                    device=self.device,
                    dtype=particles.dtype,
                )
                particles = particles + initialization_std * noise

        self.particles = particles.detach().clone()
        self.log_weights = tuple(-math.log(num_particles) for _ in range(num_particles))
        self.seed = int(seed)
        self.update_index = 0

    @property
    def num_particles(self) -> int:
        return int(self.particles.shape[0])

    @property
    def prefix_length(self) -> int:
        return int(self.particles.shape[1])

    @property
    def probabilities(self) -> tuple[float, ...]:
        return normalize_log_weights(self.log_weights)

    def reset(self, *, jitter_std: float = 0.02, seed: int | None = None) -> None:
        """Reset posterior weights and diversify prefixes around their mean."""

        if jitter_std < 0.0:
            raise SoftBeliefError("jitter_std must be non-negative")
        chosen_seed = self.seed if seed is None else int(seed)
        generator = torch.Generator(device=self.device).manual_seed(chosen_seed)
        with torch.no_grad():
            center = self.particles.mean(dim=0, keepdim=True)
            particles = center.repeat(self.num_particles, 1, 1)
            if jitter_std > 0.0:
                particles.add_(
                    jitter_std
                    * torch.randn(
                        particles.shape,
                        generator=generator,
                        device=self.device,
                        dtype=particles.dtype,
                    )
                )
            self.particles = particles.detach()
        self.log_weights = tuple(-math.log(self.num_particles) for _ in range(self.num_particles))
        self.update_index = 0

    def encode(self, text: str) -> tuple[int, ...]:
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise SoftBeliefError("text tokenized to an empty sequence")
        return ids

    def sequence_log_likelihood(
        self,
        particle_index: int,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
        *,
        require_grad: bool = False,
        prefix_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one target continuation under one soft-prefix particle."""

        if not 0 <= particle_index < self.num_particles:
            raise SoftBeliefError("particle index is out of range")
        if not prompt_ids:
            raise SoftBeliefError("prompt_ids may not be empty")
        if not target_ids:
            raise SoftBeliefError("target_ids may not be empty")

        prefix = (
            prefix_override
            if prefix_override is not None
            else self.particles[particle_index]
        )
        if prefix.shape != self.particles[particle_index].shape:
            raise SoftBeliefError("prefix_override has the wrong shape")

        token_ids = torch.tensor(
            [list(prompt_ids) + list(target_ids)],
            dtype=torch.long,
            device=self.device,
        )
        token_embeddings = self.model.get_input_embeddings()(token_ids)
        prefix_batch = prefix.unsqueeze(0)
        inputs_embeds = torch.cat((prefix_batch, token_embeddings), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=self.device
        )

        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            logits = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0]
            start = self.prefix_length + len(prompt_ids) - 1
            target_logits = logits[start : start + len(target_ids)]
            targets = torch.tensor(target_ids, dtype=torch.long, device=self.device)
            token_log_probabilities = F.log_softmax(target_logits, dim=-1).gather(
                1, targets.unsqueeze(1)
            )[:, 0]
            return token_log_probabilities.sum()

    def predictive_log_likelihoods(
        self,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> tuple[float, ...]:
        return tuple(
            float(
                self.sequence_log_likelihood(
                    index,
                    prompt_ids,
                    target_ids,
                    require_grad=False,
                ).item()
            )
            for index in range(self.num_particles)
        )

    def observe(
        self,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
        *,
        adaptation_steps: int = 3,
        learning_rate: float = 0.05,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        resample_threshold_fraction: float = 0.5,
        resample_jitter_std: float = 0.01,
    ) -> ParticleDiagnostics:
        """Update the posterior and fast prefixes from one observed transition."""

        if adaptation_steps < 0:
            raise SoftBeliefError("adaptation_steps may not be negative")
        if learning_rate <= 0.0:
            raise SoftBeliefError("learning_rate must be positive")
        if weight_decay < 0.0:
            raise SoftBeliefError("weight_decay may not be negative")
        if gradient_clip <= 0.0:
            raise SoftBeliefError("gradient_clip must be positive")
        if resample_jitter_std < 0.0:
            raise SoftBeliefError("resample_jitter_std may not be negative")

        predictive = self.predictive_log_likelihoods(prompt_ids, target_ids)
        self.log_weights, posterior = bayes_update(self.log_weights, predictive)

        adapted: list[torch.Tensor] = []
        final_losses: list[float] = []
        for particle_index in range(self.num_particles):
            prefix = self.particles[particle_index].detach().clone().requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [prefix], lr=learning_rate, weight_decay=weight_decay
            )
            final_loss = -float(predictive[particle_index])
            for _ in range(adaptation_steps):
                optimizer.zero_grad(set_to_none=True)
                negative_log_likelihood = -self.sequence_log_likelihood(
                    particle_index,
                    prompt_ids,
                    target_ids,
                    require_grad=True,
                    prefix_override=prefix,
                )
                negative_log_likelihood.backward()
                torch.nn.utils.clip_grad_norm_([prefix], gradient_clip)
                optimizer.step()
                final_loss = float(negative_log_likelihood.detach().item())
            adapted.append(prefix.detach())
            final_losses.append(final_loss)
        self.particles = torch.stack(adapted, dim=0)

        resampled = False
        if should_resample(posterior, resample_threshold_fraction):
            ancestors = systematic_resample(
                posterior,
                count=self.num_particles,
                seed=self.seed + self.update_index,
            )
            generator = torch.Generator(device=self.device).manual_seed(
                self.seed + 10_000 + self.update_index
            )
            with torch.no_grad():
                new_particles = self.particles[list(ancestors)].clone()
                if resample_jitter_std > 0.0:
                    new_particles.add_(
                        resample_jitter_std
                        * torch.randn(
                            new_particles.shape,
                            generator=generator,
                            device=self.device,
                            dtype=new_particles.dtype,
                        )
                    )
                self.particles = new_particles.detach()
            self.log_weights = tuple(
                -math.log(self.num_particles) for _ in range(self.num_particles)
            )
            posterior = self.probabilities
            resampled = True

        self.update_index += 1
        return ParticleDiagnostics(
            predictive_log_likelihoods=predictive,
            posterior_probabilities=posterior,
            entropy_bits=entropy_bits(posterior),
            effective_sample_size=effective_sample_size(posterior),
            resampled=resampled,
            adaptation_losses=tuple(final_losses),
        )

    def score_candidates(
        self,
        prompt_ids: Sequence[int],
        candidate_targets: Sequence[Sequence[int]],
    ) -> tuple[float, ...]:
        """Score target candidates under the weighted particle mixture."""

        if not candidate_targets:
            raise SoftBeliefError("at least one candidate target is required")
        weights = self.probabilities
        candidate_scores: list[float] = []
        for target_ids in candidate_targets:
            per_particle = self.predictive_log_likelihoods(prompt_ids, target_ids)
            candidate_scores.append(mixture_log_probability(per_particle, weights))
        return tuple(candidate_scores)

    def snapshot(self) -> dict[str, Any]:
        """Return a portable CPU snapshot of temporary per-game belief state."""

        return {
            "version": 1,
            "num_particles": self.num_particles,
            "prefix_length": self.prefix_length,
            "seed": self.seed,
            "update_index": self.update_index,
            "log_weights": list(self.log_weights),
            "particles": self.particles.detach().cpu(),
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        if int(snapshot.get("version", -1)) != 1:
            raise SoftBeliefError("unsupported soft-belief snapshot version")
        particles = snapshot.get("particles")
        if not isinstance(particles, torch.Tensor):
            raise SoftBeliefError("snapshot particles must be a torch.Tensor")
        if tuple(particles.shape) != tuple(self.particles.shape):
            raise SoftBeliefError("snapshot particle shape does not match this belief")
        log_weights = tuple(float(value) for value in snapshot.get("log_weights", ()))
        if len(log_weights) != self.num_particles:
            raise SoftBeliefError("snapshot has the wrong number of log weights")
        normalize_log_weights(log_weights)
        self.particles = particles.to(self.device).detach().clone()
        self.log_weights = log_weights
        self.update_index = int(snapshot.get("update_index", 0))
