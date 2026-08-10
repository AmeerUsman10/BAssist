"""Meta-learn goal signatures through raw terminal-report prefix updates.

This module is the model/evaluation half of ``raw_goal_signature_v1``.  The
data builder is intentionally separate: callers provide :class:`GoalGroup`
objects containing already-audited, exact text.  At adaptation time the only
target is one observed continuation and its ordinary mean causal-LM NLL.
Candidate goals are used only by the outer objective/readout.

Most importantly, Trial 3 is scored directly by GPT-2 as the completions
``" yes."`` and ``" no."``.  Deterministic goal replay may set ``trial3_yes``
in the data object, but is never used to manufacture a model prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
import random
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .completion_scorer import score_candidate_completions
from .meta_soft_binding import encode_text
from .meta_soft_raw_outcome_overfit import GPT2_REVISION, SUPPORT_OBJECTIVE


PROTOCOL = "raw_goal_signature_v1"
RESULT_SCHEMA_VERSION = 1
GOAL_FAMILIES = ("CONTACT", "ABSENT", "COUNT_EQ_1", "TOUCH_FOUR")
EVIDENCE_ORDERS = ((0, 1), (1, 0))
BINARY_COMPLETIONS = (" no.", " yes.")


def _family_name(value: Any) -> str:
    raw = str(getattr(value, "value", value))
    return {"COUNT": "COUNT_EQ_1", "TOUCH": "TOUCH_FOUR"}.get(raw, raw)


def from_data_group(group: Any) -> GoalGroup:
    """Adapt the audited deterministic data module to the model-only schema.

    This is the sole integration boundary with ``raw_goal_signature_data``.
    Metadata IDs are copied only to ``group_key`` and never enter model text.
    """

    from .raw_goal_signature_data import (
        FAMILIES,
        candidate_completion_text,
        goal_query_text,
        raw_support_parts,
        semantic_query_text,
    )

    candidate_families = tuple(_family_name(value) for value in group.candidate_order)
    completions = tuple(candidate_completion_text(group))
    prompt = goal_query_text(group)
    family_position = {family: index for index, family in enumerate(FAMILIES)}
    worlds: list[GoalWorld] = []
    for canonical_index, source_world in enumerate(group.worlds):
        # A fixed no-fixed-point cycle supplies another valid observed
        # signature.  Its raw reports, not its family label, enter adaptation.
        corrupt_source = group.worlds[(canonical_index + 1) % 4]

        def make_trials(report_world: Any) -> tuple[ObservedTrial, ObservedTrial]:
            result: list[ObservedTrial] = []
            for trial in group.identification_trials:
                support_prompt, target = raw_support_parts(report_world, trial)
                neutral_prompt, neutral = raw_support_parts(
                    report_world, trial, neutral_status=True
                )
                if support_prompt != neutral_prompt:
                    raise ValueError("statusless replacement changed the support prompt")
                truth_bits = tuple(
                    int(trial.terminal_by_family[family_position[family]])
                    for family in group.candidate_order
                )
                result.append(
                    ObservedTrial(support_prompt, target, neutral, truth_bits)  # type: ignore[arg-type]
                )
            return tuple(result)  # type: ignore[return-value]

        worlds.append(
            GoalWorld(
                family=_family_name(source_world.family),
                candidate_prompt=prompt,
                candidate_families=candidate_families,  # type: ignore[arg-type]
                candidate_completions=completions,  # type: ignore[arg-type]
                trials=make_trials(source_world),
                trial3_prompt=semantic_query_text(group),
                trial3_yes=bool(source_world.semantic_terminal),
                deranged_family=_family_name(corrupt_source.family),
                deranged_trials=make_trials(corrupt_source),
            )
        )
    converted = GoalGroup(str(group.group_id), tuple(worlds))  # type: ignore[arg-type]
    _validate_group(converted)
    return converted


@dataclass(frozen=True)
class ObservedTrial:
    """One exact intervention prompt and observed continuation."""

    prompt: str
    target: str
    statusless_target: str
    truth_bits: tuple[int, int, int, int]


@dataclass(frozen=True)
class GoalWorld:
    """One counterfactual world in a four-world independent group."""

    family: str
    candidate_prompt: str
    candidate_families: tuple[str, str, str, str]
    candidate_completions: tuple[str, str, str, str]
    trials: tuple[ObservedTrial, ObservedTrial]
    trial3_prompt: str
    trial3_yes: bool
    deranged_family: str
    deranged_trials: tuple[ObservedTrial, ObservedTrial]


@dataclass(frozen=True)
class GoalGroup:
    """The four correlated worlds sharing surfaces and initial prefix."""

    group_key: str
    worlds: tuple[GoalWorld, GoalWorld, GoalWorld, GoalWorld]


@dataclass(frozen=True)
class Config:
    model_name: str = "openai-community/gpt2"
    model_revision: str = GPT2_REVISION
    source_sha: str | None = None
    initialization: str = "pretrained"
    seed: int = 577_215
    epochs: int = 2
    prefix_length: int = 8
    prefix_initialization_std: float = 0.01
    inner_learning_rate: float = 0.2
    prefix_learning_rate: float = 1e-3
    model_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    outer_optimizer: str = "AdamW"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_gradient_norm: float = 1.0
    no_evidence_weight: float = 0.25
    freeze_first_n_blocks: int = 11
    bootstrap_samples: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 20_260_804


def validate_protocol_config(config: Config) -> None:
    """Fail closed on claim-relevant hyperparameter drift."""

    expected = {
        "model_name": "openai-community/gpt2",
        "model_revision": GPT2_REVISION,
        "epochs": 2,
        "prefix_length": 8,
        "prefix_initialization_std": 0.01,
        "inner_learning_rate": 0.2,
        "prefix_learning_rate": 1e-3,
        "model_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "outer_optimizer": "AdamW",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_gradient_norm": 1.0,
        "no_evidence_weight": 0.25,
        "freeze_first_n_blocks": 11,
        "bootstrap_samples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 20_260_804,
    }
    drift = {
        name: {"expected": value, "observed": getattr(config, name)}
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if drift:
        raise ValueError(f"{PROTOCOL} configuration drift: {drift}")
    if config.initialization not in {"pretrained", "random"}:
        raise ValueError(
            f"{PROTOCOL} initialization must be pretrained or random, got "
            f"{config.initialization!r}"
        )


def resolve_source_sha(config: Config) -> str | None:
    """Resolve immutable source provenance without inspecting credentials."""

    return (
        config.source_sha
        or os.environ.get("ARC_GPT2_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA")
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_sha256(model: Any, prefix: torch.Tensor) -> str:
    """Hash the complete GPT-2 state and learned initial soft prefix."""

    digest = hashlib.sha256()
    tensors = [*sorted(model.state_dict().items()), ("__soft_prefix__", prefix)]
    for name, value in tensors:
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def freeze_for_locked_evaluation(model: Any, prefix: torch.Tensor) -> str:
    """Freeze all learned state and return its pre-evaluation digest."""

    digest = checkpoint_sha256(model, prefix)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix.requires_grad_(False)
    model.eval()
    return digest


def protocol_provenance(config: Config, *, manifest_sha256: str) -> dict[str, Any]:
    """Return claim-relevant immutable configuration for result receipts."""

    validate_protocol_config(config)
    return {
        "protocol": PROTOCOL,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha": resolve_source_sha(config),
        "manifest_sha256": manifest_sha256,
        "config": asdict(config),
        "inner_update": {
            "objective": SUPPORT_OBJECTIVE,
            "candidate_count": 1,
            "reduction": "mean",
            "counterfactuals_used": False,
            "updates_per_world": 2,
        },
        "readout": {
            "goal_logits": "post_minus_same_candidate_no_update_mean_log_probability",
            "trial3": "raw_gpt2_yes_no_completion_no_third_update",
        },
        "evidence_orders": [list(order) for order in EVIDENCE_ORDERS],
    }


def token_length_audit(tokenizer: Any, groups: Sequence[GoalGroup]) -> dict[str, Any]:
    """Prove every locked equal-token-length condition with the real tokenizer."""

    candidate_lengths: set[tuple[int, ...]] = set()
    binary_lengths = tuple(len(encode_text(tokenizer, value)) for value in BINARY_COMPLETIONS)
    neutral_pairs: list[tuple[int, int]] = []
    structural_errors: list[str] = []
    for group in groups:
        try:
            _validate_group(group)
        except ValueError as exc:
            structural_errors.append(f"{group.group_key}: {exc}")
        for world in group.worlds:
            candidate_lengths.add(
                tuple(len(encode_text(tokenizer, text)) for text in world.candidate_completions)
            )
            for trial in (*world.trials, *world.deranged_trials):
                neutral_pairs.append(
                    (
                        len(encode_text(tokenizer, trial.target)),
                        len(encode_text(tokenizer, trial.statusless_target)),
                    )
                )
    return {
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "candidate_length_patterns": [list(value) for value in sorted(candidate_lengths)],
        "binary_completion_lengths": list(binary_lengths),
        "neutral_pair_count": len(neutral_pairs),
        "structural_errors": structural_errors,
        "candidate_lengths_equal": all(len(set(value)) == 1 for value in candidate_lengths),
        "binary_lengths_equal": len(set(binary_lengths)) == 1,
        "neutral_lengths_matched": all(left == right for left, right in neutral_pairs),
        "passed": (
            not structural_errors
            and all(len(set(value)) == 1 for value in candidate_lengths)
            and len(set(binary_lengths)) == 1
            and all(left == right for left, right in neutral_pairs)
        ),
    }


def _completion_scores(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    prompt: str,
    completions: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    if not completions:
        raise ValueError("completion set may not be empty")
    prompt_ids = encode_text(tokenizer, prompt)
    completion_ids = tuple(encode_text(tokenizer, value) for value in completions)
    lengths = {len(ids) for ids in completion_ids}
    if len(lengths) != 1:
        raise ValueError("candidate completions must have equal token length")
    return score_candidate_completions(
        model,
        prompt_ids,
        completion_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=len(completion_ids),
        reduction="mean",
        soft_prefix=prefix,
    )


def raw_text_adapt_prefix(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    trial: ObservedTrial,
    *,
    inner_learning_rate: float,
    device: torch.device,
    create_graph: bool,
    statusless: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Apply one candidate-count-one mean-NLL update to a soft prefix."""

    target = trial.statusless_target if statusless else trial.target
    prompt_ids = encode_text(tokenizer, trial.prompt)
    target_ids = encode_text(tokenizer, target)
    observed = score_candidate_completions(
        model,
        prompt_ids,
        (target_ids,),
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        candidate_batch_size=1,
        reduction="mean",
        soft_prefix=prefix,
    )
    if observed.shape != (1,):
        raise RuntimeError("raw inner update must score exactly one candidate")
    loss = -observed[0]
    gradient = torch.autograd.grad(
        loss,
        prefix,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    updated = prefix - inner_learning_rate * gradient
    return updated, loss, gradient, len(target_ids)


def adapt_two_trials(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    trials: Sequence[ObservedTrial],
    order: tuple[int, int],
    *,
    inner_learning_rate: float,
    device: torch.device,
    create_graph: bool,
    statusless: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Adapt sequentially; the second gradient is taken at the first update."""

    if tuple(sorted(order)) != (0, 1) or len(trials) != 2:
        raise ValueError("raw_goal_signature_v1 requires both trials exactly once")
    prefix = initial_prefix
    diagnostics: list[dict[str, Any]] = []
    for trial_index in order:
        prefix, loss, gradient, token_count = raw_text_adapt_prefix(
            model,
            tokenizer,
            prefix,
            trials[trial_index],
            inner_learning_rate=inner_learning_rate,
            device=device,
            create_graph=create_graph,
            statusless=statusless,
        )
        diagnostics.append(
            {
                "trial_index": trial_index,
                "loss": loss,
                "gradient_norm": torch.linalg.vector_norm(gradient),
                "update_l2": torch.linalg.vector_norm(
                    inner_learning_rate * gradient
                ),
                "target_tokens": token_count,
            }
        )
    return prefix, diagnostics


def goal_raw_logits(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    world: GoalWorld,
    device: torch.device,
) -> torch.Tensor:
    return _completion_scores(
        model,
        tokenizer,
        prefix,
        world.candidate_prompt,
        world.candidate_completions,
        device,
    )


def prior_subtracted_goal_logits(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    updated_prefix: torch.Tensor,
    world: GoalWorld,
    device: torch.device,
    *,
    prior_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Subtract each candidate's own no-update completion log probability."""

    if prior_logits is None:
        prior_logits = goal_raw_logits(
            model, tokenizer, initial_prefix, world, device
        )
    post = goal_raw_logits(model, tokenizer, updated_prefix, world, device)
    return post - prior_logits


def trial3_binary_logits(
    model: Any,
    tokenizer: Any,
    prefix_after_two_updates: torch.Tensor,
    world: GoalWorld,
    device: torch.device,
) -> torch.Tensor:
    """Score Trial 3 directly with GPT-2 and perform no prefix update."""

    return _completion_scores(
        model,
        tokenizer,
        prefix_after_two_updates,
        world.trial3_prompt,
        BINARY_COMPLETIONS,
        device,
    )


def set_valued_cross_entropy(logits: torch.Tensor, truth_bits: Sequence[int]) -> torch.Tensor:
    """Cross entropy against a uniform distribution on consistent candidates."""

    mask = torch.tensor(truth_bits, dtype=logits.dtype, device=logits.device)
    if logits.ndim != 1 or mask.shape != logits.shape:
        raise ValueError("truth bits must match the one-dimensional logits")
    count = mask.sum()
    if float(count.detach().item()) != 2.0:
        raise ValueError("one-bit supervision must retain exactly two candidates")
    target = mask / count
    return -(target * F.log_softmax(logits, dim=-1)).sum()


def world_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    world: GoalWorld,
    order: tuple[int, int],
    config: Config,
    device: torch.device,
    *,
    prior_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one world's prior, one-bit, final, and semantic objectives."""

    if world.family not in GOAL_FAMILIES:
        raise ValueError(f"unknown goal family: {world.family}")
    truth_index = world.candidate_families.index(world.family)
    prior = prior_logits
    if prior is None:
        prior = goal_raw_logits(model, tokenizer, initial_prefix, world, device)
    uniform = torch.full_like(prior, 0.25)
    prior_loss = -(uniform * F.log_softmax(prior, dim=-1)).sum()

    one_prefix, _, _, _ = raw_text_adapt_prefix(
        model,
        tokenizer,
        initial_prefix,
        world.trials[order[0]],
        inner_learning_rate=config.inner_learning_rate,
        device=device,
        create_graph=True,
    )
    final_prefix, _, _, _ = raw_text_adapt_prefix(
        model,
        tokenizer,
        one_prefix,
        world.trials[order[1]],
        inner_learning_rate=config.inner_learning_rate,
        device=device,
        create_graph=True,
    )
    single_logits = prior_subtracted_goal_logits(
        model, tokenizer, initial_prefix, one_prefix, world, device,
        prior_logits=prior,
    )
    single_loss = set_valued_cross_entropy(
        single_logits, world.trials[order[0]].truth_bits
    )
    final_logits = prior_subtracted_goal_logits(
        model, tokenizer, initial_prefix, final_prefix, world, device,
        prior_logits=prior,
    )
    final_loss = F.cross_entropy(
        final_logits.unsqueeze(0),
        torch.tensor([truth_index], dtype=torch.long, device=device),
    )
    semantic_logits = trial3_binary_logits(
        model, tokenizer, final_prefix, world, device
    )
    semantic_loss = F.cross_entropy(
        semantic_logits.unsqueeze(0),
        torch.tensor([int(world.trial3_yes)], dtype=torch.long, device=device),
    )
    return (
        config.no_evidence_weight * prior_loss
        + (single_loss + final_loss + semantic_loss) / 3.0,
        {
            "prior": prior_loss,
            "single": single_loss,
            "final": final_loss,
            "semantic": semantic_loss,
        },
    )


def group_meta_loss(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    group: GoalGroup,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average all four worlds and both evidence orders without leakage."""

    _validate_group(group)
    components: dict[str, list[torch.Tensor]] = {
        "prior": [], "single": [], "final": [], "semantic": []
    }
    # Every world has the same query/candidates and initial prefix, so this is
    # exactly one shared no-update readout, not an approximation.
    representative = group.worlds[0]
    prior = goal_raw_logits(model, tokenizer, initial_prefix, representative, device)
    prior_loss = -(
        torch.full_like(prior, .25) * F.log_softmax(prior, -1)
    ).sum()
    for order in EVIDENCE_ORDERS:
        # At one bit there are only two distinct raw reports (yes/no).  Cache
        # those two differentiable updates, then compute the four distinct
        # two-bit signature finals: 6 adaptations/order, 12/group.
        first_by_target: dict[str, torch.Tensor] = {}
        for world in group.worlds:
            trial = world.trials[order[0]]
            if trial.target not in first_by_target:
                first_by_target[trial.target], _, _, _ = raw_text_adapt_prefix(
                    model, tokenizer, initial_prefix, trial,
                    inner_learning_rate=config.inner_learning_rate,
                    device=device, create_graph=True,
                )
        if len(first_by_target) != 2:
            raise ValueError("one-bit counterfactuals must contain exactly yes/no reports")
        for world in group.worlds:
            one = first_by_target[world.trials[order[0]].target]
            final, _, _, _ = raw_text_adapt_prefix(
                model, tokenizer, one, world.trials[order[1]],
                inner_learning_rate=config.inner_learning_rate,
                device=device, create_graph=True,
            )
            single_logits = prior_subtracted_goal_logits(
                model, tokenizer, initial_prefix, one, world, device,
                prior_logits=prior,
            )
            final_logits = prior_subtracted_goal_logits(
                model, tokenizer, initial_prefix, final, world, device,
                prior_logits=prior,
            )
            semantic_logits = trial3_binary_logits(
                model, tokenizer, final, world, device
            )
            single_loss = set_valued_cross_entropy(
                single_logits, world.trials[order[0]].truth_bits
            )
            truth = world.candidate_families.index(world.family)
            final_loss = F.cross_entropy(
                final_logits.unsqueeze(0), torch.tensor([truth], device=device)
            )
            semantic_loss = F.cross_entropy(
                semantic_logits.unsqueeze(0),
                torch.tensor([int(world.trial3_yes)], device=device),
            )
            components["prior"].append(prior_loss)
            components["single"].append(single_loss)
            components["final"].append(final_loss)
            components["semantic"].append(semantic_loss)
    total = (
        config.no_evidence_weight * torch.stack(components["prior"]).mean()
        + torch.stack(
            components["single"] + components["final"] + components["semantic"]
        ).mean()
    )
    return total, {
        name: float(torch.stack(values).mean().detach().item())
        for name, values in components.items()
    }


def _validate_group(group: GoalGroup) -> None:
    if len(group.worlds) != 4:
        raise ValueError("each group must contain four worlds")
    if {world.family for world in group.worlds} != set(GOAL_FAMILIES):
        raise ValueError("each group must contain each goal family exactly once")
    for world in group.worlds:
        if set(world.candidate_families) != set(GOAL_FAMILIES):
            raise ValueError("candidate_families must be a permutation of all families")
        if world.deranged_family == world.family:
            raise ValueError("deranged goal must have no fixed point")
    surfaces = {
        (
            world.candidate_prompt,
            world.candidate_completions,
            tuple(trial.prompt for trial in world.trials),
            world.trial3_prompt,
        )
        for world in group.worlds
    }
    if len(surfaces) != 1:
        raise ValueError("counterfactual worlds must share model-visible surfaces")
    family_world = {world.family: world for world in group.worlds}
    for trial_index in range(2):
        prompts = {world.trials[trial_index].prompt for world in group.worlds}
        neutral = {world.trials[trial_index].statusless_target for world in group.worlds}
        truth = {world.trials[trial_index].truth_bits for world in group.worlds}
        normalized = {
            world.trials[trial_index].target
            .replace("Terminal success: yes.", "Terminal success: STATUS.")
            .replace("Terminal success: no.", "Terminal success: STATUS.")
            for world in group.worlds
        }
        if len(prompts) != 1 or len(neutral) != 1 or len(truth) != 1 or len(normalized) != 1:
            raise ValueError("counterfactual supports may differ only at terminal report")
    for world in group.worlds:
        expected = family_world[world.deranged_family]
        for trial_index in range(2):
            corrupt = world.deranged_trials[trial_index]
            source = expected.trials[trial_index]
            if (
                corrupt.prompt != source.prompt
                or corrupt.target != source.target
                or corrupt.statusless_target != source.statusless_target
            ):
                raise ValueError("deranged evidence must be another valid world signature")


def one_sided_group_bootstrap(
    values: Sequence[float], *, samples: int, confidence: float, seed: int
) -> float:
    """Deterministic one-sided percentile bound over independent groups."""

    data = tuple(float(value) for value in values)
    if not data or samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid group-bootstrap arguments")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("bootstrap values must be finite")
    rng = random.Random(seed)
    count = len(data)
    draws = sorted(
        sum(data[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    rank = int(
        ((Decimal(1) - Decimal(str(confidence))) * Decimal(samples))
        .to_integral_value(rounding=ROUND_CEILING)
    )
    return draws[max(0, rank - 1)]


def _entropy_bits(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float((-(positive * torch.log2(positive))).sum().item())


def _metric_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("metric rows may not be empty")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def validation_selection_objective(report: Mapping[str, Any], config: Config) -> float:
    """Return the frozen exact outer objective using intact validation only."""

    orders = report["orders"]
    required = ("prior_ce", "single_ce", "final_ce", "semantic_ce")
    if not orders or any(name not in order for order in orders for name in required):
        raise ValueError("validation report lacks intact objective metrics")
    prior = sum(float(order["prior_ce"]) for order in orders) / len(orders)
    six_term_mean = sum(
        float(order[name])
        for order in orders
        for name in ("single_ce", "final_ce", "semantic_ce")
    ) / (len(orders) * 3)
    return config.no_evidence_weight * prior + six_term_mean


def evaluate_validation_groups(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    groups: Sequence[GoalGroup],
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    """Compute only the frozen intact validation objective.

    Statusless and deranged controls are deliberately absent from this path so
    checkpoint selection cannot observe locked causal-control behavior.
    """

    if not groups:
        raise ValueError("validation requires groups")
    model.eval()
    order_components: list[dict[str, list[float]]] = [
        {name: [] for name in ("prior_ce", "single_ce", "final_ce", "semantic_ce")}
        for _ in EVIDENCE_ORDERS
    ]
    all_finite = True
    for group in groups:
        _validate_group(group)
        representative = group.worlds[0]
        with torch.no_grad():
            prior = goal_raw_logits(
                model, tokenizer, initial_prefix, representative, device
            )
            prior_ce = -(
                torch.full_like(prior, .25) * F.log_softmax(prior, -1)
            ).sum()
        for order_index, order in enumerate(EVIDENCE_ORDERS):
            first_by_target: dict[str, torch.Tensor] = {}
            final_by_signature: dict[tuple[str, str], torch.Tensor] = {}
            for world in group.worlds:
                first_trial = world.trials[order[0]]
                if first_trial.target not in first_by_target:
                    online = initial_prefix.detach().clone().requires_grad_(True)
                    first_by_target[first_trial.target], _, _, _ = raw_text_adapt_prefix(
                        model, tokenizer, online, first_trial,
                        inner_learning_rate=config.inner_learning_rate,
                        device=device, create_graph=False,
                    )
                signature = tuple(trial.target for trial in world.trials)
                if signature not in final_by_signature:
                    final_by_signature[signature], _, _, _ = raw_text_adapt_prefix(
                        model, tokenizer, first_by_target[first_trial.target],
                        world.trials[order[1]],
                        inner_learning_rate=config.inner_learning_rate,
                        device=device, create_graph=False,
                    )
            if len(first_by_target) != 2 or len(final_by_signature) != 4:
                raise ValueError("validation quartet must contain two reports and four signatures")
            for world in group.worlds:
                truth = world.candidate_families.index(world.family)
                one = first_by_target[world.trials[order[0]].target]
                final = final_by_signature[tuple(trial.target for trial in world.trials)]
                with torch.no_grad():
                    one_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, one, world, device,
                        prior_logits=prior,
                    )
                    final_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, final, world, device,
                        prior_logits=prior,
                    )
                    semantic_logits = trial3_binary_logits(
                        model, tokenizer, final, world, device
                    )
                    losses = {
                        "prior_ce": prior_ce,
                        "single_ce": set_valued_cross_entropy(
                            one_logits, world.trials[order[0]].truth_bits
                        ),
                        "final_ce": F.cross_entropy(
                            final_logits.unsqueeze(0),
                            torch.tensor([truth], device=device),
                        ),
                        "semantic_ce": F.cross_entropy(
                            semantic_logits.unsqueeze(0),
                            torch.tensor([int(world.trial3_yes)], device=device),
                        ),
                    }
                for name, value in losses.items():
                    all_finite &= bool(torch.isfinite(value).all().item())
                    order_components[order_index][name].append(float(value.item()))
    orders = [
        {
            name: sum(values) / len(values)
            for name, values in components.items()
        }
        for components in order_components
    ]
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "mode": "intact_validation_only",
        "group_count": len(groups),
        "orders": orders,
        "controls_computed": [],
        "all_finite": all_finite,
    }
    report["selection_objective"] = validation_selection_objective(report, config)
    return report


def evaluate_groups(
    model: Any,
    tokenizer: Any,
    initial_prefix: torch.Tensor,
    groups: Sequence[GoalGroup],
    config: Config,
    device: torch.device,
    *,
    locked: bool = False,
) -> dict[str, Any]:
    """Evaluate intact, statusless, and deranged modes at group granularity."""

    if not groups:
        raise ValueError("evaluation requires groups")
    group_keys = [group.group_key for group in groups]
    unique_group_count = len(set(group_keys))
    if locked and len(groups) != 72:
        raise ValueError("locked evaluation requires exactly 72 independent groups")
    if locked and unique_group_count != len(groups):
        raise ValueError("locked evaluation requires unique independent group keys")
    checkpoint_before = checkpoint_sha256(model, initial_prefix) if locked else None
    frozen_before = (
        not initial_prefix.requires_grad
        and all(not parameter.requires_grad for parameter in model.parameters())
    ) if locked else False
    token_audit = token_length_audit(tokenizer, groups)
    model.eval()
    group_rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, float]]] = {name: [] for name in GOAL_FAMILIES}
    deranged_family_rows: dict[str, list[dict[str, float]]] = {name: [] for name in GOAL_FAMILIES}
    family_rows_by_order = [
        {name: [] for name in GOAL_FAMILIES} for _ in EVIDENCE_ORDERS
    ]
    deranged_rows_by_order = [
        {name: [] for name in GOAL_FAMILIES} for _ in EVIDENCE_ORDERS
    ]
    all_updates_finite = True
    min_update_l2 = math.inf
    deterministic_delta = 0.0
    all_outputs_finite = True
    for group in groups:
        _validate_group(group)
        order_rows: list[dict[str, float]] = []
        order_goal_probs: dict[tuple[int, int], list[torch.Tensor]] = {}
        for order_index, order in enumerate(EVIDENCE_ORDERS):
            per_world: list[dict[str, float]] = []
            goal_prob_vectors: list[torch.Tensor] = []
            statusless_goal_logits: list[torch.Tensor] = []
            statusless_trial3_logits: list[torch.Tensor] = []
            preupdate_goal_logits: list[torch.Tensor] = []
            mode_prefix_cache: dict[tuple[bool, tuple[str, ...]], tuple[torch.Tensor, list[dict[str, Any]]]] = {}
            replay_prefix_cache: dict[tuple[bool, tuple[str, ...]], tuple[torch.Tensor, list[dict[str, Any]]]] = {}
            mode_first_cache: dict[tuple[bool, str], tuple[torch.Tensor, dict[str, Any]]] = {}
            replay_first_cache: dict[tuple[bool, str], tuple[torch.Tensor, dict[str, Any]]] = {}
            for world in group.worlds:
                truth = world.candidate_families.index(world.family)
                deranged_truth = world.candidate_families.index(world.deranged_family)
                with torch.no_grad():
                    prior_raw = goal_raw_logits(
                        model, tokenizer, initial_prefix, world, device
                    )
                    prior_prob = torch.softmax(prior_raw, dim=-1)
                preupdate_goal_logits.append(prior_raw.detach().cpu())

                def run_mode(
                    trials: Sequence[ObservedTrial],
                    statusless: bool,
                    *,
                    replay: bool = False,
                ):
                    targets = tuple(
                        trial.statusless_target if statusless else trial.target
                        for trial in trials
                    )
                    cache = replay_prefix_cache if replay else mode_prefix_cache
                    first_cache = replay_first_cache if replay else mode_first_cache
                    key = (statusless, targets)
                    if key in cache:
                        return cache[key]
                    online = initial_prefix.detach().clone().requires_grad_(True)
                    first_key = (statusless, targets[order[0]])
                    if first_key not in first_cache:
                        first, first_loss, first_gradient, first_tokens = raw_text_adapt_prefix(
                            model, tokenizer, online, trials[order[0]],
                            inner_learning_rate=config.inner_learning_rate,
                            device=device, create_graph=False, statusless=statusless,
                        )
                        first_cache[first_key] = (
                            first,
                            {
                                "trial_index": order[0],
                                "loss": first_loss,
                                "gradient_norm": torch.linalg.vector_norm(first_gradient),
                                "update_l2": torch.linalg.vector_norm(
                                    config.inner_learning_rate * first_gradient
                                ),
                                "target_tokens": first_tokens,
                            },
                        )
                    first, first_diagnostic = first_cache[first_key]
                    updated, second_loss, second_gradient, second_tokens = raw_text_adapt_prefix(
                        model, tokenizer, first, trials[order[1]],
                        inner_learning_rate=config.inner_learning_rate,
                        device=device, create_graph=False, statusless=statusless,
                    )
                    diagnostics = [
                        first_diagnostic,
                        {
                            "trial_index": order[1],
                            "loss": second_loss,
                            "gradient_norm": torch.linalg.vector_norm(second_gradient),
                            "update_l2": torch.linalg.vector_norm(
                                config.inner_learning_rate * second_gradient
                            ),
                            "target_tokens": second_tokens,
                        },
                    ]
                    cache[key] = (updated, diagnostics)
                    return cache[key]

                intact_prefix, diagnostics = run_mode(world.trials, False)
                status_prefix, status_diagnostics = run_mode(world.trials, True)
                deranged_prefix, deranged_diagnostics = run_mode(
                    world.deranged_trials, False
                )
                for item in (*diagnostics, *status_diagnostics, *deranged_diagnostics):
                    values = (item["loss"], item["gradient_norm"], item["update_l2"])
                    all_updates_finite &= all(
                        bool(torch.isfinite(value).all().item()) for value in values
                    )
                    min_update_l2 = min(min_update_l2, float(item["update_l2"].item()))

                with torch.no_grad():
                    intact_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, intact_prefix, world,
                        device, prior_logits=prior_raw,
                    )
                    status_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, status_prefix, world,
                        device, prior_logits=prior_raw,
                    )
                    deranged_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, deranged_prefix, world,
                        device, prior_logits=prior_raw,
                    )
                    intact_goal = torch.softmax(intact_logits, dim=-1).cpu()
                    status_goal = torch.softmax(status_logits, dim=-1).cpu()
                    deranged_goal = torch.softmax(deranged_logits, dim=-1).cpu()
                    intact_binary_logits = trial3_binary_logits(
                        model, tokenizer, intact_prefix, world, device
                    )
                    intact_binary = torch.softmax(intact_binary_logits, -1).cpu()
                    status_binary_logits = trial3_binary_logits(
                        model, tokenizer, status_prefix, world, device
                    )
                    status_binary = torch.softmax(status_binary_logits, -1).cpu()

                for tensor in (
                    prior_raw, prior_prob, intact_logits, status_logits,
                    deranged_logits, intact_goal, status_goal, deranged_goal,
                    intact_binary_logits, intact_binary, status_binary_logits, status_binary,
                ):
                    all_outputs_finite &= bool(torch.isfinite(tensor).all().item())

                repeat_prefix, _ = run_mode(world.trials, False, replay=True)
                repeat_status_prefix, _ = run_mode(world.trials, True, replay=True)
                repeat_deranged_prefix, _ = run_mode(
                    world.deranged_trials, False, replay=True
                )
                with torch.no_grad():
                    repeat_goal = torch.softmax(
                        prior_subtracted_goal_logits(
                            model, tokenizer, initial_prefix, repeat_prefix,
                            world, device, prior_logits=prior_raw,
                        ), -1,
                    ).cpu()
                    repeat_binary = torch.softmax(
                        trial3_binary_logits(model, tokenizer, repeat_prefix, world, device), -1
                    ).cpu()
                    repeat_status_goal = torch.softmax(
                        prior_subtracted_goal_logits(
                            model, tokenizer, initial_prefix, repeat_status_prefix,
                            world, device, prior_logits=prior_raw,
                        ), -1,
                    ).cpu()
                    repeat_status_binary = torch.softmax(
                        trial3_binary_logits(model, tokenizer, repeat_status_prefix, world, device), -1
                    ).cpu()
                    repeat_deranged_goal = torch.softmax(
                        prior_subtracted_goal_logits(
                            model, tokenizer, initial_prefix, repeat_deranged_prefix,
                            world, device, prior_logits=prior_raw,
                        ), -1,
                    ).cpu()
                deterministic_delta = max(
                    deterministic_delta,
                    float(torch.max(torch.abs(repeat_goal - intact_goal)).item()),
                    float(torch.max(torch.abs(repeat_binary - intact_binary)).item()),
                    float(torch.max(torch.abs(repeat_status_goal - status_goal)).item()),
                    float(torch.max(torch.abs(repeat_status_binary - status_binary)).item()),
                    float(torch.max(torch.abs(repeat_deranged_goal - deranged_goal)).item()),
                )
                goal_prob_vectors.append(intact_goal)
                statusless_goal_logits.append(status_logits.detach().cpu())
                statusless_trial3_logits.append(status_binary_logits.detach().cpu())
                binary_truth = int(world.trial3_yes)
                binary_probability = float(intact_binary[binary_truth].item())
                row = {
                    "single_pair_mass": 0.0,
                    "single_top2": 0.0,
                    "single_pair_imbalance": 1.0,
                    "goal_accuracy": float(int(intact_goal.argmax()) == truth),
                    "goal_probability": float(intact_goal[truth].item()),
                    "statusless_goal_accuracy": float(int(status_goal.argmax()) == truth),
                    "statusless_goal_probability": float(status_goal[truth].item()),
                    "deranged_accuracy": float(int(deranged_goal.argmax()) == deranged_truth),
                    "deranged_probability": float(deranged_goal[deranged_truth].item()),
                    "deranged_original_accuracy": float(int(deranged_goal.argmax()) == truth),
                    "deranged_original_probability": float(deranged_goal[truth].item()),
                    "trial3_accuracy": float(int(intact_binary.argmax()) == binary_truth),
                    "trial3_probability": binary_probability,
                    "trial3_brier": (binary_probability - 1.0) ** 2,
                    "statusless_trial3_accuracy": float(int(status_binary.argmax()) == binary_truth),
                    "statusless_trial3_probability": float(status_binary[binary_truth].item()),
                    "no_update_entropy_bits": _entropy_bits(prior_prob.cpu()),
                    "no_update_max_uniform_deviation": float(
                        torch.max(torch.abs(prior_prob.cpu() - 0.25)).item()
                    ),
                    "statusless_entropy_bits": _entropy_bits(status_goal),
                    "statusless_max_uniform_deviation": float(
                        torch.max(torch.abs(status_goal - 0.25)).item()
                    ),
                    "prior_ce": float(
                        -(torch.full_like(prior_raw, .25) * F.log_softmax(prior_raw, -1)).sum().item()
                    ),
                    "single_ce": 0.0,
                    "final_ce": float(
                        F.cross_entropy(
                            intact_logits.unsqueeze(0), torch.tensor([truth], device=intact_logits.device)
                        ).item()
                    ),
                    "semantic_ce": float(
                        F.cross_entropy(
                            intact_binary_logits.unsqueeze(0),
                            torch.tensor([binary_truth], device=intact_binary_logits.device),
                        ).item()
                    ),
                }

                # One-bit metrics are evaluated after the first update only.
                one_prefix = mode_first_cache[
                    (False, world.trials[order[0]].target)
                ][0]
                with torch.no_grad():
                    one_logits = prior_subtracted_goal_logits(
                        model, tokenizer, initial_prefix, one_prefix, world,
                        device, prior_logits=prior_raw,
                    )
                    one_probs = torch.softmax(one_logits, -1).cpu()
                consistent = [i for i, bit in enumerate(world.trials[order[0]].truth_bits) if bit]
                pair = one_probs[consistent]
                row["single_pair_mass"] = float(pair.sum().item())
                row["single_top2"] = float(set(one_probs.topk(2).indices.tolist()) == set(consistent))
                row["single_pair_imbalance"] = float(
                    torch.abs(pair[0] - pair[1]).item() / max(float(pair.sum().item()), 1e-12)
                )
                row["single_ce"] = float(
                    set_valued_cross_entropy(
                        one_logits,
                        world.trials[order[0]].truth_bits,
                    ).item()
                )
                per_world.append(row)
                family_rows[world.family].append(row)
                deranged_family_rows[world.deranged_family].append(row)
                family_rows_by_order[order_index][world.family].append(row)
                deranged_rows_by_order[order_index][world.deranged_family].append(row)

            order_goal_probs[order] = goal_prob_vectors
            summary = _metric_summary(per_world)
            summary["goal_accuracy_gain"] = summary["goal_accuracy"] - summary["statusless_goal_accuracy"]
            summary["goal_probability_gain"] = summary["goal_probability"] - summary["statusless_goal_probability"]
            summary["trial3_accuracy_gain"] = summary["trial3_accuracy"] - summary["statusless_trial3_accuracy"]
            summary["trial3_probability_gain"] = summary["trial3_probability"] - summary["statusless_trial3_probability"]
            summary["statusless_goal_logit_delta"] = float(
                torch.max(torch.abs(torch.stack(statusless_goal_logits) - statusless_goal_logits[0])).item()
            )
            summary["statusless_trial3_logit_delta"] = float(
                torch.max(torch.abs(torch.stack(statusless_trial3_logits) - statusless_trial3_logits[0])).item()
            )
            summary["preupdate_goal_logit_delta"] = float(
                torch.max(torch.abs(torch.stack(preupdate_goal_logits) - preupdate_goal_logits[0])).item()
            )
            order_rows.append(summary)

        aligned_a = torch.stack(order_goal_probs[EVIDENCE_ORDERS[0]])
        aligned_b = torch.stack(order_goal_probs[EVIDENCE_ORDERS[1]])
        group_rows.append(
            {
                "group_key": group.group_key,
                "orders": order_rows,
                "cross_order_agreement": float(
                    (aligned_a.argmax(-1) == aligned_b.argmax(-1)).float().mean().item()
                ),
                "cross_order_tv": float((0.5 * torch.abs(aligned_a - aligned_b).sum(-1)).mean().item()),
            }
        )

    aggregate_orders = [
        _metric_summary([group["orders"][index] for group in group_rows])
        for index in range(2)
    ]
    aggregate = _metric_summary(aggregate_orders)
    aggregate["cross_order_agreement"] = sum(row["cross_order_agreement"] for row in group_rows) / len(group_rows)
    aggregate["cross_order_tv"] = sum(row["cross_order_tv"] for row in group_rows) / len(group_rows)
    family = {name: _metric_summary(rows) for name, rows in family_rows.items()}
    deranged_family = {
        name: _metric_summary(rows) for name, rows in deranged_family_rows.items()
    }
    families_by_order = [
        {name: _metric_summary(rows) for name, rows in mapping.items()}
        for mapping in family_rows_by_order
    ]
    deranged_families_by_order = [
        {name: _metric_summary(rows) for name, rows in mapping.items()}
        for mapping in deranged_rows_by_order
    ]
    bound_names = (
        "single_pair_mass", "single_top2", "goal_accuracy", "goal_probability",
        "goal_accuracy_gain", "goal_probability_gain", "trial3_accuracy",
        "trial3_probability", "trial3_accuracy_gain", "trial3_probability_gain",
    )
    bootstrap: dict[str, float] = {}
    bootstrap_by_order: list[dict[str, float]] = [{}, {}]
    for metric_index, metric in enumerate(bound_names):
        values = [sum(float(order[metric]) for order in group["orders"]) / 2 for group in group_rows]
        bootstrap[metric] = one_sided_group_bootstrap(
            values, samples=config.bootstrap_samples, confidence=config.bootstrap_confidence,
            seed=config.bootstrap_seed + metric_index,
        )
        for order_index in range(2):
            bootstrap_by_order[order_index][metric] = one_sided_group_bootstrap(
                [float(group["orders"][order_index][metric]) for group in group_rows],
                samples=config.bootstrap_samples,
                confidence=config.bootstrap_confidence,
                seed=config.bootstrap_seed + 100 + order_index * 100 + metric_index,
            )
    checkpoint_after = checkpoint_sha256(model, initial_prefix) if locked else None
    finite = all(
        math.isfinite(value)
        for value in (*aggregate.values(), *bootstrap.values(), min_update_l2, deterministic_delta)
    )
    return {
        "group_count": len(group_rows),
        "unique_group_count": unique_group_count,
        "aggregate": aggregate,
        "orders": aggregate_orders,
        "families": family,
        "deranged_families": deranged_family,
        "families_by_order": families_by_order,
        "deranged_families_by_order": deranged_families_by_order,
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "confidence": config.bootstrap_confidence,
            "seed": config.bootstrap_seed,
            "lower_bounds": bootstrap,
            "by_order": bootstrap_by_order,
        },
        "execution": {
            "all_finite": finite and all_updates_finite and all_outputs_finite,
            "min_prefix_update_l2": min_update_l2,
            "deterministic_replay_delta": deterministic_delta,
            "trial3_updates": 0,
            "bootstrap_unit": "independent_group",
            "inner_update_objective": SUPPORT_OBJECTIVE,
            "inner_candidate_count": 1,
            "inner_reduction": "mean",
            "token_lengths_audited": bool(token_audit["passed"]),
            "token_length_audit": token_audit,
            "locked_evaluation": locked,
            "frozen_before_locked_evaluation": frozen_before,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_unchanged": locked and checkpoint_before == checkpoint_after,
        },
        "groups": group_rows,
    }


def apply_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every frozen absolute capability threshold without relaxation."""

    a = report["aggregate"]
    b = report["bootstrap"]["lower_bounds"]
    e = report["execution"]
    checks: dict[str, bool] = {}
    for index, order in enumerate(report["orders"]):
        prefix = f"order_{index + 1}"
        order_bound = report["bootstrap"]["by_order"][index]
        checks.update({
            f"{prefix}_single_pair_mass": order["single_pair_mass"] >= .75,
            f"{prefix}_single_top2": order["single_top2"] >= .70,
            f"{prefix}_single_pair_balance": order["single_pair_imbalance"] <= .15,
            f"{prefix}_goal_accuracy": order["goal_accuracy"] >= .70,
            f"{prefix}_goal_probability": order["goal_probability"] >= .60,
            f"{prefix}_goal_accuracy_gain": order["goal_accuracy_gain"] >= .35,
            f"{prefix}_goal_probability_gain": order["goal_probability_gain"] >= .25,
            f"{prefix}_statusless_goal_isolation": order["statusless_goal_logit_delta"] <= 1e-6,
            f"{prefix}_statusless_trial3_isolation": order["statusless_trial3_logit_delta"] <= 1e-6,
            f"{prefix}_preupdate_isolation": order["preupdate_goal_logit_delta"] <= 1e-6,
            f"{prefix}_statusless_goal_accuracy_exact": abs(order["statusless_goal_accuracy"] - .25) <= 1e-6,
            f"{prefix}_statusless_goal_probability_exact": abs(order["statusless_goal_probability"] - .25) <= 1e-6,
            f"{prefix}_statusless_trial3_accuracy_exact": abs(order["statusless_trial3_accuracy"] - .50) <= 1e-6,
            f"{prefix}_statusless_trial3_probability_exact": abs(order["statusless_trial3_probability"] - .50) <= 1e-6,
            f"{prefix}_deranged_accuracy": order["deranged_accuracy"] >= .70,
            f"{prefix}_deranged_probability": order["deranged_probability"] >= .60,
            f"{prefix}_deranged_reject_original_accuracy": order["deranged_original_accuracy"] <= .25,
            f"{prefix}_deranged_reject_original_probability": order["deranged_original_probability"] <= .20,
            f"{prefix}_trial3_accuracy": order["trial3_accuracy"] >= .75,
            f"{prefix}_trial3_probability": order["trial3_probability"] >= .65,
            f"{prefix}_trial3_brier": order["trial3_brier"] <= .20,
            f"{prefix}_trial3_accuracy_gain": order["trial3_accuracy_gain"] >= .25,
            f"{prefix}_trial3_probability_gain": order["trial3_probability_gain"] >= .15,
            f"{prefix}_statusless_entropy": order["statusless_entropy_bits"] >= 1.95,
            f"{prefix}_statusless_uniform": order["statusless_max_uniform_deviation"] <= .05,
            f"{prefix}_bootstrap_single_mass": order_bound["single_pair_mass"] > .50,
            f"{prefix}_bootstrap_single_top2": order_bound["single_top2"] > (1 / 6),
            f"{prefix}_bootstrap_goal_accuracy": order_bound["goal_accuracy"] > .25,
            f"{prefix}_bootstrap_goal_probability": order_bound["goal_probability"] > .25,
            f"{prefix}_bootstrap_goal_accuracy_gain": order_bound["goal_accuracy_gain"] > 0,
            f"{prefix}_bootstrap_goal_probability_gain": order_bound["goal_probability_gain"] > 0,
            f"{prefix}_bootstrap_trial3_accuracy": order_bound["trial3_accuracy"] > .50,
            f"{prefix}_bootstrap_trial3_probability": order_bound["trial3_probability"] > .50,
            f"{prefix}_bootstrap_trial3_accuracy_gain": order_bound["trial3_accuracy_gain"] > 0,
            f"{prefix}_bootstrap_trial3_probability_gain": order_bound["trial3_probability_gain"] > 0,
        })
        for family_name, metrics in report["families_by_order"][index].items():
            checks[f"{prefix}_family_{family_name}_goal_accuracy"] = metrics["goal_accuracy"] >= .55
            checks[f"{prefix}_family_{family_name}_goal_probability"] = metrics["goal_probability"] >= .45
            checks[f"{prefix}_family_{family_name}_trial3_accuracy"] = metrics["trial3_accuracy"] >= .65
            checks[f"{prefix}_family_{family_name}_trial3_probability"] = metrics["trial3_probability"] >= .60
        for family_name, metrics in report["deranged_families_by_order"][index].items():
            checks[f"{prefix}_deranged_family_{family_name}_accuracy"] = metrics["deranged_accuracy"] >= .55
            checks[f"{prefix}_deranged_family_{family_name}_probability"] = metrics["deranged_probability"] >= .45
    checks.update({
        "locked_group_count_72": report.get("group_count") == 72,
        "locked_unique_group_count_72": report.get("unique_group_count") == 72,
        "bootstrap_samples_10000": report["bootstrap"].get("samples") == 10_000,
        "bootstrap_confidence_95": report["bootstrap"].get("confidence") == .95,
        "bootstrap_seed_frozen": report["bootstrap"].get("seed") == 20_260_804,
        "single_bootstrap_mass": b["single_pair_mass"] > .50,
        "single_bootstrap_top2": b["single_top2"] > (1 / 6),
        "goal_bootstrap_accuracy": b["goal_accuracy"] > .25,
        "goal_bootstrap_probability": b["goal_probability"] > .25,
        "goal_bootstrap_accuracy_gain": b["goal_accuracy_gain"] > 0,
        "goal_bootstrap_probability_gain": b["goal_probability_gain"] > 0,
        "cross_order_agreement": a["cross_order_agreement"] >= .90,
        "cross_order_tv": a["cross_order_tv"] <= .10,
        "deranged_accuracy": a["deranged_accuracy"] >= .70,
        "deranged_probability": a["deranged_probability"] >= .60,
        "deranged_reject_original_accuracy": a["deranged_original_accuracy"] <= .25,
        "deranged_reject_original_probability": a["deranged_original_probability"] <= .20,
        "trial3_accuracy": a["trial3_accuracy"] >= .75,
        "trial3_probability": a["trial3_probability"] >= .65,
        "trial3_brier": a["trial3_brier"] <= .20,
        "trial3_bootstrap_accuracy": b["trial3_accuracy"] > .50,
        "trial3_bootstrap_probability": b["trial3_probability"] > .50,
        "trial3_accuracy_gain": a["trial3_accuracy_gain"] >= .25,
        "trial3_probability_gain": a["trial3_probability_gain"] >= .15,
        "trial3_bootstrap_accuracy_gain": b["trial3_accuracy_gain"] > 0,
        "trial3_bootstrap_probability_gain": b["trial3_probability_gain"] > 0,
        "statusless_entropy": a["statusless_entropy_bits"] >= 1.95,
        "statusless_uniform": a["statusless_max_uniform_deviation"] <= .05,
        "all_finite": bool(e["all_finite"]),
        "updates_nonzero": e["min_prefix_update_l2"] > 1e-8,
        "deterministic_replay": e["deterministic_replay_delta"] <= 1e-6,
        "no_trial3_update": e["trial3_updates"] == 0,
        "group_bootstrap": e["bootstrap_unit"] == "independent_group",
        "raw_inner_objective": e["inner_update_objective"] == SUPPORT_OBJECTIVE,
        "inner_candidate_count_one": e["inner_candidate_count"] == 1,
        "inner_mean_reduction": e["inner_reduction"] == "mean",
        "token_lengths_audited": bool(e.get("token_lengths_audited", False)),
        "checkpoint_unchanged": bool(e.get("checkpoint_unchanged", False)),
        "frozen_before_locked_evaluation": bool(e.get("frozen_before_locked_evaluation", False)),
        "locked_evaluation_mode": bool(e.get("locked_evaluation", False)),
    })
    for family_name, metrics in report["families"].items():
        checks[f"family_{family_name}_goal_accuracy"] = metrics["goal_accuracy"] >= .55
        checks[f"family_{family_name}_goal_probability"] = metrics["goal_probability"] >= .45
        checks[f"family_{family_name}_trial3_accuracy"] = metrics["trial3_accuracy"] >= .65
        checks[f"family_{family_name}_trial3_probability"] = metrics["trial3_probability"] >= .60
    for family_name, metrics in report["deranged_families"].items():
        checks[f"deranged_family_{family_name}_accuracy"] = metrics["deranged_accuracy"] >= .55
        checks[f"deranged_family_{family_name}_probability"] = metrics["deranged_probability"] >= .45
    return {"name": PROTOCOL, "passed": all(checks.values()), "checks": checks}
