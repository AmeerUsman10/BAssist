"""Stage 0.1: set-valued hidden-action algorithm distillation.

Phase 0 exposed an objective bug: when several untried actions were equally
informative, the source learner supplied one arbitrary label. GPT-2 could lower
loss by learning that tie-break rather than learning the invariant "do not
repeat an already identified action". It also saw only one probe order per game,
so a different first action at inference moved the model off its training
distribution.

Stage 0.1 removes both defects:

* every still-unidentified action is recorded as a valid target;
* every one of the 24 probe-order trajectories is generated for each base game;
* navigation decisions remain singleton targets;
* the environment and inference contract are unchanged.

The source learner and shortest-path code exist only to generate/evaluate
training histories. They are never called by the acting GPT-2 policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from typing import Iterable, Sequence

from .codec import tokens_to_text
from .phase0_hidden_action import (
    ACTION_TOKEN,
    Action,
    Direction,
    GameSpec,
    HiddenActionGame,
    SourceLearner,
    StepRecord,
    _shortest_directions,
    append_record_tokens,
    generate_game,
    initial_transcript,
)


@dataclass(frozen=True)
class SetValuedDecisionExample:
    """One action decision with all equally valid action labels preserved."""

    game_seed: int
    variant_index: int
    probe_order: tuple[Action, ...]
    level_index: int
    step_index: int
    context: str
    canonical_target: str
    valid_targets: tuple[str, ...]
    decision_phase: str
    mapping_known_count: int


def all_probe_orders() -> tuple[tuple[Action, ...], ...]:
    """Return all 24 action orders in stable lexical/enum order."""
    return tuple(tuple(order) for order in permutations(tuple(Action)))


def game_variants(game_seed: int) -> tuple[GameSpec, ...]:
    """Hold mechanics fixed while enumerating every possible probe history."""
    base = generate_game(game_seed)
    return tuple(
        replace(base, probe_order=order)
        for order in all_probe_orders()
    )


def unknown_actions(learner: SourceLearner) -> tuple[Action, ...]:
    """Actions whose cardinal meaning has not yet been observed."""
    identified = set(learner.direction_to_action.values())
    return tuple(action for action in Action if action not in identified)


def valid_actions(
    learner: SourceLearner,
    game: HiddenActionGame,
) -> tuple[tuple[Action, ...], str]:
    """Return the complete valid action set and its decision phase.

    Before all mappings are known, every untried action is equally informative.
    Once the mapping is complete, the only valid target is the action associated
    with the first step of a deterministic shortest path.
    """
    unknown = unknown_actions(learner)
    if unknown:
        return unknown, "probe"

    path = _shortest_directions(game.level, game.agent)
    if path is None:
        raise RuntimeError("generated level unexpectedly became unsolvable")
    if not path:
        raise RuntimeError("a decision was requested after reaching the goal")
    action = learner.direction_to_action.get(path[0])
    if action is None:
        raise RuntimeError("complete mapping did not contain required direction")
    return (action,), "navigate"


def build_set_valued_examples(
    spec: GameSpec,
    *,
    variant_index: int,
    max_actions: int = 128,
) -> list[SetValuedDecisionExample]:
    """Generate a complete learning history with set-valued probe labels."""
    game = HiddenActionGame(spec)
    learner = SourceLearner(spec)
    transcript = initial_transcript(spec)
    examples: list[SetValuedDecisionExample] = []

    for _ in range(max_actions):
        candidates, phase = valid_actions(learner, game)
        canonical_action = learner.choose(game)
        if canonical_action not in candidates:
            raise RuntimeError("canonical source action is outside the valid set")

        examples.append(
            SetValuedDecisionExample(
                game_seed=spec.game_seed,
                variant_index=variant_index,
                probe_order=tuple(spec.probe_order),
                level_index=game.level_index,
                step_index=game.step_index,
                context=tokens_to_text([*transcript, "<DECIDE>"]),
                canonical_target=ACTION_TOKEN[canonical_action],
                valid_targets=tuple(ACTION_TOKEN[action] for action in candidates),
                decision_phase=phase,
                mapping_known_count=len(learner.direction_to_action),
            )
        )

        record = game.step(canonical_action)
        learner.observe(record)
        next_frame = game.frame if record.status == "LEVEL_WIN" else None
        append_record_tokens(transcript, record, next_frame=next_frame)
        if record.status == "GAME_WIN":
            return examples

    raise RuntimeError("set-valued source learner exceeded action budget")


def build_all_variant_examples(game_seed: int) -> list[SetValuedDecisionExample]:
    """Generate examples for every probe order of one underlying game."""
    rows: list[SetValuedDecisionExample] = []
    for variant_index, spec in enumerate(game_variants(game_seed)):
        rows.extend(
            build_set_valued_examples(
                spec,
                variant_index=variant_index,
            )
        )
    return rows


def observe_for_metrics(learner: SourceLearner, record: StepRecord) -> None:
    """Update the evaluation-only mapping observer from one literal transition."""
    learner.observe(record)


def probe_support_counts(
    examples: Sequence[SetValuedDecisionExample],
) -> dict[str, int]:
    """Count how often each action belongs to a valid probe target set."""
    counts = {ACTION_TOKEN[action]: 0 for action in Action}
    for example in examples:
        if example.decision_phase != "probe":
            continue
        for target in example.valid_targets:
            counts[target] += 1
    return counts


def canonical_probe_counts(
    examples: Iterable[SetValuedDecisionExample],
) -> dict[str, int]:
    counts = {ACTION_TOKEN[action]: 0 for action in Action}
    for example in examples:
        if example.decision_phase == "probe":
            counts[example.canonical_target] += 1
    return counts
