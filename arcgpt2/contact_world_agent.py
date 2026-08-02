"""Closed-loop posterior agent for hidden action and contact mechanics.

The agent maintains exact executable worlds over 24 action permutations and
five contact primitives. A generic depth-limited fracture planner chooses
reachable interventions whose predicted outcomes split the posterior. Once one
world survives, generic BFS plans to the goal. The only learned input allowed is
an optional prior/scorer supplied by one GPT-2 checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

from .mechanics_v2 import (
    ContactMode,
    MechanicsProgramV2,
    MechanicsStatus,
    execute_v2,
    mapping_from_program,
    replay_v2,
    shortest_plan_v2,
)
from .phase0_hidden_action import Action, Direction
from .primitive_contact_game import (
    ContactGameSpec,
    ContactStepRecord,
    PrimitiveContactGame,
    enumerate_contact_programs,
)


class ContactScoreProvider(Protocol):
    def direction_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        action: Action,
        current_grid,
    ) -> Mapping[Direction, float]: ...

    def contact_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        current_grid,
    ) -> Mapping[ContactMode, float]: ...


class UniformContactScores:
    def direction_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec, history, action, current_grid
        return {direction: 0.0 for direction in Direction}

    def contact_scores(
        self,
        spec: ContactGameSpec,
        history: Sequence[ContactStepRecord],
        current_grid,
    ) -> Mapping[ContactMode, float]:
        del spec, history, current_grid
        return {mode: 0.0 for mode in ContactMode}


@dataclass(frozen=True)
class ContactWorldEntry:
    program: MechanicsProgramV2
    log_weight: float
    probability: float
    source: str


@dataclass(frozen=True)
class ContactActionDecision:
    action: Action
    score: float
    information_gain_bits: float
    terminal_probability: float
    predicted_outcomes: int


@dataclass(frozen=True)
class ContactPlannerConfig:
    depth: int = 2
    information_weight: float = 1.0
    terminal_reward: float = 1.0
    action_cost: float = 0.01
    no_change_penalty: float = 0.02
    discount: float = 0.8

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("contact fracture depth must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")


@dataclass(frozen=True)
class ContactAgentStep:
    action_number: int
    level_index: int
    action: Action
    decision_mode: str
    posterior_worlds: int
    posterior_entropy_bits: float
    information_gain_bits: float
    status: str


@dataclass(frozen=True)
class ContactAgentResult:
    won: bool
    levels_completed: int
    actions: int
    steps: tuple[ContactAgentStep, ...]


def normalize_worlds(
    weighted: Sequence[tuple[MechanicsProgramV2, float, str]],
) -> tuple[ContactWorldEntry, ...]:
    if not weighted:
        raise RuntimeError("contact world posterior is empty")
    maximum = max(log_weight for _, log_weight, _ in weighted)
    masses = [math.exp(log_weight - maximum) for _, log_weight, _ in weighted]
    total = sum(masses)
    return tuple(
        ContactWorldEntry(program, log_weight, mass / total, source)
        for (program, log_weight, source), mass in zip(weighted, masses, strict=True)
    )


def condition_contact_worlds(
    entries: Sequence[ContactWorldEntry] | Sequence[tuple[MechanicsProgramV2, float, str]],
    history: Sequence[ContactStepRecord],
) -> tuple[ContactWorldEntry, ...]:
    weighted: list[tuple[MechanicsProgramV2, float, str]] = []
    observations = tuple(record.as_observation() for record in history)
    for entry in entries:
        if isinstance(entry, ContactWorldEntry):
            program = entry.program
            log_weight = entry.log_weight
            source = entry.source
        else:
            program, log_weight, source = entry
        if replay_v2(program, observations).consistent:
            weighted.append((program, log_weight, source))
    return normalize_worlds(weighted)


def build_contact_worlds(
    spec: ContactGameSpec,
    history: Sequence[ContactStepRecord],
    current_grid,
    provider: ContactScoreProvider,
) -> tuple[ContactWorldEntry, ...]:
    direction = {
        action: provider.direction_scores(spec, history, action, current_grid)
        for action in Action
    }
    contacts = provider.contact_scores(spec, history, current_grid)
    weighted: list[tuple[MechanicsProgramV2, float, str]] = []
    for program in enumerate_contact_programs(spec):
        mapping = mapping_from_program(program)
        log_weight = sum(
            float(direction[action][mapping[action]]) for action in Action
        ) + float(contacts[program.contact_mode])
        weighted.append((program, log_weight, type(provider).__name__))
    return condition_contact_worlds(weighted, history)


def contact_world_entropy(entries: Sequence[ContactWorldEntry]) -> float:
    return -sum(
        entry.probability * math.log2(entry.probability)
        for entry in entries
        if entry.probability > 0.0
    )


def contact_partition(
    entries: Sequence[ContactWorldEntry],
    grid,
    action: Action,
) -> Mapping[
    tuple[Any, MechanicsStatus], tuple[ContactWorldEntry, ...]
]:
    groups: dict[tuple[Any, MechanicsStatus], list[ContactWorldEntry]] = defaultdict(list)
    for entry in entries:
        result = execute_v2(entry.program, grid, action)
        groups[(result.after, result.status)].append(entry)
    normalized: dict[tuple[Any, MechanicsStatus], tuple[ContactWorldEntry, ...]] = {}
    for key, group in groups.items():
        probability = sum(entry.probability for entry in group)
        normalized[key] = tuple(
            ContactWorldEntry(
                entry.program,
                entry.log_weight,
                entry.probability / probability,
                entry.source,
            )
            for entry in group
        )
    return normalized


def _outcome_probability(group: Sequence[ContactWorldEntry], original_total: float) -> float:
    del original_total
    # Groups are normalized internally, so callers compute mass separately from
    # the unnormalized parent entries. This helper is intentionally unused but
    # retained as a type-level reminder of that distinction.
    return sum(entry.probability for entry in group)


class ContactFracturePlanner:
    def __init__(
        self,
        entries: Sequence[ContactWorldEntry],
        config: ContactPlannerConfig | None = None,
    ) -> None:
        if not entries:
            raise ValueError("contact planner requires at least one world")
        self.entries = tuple(entries)
        self.config = config or ContactPlannerConfig()

    def choose_action(self, grid) -> ContactActionDecision:
        decisions = [self._decision(self.entries, grid, action, self.config.depth) for action in Action]
        return max(
            decisions,
            key=lambda decision: (
                decision.score,
                decision.information_gain_bits,
                -list(Action).index(decision.action),
            ),
        )

    def _decision(
        self,
        entries: Sequence[ContactWorldEntry],
        grid,
        action: Action,
        depth: int,
    ) -> ContactActionDecision:
        prior_entropy = contact_world_entropy(entries)
        raw_groups: dict[
            tuple[Any, MechanicsStatus], list[ContactWorldEntry]
        ] = defaultdict(list)
        for entry in entries:
            result = execute_v2(entry.program, grid, action)
            raw_groups[(result.after, result.status)].append(entry)

        expected_entropy = 0.0
        future_value = 0.0
        terminal_probability = 0.0
        no_change_probability = 0.0
        for (after, status), group in raw_groups.items():
            mass = sum(entry.probability for entry in group)
            normalized = tuple(
                ContactWorldEntry(
                    entry.program,
                    entry.log_weight,
                    entry.probability / mass,
                    entry.source,
                )
                for entry in group
            )
            expected_entropy += mass * contact_world_entropy(normalized)
            terminal_probability += mass * (status is MechanicsStatus.WIN)
            no_change_probability += mass * (after == grid)
            if depth > 1 and status is MechanicsStatus.ACTIVE:
                best_future = max(
                    self._decision(normalized, after, candidate, depth - 1).score
                    for candidate in Action
                )
                future_value += mass * best_future

        information_gain = prior_entropy - expected_entropy
        score = (
            self.config.information_weight * information_gain
            + self.config.terminal_reward * terminal_probability
            + self.config.discount * future_value
            - self.config.action_cost
            - self.config.no_change_penalty * no_change_probability
        )
        return ContactActionDecision(
            action=action,
            score=score,
            information_gain_bits=information_gain,
            terminal_probability=terminal_probability,
            predicted_outcomes=len(raw_groups),
        )


def choose_contact_action(
    entries: Sequence[ContactWorldEntry],
    grid,
    *,
    planner_config: ContactPlannerConfig | None = None,
) -> tuple[Action, str, float]:
    if len(entries) == 1 or contact_world_entropy(entries) < 1e-9:
        plan = shortest_plan_v2(entries[0].program, grid, max_depth=96)
        if plan:
            return plan[0], "plan", 0.0
    decision = ContactFracturePlanner(entries, planner_config).choose_action(grid)
    return decision.action, "fracture", decision.information_gain_bits


def run_contact_agent(
    spec: ContactGameSpec,
    provider: ContactScoreProvider | None = None,
    *,
    max_actions: int = 256,
    planner_depth: int = 2,
) -> ContactAgentResult:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    score_provider = provider or UniformContactScores()
    reset = getattr(score_provider, "reset", None)
    if callable(reset):
        reset()
    observer = getattr(score_provider, "observe", None)

    game = PrimitiveContactGame(spec)
    history: list[ContactStepRecord] = []
    steps: list[ContactAgentStep] = []
    levels_completed = 0
    planner_config = ContactPlannerConfig(depth=planner_depth)

    for action_number in range(1, max_actions + 1):
        entries = build_contact_worlds(
            spec,
            history,
            game.frame,
            score_provider,
        )
        entropy = contact_world_entropy(entries)
        action, mode, information = choose_contact_action(
            entries,
            game.frame,
            planner_config=planner_config,
        )
        level_before = game.level_index
        record = game.step(action)
        if callable(observer):
            observer(record)
        history.append(record)
        if record.status in {"LEVEL_WIN", "GAME_WIN"}:
            levels_completed += 1
        steps.append(
            ContactAgentStep(
                action_number=action_number,
                level_index=level_before,
                action=action,
                decision_mode=mode,
                posterior_worlds=len(entries),
                posterior_entropy_bits=entropy,
                information_gain_bits=information,
                status=record.status,
            )
        )
        if game.finished:
            return ContactAgentResult(
                True,
                levels_completed,
                action_number,
                tuple(steps),
            )
    return ContactAgentResult(False, levels_completed, max_actions, tuple(steps))


def summarize_contact_result(result: ContactAgentResult) -> dict[str, Any]:
    return {
        "won": result.won,
        "levels_completed": result.levels_completed,
        "actions": result.actions,
        "fracture_actions": sum(step.decision_mode == "fracture" for step in result.steps),
        "planning_actions": sum(step.decision_mode == "plan" for step in result.steps),
        "posterior_worlds_by_action": [step.posterior_worlds for step in result.steps],
        "posterior_entropy_by_action": [step.posterior_entropy_bits for step in result.steps],
        "action_trace": [step.action.value for step in result.steps],
        "status_trace": [step.status for step in result.steps],
    }
