"""Closed-loop posterior agent for hidden action and contact mechanics.

The agent maintains exact executable worlds over 24 action permutations and
five contact primitives. A generic fracture-sequence search navigates through
states where all surviving worlds agree until it reaches an intervention whose
outcomes split the posterior. Once one world survives, generic BFS plans to the
goal. The only learned input allowed is an optional prior/scorer supplied by one
GPT-2 checkpoint.
"""

from __future__ import annotations

from collections import defaultdict, deque
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
    fracture_navigation_depth: int = 16

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("contact fracture depth must be positive")
        if self.fracture_navigation_depth < 1:
            raise ValueError("fracture_navigation_depth must be positive")
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
    probabilities = [mass / total for mass in masses]
    # Prevent tiny drift from breaking long posterior chains and exact receipt
    # checks. The correction is much smaller than any meaningful probability.
    largest = max(range(len(probabilities)), key=probabilities.__getitem__)
    probabilities[largest] += 1.0 - sum(probabilities)
    return tuple(
        ContactWorldEntry(program, log_weight, probability, source)
        for (program, log_weight, source), probability in zip(
            weighted, probabilities, strict=True
        )
    )


def condition_contact_worlds(
    entries: Sequence[ContactWorldEntry]
    | Sequence[tuple[MechanicsProgramV2, float, str]],
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


def _raw_partition(
    entries: Sequence[ContactWorldEntry],
    grid,
    action: Action,
) -> Mapping[tuple[Any, MechanicsStatus], tuple[ContactWorldEntry, ...]]:
    groups: dict[
        tuple[Any, MechanicsStatus], list[ContactWorldEntry]
    ] = defaultdict(list)
    for entry in entries:
        result = execute_v2(entry.program, grid, action)
        groups[(result.after, result.status)].append(entry)
    return {key: tuple(group) for key, group in groups.items()}


def contact_partition(
    entries: Sequence[ContactWorldEntry],
    grid,
    action: Action,
) -> Mapping[tuple[Any, MechanicsStatus], tuple[ContactWorldEntry, ...]]:
    normalized: dict[
        tuple[Any, MechanicsStatus], tuple[ContactWorldEntry, ...]
    ] = {}
    for key, group in _raw_partition(entries, grid, action).items():
        mass = sum(entry.probability for entry in group)
        normalized[key] = tuple(
            ContactWorldEntry(
                entry.program,
                entry.log_weight,
                entry.probability / mass,
                entry.source,
            )
            for entry in group
        )
    return normalized


def action_information_gain(
    entries: Sequence[ContactWorldEntry],
    grid,
    action: Action,
) -> float:
    prior = contact_world_entropy(entries)
    expected = 0.0
    for group in _raw_partition(entries, grid, action).values():
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
        expected += mass * contact_world_entropy(normalized)
    value = prior - expected
    return 0.0 if abs(value) < 1e-14 else value


def find_fracture_sequence(
    entries: Sequence[ContactWorldEntry],
    start_grid,
    *,
    max_navigation_depth: int = 16,
) -> tuple[tuple[Action, ...], float] | None:
    """Find the shortest reachable intervention that splits the posterior.

    The search expands only *consensus* transitions: actions for which every
    surviving world predicts exactly the same active next state. Therefore no
    branch or hidden semantic decision is smuggled into navigation. The first
    non-consensus action is the experiment.
    """

    if max_navigation_depth < 1:
        raise ValueError("max_navigation_depth must be positive")
    queue: deque[tuple[Any, tuple[Action, ...]]] = deque([(start_grid, ())])
    visited = {start_grid}

    while queue:
        state, prefix = queue.popleft()
        informative = [
            (action, action_information_gain(entries, state, action))
            for action in Action
        ]
        best_action, best_gain = max(
            informative,
            key=lambda item: (item[1], -list(Action).index(item[0])),
        )
        if best_gain > 1e-12:
            return (*prefix, best_action), best_gain
        if len(prefix) >= max_navigation_depth:
            continue

        for action in Action:
            groups = _raw_partition(entries, state, action)
            if len(groups) != 1:
                continue
            (after, status), _ = next(iter(groups.items()))
            if status is MechanicsStatus.WIN or after == state or after in visited:
                continue
            visited.add(after)
            queue.append((after, (*prefix, action)))
    return None


class ContactFracturePlanner:
    """Short-horizon fallback when no consensus fracture path is reachable."""

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
        decisions = [
            self._decision(self.entries, grid, action, self.config.depth)
            for action in Action
        ]
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
        raw_groups = _raw_partition(entries, grid, action)
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
        if abs(information_gain) < 1e-14:
            information_gain = 0.0
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
    config = planner_config or ContactPlannerConfig()
    if len(entries) == 1 or contact_world_entropy(entries) < 1e-9:
        plan = shortest_plan_v2(entries[0].program, grid, max_depth=96)
        if plan:
            return plan[0], "plan", 0.0

    fracture = find_fracture_sequence(
        entries,
        grid,
        max_navigation_depth=config.fracture_navigation_depth,
    )
    if fracture is not None:
        sequence, eventual_gain = fracture
        first_action = sequence[0]
        immediate_gain = action_information_gain(entries, grid, first_action)
        del eventual_gain
        return first_action, "fracture", immediate_gain

    decision = ContactFracturePlanner(entries, config).choose_action(grid)
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
    planner_config = ContactPlannerConfig(
        depth=planner_depth,
        fracture_navigation_depth=max(8, planner_depth * 8),
    )

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
        "fracture_actions": sum(
            step.decision_mode == "fracture" for step in result.steps
        ),
        "planning_actions": sum(
            step.decision_mode == "plan" for step in result.steps
        ),
        "posterior_worlds_by_action": [
            step.posterior_worlds for step in result.steps
        ],
        "posterior_entropy_by_action": [
            step.posterior_entropy_bits for step in result.steps
        ],
        "action_trace": [step.action.value for step in result.steps],
        "status_trace": [step.status for step in result.steps],
    }
