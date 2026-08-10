"""Joint posterior over GPT-2 mechanics and terminal-goal programs.

This is the central epistemic state for the executable-program agent. GPT-2 is
the only learned source of hypotheses and priors. The module supplies exact
replay likelihoods, posterior normalization, predictive disagreement, and a
bounded Bayes-adaptive fracture search over those hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Iterable, Sequence

from .codec import Grid, normalize_grid
from .dsl import Program, execute
from .goal_dsl import GoalContext, GoalProgram, strip_terminal
from .phase0_hidden_action import Action, StepRecord


class WorldPosteriorError(ValueError):
    """Raised when the joint hypothesis posterior is invalid."""


@dataclass(frozen=True)
class WorldHypothesis:
    mechanics: Program
    goal: GoalProgram
    log_prior: float = 0.0
    source: str = "gpt2"

    @property
    def sha256(self) -> str:
        payload = f"{self.mechanics.sha256}:{self.goal.sha256}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorldEntry:
    mechanics: Program
    goal: GoalProgram
    probability: float
    log_prior: float
    source: str

    @property
    def sha256(self) -> str:
        payload = f"{self.mechanics.sha256}:{self.goal.sha256}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorldOutcome:
    grid: Grid
    terminal: bool
    changed: bool
    blocked: bool


@dataclass(frozen=True)
class WorldBranch:
    outcome: WorldOutcome
    probability: float
    entries: tuple[WorldEntry, ...]


@dataclass(frozen=True)
class WorldActionEvaluation:
    action: Action
    expected_terminal_reward: float
    information_gain_bits: float
    no_change_probability: float
    expected_future_value: float
    total_value: float
    branches: tuple[WorldBranch, ...]


@dataclass(frozen=True)
class WorldPlannerConfig:
    depth: int = 3
    terminal_reward: float = 1.0
    information_weight: float = 0.10
    action_cost: float = 0.01
    no_change_penalty: float = 0.01
    discount: float = 0.97

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise WorldPosteriorError("planner depth must be at least one")
        if not 0.0 <= self.discount <= 1.0:
            raise WorldPosteriorError("discount must lie in 0..1")


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise WorldPosteriorError("logsumexp requires values")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalize_worlds(hypotheses: Iterable[WorldHypothesis]) -> tuple[WorldEntry, ...]:
    items = tuple(hypotheses)
    if not items:
        raise WorldPosteriorError("at least one world hypothesis is required")

    grouped: dict[str, list[WorldHypothesis]] = {}
    for hypothesis in items:
        grouped.setdefault(hypothesis.sha256, []).append(hypothesis)

    merged: list[WorldHypothesis] = []
    for group in grouped.values():
        representative = group[0]
        merged.append(
            WorldHypothesis(
                mechanics=representative.mechanics,
                goal=representative.goal,
                log_prior=_logsumexp([item.log_prior for item in group]),
                source="+".join(sorted({item.source for item in group})),
            )
        )

    normalizer = _logsumexp([item.log_prior for item in merged])
    entries = tuple(
        WorldEntry(
            mechanics=item.mechanics,
            goal=item.goal,
            probability=math.exp(item.log_prior - normalizer),
            log_prior=item.log_prior,
            source=item.source,
        )
        for item in sorted(merged, key=lambda candidate: candidate.sha256)
    )
    if not math.isclose(sum(entry.probability for entry in entries), 1.0, abs_tol=1e-9):
        raise WorldPosteriorError("world posterior normalization failed")
    return entries


def _predict(entry: WorldEntry, grid: Grid, action: Action) -> WorldOutcome:
    execution = execute(strip_terminal(entry.mechanics), grid, action)
    context = GoalContext(
        before=grid,
        after=execution.after,
        action=action,
        execution=execution,
    )
    terminal = entry.goal.predicate.evaluate(context)
    return WorldOutcome(
        grid=execution.after,
        terminal=terminal,
        changed=execution.changed,
        blocked=execution.blocked,
    )


def condition_worlds_exact(
    hypotheses: Iterable[WorldHypothesis],
    history: Sequence[StepRecord],
) -> tuple[WorldEntry, ...]:
    survivors: list[WorldHypothesis] = []
    for hypothesis in hypotheses:
        entry = WorldEntry(
            mechanics=hypothesis.mechanics,
            goal=hypothesis.goal,
            probability=1.0,
            log_prior=hypothesis.log_prior,
            source=hypothesis.source,
        )
        consistent = True
        for record in history:
            outcome = _predict(entry, record.before, record.action)
            expected_terminal = record.status in {"LEVEL_WIN", "GAME_WIN"}
            if outcome.grid != record.after or outcome.terminal != expected_terminal:
                consistent = False
                break
        if consistent:
            survivors.append(hypothesis)
    if not survivors:
        raise WorldPosteriorError("all world hypotheses were contradicted")
    return normalize_worlds(survivors)


def world_entropy(entries: Sequence[WorldEntry]) -> float:
    return -sum(
        entry.probability * math.log2(entry.probability)
        for entry in entries
        if entry.probability > 0.0
    )


def world_branches(
    entries: Sequence[WorldEntry],
    grid: Sequence[Sequence[int]],
    action: Action,
) -> tuple[WorldBranch, ...]:
    if not entries:
        raise WorldPosteriorError("predictive branches require entries")
    state = normalize_grid(grid)
    grouped: dict[WorldOutcome, list[WorldEntry]] = {}
    for entry in entries:
        grouped.setdefault(_predict(entry, state, action), []).append(entry)

    branches: list[WorldBranch] = []
    for outcome, group in grouped.items():
        mass = sum(entry.probability for entry in group)
        conditional = tuple(
            WorldEntry(
                mechanics=entry.mechanics,
                goal=entry.goal,
                probability=entry.probability / mass,
                log_prior=entry.log_prior,
                source=entry.source,
            )
            for entry in group
        )
        branches.append(WorldBranch(outcome, mass, conditional))
    branches.sort(
        key=lambda branch: (
            -branch.probability,
            branch.outcome.terminal,
            branch.outcome.grid,
        )
    )
    if not math.isclose(sum(branch.probability for branch in branches), 1.0, abs_tol=1e-9):
        raise WorldPosteriorError("world branch probabilities do not sum to one")
    return tuple(branches)


def world_information_gain(
    entries: Sequence[WorldEntry],
    branches: Sequence[WorldBranch],
) -> float:
    expected_after = sum(
        branch.probability * world_entropy(branch.entries)
        for branch in branches
    )
    return max(0.0, world_entropy(entries) - expected_after)


def _posterior_key(entries: Sequence[WorldEntry]) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (entry.sha256, int(round(entry.probability * 1_000_000_000)))
            for entry in entries
        )
    )


class WorldFracturePlanner:
    """Bounded search over a posterior of mechanics/goal program pairs."""

    def __init__(
        self,
        entries: Sequence[WorldEntry],
        config: WorldPlannerConfig | None = None,
        actions: Sequence[Action] = tuple(Action),
    ) -> None:
        if not entries:
            raise WorldPosteriorError("planner requires entries")
        self.entries = tuple(entries)
        self.config = config or WorldPlannerConfig()
        self.actions = tuple(actions)
        if not self.actions:
            raise WorldPosteriorError("planner requires legal actions")
        self._registry = {_posterior_key(self.entries): self.entries}

    def evaluate_actions(self, grid: Sequence[Sequence[int]]) -> tuple[WorldActionEvaluation, ...]:
        state = normalize_grid(grid)
        evaluations = tuple(
            self._evaluate(self.entries, state, action, self.config.depth)
            for action in self.actions
        )
        return tuple(
            sorted(
                evaluations,
                key=lambda result: (-result.total_value, self.actions.index(result.action)),
            )
        )

    def choose_action(self, grid: Sequence[Sequence[int]]) -> WorldActionEvaluation:
        return self.evaluate_actions(grid)[0]

    def _evaluate(
        self,
        entries: Sequence[WorldEntry],
        state: Grid,
        action: Action,
        depth: int,
    ) -> WorldActionEvaluation:
        branches = world_branches(entries, state, action)
        information = world_information_gain(entries, branches)
        immediate = sum(
            branch.probability * self.config.terminal_reward
            for branch in branches
            if branch.outcome.terminal
        )
        no_change = sum(
            branch.probability for branch in branches if not branch.outcome.changed
        )
        future = 0.0
        if depth > 1:
            for branch in branches:
                if branch.outcome.terminal:
                    continue
                best = max(
                    self._value(
                        branch.entries,
                        branch.outcome.grid,
                        candidate,
                        depth - 1,
                    )
                    for candidate in self.actions
                )
                future += branch.probability * best
        total = (
            immediate
            + self.config.information_weight * information
            + self.config.discount * future
            - self.config.action_cost
            - self.config.no_change_penalty * no_change
        )
        return WorldActionEvaluation(
            action=action,
            expected_terminal_reward=immediate,
            information_gain_bits=information,
            no_change_probability=no_change,
            expected_future_value=future,
            total_value=total,
            branches=branches,
        )

    def _value(
        self,
        entries: Sequence[WorldEntry],
        state: Grid,
        action: Action,
        depth: int,
    ) -> float:
        key = _posterior_key(entries)
        self._registry.setdefault(key, tuple(entries))
        return self._cached_value(key, state, action, depth)

    @lru_cache(maxsize=100_000)
    def _cached_value(
        self,
        key: tuple[tuple[str, int], ...],
        state: Grid,
        action: Action,
        depth: int,
    ) -> float:
        return self._evaluate(self._registry[key], state, action, depth).total_value
