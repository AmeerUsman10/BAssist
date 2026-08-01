"""Exact posterior and fracture planning over GPT-2-proposed programs.

GPT-2 supplies candidate program text and optional log-prior scores. This module
performs only deterministic arithmetic, replay, and bounded search. It contains
no learned value function, semantic classifier, or game-specific policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable, Mapping, Sequence

from .codec import Grid, normalize_grid
from .dsl import Program, Status, execute, replay
from .phase0_hidden_action import Action, StepRecord


class PosteriorError(ValueError):
    """Raised when a posterior cannot be constructed or queried."""


@dataclass(frozen=True)
class WeightedProgram:
    program: Program
    log_prior: float = 0.0
    source: str = "gpt2"


@dataclass(frozen=True)
class PosteriorEntry:
    program: Program
    probability: float
    log_prior: float
    source: str


@dataclass(frozen=True)
class PredictionOutcome:
    grid: Grid
    status: Status
    changed: bool
    blocked: bool


@dataclass(frozen=True)
class OutcomeBranch:
    outcome: PredictionOutcome
    probability: float
    entries: tuple[PosteriorEntry, ...]


@dataclass(frozen=True)
class ActionEvaluation:
    action: Action
    expected_immediate_reward: float
    information_gain_bits: float
    expected_future_value: float
    no_change_probability: float
    total_value: float
    branches: tuple[OutcomeBranch, ...]


@dataclass(frozen=True)
class PlannerConfig:
    depth: int = 3
    terminal_reward: float = 1.0
    information_weight: float = 0.10
    action_cost: float = 0.01
    no_change_penalty: float = 0.01
    discount: float = 0.97

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise PosteriorError("planner depth must be at least one")
        if not 0.0 <= self.discount <= 1.0:
            raise PosteriorError("discount must lie in 0..1")


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise PosteriorError("logsumexp requires at least one value")
    maximum = max(values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalize_programs(programs: Iterable[WeightedProgram]) -> tuple[PosteriorEntry, ...]:
    candidates = tuple(programs)
    if not candidates:
        raise PosteriorError("posterior requires at least one candidate program")

    # Behaviorally identical programs are merged so textual duplicates do not
    # receive excess posterior mass. The canonical program hash defines exact
    # identity for the current DSL version.
    grouped: dict[str, list[WeightedProgram]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.program.sha256, []).append(candidate)

    merged: list[WeightedProgram] = []
    for group in grouped.values():
        representative = group[0]
        merged_log_prior = _logsumexp([item.log_prior for item in group])
        merged.append(
            WeightedProgram(
                program=representative.program,
                log_prior=merged_log_prior,
                source="+".join(sorted({item.source for item in group})),
            )
        )

    normalizer = _logsumexp([candidate.log_prior for candidate in merged])
    entries = tuple(
        PosteriorEntry(
            program=candidate.program,
            probability=math.exp(candidate.log_prior - normalizer),
            log_prior=candidate.log_prior,
            source=candidate.source,
        )
        for candidate in sorted(merged, key=lambda item: item.program.sha256)
    )
    probability_sum = sum(entry.probability for entry in entries)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise PosteriorError("posterior normalization failed")
    return entries


def condition_exact(
    programs: Iterable[WeightedProgram],
    history: Sequence[StepRecord],
) -> tuple[PosteriorEntry, ...]:
    """Apply a deterministic replay likelihood: one contradiction is fatal."""

    surviving = [
        candidate
        for candidate in programs
        if replay(candidate.program, history).consistent
    ]
    if not surviving:
        raise PosteriorError("all candidate programs were contradicted by history")
    return normalize_programs(surviving)


def posterior_entropy(entries: Sequence[PosteriorEntry]) -> float:
    return -sum(
        entry.probability * math.log2(entry.probability)
        for entry in entries
        if entry.probability > 0.0
    )


def predictive_branches(
    entries: Sequence[PosteriorEntry],
    grid: Sequence[Sequence[int]],
    action: Action,
) -> tuple[OutcomeBranch, ...]:
    state = normalize_grid(grid)
    if not entries:
        raise PosteriorError("predictive branches require a non-empty posterior")

    grouped: dict[PredictionOutcome, list[PosteriorEntry]] = {}
    for entry in entries:
        result = execute(entry.program, state, action)
        outcome = PredictionOutcome(
            grid=result.after,
            status=result.status,
            changed=result.changed,
            blocked=result.blocked,
        )
        grouped.setdefault(outcome, []).append(entry)

    branches: list[OutcomeBranch] = []
    for outcome, group in grouped.items():
        mass = sum(entry.probability for entry in group)
        if mass <= 0.0:
            continue
        conditional = tuple(
            PosteriorEntry(
                program=entry.program,
                probability=entry.probability / mass,
                log_prior=entry.log_prior,
                source=entry.source,
            )
            for entry in group
        )
        branches.append(
            OutcomeBranch(
                outcome=outcome,
                probability=mass,
                entries=conditional,
            )
        )
    branches.sort(
        key=lambda branch: (
            -branch.probability,
            branch.outcome.status.value,
            branch.outcome.grid,
        )
    )
    if not math.isclose(sum(branch.probability for branch in branches), 1.0, abs_tol=1e-9):
        raise PosteriorError("predictive branch probabilities do not sum to one")
    return tuple(branches)


def expected_information_gain(
    entries: Sequence[PosteriorEntry],
    branches: Sequence[OutcomeBranch],
) -> float:
    prior_entropy = posterior_entropy(entries)
    expected_after = sum(
        branch.probability * posterior_entropy(branch.entries)
        for branch in branches
    )
    return max(0.0, prior_entropy - expected_after)


def _posterior_key(entries: Sequence[PosteriorEntry]) -> tuple[tuple[str, int], ...]:
    # Quantize probabilities only for memoization; returned values still use the
    # exact floating-point entries supplied to the public evaluator.
    return tuple(
        sorted(
            (
                entry.program.sha256,
                int(round(entry.probability * 1_000_000_000)),
            )
            for entry in entries
        )
    )


class FracturePlanner:
    """Finite-horizon Bayes-adaptive search over executable GPT-2 hypotheses."""

    def __init__(
        self,
        entries: Sequence[PosteriorEntry],
        config: PlannerConfig | None = None,
        actions: Sequence[Action] = tuple(Action),
    ) -> None:
        if not entries:
            raise PosteriorError("planner requires a non-empty posterior")
        self.entries = tuple(entries)
        self.config = config or PlannerConfig()
        self.actions = tuple(actions)
        if not self.actions:
            raise PosteriorError("planner requires at least one legal action")
        self._entry_registry: dict[tuple[tuple[str, int], ...], tuple[PosteriorEntry, ...]] = {
            _posterior_key(self.entries): self.entries
        }

    def evaluate_actions(self, grid: Sequence[Sequence[int]]) -> tuple[ActionEvaluation, ...]:
        state = normalize_grid(grid)
        evaluations = tuple(
            self._evaluate_action(self.entries, state, action, self.config.depth)
            for action in self.actions
        )
        return tuple(
            sorted(
                evaluations,
                key=lambda item: (-item.total_value, self.actions.index(item.action)),
            )
        )

    def choose_action(self, grid: Sequence[Sequence[int]]) -> ActionEvaluation:
        return self.evaluate_actions(grid)[0]

    def _evaluate_action(
        self,
        entries: Sequence[PosteriorEntry],
        state: Grid,
        action: Action,
        depth: int,
    ) -> ActionEvaluation:
        branches = predictive_branches(entries, state, action)
        information_gain = expected_information_gain(entries, branches)
        immediate_reward = sum(
            branch.probability
            * (self.config.terminal_reward if branch.outcome.status is Status.WIN else 0.0)
            for branch in branches
        )
        no_change_probability = sum(
            branch.probability for branch in branches if not branch.outcome.changed
        )

        expected_future = 0.0
        if depth > 1:
            for branch in branches:
                if branch.outcome.status is Status.WIN:
                    continue
                future_best = max(
                    self._value(
                        branch.entries,
                        branch.outcome.grid,
                        candidate,
                        depth - 1,
                    )
                    for candidate in self.actions
                )
                expected_future += branch.probability * future_best

        total = (
            immediate_reward
            + self.config.information_weight * information_gain
            + self.config.discount * expected_future
            - self.config.action_cost
            - self.config.no_change_penalty * no_change_probability
        )
        return ActionEvaluation(
            action=action,
            expected_immediate_reward=immediate_reward,
            information_gain_bits=information_gain,
            expected_future_value=expected_future,
            no_change_probability=no_change_probability,
            total_value=total,
            branches=branches,
        )

    def _value(
        self,
        entries: Sequence[PosteriorEntry],
        state: Grid,
        action: Action,
        depth: int,
    ) -> float:
        key = _posterior_key(entries)
        self._entry_registry.setdefault(key, tuple(entries))
        return self._cached_value(key, state, action, depth)

    @lru_cache(maxsize=100_000)
    def _cached_value(
        self,
        posterior_key: tuple[tuple[str, int], ...],
        state: Grid,
        action: Action,
        depth: int,
    ) -> float:
        entries = self._entry_registry[posterior_key]
        return self._evaluate_action(entries, state, action, depth).total_value
