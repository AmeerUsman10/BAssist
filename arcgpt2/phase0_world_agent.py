"""Closed-loop executable-posterior agent for the hidden-action gate.

This module joins the project's pieces without adding another learned model:

- the same GPT-2 can score candidate action meanings;
- exact replay removes contradicted mechanics programs;
- a posterior fracture planner chooses informative interventions;
- once the surviving programs agree, generic search executes the inferred rule.

The environment generator's palette and primitive mechanics family are known in
this controlled gate. The hidden action permutation and layouts are not. This is
an integration test for the epistemic architecture, not an ARC-AGI-3 score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch

from .completion_scorer import score_with_contextual_calibration
from .dsl import Program, enumerate_phase0_programs, shortest_plan
from .goal_dsl import phase0_goal
from .natural_protocol import answer_text, direction_words, mapping_prompt
from .phase0_hidden_action import (
    Action,
    DIRECTION_DELTA,
    Direction,
    GameSpec,
    HiddenActionGame,
    StepRecord,
)
from .world_posterior import (
    WorldFracturePlanner,
    WorldHypothesis,
    WorldPlannerConfig,
    condition_worlds_exact,
    world_entropy,
)


class MappingScoreProvider(Protocol):
    def score(
        self,
        spec: GameSpec,
        history: Sequence[StepRecord],
        query_action: Action,
        current_grid,
    ) -> Mapping[Direction, float]: ...


class UniformMappingScores:
    """A no-learning control with equal log prior for every direction."""

    def score(
        self,
        spec: GameSpec,
        history: Sequence[StepRecord],
        query_action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec, history, query_action, current_grid
        return {direction: 0.0 for direction in Direction}


class GPT2MappingScores:
    """Use one GPT-2 checkpoint's completion likelihood as the mapping prior."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str | None = None,
        candidate_batch_size: int = 4,
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

    def score(
        self,
        spec: GameSpec,
        history: Sequence[StepRecord],
        query_action: Action,
        current_grid,
    ) -> Mapping[Direction, float]:
        del spec
        context = mapping_prompt(current_grid, history, query_action)
        null_context = mapping_prompt(
            current_grid,
            (),
            query_action,
            include_evidence=False,
        )
        prompt_ids = self.tokenizer.encode(context, add_special_tokens=False)
        null_ids = self.tokenizer.encode(null_context, add_special_tokens=False)
        candidate_ids = tuple(
            tuple(self.tokenizer.encode(answer_text(word), add_special_tokens=False))
            for word in direction_words()
        )
        with torch.no_grad():
            scores = score_with_contextual_calibration(
                self.model,
                prompt_ids,
                null_ids,
                candidate_ids,
                pad_token_id=int(self.tokenizer.pad_token_id),
                device=self.device,
                candidate_batch_size=self.candidate_batch_size,
                reduction="mean",
            )
        word_direction = {
            "north": Direction.UP,
            "south": Direction.DOWN,
            "west": Direction.LEFT,
            "east": Direction.RIGHT,
        }
        return {
            word_direction[word]: float(scores[index].item())
            for index, word in enumerate(direction_words())
        }


_DELTA_DIRECTION = {delta: direction for direction, delta in DIRECTION_DELTA.items()}


def program_mapping(program: Program) -> Mapping[Action, Direction]:
    mapping: dict[Action, Direction] = {}
    for action, rule in program.by_action.items():
        try:
            mapping[action] = _DELTA_DIRECTION[(rule.dy, rule.dx)]
        except KeyError as exc:
            raise ValueError(
                "Phase-0 mapping programs must use one-cell cardinal movement"
            ) from exc
    if set(mapping) != set(Action):
        raise ValueError("Phase-0 mapping program must define all four actions")
    return mapping


def scored_worlds(
    spec: GameSpec,
    history: Sequence[StepRecord],
    current_grid,
    score_provider: MappingScoreProvider,
):
    unary = {
        action: score_provider.score(spec, history, action, current_grid)
        for action in Action
    }
    goal = phase0_goal(spec)
    hypotheses: list[WorldHypothesis] = []
    for program in enumerate_phase0_programs(spec):
        mapping = program_mapping(program)
        log_prior = sum(float(unary[action][mapping[action]]) for action in Action)
        hypotheses.append(
            WorldHypothesis(
                mechanics=program,
                goal=goal,
                log_prior=log_prior,
                source=type(score_provider).__name__,
            )
        )
    return condition_worlds_exact(hypotheses, history)


@dataclass(frozen=True)
class AgentStep:
    action_number: int
    level_index: int
    action: Action
    decision_mode: str
    posterior_worlds: int
    posterior_entropy_bits: float
    information_gain_bits: float
    moved: bool
    status: str


@dataclass(frozen=True)
class AgentResult:
    won: bool
    levels_completed: int
    actions: int
    steps: tuple[AgentStep, ...]


def choose_action(entries, current_grid) -> tuple[Action, str, float]:
    """Explore while hypotheses disagree; plan once one world remains."""

    if len(entries) == 1 or world_entropy(entries) < 1e-9:
        plan = shortest_plan(entries[0].mechanics, current_grid, max_depth=64)
        if plan:
            return plan[0], "plan", 0.0

    planner = WorldFracturePlanner(
        entries,
        WorldPlannerConfig(
            depth=1,
            terminal_reward=2.0,
            information_weight=1.0,
            action_cost=0.01,
            no_change_penalty=0.02,
            discount=0.0,
        ),
    )
    decision = planner.choose_action(current_grid)
    return decision.action, "fracture", decision.information_gain_bits


def run_agent(
    spec: GameSpec,
    score_provider: MappingScoreProvider | None = None,
    *,
    max_actions: int = 128,
) -> AgentResult:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    provider = score_provider or UniformMappingScores()
    reset = getattr(provider, "reset", None)
    if callable(reset):
        reset()
    observer = getattr(provider, "observe", None)

    game = HiddenActionGame(spec)
    history: list[StepRecord] = []
    steps: list[AgentStep] = []
    levels_completed = 0

    for action_number in range(1, max_actions + 1):
        current_grid = game.frame
        entries = scored_worlds(spec, history, current_grid, provider)
        entropy = world_entropy(entries)
        action, mode, information = choose_action(entries, current_grid)
        level_before = game.level_index
        record = game.step(action)
        if callable(observer):
            observer(record)
        history.append(record)
        if record.status in {"LEVEL_WIN", "GAME_WIN"}:
            levels_completed += 1
        steps.append(
            AgentStep(
                action_number=action_number,
                level_index=level_before,
                action=action,
                decision_mode=mode,
                posterior_worlds=len(entries),
                posterior_entropy_bits=entropy,
                information_gain_bits=information,
                moved=record.moved,
                status=record.status,
            )
        )
        if game.finished:
            return AgentResult(True, levels_completed, action_number, tuple(steps))
    return AgentResult(False, levels_completed, max_actions, tuple(steps))


def summarize_result(result: AgentResult) -> dict[str, object]:
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
