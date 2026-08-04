"""Stride-aware movement binding and navigation-plan generation.

ARC movement actions need not translate an object by one rendered cell. LS20,
for example, exposes exact pure-axis translations of five cells per action. This
module accepts any non-zero pure-axis stride while reusing the collision-aware
planner from :mod:`instella_arc.navigation`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from .action_belief import ActionBeliefState
from .navigation import (
    AgentHypothesis,
    NavigationPlan,
    locate_agent_components,
    plausible_targets,
    shortest_navigation_plan,
)
from .world_state import GridFacts


Coordinate = tuple[int, int]


def cardinal_direction(vector: tuple[int, int]) -> str | None:
    delta_row, delta_column = vector
    if delta_row < 0 and delta_column == 0:
        return "north"
    if delta_row > 0 and delta_column == 0:
        return "south"
    if delta_row == 0 and delta_column < 0:
        return "west"
    if delta_row == 0 and delta_column > 0:
        return "east"
    return None


@dataclass(frozen=True)
class StrideMovementBinding:
    action: str
    delta_row: int
    delta_column: int
    confidence: float
    support: int

    @property
    def direction(self) -> str:
        direction = cardinal_direction((self.delta_row, self.delta_column))
        if direction is None:
            raise ValueError("movement binding is not pure-axis cardinal")
        return direction

    @property
    def stride(self) -> int:
        return abs(self.delta_row) + abs(self.delta_column)


def _observation_cardinal_vectors(observation) -> tuple[tuple[int, int], ...]:
    vectors = tuple(
        vector
        for vector in observation.effect.translation_vectors
        if cardinal_direction(vector) is not None
    )
    if len(vectors) <= 1:
        return vectors

    # Prefer moved-component metadata when a transition contains several
    # translations. Small controllable components tend to expose a consistent
    # vector while large background effects vary. This ranks evidence; it does
    # not assign semantic colors.
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, Mapping):
        weighted: Counter[tuple[int, int]] = Counter()
        for moved in metadata.get("moved_components", []):
            try:
                vector = (int(moved["delta_row"]), int(moved["delta_column"]))
                shape = tuple(tuple(cell) for cell in moved["normalized_shape"])
            except (KeyError, TypeError, ValueError):
                continue
            if cardinal_direction(vector) is None:
                continue
            # Integer inverse-area weighting retains determinism.
            weight = max(1, 64 // max(len(shape), 1))
            weighted[vector] += weight
        if weighted:
            best_count = max(weighted.values())
            best = tuple(sorted(vector for vector, count in weighted.items() if count == best_count))
            if len(best) == 1:
                return best
    return ()


def infer_stride_movement_bindings(
    belief: ActionBeliefState,
    *,
    minimum_support: int = 1,
) -> tuple[StrideMovementBinding, ...]:
    provisional: list[StrideMovementBinding] = []
    for action, profile in sorted(belief.profiles.items()):
        vector_counts: Counter[tuple[int, int]] = Counter()
        for observation in profile.observations:
            vectors = _observation_cardinal_vectors(observation)
            if len(vectors) == 1:
                vector_counts[vectors[0]] += 1
        if not vector_counts:
            continue
        vector, support = vector_counts.most_common(1)[0]
        total = sum(vector_counts.values())
        confidence = support / total
        if support < minimum_support or confidence < 0.5:
            continue
        provisional.append(
            StrideMovementBinding(
                action=action,
                delta_row=vector[0],
                delta_column=vector[1],
                confidence=confidence,
                support=support,
            )
        )

    # Hidden directional controls should be one action per direction. Resolve
    # collisions by support, confidence, then shorter stride.
    by_direction: dict[str, StrideMovementBinding] = {}
    for binding in provisional:
        current = by_direction.get(binding.direction)
        if current is None or (
            binding.support,
            binding.confidence,
            -binding.stride,
            binding.action,
        ) > (
            current.support,
            current.confidence,
            -current.stride,
            current.action,
        ):
            by_direction[binding.direction] = binding
    return tuple(
        sorted(by_direction.values(), key=lambda binding: (binding.direction, binding.action))
    )


def infer_stride_agent_hypotheses(
    belief: ActionBeliefState,
    *,
    limit: int = 8,
) -> tuple[AgentHypothesis, ...]:
    counts: Counter[tuple[int, tuple[Coordinate, ...]]] = Counter()
    total = 0
    for observation in belief.transitions:
        metadata = getattr(observation, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        for moved in metadata.get("moved_components", []):
            try:
                vector = (int(moved["delta_row"]), int(moved["delta_column"]))
                color = int(moved["color"])
                shape = tuple(tuple(cell) for cell in moved["normalized_shape"])
            except (KeyError, TypeError, ValueError):
                continue
            if cardinal_direction(vector) is None:
                continue
            counts[(color, shape)] += 1
            total += 1
    return tuple(
        AgentHypothesis(
            color=color,
            shape=shape,
            confidence=support / max(total, 1),
            support=support,
        )
        for (color, shape), support in counts.most_common(limit)
    )


def stride_agent_hypotheses(
    facts: GridFacts,
    belief: ActionBeliefState,
    *,
    limit: int = 8,
) -> tuple[AgentHypothesis, ...]:
    observed = infer_stride_agent_hypotheses(belief, limit=limit)
    if observed:
        return observed
    color_counts = dict(facts.color_counts)
    majority_color = max(facts.color_counts, key=lambda pair: pair[1])[0]
    candidates = [
        component
        for component in facts.components
        if component.color != majority_color
    ]
    candidates.sort(
        key=lambda component: (
            color_counts[component.color],
            component.area,
            component.color,
            component.top,
            component.left,
        )
    )
    return tuple(
        AgentHypothesis(
            color=component.color,
            shape=component.normalized_shape,
            confidence=1.0 / max(len(candidates), 1),
            support=0,
        )
        for component in candidates[:limit]
    )


def generate_stride_navigation_plans(
    facts: GridFacts,
    belief: ActionBeliefState,
    *,
    max_agents: int = 6,
    max_targets_per_agent: int = 16,
    max_plans: int = 64,
) -> tuple[NavigationPlan, ...]:
    bindings = infer_stride_movement_bindings(belief)
    if len(bindings) < 2:
        return ()
    hypotheses = stride_agent_hypotheses(facts, belief, limit=max_agents)
    plans: list[NavigationPlan] = []
    for hypothesis in hypotheses:
        for component in locate_agent_components(facts, hypothesis):
            for target in plausible_targets(
                facts, component, limit=max_targets_per_agent
            ):
                plan = shortest_navigation_plan(
                    facts,
                    agent_component=component,
                    agent_hypothesis=hypothesis,
                    target=target,
                    bindings=bindings,
                )
                if plan is not None:
                    plans.append(plan)
    plans.sort(
        key=lambda plan: (
            plan.length,
            -plan.target.heuristic_priority,
            -plan.agent.confidence,
            plan.target.color,
            plan.target.relation,
            plan.action_sequence,
        )
    )
    return tuple(plans[:max_plans])
