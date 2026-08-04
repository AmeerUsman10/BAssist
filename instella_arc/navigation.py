"""Exact movement semantics and collision-aware navigation candidates.

The navigation specialist activates only after observed component translations
support a movement interpretation. It never assumes action numbers or colors
carry semantics. It constructs several explicit target/contact hypotheses and
lets terminal/progress evidence eliminate them online.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .action_belief import ActionBeliefState
from .world_state import Component, GridFacts


Coordinate = tuple[int, int]


CARDINAL_VECTOR_NAME: dict[tuple[int, int], str] = {
    (-1, 0): "north",
    (1, 0): "south",
    (0, -1): "west",
    (0, 1): "east",
}


@dataclass(frozen=True)
class MovementBinding:
    action: str
    delta_row: int
    delta_column: int
    confidence: float
    support: int

    @property
    def direction(self) -> str:
        return CARDINAL_VECTOR_NAME[(self.delta_row, self.delta_column)]


@dataclass(frozen=True)
class AgentHypothesis:
    color: int
    shape: tuple[Coordinate, ...]
    confidence: float
    support: int


@dataclass(frozen=True)
class NavigationTarget:
    component_index: int
    color: int
    cells: tuple[Coordinate, ...]
    relation: str
    heuristic_priority: float


@dataclass(frozen=True)
class NavigationPlan:
    agent: AgentHypothesis
    target: NavigationTarget
    start_top_left: Coordinate
    destination_top_left: Coordinate
    action_sequence: tuple[str, ...]
    position_sequence: tuple[Coordinate, ...]
    assumed_passable_colors: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.action_sequence)


def infer_movement_bindings(
    belief: ActionBeliefState,
    *,
    minimum_support: int = 1,
) -> tuple[MovementBinding, ...]:
    bindings: list[MovementBinding] = []
    used_vectors: set[tuple[int, int]] = set()
    for action, profile in sorted(belief.profiles.items()):
        vector_counts: Counter[tuple[int, int]] = Counter()
        for observation in profile.observations:
            vectors = observation.effect.translation_vectors
            if len(vectors) == 1 and vectors[0] in CARDINAL_VECTOR_NAME:
                vector_counts[vectors[0]] += 1
        if not vector_counts:
            continue
        vector, support = vector_counts.most_common(1)[0]
        total = sum(vector_counts.values())
        confidence = support / max(total, 1)
        if support < minimum_support or confidence < 0.5:
            continue
        # A hidden action mapping is normally one-to-one. Keep the stronger
        # binding when contradictory actions appear to share a vector.
        if vector in used_vectors:
            continue
        used_vectors.add(vector)
        bindings.append(
            MovementBinding(
                action=action,
                delta_row=vector[0],
                delta_column=vector[1],
                confidence=confidence,
                support=support,
            )
        )
    return tuple(
        sorted(bindings, key=lambda binding: (binding.direction, binding.action))
    )


def infer_agent_hypotheses(
    belief: ActionBeliefState,
    *,
    limit: int = 8,
) -> tuple[AgentHypothesis, ...]:
    counts: Counter[tuple[int, tuple[Coordinate, ...]]] = Counter()
    totals = 0
    for observation in belief.transitions:
        for translation in observation.effect.translation_vectors:
            if translation not in CARDINAL_VECTOR_NAME:
                continue
        # Recover moved component shape/color from literal transition evidence
        # is not available in EffectSignature, so use the transition receipt's
        # stable signatures recorded by action observation metadata when present.
        # Older observations simply contribute no agent hypothesis.
        metadata = getattr(observation, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        for moved in metadata.get("moved_components", []):
            try:
                color = int(moved["color"])
                shape = tuple(tuple(cell) for cell in moved["normalized_shape"])
            except (KeyError, TypeError, ValueError):
                continue
            counts[(color, shape)] += 1
            totals += 1
    hypotheses = [
        AgentHypothesis(
            color=color,
            shape=shape,
            confidence=support / max(totals, 1),
            support=support,
        )
        for (color, shape), support in counts.most_common(limit)
    ]
    return tuple(hypotheses)


def agent_hypotheses_from_current_and_bindings(
    current: GridFacts,
    belief: ActionBeliefState,
    *,
    limit: int = 8,
) -> tuple[AgentHypothesis, ...]:
    """Fallback agent candidates when older receipts lack component metadata.

    Candidate colors are those participating in observed translations when such
    metadata exists; otherwise small non-majority components are retained with
    low confidence for model ranking and exact execution tests.
    """

    observed = infer_agent_hypotheses(belief, limit=limit)
    if observed:
        return observed
    color_counts = dict(current.color_counts)
    majority_color = max(current.color_counts, key=lambda pair: pair[1])[0]
    candidates = [
        component
        for component in current.components
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


def locate_agent_components(
    facts: GridFacts,
    hypothesis: AgentHypothesis,
) -> tuple[Component, ...]:
    return tuple(
        component
        for component in facts.components
        if component.color == hypothesis.color
        and component.normalized_shape == hypothesis.shape
    )


def plausible_targets(
    facts: GridFacts,
    agent_component: Component,
    *,
    limit: int = 24,
) -> tuple[NavigationTarget, ...]:
    color_counts = dict(facts.color_counts)
    majority_color = max(facts.color_counts, key=lambda pair: pair[1])[0]
    targets: list[NavigationTarget] = []
    for index, component in enumerate(facts.components):
        if component == agent_component:
            continue
        if component.color == majority_color and component.area > 1:
            continue
        rarity = 1.0 / max(color_counts[component.color], 1)
        distance = min(
            abs(row - agent_component.top) + abs(column - agent_component.left)
            for row, column in component.cells
        )
        for relation, relation_bonus in (
            ("overlap", 3.0),
            ("adjacent", 2.0),
            ("touch", 1.0),
        ):
            targets.append(
                NavigationTarget(
                    component_index=index,
                    color=component.color,
                    cells=component.cells,
                    relation=relation,
                    heuristic_priority=(
                        relation_bonus + rarity - 0.01 * distance
                    ),
                )
            )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                -target.heuristic_priority,
                target.color,
                target.component_index,
                target.relation,
            ),
        )[:limit]
    )


def _placed_cells(
    top_left: Coordinate,
    shape: Sequence[Coordinate],
) -> frozenset[Coordinate]:
    top, left = top_left
    return frozenset((top + row, left + column) for row, column in shape)


def _relation_satisfied(
    placed: frozenset[Coordinate],
    target_cells: frozenset[Coordinate],
    relation: str,
) -> bool:
    overlap = bool(placed & target_cells)
    if relation == "overlap":
        return overlap
    adjacent = any(
        (row + delta_row, column + delta_column) in target_cells
        for row, column in placed
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    if relation == "adjacent":
        return adjacent and not overlap
    if relation == "touch":
        diagonal_or_cardinal = any(
            (row + delta_row, column + delta_column) in target_cells
            for row, column in placed
            for delta_row in (-1, 0, 1)
            for delta_column in (-1, 0, 1)
            if delta_row or delta_column
        )
        return (diagonal_or_cardinal or overlap)
    raise ValueError(f"unknown target relation: {relation}")


def _valid_placement(
    top_left: Coordinate,
    shape: Sequence[Coordinate],
    *,
    height: int,
    width: int,
    occupied: frozenset[Coordinate],
    target_cells: frozenset[Coordinate],
    allow_target_overlap: bool,
) -> bool:
    cells = _placed_cells(top_left, shape)
    if any(
        row < 0 or row >= height or column < 0 or column >= width
        for row, column in cells
    ):
        return False
    blocked = occupied - (target_cells if allow_target_overlap else frozenset())
    return not bool(cells & blocked)


def shortest_navigation_plan(
    facts: GridFacts,
    *,
    agent_component: Component,
    agent_hypothesis: AgentHypothesis,
    target: NavigationTarget,
    bindings: Sequence[MovementBinding],
    passable_colors: Iterable[int] | None = None,
    max_expansions: int = 20_000,
) -> NavigationPlan | None:
    vector_to_action = {
        (binding.delta_row, binding.delta_column): binding.action
        for binding in bindings
    }
    if not vector_to_action:
        return None
    majority_color = max(facts.color_counts, key=lambda pair: pair[1])[0]
    passable = set(passable_colors or (majority_color,))
    agent_cells = frozenset(agent_component.cells)
    target_cells = frozenset(target.cells)
    occupied = frozenset(
        cell
        for component in facts.components
        if component.color not in passable
        for cell in component.cells
        if cell not in agent_cells
    )
    start = (agent_component.top, agent_component.left)
    queue: deque[Coordinate] = deque([start])
    parent: dict[Coordinate, tuple[Coordinate, str] | None] = {start: None}
    expansions = 0

    while queue and expansions < max_expansions:
        current = queue.popleft()
        expansions += 1
        placed = _placed_cells(current, agent_hypothesis.shape)
        if _relation_satisfied(placed, target_cells, target.relation):
            positions: list[Coordinate] = [current]
            actions: list[str] = []
            cursor = current
            while parent[cursor] is not None:
                previous, action = parent[cursor]
                actions.append(action)
                positions.append(previous)
                cursor = previous
            actions.reverse()
            positions.reverse()
            return NavigationPlan(
                agent=agent_hypothesis,
                target=target,
                start_top_left=start,
                destination_top_left=current,
                action_sequence=tuple(actions),
                position_sequence=tuple(positions),
                assumed_passable_colors=tuple(sorted(passable)),
            )

        for vector, action in sorted(vector_to_action.items()):
            next_position = (
                current[0] + vector[0],
                current[1] + vector[1],
            )
            if next_position in parent:
                continue
            if not _valid_placement(
                next_position,
                agent_hypothesis.shape,
                height=facts.height,
                width=facts.width,
                occupied=occupied,
                target_cells=target_cells,
                allow_target_overlap=(target.relation in {"overlap", "touch"}),
            ):
                continue
            parent[next_position] = (current, action)
            queue.append(next_position)
    return None


def generate_navigation_plans(
    facts: GridFacts,
    belief: ActionBeliefState,
    *,
    max_agents: int = 6,
    max_targets_per_agent: int = 16,
    max_plans: int = 64,
) -> tuple[NavigationPlan, ...]:
    bindings = infer_movement_bindings(belief)
    if len(bindings) < 2:
        return ()
    agent_hypotheses = agent_hypotheses_from_current_and_bindings(
        facts, belief, limit=max_agents
    )
    plans: list[NavigationPlan] = []
    for hypothesis in agent_hypotheses:
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
