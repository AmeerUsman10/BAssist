"""Deterministic transition-frontier baseline for official ARC-AGI-3 games.

The policy in this module is deliberately game-general.  It sees only exact
official observations, opaque legal action names, and transition novelty.  It
does not inspect game identifiers, attach semantics to ``ACTION1``-style names,
or import a learned model.  An optional external direction-score provider may
rank otherwise tied simple actions; this keeps the module inside the project's
one-model boundary while allowing the pinned GPT-2 path to be connected later.

The official SDK is intentionally duck-typed.  Importing this module requires
neither ``arc_agi`` nor ``torch``.  ``run_official_game`` expects the documented
``observation_space``, ``action_space``, ``reset()``, and
``step(action, data=...)`` surface and therefore also works with strict fakes.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeAlias

from .codec import Grid, encode_delta, encode_grid
from .official_observation import OfficialFrameSequence


PROTOCOL = "official_transition_frontier_v1"
RECEIPT_SCHEMA_VERSION = 1
RESET_ACTION = "RESET"

_GAME_OVER_STATES = {"GAME_OVER"}
_NOT_STARTED_STATES = {"NOT_PLAYED", "NOT_STARTED"}
_WIN_STATES = {"WIN", "WON", "GAME_WIN"}

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


class OfficialScoreAgentError(ValueError):
    """Raised when the SDK surface or policy state violates the protocol."""


class DirectionScoreProvider(Protocol):
    """Minimal optional single-model adapter.

    Implementations may import and own GPT-2 outside this module.  Provider
    errors are converted into explicit receipt entries and the exact-state
    policy continues without learned scores.
    """

    def reset(self) -> None: ...

    def observe(
        self,
        action_name: str,
        before_grid: Grid | None,
        after_grid: Grid | None,
        status: str,
    ) -> JSONValue: ...

    def direction_scores(self, action_name: str) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class AgentConfig:
    """Frozen action and click-frontier bounds for one game run."""

    max_actions: int = 160
    max_resets: int = 3
    max_click_candidates_per_action: int = 64
    low_discrepancy_points: int = 48
    provider_confidence_margin: float = 0.25

    def validate(self) -> None:
        if self.max_actions < 0:
            raise OfficialScoreAgentError("max_actions must be non-negative")
        if self.max_resets < 0:
            raise OfficialScoreAgentError("max_resets must be non-negative")
        if self.max_click_candidates_per_action < 1:
            raise OfficialScoreAgentError(
                "max_click_candidates_per_action must be positive"
            )
        if not 0 <= self.low_discrepancy_points <= self.max_click_candidates_per_action:
            raise OfficialScoreAgentError(
                "low_discrepancy_points must lie within the click-candidate bound"
            )
        if (
            not math.isfinite(self.provider_confidence_margin)
            or self.provider_confidence_margin < 0.0
        ):
            raise OfficialScoreAgentError(
                "provider_confidence_margin must be finite and non-negative"
            )


@dataclass(frozen=True)
class ActionChoice:
    """One SDK action plus its deterministic, JSON-safe complex-action data."""

    action: Any = field(repr=False, compare=False)
    action_name: str
    simple: bool
    x: int | None = None
    y: int | None = None
    candidate_source: str | None = None

    @property
    def data(self) -> dict[str, int]:
        if self.simple:
            return {}
        if self.x is None or self.y is None:
            raise OfficialScoreAgentError("complex action is missing x/y coordinates")
        return {"x": self.x, "y": self.y}

    @property
    def signature(self) -> str:
        if self.simple:
            return self.action_name
        return f"{self.action_name}@x={self.x},y={self.y}"


@dataclass
class _Node:
    state_sha256: str
    levels_completed: int | None
    environment_state: str
    visits: int = 0
    available_signatures: set[str] = field(default_factory=set)
    tried_signatures: set[str] = field(default_factory=set)


@dataclass
class _Edge:
    before: str
    signature: str
    action_name: str
    data: dict[str, int]
    after: str
    count: int = 0
    changed_count: int = 0
    novel_state_hits: int = 0


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _state_token(value: str) -> str:
    return str(value).rsplit(".", 1)[-1].upper()


def _is_simple_action(action: Any) -> bool:
    marker = getattr(action, "is_simple", None)
    if callable(marker):
        return bool(marker())
    if marker is not None:
        return bool(marker)
    requires_data = getattr(action, "requires_data", None)
    if requires_data is not None:
        return not bool(requires_data)
    # The SDK exposes ``is_simple``.  Defaulting unknown action objects to simple
    # is safer than inventing coordinates for an unrelated third-party wrapper.
    return True


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_state_sha256(observation: OfficialFrameSequence) -> str:
    """Hash every policy-visible exact field while excluding ``game_id``."""

    return _canonical_sha256(
        {
            "state": observation.state,
            "levels_completed": observation.levels_completed,
            "win_levels": observation.win_levels,
            "full_reset": observation.full_reset,
            "available_actions": list(observation.available_actions),
            "final_grid": (
                [list(row) for row in observation.final_grid]
                if observation.final_grid is not None
                else None
            ),
        }
    )


def _grid_sha256(grid: Grid | None) -> str | None:
    if grid is None:
        return None
    return hashlib.sha256(encode_grid(grid).encode("utf-8")).hexdigest()


def _observation_receipt(observation: OfficialFrameSequence) -> dict[str, Any]:
    return {
        "frame_sequence_sha256": observation.sha256,
        "policy_state_sha256": _policy_state_sha256(observation),
        "final_grid_sha256": _grid_sha256(observation.final_grid),
        "environment_state": observation.state,
        "levels_completed": observation.levels_completed,
        "win_levels": observation.win_levels,
        "full_reset": observation.full_reset,
        "available_actions": list(observation.available_actions),
        "rendered_frame_count": len(observation.rendered_frames),
        "shape": list(observation.shape) if observation.shape is not None else None,
    }


def _modal_color(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    # A numeric tie break is independent of encounter order.
    return min(counts, key=lambda color: (-counts[color], color))


def _component_representatives(grid: Grid) -> tuple[tuple[int, int], ...]:
    """Return one real cell nearest the centroid of each non-modal component."""

    height = len(grid)
    width = len(grid[0])
    modal = _modal_color(grid)
    remaining = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != modal
    }
    components: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        color = grid[start[0]][start[1]]
        queue = [start]
        remaining.remove(start)
        cells: list[tuple[int, int]] = []
        while queue:
            row, column = queue.pop()
            cells.append((row, column))
            for neighbor in (
                (row - 1, column),
                (row, column - 1),
                (row, column + 1),
                (row + 1, column),
            ):
                if neighbor in remaining and grid[neighbor[0]][neighbor[1]] == color:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        mean_row = sum(row for row, _ in cells) / len(cells)
        mean_column = sum(column for _, column in cells) / len(cells)
        representative = min(
            cells,
            key=lambda cell: (
                (cell[0] - mean_row) ** 2 + (cell[1] - mean_column) ** 2,
                cell[0],
                cell[1],
            ),
        )
        first = min(cells)
        components.append(
            ((first[0], first[1], color, -len(cells)), representative)
        )
    return tuple(representative for _, representative in sorted(components))


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def _low_discrepancy_points(
    height: int,
    width: int,
    count: int,
) -> tuple[tuple[int, int], ...]:
    """Map a deterministic base-2/base-3 Halton prefix onto grid cells."""

    if min(height, width) <= 0 or count <= 0:
        return ()
    target = min(count, height * width)
    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # Quantization can repeat Halton cells.  The bound is ample for 64x64 ARC
    # frames; row-major completion is a deterministic total-coverage fallback.
    limit = max(target * 32, height * width * 4)
    for index in range(1, limit + 1):
        column = min(width - 1, int(_radical_inverse(index, 2) * width))
        row = min(height - 1, int(_radical_inverse(index, 3) * height))
        point = (row, column)
        if point not in seen:
            seen.add(point)
            points.append(point)
            if len(points) == target:
                return tuple(points)
    for row in range(height):
        for column in range(width):
            point = (row, column)
            if point not in seen:
                points.append(point)
                if len(points) == target:
                    return tuple(points)
    return tuple(points)


def click_candidates(
    grid: Grid,
    *,
    maximum: int,
    low_discrepancy_points: int,
) -> tuple[tuple[int, int, str], ...]:
    """Return ``(row, column, source)`` candidates, components before coverage."""

    if maximum < 1:
        raise OfficialScoreAgentError("maximum click candidates must be positive")
    candidates: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for row, column in _component_representatives(grid):
        if (row, column) not in seen:
            candidates.append((row, column, "non_modal_component"))
            seen.add((row, column))
            if len(candidates) == maximum:
                return tuple(candidates)
    for row, column in _low_discrepancy_points(
        len(grid), len(grid[0]), low_discrepancy_points
    ):
        if (row, column) not in seen:
            candidates.append((row, column, "low_discrepancy"))
            seen.add((row, column))
            if len(candidates) == maximum:
                break
    return tuple(candidates)


class TransitionFrontierAgent:
    """Exact-state frontier policy with an incrementally learned reversible graph."""

    def __init__(
        self,
        config: AgentConfig,
        direction_score_provider: DirectionScoreProvider | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.direction_score_provider = direction_score_provider
        self._current: OfficialFrameSequence | None = None
        self._initial: OfficialFrameSequence | None = None
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[tuple[str, str, str], _Edge] = {}
        self._signature_uses: Counter[str] = Counter()
        self._signature_novel_transitions: Counter[str] = Counter()
        self._events: list[dict[str, Any]] = []
        self._resets = 0
        self._steps = 0
        self._level_changes = 0
        self._stop_reason: str | None = None
        self._provider_failures: list[dict[str, Any]] = []
        self._provider_observations: list[dict[str, Any]] = []
        self._provider_scores_receipts: list[dict[str, Any]] = []
        self._provider_score_calls = 0
        self._provider_reset_calls = 0

    @property
    def current(self) -> OfficialFrameSequence:
        if self._current is None:
            raise OfficialScoreAgentError("agent has not observed an initial frame")
        return self._current

    @property
    def actions_taken(self) -> int:
        return len(self._events)

    @property
    def resets_used(self) -> int:
        return self._resets

    def begin(self, frame_data: Any, *, bootstrap_reset: bool = False) -> None:
        if self._current is not None:
            raise OfficialScoreAgentError("agent may begin only once")
        observation = OfficialFrameSequence.from_frame_data(frame_data)
        self._initial = observation
        self._current = observation
        self._register_node(observation)
        self._provider_reset()
        if bootstrap_reset:
            self._resets += 1
            provider_observation = self._provider_observe(
                RESET_ACTION, None, observation
            )
            self._events.append(
                {
                    "index": 1,
                    "kind": "reset",
                    "action": RESET_ACTION,
                    "data": {},
                    "signature": RESET_ACTION,
                    "candidate_source": None,
                    "before": None,
                    "after": _observation_receipt(observation),
                    "persistent_delta": None,
                    "state_changed": None,
                    "shape_changed": None,
                    "novel_state": True,
                    "novel_transition": True,
                    "level_change": False,
                    "terminal": self._terminal_kind(observation),
                    "provider_observation": provider_observation,
                }
            )

    def _provider_failure(
        self,
        phase: str,
        exc: BaseException,
        *,
        action_name: str | None = None,
    ) -> None:
        self._provider_failures.append(
            {
                "index": len(self._provider_failures) + 1,
                "phase": phase,
                "action": action_name,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "fallback": "exact_state_frontier",
            }
        )

    def _provider_reset(self) -> None:
        if self.direction_score_provider is None:
            return
        self._provider_reset_calls += 1
        try:
            self.direction_score_provider.reset()
        except Exception as exc:
            self._provider_failure("reset", exc)

    def _provider_observe(
        self,
        action_name: str,
        before: OfficialFrameSequence | None,
        after: OfficialFrameSequence,
    ) -> dict[str, Any] | None:
        if self.direction_score_provider is None:
            return None
        try:
            value = self.direction_score_provider.observe(
                action_name,
                before.final_grid if before is not None else None,
                after.final_grid,
                after.state,
            )
            # Providers may return a compact provenance receipt, but never an
            # opaque model object or tensor on the durable evidence surface.
            json.dumps(value, sort_keys=True, allow_nan=False)
            receipt = {
                "index": len(self._provider_observations) + 1,
                "action": action_name,
                "receipt": value,
            }
            self._provider_observations.append(receipt)
            return receipt
        except Exception as exc:
            self._provider_failure("observe", exc, action_name=action_name)
            return None

    def _register_node(self, observation: OfficialFrameSequence) -> tuple[str, bool]:
        key = _policy_state_sha256(observation)
        novel = key not in self._nodes
        if novel:
            self._nodes[key] = _Node(
                state_sha256=key,
                levels_completed=observation.levels_completed,
                environment_state=observation.state,
            )
        self._nodes[key].visits += 1
        return key, novel

    @staticmethod
    def _terminal_kind(observation: OfficialFrameSequence) -> str | None:
        token = _state_token(observation.state)
        if token in _WIN_STATES:
            return "win"
        if token in _GAME_OVER_STATES:
            return "game_over"
        return None

    def _actual_actions(self, action_space: Iterable[Any]) -> tuple[Any, ...]:
        available = set(self.current.available_actions)
        by_name: dict[str, Any] = {}
        for action in action_space:
            name = _enum_name(action)
            if available and name not in available:
                continue
            if name in by_name:
                raise OfficialScoreAgentError(f"duplicate legal action name: {name}")
            by_name[name] = action
        return tuple(by_name[name] for name in sorted(by_name))

    def _choices(self, action_space: Iterable[Any]) -> tuple[ActionChoice, ...]:
        choices: list[ActionChoice] = []
        for action in self._actual_actions(action_space):
            name = _enum_name(action)
            if name.upper() == RESET_ACTION:
                continue
            if _is_simple_action(action):
                choices.append(ActionChoice(action, name, True))
                continue
            grid = self.current.final_grid
            if grid is None:
                continue
            for row, column, source in click_candidates(
                grid,
                maximum=self.config.max_click_candidates_per_action,
                low_discrepancy_points=self.config.low_discrepancy_points,
            ):
                choices.append(
                    ActionChoice(
                        action,
                        name,
                        False,
                        x=column,
                        y=row,
                        candidate_source=source,
                    )
                )
        return tuple(choices)

    def _provider_scores(
        self, choices: Sequence[ActionChoice]
    ) -> dict[str, tuple[float, str | None]]:
        """Return confidence/direction hints without assigning goal semantics."""

        simple_names = tuple(
            sorted({choice.action_name for choice in choices if choice.simple})
        )
        if self.direction_score_provider is None or not simple_names:
            return {}
        scores: dict[str, tuple[float, str | None]] = {}
        for name in simple_names:
            self._provider_score_calls += 1
            try:
                raw = self.direction_score_provider.direction_scores(name)
                if not isinstance(raw, Mapping):
                    raise TypeError("direction_scores must return a mapping")
                values = sorted(
                    (
                        (str(direction).upper(), float(value))
                        for direction, value in raw.items()
                    ),
                    key=lambda item: item[0],
                )
                if not values:
                    scores[name] = (0.0, None)
                    self._provider_scores_receipts.append(
                        {
                            "call": self._provider_score_calls,
                            "action": name,
                            "scores": {},
                            "top_direction": None,
                            "margin": 0.0,
                            "high_confidence": False,
                        }
                    )
                    continue
                if not all(math.isfinite(value) for _, value in values):
                    raise ValueError("direction scores must be finite")
                ranked = sorted(values, key=lambda item: (-item[1], item[0]))
                second = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
                raw_margin = max(0.0, ranked[0][1] - second)
                confidence = (
                    raw_margin
                    if raw_margin >= self.config.provider_confidence_margin
                    else 0.0
                )
                scores[name] = (confidence, ranked[0][0])
                self._provider_scores_receipts.append(
                    {
                        "call": self._provider_score_calls,
                        "action": name,
                        "scores": {direction: value for direction, value in values},
                        "top_direction": ranked[0][0],
                        "margin": raw_margin,
                        "high_confidence": confidence > 0.0,
                    }
                )
            except Exception as exc:
                self._provider_failure("direction_scores", exc, action_name=name)
                scores[name] = (0.0, None)
        return scores

    def _novelty_rate(self, signature: str) -> float:
        uses = self._signature_uses[signature]
        return self._signature_novel_transitions[signature] / uses if uses else 0.0

    def _choice_rank(
        self,
        choice: ActionChoice,
        scores: Mapping[str, tuple[float, str | None]],
    ) -> tuple[Any, ...]:
        # Simple opaque interventions precede the larger coordinate frontier.
        # Without a provider, least global use and learned novelty are the only
        # non-lexical tie breaks.  A provider can break otherwise equal actions.
        return (
            0 if choice.simple else 1,
            0 if choice.candidate_source == "non_modal_component" else 1,
            self._signature_uses[choice.signature],
            -self._novelty_rate(choice.signature),
            -scores.get(choice.action_name, (0.0, None))[0],
            choice.signature,
        )

    def _edge_is_reversible(self, edge: _Edge) -> bool:
        if edge.before == edge.after:
            return False
        return any(
            candidate.before == edge.after and candidate.after == edge.before
            for candidate in self._edges.values()
        )

    def _route_to_frontier(
        self,
        current_key: str,
        choices_by_signature: Mapping[str, ActionChoice],
    ) -> ActionChoice | None:
        """Return the first action on a deterministic reversible BFS route."""

        queue: deque[tuple[str, tuple[str, ...]]] = deque([(current_key, ())])
        visited = {current_key}
        while queue:
            node_key, path = queue.popleft()
            if path:
                node = self._nodes[node_key]
                if node.available_signatures - node.tried_signatures:
                    return choices_by_signature.get(path[0])
            outgoing = sorted(
                (
                    edge
                    for edge in self._edges.values()
                    if edge.before == node_key and self._edge_is_reversible(edge)
                ),
                key=lambda edge: (edge.signature, edge.after),
            )
            for edge in outgoing:
                if edge.after in visited:
                    continue
                if not path and edge.signature not in choices_by_signature:
                    continue
                visited.add(edge.after)
                queue.append((edge.after, (*path, edge.signature)))
        return None

    def choose(self, action_space: Iterable[Any]) -> ActionChoice | None:
        if self.actions_taken >= self.config.max_actions:
            return None
        choices = self._choices(action_space)
        if not choices:
            return None
        current_key = _policy_state_sha256(self.current)
        node = self._nodes[current_key]
        node.available_signatures = {choice.signature for choice in choices}
        scores = self._provider_scores(choices)
        untried = [
            choice for choice in choices if choice.signature not in node.tried_signatures
        ]
        if untried:
            return min(untried, key=lambda choice: self._choice_rank(choice, scores))

        choices_by_signature = {choice.signature: choice for choice in choices}
        routed = self._route_to_frontier(current_key, choices_by_signature)
        if routed is not None:
            return routed

        # A fully explored component can still be stochastic or have hidden
        # state. Prefer actions that previously produced new exact transitions,
        # then use the same deterministic least-use ordering.
        return min(
            choices,
            key=lambda choice: (
                -self._novelty_rate(choice.signature),
                *self._choice_rank(choice, scores),
            ),
        )

    def _record(
        self,
        *,
        action_name: str,
        signature: str,
        data: dict[str, int],
        candidate_source: str | None,
        frame_data: Any,
        kind: str,
    ) -> None:
        if self.actions_taken >= self.config.max_actions:
            raise OfficialScoreAgentError("record would exceed the action budget")
        before = self.current
        before_key = _policy_state_sha256(before)
        after = OfficialFrameSequence.from_frame_data(frame_data)
        after_key, novel_state = self._register_node(after)
        node = self._nodes[before_key]
        node.tried_signatures.add(signature)
        transition_key = (before_key, signature, after_key)
        novel_transition = transition_key not in self._edges
        if novel_transition:
            self._edges[transition_key] = _Edge(
                before=before_key,
                signature=signature,
                action_name=action_name,
                data=dict(data),
                after=after_key,
            )
        edge = self._edges[transition_key]
        changed = before.final_grid != after.final_grid
        shape_changed = before.shape != after.shape
        edge.count += 1
        edge.changed_count += int(changed)
        edge.novel_state_hits += int(novel_state)
        self._signature_uses[signature] += 1
        if novel_transition and before_key != after_key:
            self._signature_novel_transitions[signature] += 1
        level_change = (
            before.levels_completed is not None
            and after.levels_completed is not None
            and before.levels_completed != after.levels_completed
        )
        self._level_changes += int(level_change)
        provider_observation = self._provider_observe(action_name, before, after)
        self._events.append(
            {
                "index": self.actions_taken + 1,
                "kind": kind,
                "action": action_name,
                "data": dict(sorted(data.items())),
                "signature": signature,
                "candidate_source": candidate_source,
                "before": _observation_receipt(before),
                "after": _observation_receipt(after),
                "persistent_delta": (
                    encode_delta(before.final_grid, after.final_grid)
                    if (
                        before.final_grid is not None
                        and after.final_grid is not None
                        and not shape_changed
                    )
                    else None
                ),
                "state_changed": changed,
                "shape_changed": shape_changed,
                "novel_state": novel_state,
                "novel_transition": novel_transition,
                "level_change": level_change,
                "terminal": self._terminal_kind(after),
                "provider_observation": provider_observation,
            }
        )
        self._current = after

    def record_step(self, choice: ActionChoice, frame_data: Any) -> None:
        self._record(
            action_name=choice.action_name,
            signature=choice.signature,
            data=choice.data,
            candidate_source=choice.candidate_source,
            frame_data=frame_data,
            kind="step",
        )
        self._steps += 1

    def record_reset(self, frame_data: Any) -> None:
        if self._resets >= self.config.max_resets:
            raise OfficialScoreAgentError("reset would exceed max_resets")
        self._record(
            action_name=RESET_ACTION,
            signature=RESET_ACTION,
            data={},
            candidate_source=None,
            frame_data=frame_data,
            kind="reset",
        )
        self._resets += 1

    def can_reset(self) -> bool:
        return (
            # In arc-agi 0.9.9 a RESET before any ordinary action is a full
            # reset and starts another scorecard run.  The public protocol
            # permits one run per game, so only level resets after real play
            # are eligible.
            self._steps > 0
            and self._resets < self.config.max_resets
            and self.actions_taken < self.config.max_actions
        )

    def finish(self, stop_reason: str) -> dict[str, Any]:
        if self._initial is None or self._current is None:
            raise OfficialScoreAgentError("cannot finish before begin")
        self._stop_reason = stop_reason
        reversible_keys = {
            (edge.before, edge.signature, edge.after)
            for edge in self._edges.values()
            if self._edge_is_reversible(edge)
        }
        edges = [
            {
                "before": edge.before,
                "signature": edge.signature,
                "action": edge.action_name,
                "data": dict(sorted(edge.data.items())),
                "after": edge.after,
                "count": edge.count,
                "changed_count": edge.changed_count,
                "novel_state_hits": edge.novel_state_hits,
                "reversible": (edge.before, edge.signature, edge.after)
                in reversible_keys,
            }
            for edge in sorted(
                self._edges.values(),
                key=lambda edge: (edge.before, edge.signature, edge.after),
            )
        ]
        nodes = [
            {
                "state_sha256": node.state_sha256,
                "levels_completed": node.levels_completed,
                "environment_state": node.environment_state,
                "visits": node.visits,
                "available_signatures": sorted(node.available_signatures),
                "tried_signatures": sorted(node.tried_signatures),
            }
            for node in sorted(self._nodes.values(), key=lambda node: node.state_sha256)
        ]
        terminal = self._terminal_kind(self.current)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scope": (
                "Game-general exact-state transition-frontier baseline; not an "
                "ARC-AGI-3 capability claim or competition submission."
            ),
            "competition_submission_performed": False,
            "model_weights_persisted": False,
            "reported_game_id": self._initial.game_id,
            "policy_game_id_used": False,
            "direction_score_provider_used": self.direction_score_provider is not None,
            "learned_model_count": int(self.direction_score_provider is not None),
            "config": asdict(self.config),
            "status": "won" if terminal == "win" else "stopped",
            "stop_reason": stop_reason,
            "actions_taken": self.actions_taken,
            "step_actions": self._steps,
            "resets": self._resets,
            "level_changes": self._level_changes,
            "initial_observation": _observation_receipt(self._initial),
            "final_observation": _observation_receipt(self.current),
            "actions": list(self._events),
            "provider": {
                "enabled": self.direction_score_provider is not None,
                "reset_calls": self._provider_reset_calls,
                "score_calls": self._provider_score_calls,
                "score_receipts": list(self._provider_scores_receipts),
                "observations": list(self._provider_observations),
                "failures": list(self._provider_failures),
                "fallback_used": bool(self._provider_failures),
                "policy_use": (
                    "confidence-only tie break among opaque simple actions; no goal semantics"
                ),
            },
            "graph": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "reversible_edge_count": len(reversible_keys),
                "nodes": nodes,
                "edges": edges,
            },
            "checks": {
                "action_budget_respected": self.actions_taken
                <= self.config.max_actions,
                "reset_budget_respected": self._resets <= self.config.max_resets,
                "consecutive_action_indices": [
                    event["index"] for event in self._events
                ]
                == list(range(1, self.actions_taken + 1)),
                "exact_final_frame_normalization": True,
                "game_id_excluded_from_policy_state": True,
                "learned_model_count_at_most_one": int(
                    self.direction_score_provider is not None
                )
                <= 1,
            },
        }
        # Fail before returning if any accidental object or non-finite score has
        # escaped the strict evidence surface.
        json.dumps(receipt, sort_keys=True, allow_nan=False)
        return receipt


def run_official_game(
    environment: Any,
    *,
    config: AgentConfig | None = None,
    direction_score_provider: DirectionScoreProvider | None = None,
) -> dict[str, Any]:
    """Run one bounded game through the official SDK-compatible interface."""

    resolved = config or AgentConfig()
    resolved.validate()
    agent = TransitionFrontierAgent(resolved, direction_score_provider)
    initial = getattr(environment, "observation_space", None)
    if initial is None:
        if resolved.max_actions < 1 or resolved.max_resets < 1:
            raise OfficialScoreAgentError(
                "an uninitialized environment requires one budgeted reset"
            )
        initial = environment.reset()
        if initial is None:
            raise OfficialScoreAgentError("environment.reset() returned None")
        agent.begin(initial, bootstrap_reset=True)
    else:
        agent.begin(initial)

    while agent.actions_taken < resolved.max_actions:
        state = _state_token(agent.current.state)
        if state in _WIN_STATES:
            return agent.finish("win")
        if state in _GAME_OVER_STATES or state in _NOT_STARTED_STATES:
            if not agent.can_reset():
                reason = "game_over" if state in _GAME_OVER_STATES else "not_started"
                return agent.finish(reason)
            observation = environment.reset()
            if observation is None:
                raise OfficialScoreAgentError("environment.reset() returned None")
            agent.record_reset(observation)
            continue

        choice = agent.choose(getattr(environment, "action_space", ()))
        if choice is None:
            if agent.can_reset():
                observation = environment.reset()
                if observation is None:
                    raise OfficialScoreAgentError("environment.reset() returned None")
                agent.record_reset(observation)
                continue
            return agent.finish("frontier_exhausted")
        observation = environment.step(choice.action, data=choice.data)
        if observation is None:
            raise OfficialScoreAgentError("environment.step() returned None")
        agent.record_step(choice, observation)

    if _state_token(agent.current.state) in _WIN_STATES:
        return agent.finish("win")
    return agent.finish("action_budget_exhausted")


__all__ = [
    "ActionChoice",
    "AgentConfig",
    "DirectionScoreProvider",
    "JSONValue",
    "OfficialScoreAgentError",
    "PROTOCOL",
    "RECEIPT_SCHEMA_VERSION",
    "TransitionFrontierAgent",
    "click_candidates",
    "run_official_game",
]
