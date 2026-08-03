"""Online action-effect beliefs and deterministic experiment selection.

The belief state stores literal observed effect signatures. It does not assign a
semantic label such as "move north" unless an exact component translation
supports it. Unseen or contradictory actions remain uncertain and are preferred
for safe information-gathering probes before the agent commits to a plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .world_state import GridFacts, TransitionFacts


@dataclass(frozen=True)
class EffectSignature:
    unchanged: bool
    changed_cell_count: int
    translation_vectors: tuple[tuple[int, int], ...]
    colors_added: tuple[tuple[int, int], ...]
    colors_removed: tuple[tuple[int, int], ...]
    level_progress: int
    terminal_state: str

    @classmethod
    def from_transition(
        cls,
        transition: TransitionFacts,
        *,
        level_progress: int = 0,
        terminal_state: str = "UNKNOWN",
    ) -> "EffectSignature":
        return cls(
            unchanged=transition.unchanged,
            changed_cell_count=transition.changed_cell_count,
            translation_vectors=transition.translation_vectors,
            colors_added=transition.colors_added,
            colors_removed=transition.colors_removed,
            level_progress=int(level_progress),
            terminal_state=str(terminal_state),
        )

    @property
    def canonical_key(self) -> str:
        return json.dumps(
            {
                "unchanged": self.unchanged,
                "changed_cell_count": self.changed_cell_count,
                "translation_vectors": self.translation_vectors,
                "colors_added": self.colors_added,
                "colors_removed": self.colors_removed,
                "level_progress": self.level_progress,
                "terminal_state": self.terminal_state,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionObservation:
    action: str
    coordinate: tuple[int, int] | None
    before_sha256: str
    after_sha256: str
    effect: EffectSignature


@dataclass
class ActionProfile:
    action: str
    observations: list[ActionObservation] = field(default_factory=list)

    def add(self, observation: ActionObservation) -> None:
        if observation.action != self.action:
            raise ValueError("observation action does not match profile")
        self.observations.append(observation)

    @property
    def trials(self) -> int:
        return len(self.observations)

    @property
    def effect_counts(self) -> Counter[str]:
        return Counter(observation.effect.canonical_key for observation in self.observations)

    @property
    def no_change_rate(self) -> float | None:
        if not self.observations:
            return None
        return sum(observation.effect.unchanged for observation in self.observations) / len(
            self.observations
        )

    @property
    def progress_rate(self) -> float | None:
        if not self.observations:
            return None
        return sum(observation.effect.level_progress > 0 for observation in self.observations) / len(
            self.observations
        )

    @property
    def empirical_entropy_bits(self) -> float:
        counts = self.effect_counts
        total = sum(counts.values())
        if total == 0:
            return math.log2(16.0)
        return -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )

    @property
    def translation_vectors(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                {
                    vector
                    for observation in self.observations
                    for vector in observation.effect.translation_vectors
                }
            )
        )

    @property
    def consistency(self) -> float:
        counts = self.effect_counts
        total = sum(counts.values())
        return max(counts.values()) / total if total else 0.0


@dataclass
class ActionBeliefState:
    profiles: dict[str, ActionProfile] = field(default_factory=dict)
    visited_states: Counter[str] = field(default_factory=Counter)
    transitions: list[ActionObservation] = field(default_factory=list)

    def profile(self, action: str) -> ActionProfile:
        key = str(action)
        if key not in self.profiles:
            self.profiles[key] = ActionProfile(action=key)
        return self.profiles[key]

    def observe_state(self, grid_sha256: str) -> None:
        self.visited_states[str(grid_sha256)] += 1

    def add_transition(self, observation: ActionObservation) -> None:
        self.profile(observation.action).add(observation)
        self.transitions.append(observation)
        self.observe_state(observation.after_sha256)

    def state_visit_count(self, grid_sha256: str) -> int:
        return int(self.visited_states.get(grid_sha256, 0))

    def action_summary(self, legal_actions: Sequence[str]) -> str:
        lines: list[str] = []
        for action in legal_actions:
            profile = self.profile(action)
            lines.append(
                f"ACTION {action} TRIALS={profile.trials} "
                f"ENTROPY_BITS={profile.empirical_entropy_bits:.3f} "
                f"CONSISTENCY={profile.consistency:.3f} "
                f"NO_CHANGE_RATE={profile.no_change_rate} "
                f"PROGRESS_RATE={profile.progress_rate} "
                f"TRANSLATIONS={profile.translation_vectors}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbeChoice:
    action: str
    coordinate: tuple[int, int] | None
    score: float
    reasons: tuple[str, ...]


def candidate_coordinates(facts: GridFacts, *, limit: int = 32) -> tuple[tuple[int, int], ...]:
    """Return deterministic high-information coordinates for complex actions."""

    candidates: list[tuple[int, int]] = []

    def add(row: int, column: int) -> None:
        row = min(max(int(row), 0), facts.height - 1)
        column = min(max(int(column), 0), facts.width - 1)
        coordinate = (row, column)
        if coordinate not in candidates:
            candidates.append(coordinate)

    add(facts.height // 2, facts.width // 2)
    add(0, 0)
    add(0, facts.width - 1)
    add(facts.height - 1, 0)
    add(facts.height - 1, facts.width - 1)

    # Small and rare components often represent controllable agents, goals,
    # switches, or obstacles. This is only a coordinate prior, not a semantic
    # assertion; every effect is verified after the click.
    color_count = dict(facts.color_counts)
    ordered_components = sorted(
        facts.components,
        key=lambda component: (
            color_count[component.color],
            component.area,
            component.color,
            component.top,
            component.left,
        ),
    )
    for component in ordered_components:
        centroid_row, centroid_column = component.centroid
        add(round(centroid_row), round(centroid_column))
        add(component.top, component.left)
        add(component.bottom, component.right)
        if len(candidates) >= limit:
            break
    return tuple(candidates[:limit])


def probe_score(
    *,
    profile: ActionProfile,
    coordinate_trials: int,
    current_visit_count: int,
    is_reset: bool,
    recent_same_action: bool,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = 0.0
    if profile.trials == 0:
        score += 8.0
        reasons.append("untried-action")
    else:
        score += min(profile.empirical_entropy_bits, 4.0)
        reasons.append("effect-uncertainty")
        if profile.consistency < 0.75:
            score += 2.0
            reasons.append("context-dependent-effect")
        if profile.no_change_rate is not None and profile.no_change_rate >= 0.75:
            score -= 1.5
            reasons.append("usually-no-change")
        if profile.progress_rate:
            score += 5.0 * profile.progress_rate
            reasons.append("observed-progress")
    if coordinate_trials == 0:
        score += 2.0
        reasons.append("untried-coordinate")
    score -= 0.5 * min(current_visit_count, 8)
    if current_visit_count:
        reasons.append("state-repetition-penalty")
    if recent_same_action:
        score -= 1.0
        reasons.append("repeat-action-penalty")
    if is_reset:
        score -= 20.0
        reasons.append("reset-penalty")
    return score, tuple(reasons)


def choose_probe(
    belief: ActionBeliefState,
    *,
    legal_actions: Sequence[str],
    current_facts: GridFacts,
    complex_actions: Iterable[str] = (),
) -> ProbeChoice:
    if not legal_actions:
        raise ValueError("at least one legal action is required")
    complex_set = {str(action) for action in complex_actions}
    coordinates = candidate_coordinates(current_facts)
    recent_action = belief.transitions[-1].action if belief.transitions else None
    choices: list[ProbeChoice] = []
    for action in legal_actions:
        action = str(action)
        profile = belief.profile(action)
        action_coordinates: tuple[tuple[int, int] | None, ...]
        if action in complex_set:
            action_coordinates = coordinates or ((current_facts.height // 2, current_facts.width // 2),)
        else:
            action_coordinates = (None,)
        for coordinate in action_coordinates:
            coordinate_trials = sum(
                observation.coordinate == coordinate
                for observation in profile.observations
            )
            score, reasons = probe_score(
                profile=profile,
                coordinate_trials=coordinate_trials,
                current_visit_count=belief.state_visit_count(current_facts.sha256),
                is_reset=(action.upper() in {"RESET", "A0", "ACTION0"}),
                recent_same_action=(recent_action == action),
            )
            choices.append(
                ProbeChoice(
                    action=action,
                    coordinate=coordinate,
                    score=score,
                    reasons=reasons,
                )
            )
    return max(
        choices,
        key=lambda choice: (
            choice.score,
            choice.action,
            choice.coordinate if choice.coordinate is not None else (-1, -1),
        ),
    )
