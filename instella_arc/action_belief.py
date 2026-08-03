"""Online action-effect beliefs and deterministic experiment selection.

The belief state stores literal observed effect signatures. It does not assign a
semantic label such as "move north" unless an exact component translation
supports it. Complex click actions are explored per full observation state,
including animation-only evidence, rather than globally exhausting coordinates
and repeating one corner forever.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .world_state import GridFacts, TransitionFacts, grid_facts


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


def _metadata(observation: Any) -> Mapping[str, Any]:
    value = getattr(observation, "metadata", None)
    return value if isinstance(value, Mapping) else {}


def observation_before_state(observation: Any) -> str:
    metadata = _metadata(observation)
    return str(
        metadata.get("observation_before_sha256")
        or getattr(observation, "before_sha256")
    )


def observation_after_state(observation: Any) -> str:
    metadata = _metadata(observation)
    return str(
        metadata.get("observation_after_sha256")
        or getattr(observation, "after_sha256")
    )


def observation_is_informative(observation: Any) -> bool:
    metadata = _metadata(observation)
    animation_deltas = metadata.get("animation_deltas") or ()
    rendered_count = int(metadata.get("rendered_frame_count") or 0)
    effect = getattr(observation, "effect")
    return bool(
        not effect.unchanged
        or effect.level_progress > 0
        or animation_deltas
        or rendered_count > 1
        or observation_before_state(observation)
        != observation_after_state(observation)
    )


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
    def informative_rate(self) -> float | None:
        if not self.observations:
            return None
        return sum(observation_is_informative(observation) for observation in self.observations) / len(
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

    def observe_state(self, state_sha256: str) -> None:
        self.visited_states[str(state_sha256)] += 1

    def add_transition(self, observation: ActionObservation) -> None:
        self.profile(observation.action).add(observation)
        self.transitions.append(observation)
        self.observe_state(observation_after_state(observation))

    def state_visit_count(self, state_sha256: str) -> int:
        return int(self.visited_states.get(state_sha256, 0))

    def action_summary(self, legal_actions: Sequence[str]) -> str:
        lines: list[str] = []
        for action in legal_actions:
            profile = self.profile(action)
            lines.append(
                f"ACTION {action} TRIALS={profile.trials} "
                f"ENTROPY_BITS={profile.empirical_entropy_bits:.3f} "
                f"CONSISTENCY={profile.consistency:.3f} "
                f"NO_CHANGE_RATE={profile.no_change_rate} "
                f"INFORMATIVE_RATE={profile.informative_rate} "
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
    priority_index: int = 0


def candidate_coordinates(facts: GridFacts, *, limit: int = 96) -> tuple[tuple[int, int], ...]:
    """Return deterministic high-information coordinates for complex actions."""

    candidates: list[tuple[int, int]] = []

    def add(row: int, column: int) -> None:
        row = min(max(int(row), 0), facts.height - 1)
        column = min(max(int(column), 0), facts.width - 1)
        coordinate = (row, column)
        if coordinate not in candidates:
            candidates.append(coordinate)

    # Center and rare/small components are higher priority than corners.
    add(facts.height // 2, facts.width // 2)
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
        add(component.top, component.right)
        add(component.bottom, component.left)
        add(component.bottom, component.right)
        if len(candidates) >= limit // 2:
            break

    # A coarse lattice keeps large click boards explorable even when connected
    # components merge into broad regions. Use cell centers, not boundaries.
    divisions = 8 if max(facts.height, facts.width) >= 64 else 4
    for row_index in range(divisions):
        row = round((row_index + 0.5) * facts.height / divisions - 0.5)
        for column_index in range(divisions):
            column = round((column_index + 0.5) * facts.width / divisions - 0.5)
            add(row, column)
            if len(candidates) >= limit:
                return tuple(candidates[:limit])

    add(0, 0)
    add(0, facts.width - 1)
    add(facts.height - 1, 0)
    add(facts.height - 1, facts.width - 1)
    return tuple(candidates[:limit])


def sequence_candidate_coordinates(
    rendered_frames: Sequence[Sequence[Sequence[int]]],
    *,
    limit: int = 160,
) -> tuple[tuple[int, int], ...]:
    """Union candidates from every animation frame, newest evidence first."""

    candidates: list[tuple[int, int]] = []
    for frame in reversed(tuple(rendered_frames)):
        for coordinate in candidate_coordinates(grid_facts(frame), limit=96):
            if coordinate not in candidates:
                candidates.append(coordinate)
            if len(candidates) >= limit:
                return tuple(candidates)
    return tuple(candidates)


def probe_score(
    *,
    profile: ActionProfile,
    coordinate_trials: int,
    current_visit_count: int,
    is_reset: bool,
    recent_same_action: bool,
    coordinate_informative_rate: float = 0.0,
    coordinate_novel_state_rate: float = 0.0,
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
        if (
            profile.no_change_rate is not None
            and profile.no_change_rate >= 0.75
            and (profile.informative_rate or 0.0) < 0.25
        ):
            score -= 1.5
            reasons.append("usually-uninformative")
        if profile.progress_rate:
            score += 5.0 * profile.progress_rate
            reasons.append("observed-progress")
    if coordinate_trials == 0:
        score += 3.0
        reasons.append("untried-coordinate-in-state")
    else:
        score -= min(3.0, 0.75 * coordinate_trials)
        reasons.append("coordinate-repeat-penalty")
        if coordinate_informative_rate > 0.0:
            score += 4.0 * coordinate_informative_rate
            reasons.append("coordinate-produced-evidence")
        if coordinate_novel_state_rate > 0.0:
            score += 3.0 * coordinate_novel_state_rate
            reasons.append("coordinate-produced-new-state")
    score -= 0.35 * min(current_visit_count, 10)
    if current_visit_count > 1:
        reasons.append("state-repetition-penalty")
    if recent_same_action:
        score -= 0.5
        reasons.append("repeat-action-penalty")
    if is_reset:
        score -= 20.0
        reasons.append("reset-penalty")
    return score, tuple(reasons)


def _coordinate_history(
    profile: ActionProfile,
    *,
    coordinate: tuple[int, int] | None,
    current_state_id: str,
) -> tuple[int, float, float]:
    matched = [
        observation
        for observation in profile.observations
        if observation.coordinate == coordinate
        and observation_before_state(observation) == current_state_id
    ]
    if not matched:
        return 0, 0.0, 0.0
    informative_rate = sum(observation_is_informative(item) for item in matched) / len(matched)
    novel_rate = sum(
        observation_after_state(item) != current_state_id for item in matched
    ) / len(matched)
    return len(matched), informative_rate, novel_rate


def choose_probe(
    belief: ActionBeliefState,
    *,
    legal_actions: Sequence[str],
    current_facts: GridFacts,
    complex_actions: Iterable[str] = (),
    current_state_id: str | None = None,
    coordinate_candidates: Sequence[tuple[int, int]] | None = None,
) -> ProbeChoice:
    if not legal_actions:
        raise ValueError("at least one legal action is required")
    complex_set = {str(action) for action in complex_actions}
    state_id = str(current_state_id or current_facts.sha256)
    coordinates = tuple(coordinate_candidates or candidate_coordinates(current_facts))
    recent_action = belief.transitions[-1].action if belief.transitions else None
    choices: list[ProbeChoice] = []
    for action_index, action in enumerate(legal_actions):
        action = str(action)
        profile = belief.profile(action)
        action_coordinates: tuple[tuple[int, int] | None, ...]
        if action in complex_set:
            action_coordinates = coordinates or (
                (current_facts.height // 2, current_facts.width // 2),
            )
        else:
            action_coordinates = (None,)
        for coordinate_index, coordinate in enumerate(action_coordinates):
            if action in complex_set:
                trials, informative_rate, novel_rate = _coordinate_history(
                    profile,
                    coordinate=coordinate,
                    current_state_id=state_id,
                )
            else:
                trials = profile.trials
                informative_rate = profile.informative_rate or 0.0
                novel_rate = 0.0
            score, reasons = probe_score(
                profile=profile,
                coordinate_trials=trials,
                current_visit_count=belief.state_visit_count(state_id),
                is_reset=(action.upper() in {"RESET", "A0", "ACTION0"}),
                recent_same_action=(recent_action == action),
                coordinate_informative_rate=informative_rate,
                coordinate_novel_state_rate=novel_rate,
            )
            choices.append(
                ProbeChoice(
                    action=action,
                    coordinate=coordinate,
                    score=score,
                    reasons=reasons,
                    priority_index=action_index * max(len(coordinates), 1) + coordinate_index,
                )
            )
    return max(
        choices,
        key=lambda choice: (
            choice.score,
            -choice.priority_index,
        ),
    )
