"""Closed-loop ARC controller combining exact evidence and one model ranker.

The controller has two memory scopes:

* game-global: action-effect beliefs and discovered movement semantics;
* level-local: current navigation plan, tried target hypotheses, and state visits.

Every action is followed by an exact receipt. Plans are invalidated immediately
when the observed transition contradicts their predicted movement or when the
same state repeats without progress. Complex-action probes use the complete
rendered observation sequence, so temporary animation evidence is not discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Protocol, Sequence

from arcgpt2.official_observation import OfficialFrameSequence

from .action_belief import (
    ActionBeliefState,
    ProbeChoice,
    choose_probe,
    sequence_candidate_coordinates,
)
from .navigation import NavigationPlan, generate_navigation_plans
from .receipts import RichActionObservation, rich_action_observation
from .world_state import GridFacts, grid_facts


@dataclass(frozen=True)
class PlannedAction:
    action: str
    coordinate: tuple[int, int] | None
    source: str
    purpose: str
    plan_id: str | None = None
    expected_delta: tuple[int, int] | None = None

    @property
    def reasoning(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "purpose": self.purpose,
            "plan_id": self.plan_id,
            "expected_delta": (
                list(self.expected_delta) if self.expected_delta is not None else None
            ),
        }


@dataclass(frozen=True)
class RankedPlan:
    index: int
    score: float
    reason: str


class PlanRanker(Protocol):
    def rank(
        self,
        *,
        observation: OfficialFrameSequence,
        facts: GridFacts,
        belief: ActionBeliefState,
        plans: Sequence[NavigationPlan],
    ) -> RankedPlan: ...


@dataclass
class HeuristicPlanRanker:
    """Deterministic fallback used when no model ranker is loaded."""

    def rank(
        self,
        *,
        observation: OfficialFrameSequence,
        facts: GridFacts,
        belief: ActionBeliefState,
        plans: Sequence[NavigationPlan],
    ) -> RankedPlan:
        del observation, facts, belief
        if not plans:
            raise ValueError("cannot rank an empty plan list")
        plan = plans[0]
        return RankedPlan(
            index=0,
            score=-float(plan.length),
            reason="shortest executable navigation hypothesis",
        )


@dataclass
class ActiveNavigationPlan:
    plan: NavigationPlan
    plan_id: str
    next_index: int = 0
    state_before_action: str | None = None

    @property
    def complete(self) -> bool:
        return self.next_index >= len(self.plan.action_sequence)

    def next_action(self) -> tuple[str, tuple[int, int]]:
        if self.complete:
            raise IndexError("navigation plan is complete")
        action = self.plan.action_sequence[self.next_index]
        before = self.plan.position_sequence[self.next_index]
        after = self.plan.position_sequence[self.next_index + 1]
        return action, (after[0] - before[0], after[1] - before[1])


@dataclass
class ControllerState:
    belief: ActionBeliefState = field(default_factory=ActionBeliefState)
    current: OfficialFrameSequence | None = None
    last_action: PlannedAction | None = None
    active_navigation: ActiveNavigationPlan | None = None
    failed_plan_ids: set[str] = field(default_factory=set)
    completed_plan_ids: set[str] = field(default_factory=set)
    receipts: list[RichActionObservation] = field(default_factory=list)
    level_epoch: int = 0
    actions_this_level: int = 0
    total_actions: int = 0


def navigation_plan_id(plan: NavigationPlan) -> str:
    payload = {
        "agent": asdict(plan.agent),
        "target": asdict(plan.target),
        "start_top_left": plan.start_top_left,
        "destination_top_left": plan.destination_top_left,
        "action_sequence": plan.action_sequence,
        "assumed_passable_colors": plan.assumed_passable_colors,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class ClosedLoopController:
    ranker: PlanRanker = field(default_factory=HeuristicPlanRanker)
    max_navigation_candidates: int = 64
    max_actions_per_level: int = 72
    state: ControllerState = field(default_factory=ControllerState)

    def initialize(self, observation: OfficialFrameSequence) -> None:
        if observation.final_grid is None:
            raise ValueError("controller requires a persistent initial grid")
        self.state.current = observation
        self.state.belief.observe_state(observation.sha256)
        self.state.actions_this_level = 0

    def observe(
        self,
        observation: OfficialFrameSequence,
    ) -> RichActionObservation | None:
        previous = self.state.current
        action = self.state.last_action
        if previous is None:
            self.initialize(observation)
            return None
        if action is None:
            self.state.current = observation
            self.state.belief.observe_state(observation.sha256)
            return None

        receipt = rich_action_observation(
            action=action.action,
            coordinate=action.coordinate,
            previous=previous,
            current=observation,
        )
        self.state.receipts.append(receipt)
        # RichActionObservation is intentionally duck-compatible with the
        # ActionObservation protocol and carries additional component evidence.
        self.state.belief.add_transition(receipt)  # type: ignore[arg-type]
        self.state.current = observation

        progressed = receipt.effect.level_progress > 0
        if progressed:
            if self.state.active_navigation is not None:
                self.state.completed_plan_ids.add(
                    self.state.active_navigation.plan_id
                )
            self._reset_level_local()
            self.state.level_epoch += 1
        else:
            self._validate_active_plan(receipt)
        self.state.last_action = None
        return receipt

    def _reset_level_local(self) -> None:
        self.state.active_navigation = None
        self.state.failed_plan_ids.clear()
        self.state.actions_this_level = 0

    def _validate_active_plan(self, receipt: RichActionObservation) -> None:
        active = self.state.active_navigation
        last = self.state.last_action
        if active is None or last is None or last.plan_id != active.plan_id:
            return
        expected = last.expected_delta
        observed = receipt.effect.translation_vectors
        contradiction = False
        if expected is not None:
            contradiction = expected not in observed
        if receipt.effect.unchanged:
            contradiction = True
        if contradiction:
            self.state.failed_plan_ids.add(active.plan_id)
            self.state.active_navigation = None
            return
        active.next_index += 1
        if active.complete:
            # Reaching a hypothesized target without level progress falsifies the
            # target relation for the current level.
            self.state.failed_plan_ids.add(active.plan_id)
            self.state.active_navigation = None

    def _current_facts(self) -> GridFacts:
        if self.state.current is None or self.state.current.final_grid is None:
            raise RuntimeError("controller has no current persistent grid")
        return grid_facts(self.state.current.final_grid)

    def _legal_actions(self) -> tuple[str, ...]:
        if self.state.current is None:
            return ()
        return tuple(self.state.current.available_actions)

    def _complex_actions(self, action_complexity: dict[str, bool] | None) -> tuple[str, ...]:
        if not action_complexity:
            return ()
        return tuple(
            action for action, complex_flag in action_complexity.items() if complex_flag
        )

    def _continue_navigation(self, facts: GridFacts) -> PlannedAction | None:
        active = self.state.active_navigation
        if active is None or active.complete:
            return None
        action, expected_delta = active.next_action()
        if action not in self._legal_actions():
            self.state.failed_plan_ids.add(active.plan_id)
            self.state.active_navigation = None
            return None
        if self.state.current is not None:
            active.state_before_action = self.state.current.sha256
        return PlannedAction(
            action=action,
            coordinate=None,
            source="navigation",
            purpose=(
                f"execute target color {active.plan.target.color} "
                f"relation {active.plan.target.relation}; step "
                f"{active.next_index + 1}/{active.plan.length}"
            ),
            plan_id=active.plan_id,
            expected_delta=expected_delta,
        )

    def _start_navigation(self, facts: GridFacts) -> PlannedAction | None:
        if self.state.current is None:
            return None
        plans = generate_navigation_plans(
            facts,
            self.state.belief,
            max_plans=self.max_navigation_candidates,
        )
        candidates: list[NavigationPlan] = []
        for plan in plans:
            plan_id = navigation_plan_id(plan)
            if (
                plan_id in self.state.failed_plan_ids
                or plan_id in self.state.completed_plan_ids
                or not plan.action_sequence
            ):
                continue
            candidates.append(plan)
        if not candidates:
            return None
        ranked = self.ranker.rank(
            observation=self.state.current,
            facts=facts,
            belief=self.state.belief,
            plans=candidates,
        )
        if ranked.index < 0 or ranked.index >= len(candidates):
            raise ValueError("plan ranker returned an invalid candidate index")
        plan = candidates[ranked.index]
        active = ActiveNavigationPlan(
            plan=plan,
            plan_id=navigation_plan_id(plan),
        )
        self.state.active_navigation = active
        return self._continue_navigation(facts)

    def _probe(
        self,
        facts: GridFacts,
        *,
        action_complexity: dict[str, bool] | None,
    ) -> PlannedAction:
        if self.state.current is None:
            raise RuntimeError("controller has no current observation")
        coordinates = sequence_candidate_coordinates(
            self.state.current.rendered_frames
        )
        choice: ProbeChoice = choose_probe(
            self.state.belief,
            legal_actions=self._legal_actions(),
            current_facts=facts,
            complex_actions=self._complex_actions(action_complexity),
            current_state_id=self.state.current.sha256,
            coordinate_candidates=coordinates,
        )
        return PlannedAction(
            action=choice.action,
            coordinate=choice.coordinate,
            source="probe",
            purpose=";".join(choice.reasons),
        )

    def choose_action(
        self,
        *,
        action_complexity: dict[str, bool] | None = None,
    ) -> PlannedAction:
        if self.state.current is None:
            raise RuntimeError("initialize controller before choosing an action")
        if self.state.last_action is not None:
            raise RuntimeError("observe the result of the last action first")
        if self.state.actions_this_level >= self.max_actions_per_level:
            reset_actions = [
                action
                for action in self._legal_actions()
                if action.upper() in {"RESET", "A0", "ACTION0"}
            ]
            if reset_actions:
                decision = PlannedAction(
                    action=reset_actions[0],
                    coordinate=None,
                    source="level-budget",
                    purpose="level action budget exhausted; request level reset",
                )
            else:
                decision = self._probe(
                    self._current_facts(),
                    action_complexity=action_complexity,
                )
        else:
            facts = self._current_facts()
            decision = (
                self._continue_navigation(facts)
                or self._start_navigation(facts)
                or self._probe(facts, action_complexity=action_complexity)
            )

        self.state.last_action = decision
        self.state.actions_this_level += 1
        self.state.total_actions += 1
        return decision

    def summary(self) -> dict[str, Any]:
        current = self.state.current
        return {
            "schema": "instella_arc.closed_loop_controller.v2",
            "game_id": current.game_id if current else None,
            "state": current.state if current else None,
            "levels_completed": current.levels_completed if current else None,
            "level_epoch": self.state.level_epoch,
            "actions_this_level": self.state.actions_this_level,
            "total_actions": self.state.total_actions,
            "receipts": len(self.state.receipts),
            "failed_plans": len(self.state.failed_plan_ids),
            "completed_plans": len(self.state.completed_plan_ids),
            "active_plan_id": (
                self.state.active_navigation.plan_id
                if self.state.active_navigation is not None
                else None
            ),
            "action_profiles": {
                action: {
                    "trials": profile.trials,
                    "consistency": profile.consistency,
                    "no_change_rate": profile.no_change_rate,
                    "informative_rate": profile.informative_rate,
                    "progress_rate": profile.progress_rate,
                    "translation_vectors": profile.translation_vectors,
                }
                for action, profile in sorted(self.state.belief.profiles.items())
            },
        }
