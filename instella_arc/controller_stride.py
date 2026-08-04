"""Closed-loop controller variant with pure-axis variable-stride navigation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .controller import (
    ActiveNavigationPlan,
    ClosedLoopController,
    PlannedAction,
    navigation_plan_id,
)
from .navigation import NavigationPlan
from .navigation_stride import generate_stride_navigation_plans
from .receipts import RichActionObservation
from .world_state import GridFacts


def target_hypothesis_key(plan: NavigationPlan) -> tuple:
    return (
        plan.agent.color,
        plan.agent.shape,
        plan.target.color,
        plan.target.cells,
        plan.target.relation,
    )


@dataclass
class StrideClosedLoopController(ClosedLoopController):
    """Use observed rendered-cell stride and remember disproven level goals."""

    failed_target_hypotheses: set[tuple] = field(default_factory=set)

    def _reset_level_local(self) -> None:
        super()._reset_level_local()
        self.failed_target_hypotheses.clear()

    def _validate_active_plan(self, receipt: RichActionObservation) -> None:
        active = self.state.active_navigation
        last = self.state.last_action
        if active is None or last is None or last.plan_id != active.plan_id:
            return
        expected = last.expected_delta
        observed = receipt.effect.translation_vectors
        contradiction = receipt.effect.unchanged or (
            expected is not None and expected not in observed
        )
        if contradiction:
            self.state.failed_plan_ids.add(active.plan_id)
            self.state.active_navigation = None
            return
        active.next_index += 1
        if active.complete:
            # The exact path executed as predicted but did not increase the level
            # count. Reject the underlying target relation for this level, not
            # merely this start-state-specific path.
            self.state.failed_plan_ids.add(active.plan_id)
            self.failed_target_hypotheses.add(
                target_hypothesis_key(active.plan)
            )
            self.state.active_navigation = None

    def _start_navigation(self, facts: GridFacts) -> PlannedAction | None:
        if self.state.current is None:
            return None
        plans = generate_stride_navigation_plans(
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
                or target_hypothesis_key(plan) in self.failed_target_hypotheses
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
        self.state.active_navigation = ActiveNavigationPlan(
            plan=plan,
            plan_id=navigation_plan_id(plan),
        )
        return self._continue_navigation(facts)

    def summary(self):
        payload = super().summary()
        payload["schema"] = "instella_arc.stride_closed_loop_controller.v1"
        payload["failed_target_hypotheses"] = len(
            self.failed_target_hypotheses
        )
        return payload
