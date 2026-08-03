"""Closed-loop controller variant with pure-axis variable-stride navigation."""

from __future__ import annotations

from .controller import (
    ActiveNavigationPlan,
    ClosedLoopController,
    PlannedAction,
    navigation_plan_id,
)
from .navigation import NavigationPlan
from .navigation_stride import generate_stride_navigation_plans
from .world_state import GridFacts


class StrideClosedLoopController(ClosedLoopController):
    """Use observed rendered-cell stride rather than assuming one-cell motion."""

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
