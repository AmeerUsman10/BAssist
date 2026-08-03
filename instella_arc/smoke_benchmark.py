"""Bounded frozen-checkpoint benchmark for the first expensive GPU load.

A full synthetic benchmark can contain hundreds of candidate-scoring forwards.
The smoke profile selects one no-evidence case and one matched maximum-evidence
case across every control mode. It is sufficient to validate model loading,
log-probability scoring, and evidence sensitivity before spending more quota.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from .benchmark import (
    Case,
    ScoringBackend,
    build_action_cases,
    build_contact_cases,
    build_goal_cases,
    run_case,
    summarize,
)


def select_smoke_cases(cases: Sequence[Case]) -> list[Case]:
    if not cases:
        raise ValueError("smoke selection requires at least one case")
    selected: list[Case] = []
    for task in sorted({case.task for case in cases}):
        task_cases = [case for case in cases if case.task == task]
        min_depth = min(case.evidence_depth for case in task_cases)
        max_depth = max(case.evidence_depth for case in task_cases)

        no_evidence = next(
            case
            for case in task_cases
            if case.evidence_depth == min_depth and case.mode == "intact"
        )
        selected.append(no_evidence)

        max_intact = next(
            case
            for case in task_cases
            if case.evidence_depth == max_depth and case.mode == "intact"
        )
        identity = max_intact.case_id.rsplit(":", 1)[0]
        matched = [
            case
            for case in task_cases
            if case.evidence_depth == max_depth
            and case.case_id.rsplit(":", 1)[0] == identity
        ]
        selected.extend(sorted(matched, key=lambda case: case.mode))

    unique: list[Case] = []
    seen: set[str] = set()
    for case in selected:
        if case.case_id not in seen:
            unique.append(case)
            seen.add(case.case_id)
    return unique


def run_smoke_benchmark(
    backend: ScoringBackend,
    *,
    tasks: Sequence[str],
    seed_base: int,
) -> dict[str, Any]:
    builders = {
        "action": build_action_cases,
        "goal": build_goal_cases,
        "contact": build_contact_cases,
    }
    all_cases: list[Case] = []
    for task in tasks:
        try:
            builder = builders[task]
        except KeyError as exc:
            raise ValueError(f"unknown task: {task}") from exc
        all_cases.extend(builder((seed_base,)))
    cases = select_smoke_cases(all_cases)
    results = [run_case(backend, case) for case in cases]
    return {
        "schema": "instella_arc.frozen_benchmark.v1",
        "profile": "smoke",
        "backend": dict(getattr(backend, "metadata", {})),
        "tasks": list(tasks),
        "seed_base": seed_base,
        "games_per_task": 1,
        "generated_case_count": len(all_cases),
        "case_count": len(results),
        "selected_case_ids": [case.case_id for case in cases],
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
