"""Compact latent-goal dataset builder for runtime smoke experiments.

The full curriculum uses the complete bounded atomic Goal DSL. A CPU smoke run
only needs enough mutually competing predicates to verify the scoring,
set-valued objective, information controls, and held-out evaluation path. This
entry point restricts candidates to CONTACT and ABSENT for every observed color,
then delegates split construction to the canonical builder.
"""

from __future__ import annotations

from typing import Iterable

from . import build_goal_version_dataset as base
from .goal_dsl import ColorAbsent, ContactedColor, GoalProgram


def compact_candidate_goals(spec) -> tuple[GoalProgram, ...]:
    colors: Iterable[int] = (
        spec.palette.background,
        spec.palette.wall,
        spec.palette.agent,
        spec.palette.goal,
    )
    candidates = [
        GoalProgram(predicate)
        for color in sorted(set(int(value) for value in colors))
        for predicate in (ContactedColor(color), ColorAbsent(color))
    ]
    unique = {candidate.sha256: candidate for candidate in candidates}
    return tuple(unique[key] for key in sorted(unique))


def main() -> None:
    base.candidate_goals = compact_candidate_goals
    base.main()


if __name__ == "__main__":
    main()
