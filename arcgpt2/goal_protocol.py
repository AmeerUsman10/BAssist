"""Natural-language prompts and completions for latent ARC goal predicates.

The serializer exposes only mechanically exact facts: full grids, exact changed
cells, action labels, and environment terminal reports. It does not identify an
agent, target, object, or likely goal. Candidate meanings are supplied by the
same bounded Goal DSL used for exact replay.
"""

from __future__ import annotations

from typing import Sequence

from .goal_dsl import (
    ColorAbsent,
    ColorCount,
    ColorsTouch,
    Comparison,
    ContactedColor,
    GoalProgram,
)
from .natural_protocol import grid_text, transition_text
from .phase0_hidden_action import StepRecord


def goal_candidate_text(goal: GoalProgram) -> str:
    """Return one ordinary-language completion for a typed goal program."""

    predicate = goal.predicate
    if isinstance(predicate, ContactedColor):
        return f" success occurs when the action contacts color {predicate.color}."
    if isinstance(predicate, ColorAbsent):
        return f" success occurs when color {predicate.color} is absent after the action."
    if isinstance(predicate, ColorCount):
        comparison = {
            Comparison.EQ: "exactly",
            Comparison.LE: "at most",
            Comparison.GE: "at least",
        }[predicate.comparison]
        noun = "cell has" if predicate.value == 1 else "cells have"
        return (
            f" success occurs when {comparison} {predicate.value} {noun} "
            f"color {predicate.color} after the action."
        )
    if isinstance(predicate, ColorsTouch):
        connectivity = (
            "edge-neighbor"
            if predicate.connectivity.value == "FOUR"
            else "edge-or-corner-neighbor"
        )
        return (
            f" success occurs when a cell of color {predicate.left} and a cell "
            f"of color {predicate.right} are {connectivity} touching after the action."
        )
    # Compound goals are deliberately deferred until atomic calibration works.
    raise ValueError(
        f"no natural-language completion is defined for {type(predicate).__name__}"
    )


def goal_history_text(
    records: Sequence[StepRecord],
    *,
    include_terminal_reports: bool = True,
    displayed_terminal_reports: Sequence[bool] | None = None,
) -> str:
    """Serialize exact attempts, including each pre-action grid.

    Supplying ``displayed_terminal_reports`` creates a controlled corruption of
    status labels while preserving actions and state transitions.
    """

    if displayed_terminal_reports is not None and len(displayed_terminal_reports) != len(records):
        raise ValueError("displayed terminal reports must match the record count")
    if not records:
        return "No action-outcome observations are available."

    sections: list[str] = []
    for index, record in enumerate(records, start=1):
        sections.append(f"OBSERVATION {index}\nBEFORE ACTION\n{grid_text(record.before)}")
        exact = transition_text(record)
        if not include_terminal_reports:
            exact = "\n".join(
                line
                for line in exact.splitlines()
                if not line.startswith("The environment reported a terminal success:")
            )
            exact += "\nThe terminal report is hidden."
        elif displayed_terminal_reports is not None:
            lines = [
                line
                for line in exact.splitlines()
                if not line.startswith("The environment reported a terminal success:")
            ]
            shown = "yes" if displayed_terminal_reports[index - 1] else "no"
            lines.append(f"The environment reported a terminal success: {shown}.")
            exact = "\n".join(lines)
        sections.append(exact)
    return "\n\n".join(sections)


def goal_prompt(
    initial_grid,
    records: Sequence[StepRecord],
    *,
    include_terminal_reports: bool = True,
    displayed_terminal_reports: Sequence[bool] | None = None,
) -> str:
    """Build a candidate-scoring prompt for an unknown terminal rule."""

    return "\n\n".join(
        (
            "You are studying an unfamiliar deterministic grid game.",
            "A terminal success rule is fixed across the game's levels. Infer only what the literal observations support. Several rules may remain possible before enough evidence is observed.",
            "INITIAL GRID\n" + grid_text(initial_grid),
            "ACTION HISTORY\n"
            + goal_history_text(
                records,
                include_terminal_reports=include_terminal_reports,
                displayed_terminal_reports=displayed_terminal_reports,
            ),
            "QUESTION\nComplete the sentence with a terminal rule consistent with every observation.",
            "ANSWER:",
        )
    )


def rotate_terminal_reports(records: Sequence[StepRecord]) -> tuple[bool, ...]:
    """Move every terminal label one step forward as a deterministic control."""

    labels = tuple(record.status in {"LEVEL_WIN", "GAME_WIN"} for record in records)
    if len(labels) <= 1:
        return tuple(not value for value in labels)
    return labels[-1:] + labels[:-1]
