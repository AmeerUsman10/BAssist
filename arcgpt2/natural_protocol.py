"""Exact natural-language serialization for GPT-2.

The symbolic codec is ideal for lossless storage, but most of its tokens are new
to GPT-2. This protocol expresses the same mechanically available facts using
ordinary English, numbers, and punctuation already present in GPT-2's
pretraining distribution. It adds no object, goal, or movement labels.
"""

from __future__ import annotations

from typing import Sequence

from .codec import normalize_grid
from .phase0_hidden_action import Action, StepRecord


_DIRECTION_WORDS = ("north", "south", "west", "east")
_DIRECTION_ALIASES = {
    "north": "north",
    "south": "south",
    "west": "west",
    "east": "east",
    "up": "north",
    "down": "south",
    "left": "west",
    "right": "east",
}


def grid_text(grid: Sequence[Sequence[int]]) -> str:
    """Return an exact row-major grid description."""

    frame = normalize_grid(grid)
    lines = [f"The grid has {len(frame)} rows and {len(frame[0])} columns."]
    for row_index, row in enumerate(frame):
        values = " ".join(str(value) for value in row)
        lines.append(f"Row {row_index}: {values}.")
    return "\n".join(lines)


def changed_cells(record: StepRecord) -> list[tuple[int, int, int, int]]:
    """List exact `(row, column, old, new)` cell changes in canonical order."""

    before = record.before
    after = record.after
    return [
        (row, column, before[row][column], after[row][column])
        for row in range(len(before))
        for column in range(len(before[0]))
        if before[row][column] != after[row][column]
    ]


def transition_text(record: StepRecord, *, displayed_action: Action | None = None) -> str:
    """Describe one exact transition without assigning semantic roles."""

    action = displayed_action or record.action
    changes = changed_cells(record)
    lines = [f"I applied {action.value}."]
    if not changes:
        lines.append("No grid cell changed.")
    else:
        lines.append(f"Exactly {len(changes)} grid cells changed:")
        for row, column, old, new in changes:
            lines.append(
                f"- Row {row}, column {column} changed from color {old} to color {new}."
            )
    terminal = "yes" if record.status in {"LEVEL_WIN", "GAME_WIN"} else "no"
    lines.append(f"The environment reported a terminal success: {terminal}.")
    return "\n".join(lines)


def mapping_prompt(
    initial_grid: Sequence[Sequence[int]],
    records: Sequence[StepRecord],
    query_action: Action,
    *,
    displayed_actions: Sequence[Action] | None = None,
    include_evidence: bool = True,
) -> str:
    """Build a natural-text action-semantics query.

    `displayed_actions` can relabel the evidence for corruption controls while
    the underlying frames remain unchanged.
    """

    if displayed_actions is not None and len(displayed_actions) != len(records):
        raise ValueError("displayed_actions must match the number of records")

    sections = [
        "You are studying an unfamiliar deterministic grid game.",
        "Actions A1, A2, A3, and A4 are a hidden permutation of north, south, west, and east.",
        "Use only the exact observations below. Do not guess from the action number.",
        "INITIAL GRID\n" + grid_text(initial_grid),
    ]
    if include_evidence:
        for index, record in enumerate(records, start=1):
            displayed = displayed_actions[index - 1] if displayed_actions is not None else None
            sections.append(
                f"OBSERVATION {index}\n" + transition_text(record, displayed_action=displayed)
            )
    else:
        sections.append("No action-outcome observations are available.")

    sections.extend(
        (
            f"QUESTION\nWhich direction does {query_action.value} represent?",
            "Answer with exactly one word: north, south, west, or east.",
            "ANSWER:",
        )
    )
    return "\n\n".join(sections)


def rotate_action_labels(records: Sequence[StepRecord]) -> tuple[Action, ...]:
    """Deterministically corrupt action labels while preserving all outcomes."""

    rotation = {
        Action.A1: Action.A2,
        Action.A2: Action.A3,
        Action.A3: Action.A4,
        Action.A4: Action.A1,
    }
    return tuple(rotation[record.action] for record in records)


def canonical_direction_word(direction: str) -> str:
    normalized = direction.strip().lower()
    try:
        return _DIRECTION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"invalid cardinal direction: {direction!r}") from exc


def answer_text(direction: str) -> str:
    # Leading whitespace gives GPT-2 its ordinary word-boundary tokenization.
    return " " + canonical_direction_word(direction)


def direction_words() -> tuple[str, ...]:
    return _DIRECTION_WORDS
