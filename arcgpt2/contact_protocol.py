"""Exact natural-language protocol for hidden contact mechanics.

The protocol names the color whose behavior is being queried but supplies no
learned object labels or effect classification. All evidence is exact grid
state, action label, changed cells, and terminal status.
"""

from __future__ import annotations

from typing import Sequence

from .mechanics_v2 import ContactMode
from .natural_protocol import grid_text, transition_text
from .primitive_contact_game import ContactStepRecord


def contact_completion(mode: ContactMode, interaction_color: int) -> str:
    """Return one ordinary-language completion for a typed contact primitive."""

    prefix = f" contact with color {interaction_color}"
    descriptions = {
        ContactMode.BLOCK: (
            f"{prefix} blocks movement and leaves the grid unchanged."
        ),
        ContactMode.COLLECT: (
            f"{prefix} moves the controlled cell into that location and removes the contacted cell."
        ),
        ContactMode.PUSH: (
            f"{prefix} pushes the contacted cell one step forward and moves the controlled cell into its former location."
        ),
        ContactMode.ERASE: (
            f"{prefix} removes the contacted cell while the controlled cell stays where it was."
        ),
        ContactMode.SWAP: (
            f"{prefix} swaps the controlled cell and the contacted cell."
        ),
    }
    return descriptions[mode]


def contact_history_text(records: Sequence[ContactStepRecord]) -> str:
    if not records:
        return "No action-outcome observations are available."
    sections: list[str] = []
    for index, record in enumerate(records, start=1):
        sections.append(
            f"OBSERVATION {index}\nBEFORE ACTION\n{grid_text(record.before)}\n"
            + transition_text(record)  # structural protocol; ContactStepRecord exposes the same fields
        )
    return "\n\n".join(sections)


def contact_prompt(
    initial_grid,
    records: Sequence[ContactStepRecord],
    interaction_color: int,
) -> str:
    return "\n\n".join(
        (
            "You are studying an unfamiliar deterministic grid game.",
            "Actions A1, A2, A3, and A4 have hidden directional meanings. One persistent rule determines what happens when the controlled moving cell contacts the cell color named in the question.",
            "Use only exact interventions and outcomes. Several rules may remain possible before direct contact is observed.",
            "INITIAL GRID\n" + grid_text(initial_grid),
            "ACTION HISTORY\n" + contact_history_text(records),
            f"QUESTION\nComplete the sentence with the rule for color {interaction_color}.",
            "ANSWER:",
        )
    )
