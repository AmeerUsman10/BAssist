"""Plain-text interaction protocol used to train and run one GPT-2 model.

The protocol intentionally uses GPT-2's original byte-pair tokenizer. No new
learned embeddings or auxiliary tokenizer model are introduced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .codec import Grid, encode_delta, encode_grid

ACTION_PATTERN = re.compile(r"\bA([0-7])\b", flags=re.IGNORECASE)
MAPPING_ITEM_PATTERN = re.compile(r"\bA([1-4])\s*=\s*([NESW?])\b", flags=re.IGNORECASE)
COORDINATE_PATTERN = re.compile(
    r"\bX\s*=?\s*(\d{1,2})\s*[,; ]+\s*Y\s*=?\s*(\d{1,2})\b",
    flags=re.IGNORECASE,
)

DIRECTION_ORDER = ("N", "E", "S", "W")
ACTION_ORDER = ("A1", "A2", "A3", "A4")


@dataclass(frozen=True)
class Transition:
    """One literal state/action/result observation."""

    before: Grid
    action: str
    after: Grid

    @property
    def delta(self) -> str:
        return encode_delta(self.before, self.after)


def normalize_action(action: str | int) -> str:
    """Normalize an action identifier to ``A0`` through ``A7``."""
    if isinstance(action, int):
        value = action
    else:
        match = ACTION_PATTERN.search(str(action).strip())
        if match is None:
            raise ValueError(f"invalid action: {action!r}")
        value = int(match.group(1))
    if not 0 <= value <= 7:
        raise ValueError("action id must be between 0 and 7")
    return f"A{value}"


def format_history(transitions: Sequence[Transition], limit: int | None = None) -> str:
    """Serialize literal transitions, optionally retaining only the newest ones."""
    selected = list(transitions[-limit:] if limit is not None else transitions)
    if not selected:
        return "NONE"
    blocks: list[str] = []
    for index, transition in enumerate(selected, start=1):
        blocks.extend(
            [
                f"TURN {index}",
                f"OBS {encode_grid(transition.before)}",
                f"ACT {normalize_action(transition.action)}",
                f"NEXT {transition.delta}",
            ]
        )
    return "\n".join(blocks)


def format_mapping(mapping: Mapping[str, str], unknown: str = "?") -> str:
    """Serialize the four simple-action meanings in a fixed order."""
    parts: list[str] = []
    for action in ACTION_ORDER:
        direction = str(mapping.get(action, unknown)).upper()
        if direction not in {*DIRECTION_ORDER, "?"}:
            raise ValueError(f"invalid direction for {action}: {direction}")
        parts.append(f"{action}={direction}")
    return ";".join(parts)


def parse_mapping(text: str) -> dict[str, str]:
    """Extract the most recent valid assignment for every simple action.

    Missing fields are represented explicitly as ``?``. Returning a complete,
    fixed-order mapping prevents the recurrent state from silently deleting an
    unknown action and makes parse/format round-trips lossless.
    """

    mapping: dict[str, str] = {action: "?" for action in ACTION_ORDER}
    for match in MAPPING_ITEM_PATTERN.finditer(text):
        mapping[f"A{match.group(1)}"] = match.group(2).upper()
    return mapping


def parse_action(text: str, available: Iterable[str | int] | None = None) -> str | None:
    """Extract the last action token and optionally require it to be available."""
    matches = list(ACTION_PATTERN.finditer(text))
    if not matches:
        return None
    allowed = (
        {normalize_action(action) for action in available}
        if available is not None
        else None
    )
    for match in reversed(matches):
        action = f"A{int(match.group(1))}"
        if allowed is None or action in allowed:
            return action
    return None


def parse_coordinate(text: str, width: int = 64, height: int = 64) -> tuple[int, int] | None:
    """Extract and bounds-check an ``X n Y n`` coordinate pair."""
    matches = list(COORDINATE_PATTERN.finditer(text))
    if not matches:
        return None
    x = int(matches[-1].group(1))
    y = int(matches[-1].group(2))
    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def memory_prompt(
    transitions: Sequence[Transition],
    current_grid: Grid,
    available_actions: Sequence[str | int] = ACTION_ORDER,
    history_limit: int = 8,
) -> str:
    """Prompt the same GPT-2 to rewrite its compact recurrent memory."""
    available = " ".join(normalize_action(action) for action in available_actions)
    return (
        "[[TASK]] MEMORY\n"
        "Infer only action meanings supported by literal transitions. "
        "Use ? when unknown.\n"
        "[[HISTORY]]\n"
        f"{format_history(transitions, limit=history_limit)}\n"
        "[[/HISTORY]]\n"
        f"[[CURRENT]] {encode_grid(current_grid)}\n"
        f"[[AVAILABLE]] {available}\n"
        "[[OUTPUT]]\n"
        "[[MEMORY]]"
    )


def action_prompt(
    transitions: Sequence[Transition],
    current_grid: Grid,
    memory: str,
    available_actions: Sequence[str | int] = ACTION_ORDER,
    history_limit: int = 8,
) -> str:
    """Prompt GPT-2 for the next action using its own memory packet."""
    available = " ".join(normalize_action(action) for action in available_actions)
    return (
        "[[TASK]] ACTION\n"
        "Choose one available action. Reach the value-3 goal with few actions. "
        "If the needed direction is unknown, test an action whose meaning is ?.\n"
        "[[HISTORY]]\n"
        f"{format_history(transitions, limit=history_limit)}\n"
        "[[/HISTORY]]\n"
        f"[[CURRENT]] {encode_grid(current_grid)}\n"
        f"[[MEMORY]] {memory.strip()} [[/MEMORY]]\n"
        f"[[AVAILABLE]] {available}\n"
        "[[OUTPUT]]\n"
        "[[ACTION]]"
    )


def prediction_prompt(
    transitions: Sequence[Transition],
    current_grid: Grid,
    action: str | int,
    memory: str,
    history_limit: int = 8,
) -> str:
    """Prompt GPT-2 to predict a literal next-state delta for one action."""
    return (
        "[[TASK]] PREDICT\n"
        "Predict the literal changed cells. Use D[h,w]:SAME if no cell changes.\n"
        "[[HISTORY]]\n"
        f"{format_history(transitions, limit=history_limit)}\n"
        "[[/HISTORY]]\n"
        f"[[CURRENT]] {encode_grid(current_grid)}\n"
        f"[[MEMORY]] {memory.strip()} [[/MEMORY]]\n"
        f"[[ACTION]] {normalize_action(action)} [[/ACTION]]\n"
        "[[OUTPUT]]\n"
        "[[NEXT]]"
    )


def coordinate_prompt(
    transitions: Sequence[Transition],
    current_grid: Grid,
    memory: str,
    history_limit: int = 8,
) -> str:
    """Prompt GPT-2 for coordinates after it has selected complex action A6."""
    return (
        "[[TASK]] COORDINATE\n"
        "Choose one grid coordinate for A6 and answer as X n Y n.\n"
        "[[HISTORY]]\n"
        f"{format_history(transitions, limit=history_limit)}\n"
        "[[/HISTORY]]\n"
        f"[[CURRENT]] {encode_grid(current_grid)}\n"
        f"[[MEMORY]] {memory.strip()} [[/MEMORY]]\n"
        "[[ACTION]] A6 [[/ACTION]]\n"
        "[[OUTPUT]]\n"
        "[[COORD]]"
    )
