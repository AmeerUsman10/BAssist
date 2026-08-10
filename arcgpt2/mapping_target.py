"""Compact atomic target for the Phase-0 hidden action mapping.

The first program-induction smoke run trained on a full ARC-DSL program. Most
target tokens were identical across all 24 candidates, so the useful mapping
signal was diluted by syntax and shared color-role fields. This module isolates
the four latent action effects while retaining an exact deterministic expansion
back to a complete executable Program.
"""

from __future__ import annotations

from typing import Mapping

from .dsl import DSLError, MoveColorRule, Program
from .phase0_hidden_action import Action, Direction, DIRECTION_DELTA


MAPPING_SPECIAL_TOKENS = [
    "<MAP>",
    "</MAP>",
    "<UP>",
    "<DOWN>",
    "<LEFT>",
    "<RIGHT>",
]

_DIRECTION_TOKEN: dict[Direction, str] = {
    Direction.UP: "<UP>",
    Direction.DOWN: "<DOWN>",
    Direction.LEFT: "<LEFT>",
    Direction.RIGHT: "<RIGHT>",
}
_TOKEN_DIRECTION = {token: direction for direction, token in _DIRECTION_TOKEN.items()}
_DELTA_DIRECTION = {delta: direction for direction, delta in DIRECTION_DELTA.items()}


def program_mapping(program: Program) -> dict[Action, Direction]:
    mapping: dict[Action, Direction] = {}
    for action in Action:
        rule = program.by_action.get(action)
        if rule is None:
            raise DSLError(f"program lacks {action.value}")
        direction = _DELTA_DIRECTION.get((rule.dy, rule.dx))
        if direction is None:
            raise DSLError("compact Phase-0 mapping supports cardinal unit moves only")
        mapping[action] = direction
    if len(set(mapping.values())) != len(Action):
        raise DSLError("Phase-0 mapping must be a direction permutation")
    return mapping


def compact_mapping(program: Program) -> str:
    mapping = program_mapping(program)
    tokens = ["<MAP>"]
    for action in Action:
        tokens.extend((f"<{action.value}>", _DIRECTION_TOKEN[mapping[action]]))
    tokens.append("</MAP>")
    return " ".join(tokens)


def parse_compact_mapping(text: str) -> dict[Action, Direction]:
    tokens = text.strip().split()
    expected_length = 2 + 2 * len(Action)
    if len(tokens) != expected_length:
        raise DSLError(
            f"compact mapping requires {expected_length} tokens, received {len(tokens)}"
        )
    if tokens[0] != "<MAP>" or tokens[-1] != "</MAP>":
        raise DSLError("compact mapping must be enclosed by <MAP> and </MAP>")

    mapping: dict[Action, Direction] = {}
    cursor = 1
    for expected_action in Action:
        action_token = tokens[cursor]
        direction_token = tokens[cursor + 1]
        cursor += 2
        if action_token != f"<{expected_action.value}>":
            raise DSLError(
                f"expected <{expected_action.value}>, received {action_token!r}"
            )
        direction = _TOKEN_DIRECTION.get(direction_token)
        if direction is None:
            raise DSLError(f"invalid direction token: {direction_token!r}")
        mapping[expected_action] = direction
    if len(set(mapping.values())) != len(Action):
        raise DSLError("compact mapping must assign each direction exactly once")
    return mapping


def expand_mapping(mapping: Mapping[Action, Direction], template: Program) -> Program:
    """Replace only action displacements while preserving all other mechanics."""

    if set(mapping) != set(Action) or len(set(mapping.values())) != len(Action):
        raise DSLError("mapping must be a complete action-direction permutation")
    rules: list[MoveColorRule] = []
    for action in Action:
        source = template.by_action[action]
        dy, dx = DIRECTION_DELTA[mapping[action]]
        rules.append(
            MoveColorRule(
                action=action,
                moving_color=source.moving_color,
                dy=dy,
                dx=dx,
                background_color=source.background_color,
                blocking_colors=source.blocking_colors,
                win_on_colors=source.win_on_colors,
            )
        )
    return Program(tuple(rules), version=template.version)
