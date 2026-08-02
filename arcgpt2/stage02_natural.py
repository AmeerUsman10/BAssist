"""Natural-token, compact exact-codec variant of Stage 0.2.

The first Stage-0.2 protocol used bare character targets (``N/E/S/W/?``) and a
long header.  A token audit showed that GPT-2 already has distinct one-token
representations for the natural continuations `` north/east/south/west/unknown``
and `` one/two/three/four``.  This variant uses those pretrained lexical
representations while preserving the exact same canonical labels, environment,
and one-GPT-2 purity contract.

The grid codec is still deterministic and lossless.  It lists a mechanically
chosen baseline value and every exceptional cell by exact row, column, and
value; it does not identify objects, infer rules, plan, or choose actions.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import stage02_decomposed as base
from .phase0_hidden_action import Action
from .stage02_sparse import parse_sparse_grid_text, sparse_grid_text

DIRECTION_WORD = {
    "N": "north",
    "E": "east",
    "S": "south",
    "W": "west",
    "?": "unknown",
}
ACTION_WORD = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
}
LABEL_SURFACE = {
    "N": " north",
    "E": " east",
    "S": " south",
    "W": " west",
    "?": " unknown",
    "1": " one",
    "2": " two",
    "3": " three",
    "4": " four",
}

COMPACT_WORLD_HEADER = (
    "Exact grid record. Grid format is HxW;bV;cells: every unlisted cell has "
    "value V and each listed cell is rROWcCOLUMN=VALUE. Values: 0 empty, "
    "1 wall, 2 mover, 3 goal. Row numbers increase south; column numbers "
    "increase east. In each game actions one, two, three, four are a hidden "
    "permutation of north, east, south, west. Infer it only from observed "
    "cell changes."
)


def mapping_prompt(history: str, action: Action) -> str:
    digit = base.ACTION_DIGIT[action]
    return (
        history
        + f"\nQuestion: What direction does action {ACTION_WORD[digit]} move? "
          "If the observations do not reveal it, answer unknown. Answer exactly "
          "north, east, south, west, or unknown.\nAnswer:"
    )


def need_prompt(history: str) -> str:
    return (
        history
        + "\nQuestion: If any action meaning is still unknown, answer unknown. "
          "Otherwise choose the direction that reduces the larger absolute "
          "row or column distance from mover to goal; use vertical on a tie. "
          "Answer exactly north, east, south, west, or unknown.\nAnswer:"
    )


def mapping_summary(labels: Mapping[Action, str]) -> str:
    return ", ".join(
        f"{ACTION_WORD[base.ACTION_DIGIT[action]]}="
        f"{DIRECTION_WORD[labels.get(action, '?')]}"
        for action in Action
    )


def compose_prompt(labels: Mapping[Action, str], needed_direction: str) -> str:
    return (
        "Action meanings: "
        + mapping_summary(labels)
        + f". Needed direction: {DIRECTION_WORD[needed_direction]}. "
          "If any meaning is unknown, choose the lowest-numbered unknown action. "
          "Otherwise choose the action matching the needed direction. Answer "
          "exactly one, two, three, or four.\nAnswer:"
    )


def direct_prompt(history: str) -> str:
    return (
        history
        + "\nQuestion: Choose the next action. Infer action meanings from observed "
          "changes. If any meaning is unknown, choose the lowest-numbered unknown "
          "action. Otherwise choose the action moving toward the goal by reducing "
          "the larger absolute row or column distance, using vertical on a tie. "
          "Answer exactly one, two, three, or four.\nAnswer:"
    )


# Install only deterministic representation and wording changes.  The base
# curriculum and decision labels remain identical.
base.grid_text = sparse_grid_text
base.parse_grid_text = parse_sparse_grid_text
base.WORLD_HEADER = COMPACT_WORLD_HEADER
base.mapping_prompt = mapping_prompt
base.need_prompt = need_prompt
base.mapping_summary = mapping_summary
base.compose_prompt = compose_prompt
base.direct_prompt = direct_prompt

build_dataset = base.build_dataset
build_rows = base.build_rows
format_history = base.format_history
make_stage02_spec = base.make_stage02_spec
simulate_context = base.simulate_context


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
