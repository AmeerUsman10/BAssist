from __future__ import annotations

from arcgpt2.phase0_hidden_action import Action, Direction
from arcgpt2 import stage02_decomposed as base
from arcgpt2.stage02_natural import (
    ACTION_WORD,
    DIRECTION_WORD,
    LABEL_SURFACE,
    compose_prompt,
    direct_prompt,
    mapping_prompt,
    need_prompt,
)


def test_natural_surfaces_cover_every_canonical_label() -> None:
    assert set(LABEL_SURFACE) == {"N", "E", "S", "W", "?", "1", "2", "3", "4"}
    assert all(surface.startswith(" ") for surface in LABEL_SURFACE.values())
    assert len(set(LABEL_SURFACE.values())) == len(LABEL_SURFACE)


def test_natural_protocol_installs_compact_sparse_history() -> None:
    context = base.simulate_context(
        0,
        (Action.A1,),
        goal_direction=Direction.UP,
        variant_id="natural",
    )
    history = base.format_history(
        context.records,
        context.current_grid,
        level_index=context.level_index,
    )
    assert "Exact grid record" in history
    assert ";b" in history
    assert "action 1;" in history


def test_natural_prompts_request_lexical_answers() -> None:
    history = "history"
    assert mapping_prompt(history, Action.A1).endswith("Answer:")
    assert "north, east, south, west, or unknown" in mapping_prompt(history, Action.A1)
    assert "lowest-numbered unknown" in compose_prompt(
        {action: "?" for action in Action}, "?"
    )
    assert "one, two, three, or four" in direct_prompt(history)
    assert "answer unknown" in need_prompt(history)


def test_word_maps_are_bijective_for_protocol_labels() -> None:
    assert set(DIRECTION_WORD) == {"N", "E", "S", "W", "?"}
    assert set(ACTION_WORD) == {"1", "2", "3", "4"}
    assert len(set(DIRECTION_WORD.values())) == 5
    assert len(set(ACTION_WORD.values())) == 4
