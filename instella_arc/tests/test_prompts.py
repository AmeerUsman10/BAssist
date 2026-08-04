from __future__ import annotations

import pytest

from instella_arc.prompts import (
    TaskKind,
    build_prompt,
    extract_final_json,
    extract_legal_action,
    normalize_actions,
)


def test_action_normalization_is_stable_and_deduplicated() -> None:
    assert normalize_actions((1, "A2", "ACTION1", "action7")) == (
        "A1",
        "A2",
        "A7",
    )
    with pytest.raises(ValueError):
        normalize_actions(("north",))


def test_prompt_contains_exact_evidence_legal_actions_and_schema() -> None:
    prompt = build_prompt(
        TaskKind.PROPOSE_EXPERIMENT,
        "FRAME 0 exact-grid",
        legal_actions=("A1", "A4", "A6"),
        query="Separate H1 from H2.",
    )
    rendered = prompt.plain_text()
    assert "FRAME 0 exact-grid" in rendered
    assert "A1 A4 A6" in rendered
    assert "Separate H1 from H2" in rendered
    assert "<FINAL>" in rendered


def test_final_json_and_action_parser_use_the_last_valid_block() -> None:
    text = (
        '<FINAL>{"action":"A2"}</FINAL>\n'
        'revision\n<FINAL>{"action":"A4","x":3,"y":5}</FINAL>'
    )
    assert extract_final_json(text)["action"] == "A4"
    assert extract_legal_action(text, ("A1", "A4")) == ("A4", 3, 5)


def test_action_parser_falls_back_to_last_legal_plain_token() -> None:
    text = "I considered A2 but the final legal experiment is A3."
    assert extract_legal_action(text, ("A1", "A3")) == ("A3", None, None)
    with pytest.raises(ValueError):
        extract_legal_action("A2", ("A1", "A3"))
