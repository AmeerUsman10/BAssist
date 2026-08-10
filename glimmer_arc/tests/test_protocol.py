from __future__ import annotations

import pytest

from glimmer_arc.protocol import (
    ProtocolError,
    choose_cfs_action,
    extract_json_object,
    parse_cfs_packet,
    parse_native_plan,
)


def test_extract_json_object_accepts_one_fence() -> None:
    assert extract_json_object('```json\n{"x":1}\n```') == {"x": 1}


def test_extract_json_object_rejects_prose() -> None:
    with pytest.raises(ProtocolError):
        extract_json_object('answer: {"x":1}')


def test_native_plan_rejects_unknown_signature() -> None:
    with pytest.raises(ProtocolError):
        parse_native_plan(
            {"plan": [{"signature": "BAD", "confidence": 1.0}]},
            allowed=["ACTION1"],
            maximum=2,
        )


def test_cfs_controller_uses_disagreement_and_progress() -> None:
    packet = {
        "hypotheses": [
            {
                "id": "H1",
                "weight": 1,
                "predictions": [
                    {"signature": "A", "outcome": "PROGRESS", "confidence": 0.9},
                    {"signature": "B", "outcome": "NO_CHANGE", "confidence": 0.9},
                ],
            },
            {
                "id": "H2",
                "weight": 1,
                "predictions": [
                    {"signature": "A", "outcome": "CHANGE", "confidence": 0.8},
                    {"signature": "B", "outcome": "NO_CHANGE", "confidence": 0.9},
                ],
            },
        ],
        "unsafe_signatures": [],
    }
    hypotheses, unsafe = parse_cfs_packet(packet, allowed=["A", "B"])
    chosen, scores = choose_cfs_action(hypotheses, unsafe, ["A", "B"])
    assert chosen == "A"
    assert scores["A"]["score"] > scores["B"]["score"]
