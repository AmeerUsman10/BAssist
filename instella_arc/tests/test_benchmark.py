from __future__ import annotations

import math

import pytest

from instella_arc.backend import MockBackend
from instella_arc.benchmark import (
    Case,
    build_action_cases,
    run_case,
    summarize,
)
from instella_arc.prompts import Prompt, TaskKind


def _prompt() -> Prompt:
    return Prompt(
        task=TaskKind.INFER_ACTION,
        messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "evidence"},
        ),
        legal_actions=(),
    )


def test_uniform_scores_preserve_a_set_valued_target() -> None:
    case = Case(
        case_id="test",
        task="action_binding",
        mode="intact",
        evidence_depth=0,
        prompt=_prompt(),
        completions=(" north", " south", " west", " east"),
        target=(0.25, 0.25, 0.25, 0.25),
        truth_index=0,
        metadata={},
    )
    result = run_case(MockBackend(), case)
    assert result.probabilities == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert result.consistent_mass == pytest.approx(1.0)
    assert result.set_cross_entropy == pytest.approx(math.log(4.0))
    assert result.brier == pytest.approx(0.0)


def test_consistent_candidate_receives_high_mass() -> None:
    completions = (" north", " south", " west", " east")
    backend = MockBackend(scores={" east": 8.0})
    case = Case(
        case_id="test2",
        task="action_binding",
        mode="intact",
        evidence_depth=4,
        prompt=_prompt(),
        completions=completions,
        target=(0.0, 0.0, 0.0, 1.0),
        truth_index=3,
        metadata={},
    )
    result = run_case(backend, case)
    assert result.consistent_mass > 0.99
    assert result.top1_consistent
    assert result.truth_rank == 1


def test_action_case_builder_emits_intact_amnesic_and_shuffled_controls() -> None:
    cases = build_action_cases((990_001,))
    assert cases
    assert {case.mode for case in cases} == {"intact", "amnesic", "shuffled"}
    assert {case.evidence_depth for case in cases} == {0, 1, 2, 4}
    assert all(len(case.completions) == 4 for case in cases)
    assert all(sum(case.target) == pytest.approx(1.0) for case in cases)


def test_summary_reports_intact_minus_control_deltas() -> None:
    completions = (" north", " south", " west", " east")
    results = []
    for mode, selected in (("intact", " east"), ("amnesic", " north"), ("shuffled", " west")):
        case = Case(
            case_id=mode,
            task="action_binding",
            mode=mode,
            evidence_depth=4,
            prompt=_prompt(),
            completions=completions,
            target=(0.0, 0.0, 0.0, 1.0),
            truth_index=3,
            metadata={},
        )
        results.append(run_case(MockBackend(scores={selected: 8.0}), case))
    report = summarize(results)
    delta = report["evidence_deltas"]["action_binding"]
    assert delta["intact_minus_amnesic"] > 0.9
    assert delta["intact_minus_shuffled"] > 0.9
