from __future__ import annotations

import pytest

from arcgpt2.dsl import (
    DSLError,
    Program,
    Status,
    canonical_program,
    disagreement_partition,
    entropy_of_partition,
    enumerate_phase0_programs,
    execute,
    filter_consistent,
    most_informative_action,
    parse_program,
    program_from_phase0_spec,
    replay,
    roundtrip_program,
    shortest_plan,
)
from arcgpt2.phase0_hidden_action import Action, HiddenActionGame, generate_game, simulate_source_history


def test_phase0_program_roundtrips_canonically() -> None:
    program = program_from_phase0_spec(generate_game(127))
    restored = roundtrip_program(program)
    assert restored == program
    assert canonical_program(restored) == canonical_program(program)
    assert parse_program(canonical_program(program)) == program


def test_ground_truth_program_exactly_replays_many_games() -> None:
    for seed in range(100):
        spec = generate_game(50_000 + seed)
        program = program_from_phase0_spec(spec)
        records = simulate_source_history(spec)
        result = replay(program, records)
        assert result.consistent
        assert result.checked == len(records)
        assert result.mismatch is None


def test_all_24_hidden_action_programs_are_unique() -> None:
    spec = generate_game(8)
    programs = enumerate_phase0_programs(spec)
    assert len(programs) == 24
    assert len({program.sha256 for program in programs}) == 24


def test_observations_eliminate_action_mapping_hypotheses() -> None:
    spec = generate_game(91)
    all_programs = enumerate_phase0_programs(spec)
    records = simulate_source_history(spec)

    live = all_programs
    expected_counts = [6, 2, 1, 1]
    for index, expected in enumerate(expected_counts):
        live = filter_consistent(live, records[: index + 1])
        assert len(live) == expected

    assert live[0] == program_from_phase0_spec(spec)


def test_initial_actions_have_two_bits_of_mapping_information() -> None:
    spec = generate_game(93)
    programs = enumerate_phase0_programs(spec)
    game = HiddenActionGame(spec)

    for action in Action:
        groups = disagreement_partition(programs, game.frame, action)
        assert sorted(len(group) for group in groups.values()) == [6, 6, 6, 6]
        assert entropy_of_partition(groups) == pytest.approx(2.0)

    action, information = most_informative_action(programs, game.frame)
    assert action is Action.A1
    assert information == pytest.approx(2.0)


def test_wrong_program_is_rejected_with_a_concrete_counterexample() -> None:
    spec = generate_game(101)
    records = simulate_source_history(spec)
    truth = program_from_phase0_spec(spec)
    wrong = next(program for program in enumerate_phase0_programs(spec) if program != truth)
    result = replay(wrong, records)
    assert not result.consistent
    assert result.mismatch is not None
    assert result.mismatch.differing_cells > 0 or (
        result.mismatch.expected_terminal != result.mismatch.predicted_terminal
    )


def test_program_planning_reaches_terminal_state() -> None:
    spec = generate_game(111)
    game = HiddenActionGame(spec)
    program = program_from_phase0_spec(spec)
    plan = shortest_plan(program, game.frame, max_depth=32)
    assert plan

    state = game.frame
    terminal = False
    for action in plan:
        result = execute(program, state, action)
        state = result.after
        terminal = result.status is Status.WIN
    assert terminal


def test_parser_rejects_noncanonical_or_unsafe_programs() -> None:
    with pytest.raises(DSLError):
        parse_program("ARC-DSL 1\nEND")
    with pytest.raises(DSLError):
        parse_program(
            "ARC-DSL 1\n"
            "RULE A1 MOVE C2 DY 0 DX 0 BG C0 BLOCK C1 WIN C3 ;\n"
            "END"
        )
    with pytest.raises(DSLError):
        parse_program(
            "ARC-DSL 1\n"
            "RULE A9 MOVE C2 DY 1 DX 0 BG C0 BLOCK C1 WIN C3 ;\n"
            "END"
        )
