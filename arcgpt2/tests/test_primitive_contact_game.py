from __future__ import annotations

from arcgpt2.mechanics_v2 import ContactMode, filter_consistent_v2
from arcgpt2.phase0_hidden_action import Direction
from arcgpt2.primitive_contact_game import (
    PrimitiveContactGame,
    enumerate_contact_programs,
    generate_contact_game,
    program_from_contact_spec,
    render_contact_level,
    simulate_truth_plan,
)


def test_generator_is_reproducible_and_truth_program_is_in_version_space() -> None:
    for seed in range(12):
        first = generate_contact_game(200_000 + seed)
        second = generate_contact_game(200_000 + seed)
        assert first == second
        candidates = enumerate_contact_programs(first)
        assert len(candidates) == 24 * len(ContactMode)
        truth = program_from_contact_spec(first)
        assert truth in candidates
        assert len({candidate.sha256 for candidate in candidates}) == len(candidates)


def test_laboratory_identifies_mapping_without_touching_the_object() -> None:
    spec = generate_contact_game(200_100)
    game = PrimitiveContactGame(spec)
    initial = game.frame
    records = [game.step(action) for action in spec.probe_order]
    assert all(record.status == "ACTIVE" for record in records)
    assert all(record.contacted_color != spec.palette.interaction for record in records)
    assert game.frame == initial

    observations = tuple(record.as_observation() for record in records)
    surviving = filter_consistent_v2(enumerate_contact_programs(spec), observations)
    # Four unique cardinal probes identify the mapping but leave all five
    # contact modes alive.
    assert len(surviving) == len(ContactMode)
    assert {program.contact_mode for program in surviving} == set(ContactMode)


def test_one_controlled_contact_identifies_the_hidden_contact_mode() -> None:
    spec = generate_contact_game(200_101)
    game = PrimitiveContactGame(spec)
    records = [game.step(action) for action in spec.probe_order]
    right_action = next(
        action
        for action, direction in spec.action_to_direction.items()
        if direction is Direction.RIGHT
    )
    records.append(game.step(right_action))
    contact = game.step(right_action)
    records.append(contact)
    assert contact.contacted_color == spec.palette.interaction

    surviving = filter_consistent_v2(
        enumerate_contact_programs(spec),
        tuple(record.as_observation() for record in records),
    )
    assert surviving == (program_from_contact_spec(spec),)


def test_truth_program_completes_all_generated_levels() -> None:
    for seed in range(10):
        spec = generate_contact_game(200_200 + seed)
        records = simulate_truth_plan(spec, max_actions=256)
        assert records[-1].status == "GAME_WIN"
        assert sum(record.status in {"LEVEL_WIN", "GAME_WIN"} for record in records) == len(
            spec.levels
        )


def test_rendering_contains_exactly_one_agent_goal_and_interaction() -> None:
    spec = generate_contact_game(200_300)
    for level in spec.levels:
        grid = render_contact_level(level, spec.palette)
        flattened = [value for row in grid for value in row]
        assert flattened.count(spec.palette.agent) == 1
        assert flattened.count(spec.palette.goal) == 1
        assert flattened.count(spec.palette.interaction) == len(level.interactions)


def test_contact_modes_are_balanced_over_a_seed_window() -> None:
    counts = {mode: 0 for mode in ContactMode}
    for seed in range(100):
        counts[generate_contact_game(201_000 + seed).contact_mode] += 1
    # This is only a generator regression, not a statistical claim. The bound
    # catches a missing enum branch while keeping unit tests inexpensive.
    assert all(count >= 8 for count in counts.values())
