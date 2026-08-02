from __future__ import annotations

from pathlib import Path

from arcgpt2.phase0_hidden_action import Action, Direction
from arcgpt2.stage03_components import (
    action_for_direction,
    build_dataset,
    compose_prompt_natural,
    direct_prompt_natural,
    need_prompt,
    needed_direction,
    render_grid,
    rows_for_game,
    transition_prompt,
)


def test_needed_direction_uses_larger_axis_and_vertical_tie() -> None:
    assert needed_direction((4, 4), (1, 4)) == Direction.UP
    assert needed_direction((4, 4), (4, 8)) == Direction.RIGHT
    assert needed_direction((4, 4), (7, 6)) == Direction.DOWN
    assert needed_direction((4, 4), (2, 2)) == Direction.UP


def test_component_prompts_are_exact_and_natural() -> None:
    before = render_grid(7, 7, (3, 3), (6, 6))
    after = render_grid(7, 7, (2, 3), (6, 6))
    transition = transition_prompt(before, after, Action.A1)
    assert "Before:" in transition and "After:" in transition
    assert "action: one" in transition.lower()
    assert transition.endswith("Answer:")
    assert "north, east, south, or west" in transition
    assert need_prompt(before).endswith("Answer:")


def test_action_composition_matches_mapping() -> None:
    mapping = {
        Action.A1: Direction.LEFT,
        Action.A2: Direction.DOWN,
        Action.A3: Direction.RIGHT,
        Action.A4: Direction.UP,
    }
    assert action_for_direction(mapping, Direction.RIGHT) == Action.A3
    prompt = compose_prompt_natural(mapping, Direction.RIGHT)
    assert "three=right" not in prompt
    assert "three=east" in prompt
    direct = direct_prompt_natural(
        mapping, render_grid(9, 9, (4, 4), (4, 7))
    )
    assert "Memory:" in direct
    assert "three=east" in direct


def test_game_rows_have_balanced_component_shapes() -> None:
    rows = rows_for_game(0, variants=4)
    tasks = [row["task"] for row in rows]
    assert tasks.count("mapping") == 16
    assert tasks.count("need") == 4
    assert tasks.count("compose") == 4
    assert tasks.count("direct") == 4
    assert all(row["target"] in row["candidates"] for row in rows)


def test_dataset_is_reproducible(tmp_path: Path) -> None:
    first = build_dataset(
        tmp_path / "first",
        train_seed_start=0,
        train_games=2,
        validation_seed_start=24,
        validation_games=1,
        test_seed_start=100,
        test_games=1,
        variants=2,
    )
    second = build_dataset(
        tmp_path / "second",
        train_seed_start=0,
        train_games=2,
        validation_seed_start=24,
        validation_games=1,
        test_seed_start=100,
        test_games=1,
        variants=2,
    )
    for split in ("train", "validation", "test"):
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        assert first["splits"][split]["rows"] > 0
