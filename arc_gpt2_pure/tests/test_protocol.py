from arc_gpt2.curriculum import generate_episode
from arc_gpt2.protocol import (
    action_prompt,
    format_mapping,
    memory_prompt,
    parse_action,
    parse_coordinate,
    parse_mapping,
)


def test_mapping_format_and_parse() -> None:
    text = format_mapping({"A1": "N", "A2": "E", "A4": "W"})
    assert text == "A1=N;A2=E;A3=?;A4=W"
    assert parse_mapping(text) == {
        "A1": "N",
        "A2": "E",
        "A3": "?",
        "A4": "W",
    }


def test_parse_mapping_prefers_latest_unknown_assignment() -> None:
    text = "A1=N;A2=E;A3=N;A4=W then later A3=?"
    assert parse_mapping(text)["A3"] == "?"


def test_parse_action_uses_last_available_action() -> None:
    text = "consider A2, reject A3, final A4"
    assert parse_action(text) == "A4"
    assert parse_action(text, available=["A1", "A2"]) == "A2"
    assert parse_action("no action here") is None


def test_parse_coordinate_bounds() -> None:
    assert parse_coordinate("X 7 Y 3", width=8, height=5) == (7, 3)
    assert parse_coordinate("X=8, Y=3", width=8, height=5) is None


def test_prompts_are_self_contained() -> None:
    episode = generate_episode(19, probe_count=2)
    memory = format_mapping(episode.known_mapping)
    memory_text = memory_prompt(episode.transitions, episode.current_grid)
    action_text = action_prompt(
        episode.transitions,
        episode.current_grid,
        memory=memory,
    )
    assert "[[TASK]] MEMORY" in memory_text
    assert "[[CURRENT]] G[" in memory_text
    assert "[[MEMORY]]" in memory_text
    assert "[[TASK]] ACTION" in action_text
    assert memory in action_text
    assert "[[ACTION]]" in action_text
