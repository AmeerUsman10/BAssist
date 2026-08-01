from arc_gpt2.codec import apply_delta
from arc_gpt2.curriculum import (
    ACTION_ORDER,
    DIRECTION_ORDER,
    generate_episode,
    move_agent,
    shortest_direction,
)


def test_episode_is_deterministic() -> None:
    first = generate_episode(123456, probe_count=2)
    second = generate_episode(123456, probe_count=2)
    assert first.to_dict() == second.to_dict()


def test_mapping_is_a_permutation() -> None:
    episode = generate_episode(11, probe_count=4)
    assert set(episode.mapping) == set(ACTION_ORDER)
    assert set(episode.mapping.values()) == set(DIRECTION_ORDER)
    assert episode.known_mapping == episode.mapping


def test_full_mapping_selects_shortest_path_action() -> None:
    episode = generate_episode(99, probe_count=4)
    expected = next(
        action
        for action, direction in episode.mapping.items()
        if direction == shortest_direction(episode.current_grid)
    )
    assert episode.decision_kind == "navigate"
    assert episode.target_action == expected


def test_missing_required_direction_selects_unknown_probe() -> None:
    # Search a bounded deterministic range because the observed subset is sampled.
    episode = next(
        candidate
        for seed in range(200)
        if (candidate := generate_episode(seed, probe_count=1)).decision_kind == "probe"
    )
    assert episode.known_mapping[episode.target_action] == "?"
    assert episode.target_action in ACTION_ORDER


def test_target_delta_matches_environment_transition() -> None:
    episode = generate_episode(501, probe_count=3)
    after = move_agent(
        episode.current_grid,
        episode.mapping[episode.target_action],
    )
    assert apply_delta(episode.current_grid, episode.target_delta) == after


def test_probe_transitions_reveal_observed_mapping() -> None:
    episode = generate_episode(881, probe_count=4)
    for transition in episode.transitions:
        expected = move_agent(transition.before, episode.mapping[transition.action])
        assert transition.after == expected
