from __future__ import annotations

from collections import Counter

import pytest

from arcgpt2.meta_soft_contrastive import counterfactual_probe_records
from arcgpt2.meta_soft_single_binding import (
    corrupted_target,
    deranged_direction,
    make_single_episode,
)
from arcgpt2.meta_soft_binding import transition_prompt
from arcgpt2.phase0_hidden_action import Action, Direction


@pytest.mark.parametrize("action", tuple(Action))
def test_single_binding_counterfactuals_start_from_one_identical_world(
    action: Action,
) -> None:
    """Changing only the hidden mapping must not change the support prompt."""

    episodes = tuple(
        make_single_episode(940_001, variant_index, 24, action)
        for variant_index in range(24)
    )

    assert {episode.direction for episode in episodes} == set(Direction)
    assert {episode.record.action for episode in episodes} == {action}
    assert {episode.record.level_index for episode in episodes} == {0}
    assert {episode.record.step_index for episode in episodes} == {0}
    assert len({episode.record.before for episode in episodes}) == 1
    assert len({transition_prompt(episode.record) for episode in episodes}) == 1

    # All six permutations that bind the queried action to one direction must
    # produce the same exact result, while the four directions remain distinct.
    outcomes_by_direction = {
        direction: {
            episode.record.after
            for episode in episodes
            if episode.direction is direction
        }
        for direction in Direction
    }
    assert all(len(outcomes) == 1 for outcomes in outcomes_by_direction.values())
    assert len({next(iter(outcomes)) for outcomes in outcomes_by_direction.values()}) == 4


def test_direction_derangement_is_balanced_and_has_no_fixed_point() -> None:
    mapping = {direction: deranged_direction(direction) for direction in Direction}

    assert all(source is not target for source, target in mapping.items())
    assert Counter(mapping.values()) == Counter(Direction)


@pytest.mark.parametrize("action", tuple(Action))
def test_corrupted_target_is_the_balanced_injected_counterfactual(
    action: Action,
) -> None:
    episodes = {
        episode.direction: episode
        for episode in (
            make_single_episode(940_002, variant_index, 24, action)
            for variant_index in range(24)
        )
    }
    assert set(episodes) == set(Direction)

    injected_directions: list[Direction] = []
    for true_direction, episode in episodes.items():
        target, injected_direction = corrupted_target(
            episode.record,
            true_direction,
        )
        candidates = counterfactual_probe_records(episode.record)
        expected = candidates[tuple(Direction).index(injected_direction)]

        assert injected_direction is deranged_direction(true_direction)
        assert injected_direction is not true_direction
        assert target.before == episode.record.before
        assert target.action is action
        assert target.after == expected.after
        assert target.after != episode.record.after
        assert target.status == expected.status
        injected_directions.append(injected_direction)

    assert Counter(injected_directions) == Counter(Direction)
