from __future__ import annotations

from arcgpt2.meta_soft_second_order import bounded_make_episode, outcome_only_target
from arcgpt2.phase0_hidden_action import HiddenActionGame, generate_game


def test_outcome_target_does_not_repeat_the_known_action() -> None:
    spec = generate_game(93_001)
    game = HiddenActionGame(spec)
    action = tuple(spec.action_to_direction)[0]
    record = game.step(action)
    text = outcome_only_target(record)
    assert action.value not in text
    assert "terminal success" in text
    assert "grid cell changed" in text or "grid cells changed" in text


def test_second_order_smoke_can_bound_support_history(monkeypatch) -> None:
    monkeypatch.setenv("ARC_GPT2_META_MAX_PROBES", "2")
    episode = bounded_make_episode(93_002, 0, 3)
    assert len(episode.records) == 2
    assert episode.records[0].action != episode.records[1].action


def test_invalid_support_bound_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv("ARC_GPT2_META_MAX_PROBES", "0")
    try:
        bounded_make_episode(93_003, 0, 2)
    except ValueError as exc:
        assert "ARC_GPT2_META_MAX_PROBES" in str(exc)
    else:
        raise AssertionError("invalid support bound should raise ValueError")
