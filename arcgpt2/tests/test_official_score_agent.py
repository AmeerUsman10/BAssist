from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from types import SimpleNamespace
from typing import Any, Callable

from arcgpt2.official_score_agent import (
    AgentConfig,
    PROTOCOL,
    click_candidates,
    run_official_game,
)


class GameState(Enum):
    NOT_PLAYED = 0
    IN_PROGRESS = 1
    GAME_OVER = 2
    WIN = 3


@dataclass(frozen=True)
class FakeAction:
    name: str
    simple: bool = True

    def is_simple(self) -> bool:
        return self.simple


RESET = FakeAction("RESET")
A = FakeAction("A")
B = FakeAction("B")
CLICK = FakeAction("CLICK", simple=False)


def _grid(value: int, *, size: int = 3) -> list[list[int]]:
    return [[value for _ in range(size)] for _ in range(size)]


def _frame(
    grid: list[list[int]],
    *,
    state: GameState = GameState.IN_PROGRESS,
    levels: int = 0,
    win_levels: int = 1,
    actions: tuple[FakeAction, ...] = (RESET, A, B),
    game_id: str = "fake-game",
) -> Any:
    return SimpleNamespace(
        game_id=game_id,
        state=state,
        levels_completed=levels,
        win_levels=win_levels,
        full_reset=False,
        available_actions=list(actions),
        frame=[grid],
    )


class FakeEnvironment:
    def __init__(
        self,
        initial: Any,
        transition: Callable[[str, dict[str, int], int], Any],
        *,
        actions: tuple[FakeAction, ...] = (RESET, A, B),
        reset_frame: Any | Callable[[int], Any] | None = None,
    ) -> None:
        self.observation_space = initial
        self.action_space = list(actions)
        self._transition = transition
        self._reset_frame = reset_frame if reset_frame is not None else initial
        self.calls: list[tuple[str, dict[str, int]]] = []
        self.reset_calls = 0

    def step(self, action: FakeAction, data=None):
        payload = dict(data or {})
        self.calls.append((action.name, payload))
        observation = self._transition(action.name, payload, len(self.calls))
        self.observation_space = observation
        return observation

    def reset(self):
        self.reset_calls += 1
        self.calls.append(("RESET", {}))
        observation = (
            self._reset_frame(self.reset_calls)
            if callable(self._reset_frame)
            else self._reset_frame
        )
        self.observation_space = observation
        return observation


def test_simple_frontier_learns_reversible_graph_and_reaches_new_state() -> None:
    start = _frame(_grid(0))
    middle = _frame(_grid(1))
    won = _frame(_grid(2), state=GameState.WIN, levels=1)

    state = {"name": "start"}

    def transition(action: str, _data: dict[str, int], _index: int):
        if state["name"] == "start" and action == "A":
            state["name"] = "middle"
            return middle
        if state["name"] == "middle" and action == "B":
            state["name"] = "start"
            return start
        if state["name"] == "start" and action == "B":
            state["name"] = "won"
            return won
        raise AssertionError((state["name"], action))

    env = FakeEnvironment(start, transition)
    receipt = run_official_game(env, config=AgentConfig(max_actions=6))

    assert [name for name, _ in env.calls] == ["A", "B", "B"]
    assert receipt["status"] == "won"
    assert receipt["stop_reason"] == "win"
    assert receipt["actions_taken"] == 3
    assert receipt["graph"]["reversible_edge_count"] == 2
    assert any(edge["reversible"] for edge in receipt["graph"]["edges"])


def test_game_over_reset_counts_against_budget_and_preserves_frontier() -> None:
    start = _frame(_grid(0))
    failed = _frame(_grid(9), state=GameState.GAME_OVER)
    won = _frame(_grid(2), state=GameState.WIN, levels=1)

    attempts = {"failed": False}

    def transition(action: str, _data: dict[str, int], _index: int):
        if action == "A" and not attempts["failed"]:
            attempts["failed"] = True
            return failed
        if action == "B" and attempts["failed"]:
            return won
        raise AssertionError(action)

    env = FakeEnvironment(start, transition, reset_frame=start)
    receipt = run_official_game(
        env,
        config=AgentConfig(max_actions=3, max_resets=1),
    )

    assert env.calls == [("A", {}), ("RESET", {}), ("B", {})]
    assert receipt["status"] == "won"
    assert receipt["actions_taken"] == 3
    assert receipt["step_actions"] == 2
    assert receipt["resets"] == 1
    assert [event["kind"] for event in receipt["actions"]] == [
        "step",
        "reset",
        "step",
    ]


def test_level_change_opens_a_fresh_state_frontier_without_stopping() -> None:
    level_zero = _frame(_grid(0), levels=0, win_levels=2)
    level_one = _frame(_grid(1, size=4), levels=1, win_levels=2)
    won = _frame(_grid(2, size=4), state=GameState.WIN, levels=2, win_levels=2)

    def transition(action: str, _data: dict[str, int], index: int):
        if index == 1:
            assert action == "A"
            return level_one
        assert action == "B"
        return won

    env = FakeEnvironment(level_zero, transition)
    receipt = run_official_game(env, config=AgentConfig(max_actions=4))

    assert [name for name, _ in env.calls] == ["A", "B"]
    assert receipt["level_changes"] == 2
    assert receipt["actions"][0]["level_change"] is True
    assert receipt["actions"][0]["shape_changed"] is True
    assert receipt["actions"][0]["persistent_delta"] is None
    assert receipt["actions"][1]["level_change"] is True
    assert receipt["final_observation"]["levels_completed"] == 2


def test_complex_actions_try_non_modal_components_before_grid_coverage() -> None:
    grid = _grid(0, size=5)
    grid[1][1] = 7
    grid[1][2] = 7
    initial = _frame(grid, actions=(RESET, CLICK))

    def transition(action: str, data: dict[str, int], _index: int):
        assert action == "CLICK"
        return initial

    env = FakeEnvironment(initial, transition, actions=(RESET, CLICK))
    receipt = run_official_game(
        env,
        config=AgentConfig(
            max_actions=2,
            max_resets=0,
            max_click_candidates_per_action=8,
            low_discrepancy_points=6,
        ),
    )

    assert env.calls[0] == ("CLICK", {"x": 1, "y": 1})
    assert receipt["actions"][0]["candidate_source"] == "non_modal_component"
    assert receipt["actions"][1]["candidate_source"] == "low_discrepancy"
    assert all(0 <= data["x"] < 5 and 0 <= data["y"] < 5 for _, data in env.calls)
    candidates = click_candidates(grid, maximum=8, low_discrepancy_points=6)
    assert candidates[0] == (1, 1, "non_modal_component")


def test_policy_and_receipt_are_bit_deterministic_and_ignore_game_id() -> None:
    def build(game_id: str):
        initial = _frame(_grid(0), game_id=game_id)

        def transition(action: str, _data: dict[str, int], index: int):
            value = 1 if action == "A" else 2
            state = GameState.WIN if index == 3 else GameState.IN_PROGRESS
            return _frame(_grid(value), state=state, levels=int(state is GameState.WIN), game_id=game_id)

        return FakeEnvironment(initial, transition)

    first_env = build("receipt-a")
    second_env = build("receipt-a")
    first = run_official_game(first_env, config=AgentConfig(max_actions=3))
    second = run_official_game(second_env, config=AgentConfig(max_actions=3))
    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )

    different_id_env = build("receipt-b")
    different = run_official_game(
        different_id_env, config=AgentConfig(max_actions=3)
    )
    assert first_env.calls == different_id_env.calls
    assert first["initial_observation"]["policy_state_sha256"] == different[
        "initial_observation"
    ]["policy_state_sha256"]
    assert first["reported_game_id"] != different["reported_game_id"]
    assert first["policy_game_id_used"] is False


def test_action_budget_is_strict_for_a_nonterminating_environment() -> None:
    initial = _frame(_grid(0), actions=(RESET, A))

    def transition(_action: str, _data: dict[str, int], _index: int):
        return initial

    env = FakeEnvironment(initial, transition, actions=(RESET, A))
    receipt = run_official_game(
        env,
        config=AgentConfig(max_actions=3, max_resets=0),
    )

    assert len(env.calls) == 3
    assert receipt["actions_taken"] == 3
    assert receipt["stop_reason"] == "action_budget_exhausted"
    assert receipt["checks"]["action_budget_respected"] is True
    assert [event["index"] for event in receipt["actions"]] == [1, 2, 3]


def test_reset_only_initial_state_does_not_start_a_second_scorecard_run() -> None:
    initial = _frame(_grid(0), actions=(RESET,))

    def transition(_action: str, _data: dict[str, int], _index: int):
        raise AssertionError("no ordinary action is available")

    env = FakeEnvironment(initial, transition, actions=(RESET,))
    receipt = run_official_game(
        env,
        config=AgentConfig(max_actions=3, max_resets=3),
    )

    assert env.calls == []
    assert env.reset_calls == 0
    assert receipt["actions_taken"] == 0
    assert receipt["resets"] == 0
    assert receipt["stop_reason"] == "frontier_exhausted"


class FailingProvider:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.observations: list[str] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def observe(self, action_name, before_grid, after_grid, status):
        del before_grid, after_grid, status
        self.observations.append(action_name)
        return {"observed": action_name}

    def direction_scores(self, action_name):
        if action_name == "A":
            raise RuntimeError("provider unavailable")
        return {"UP": 0.8, "DOWN": 0.2}


def test_provider_contract_records_receipts_and_explicit_fallback() -> None:
    initial = _frame(_grid(0))
    won = _frame(_grid(1), state=GameState.WIN, levels=1)

    def transition(_action: str, _data: dict[str, int], _index: int):
        return won

    provider = FailingProvider()
    env = FakeEnvironment(initial, transition)
    receipt = run_official_game(
        env,
        config=AgentConfig(max_actions=1),
        direction_score_provider=provider,
    )

    assert provider.reset_calls == 1
    assert provider.observations == [env.calls[0][0]]
    assert receipt["direction_score_provider_used"] is True
    assert receipt["learned_model_count"] == 1
    assert receipt["provider"]["fallback_used"] is True
    assert receipt["provider"]["failures"] == [
        {
            "index": 1,
            "phase": "direction_scores",
            "action": "A",
            "error_type": "RuntimeError",
            "error": "provider unavailable",
            "fallback": "exact_state_frontier",
        }
    ]
    assert receipt["provider"]["observations"][0]["receipt"] == {
        "observed": env.calls[0][0]
    }


def test_receipt_has_strict_json_types_and_contract_checks() -> None:
    initial = _frame(_grid(0), actions=(RESET, A))
    won = _frame(
        _grid(1),
        state=GameState.WIN,
        levels=1,
        actions=(RESET, A),
    )
    env = FakeEnvironment(
        initial,
        lambda _action, _data, _index: won,
        actions=(RESET, A),
    )
    receipt = run_official_game(env, config=AgentConfig(max_actions=1))

    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == 1
    assert decoded["protocol"] == PROTOCOL
    assert decoded["status"] == "won"
    assert decoded["checks"] == {
        "action_budget_respected": True,
        "consecutive_action_indices": True,
        "exact_final_frame_normalization": True,
        "game_id_excluded_from_policy_state": True,
        "learned_model_count_at_most_one": True,
        "reset_budget_respected": True,
    }
    assert len(decoded["initial_observation"]["policy_state_sha256"]) == 64
    assert len(decoded["final_observation"]["final_grid_sha256"]) == 64
    assert decoded["competition_submission_performed"] is False
