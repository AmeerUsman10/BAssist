from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from arcgpt2.official_public_score import (
    EXPECTED_ARC_AGI_VERSION,
    EXPECTED_GAME_IDS,
    EXPECTED_MANIFEST_CANONICAL_SHA256,
    EXPECTED_MANIFEST_SHA256,
    MAX_ACTIONS_PER_GAME,
    OfficialPublicScoreError,
    load_public_manifest,
    run_official_public_score,
    run_one_game,
    sanitize_scorecard,
    validate_environment_ids,
    validate_scorecard_consistency,
    validate_sdk_version,
)


CARD_SECRET = "CARD-SECRET-93dd"
REGISTERED_SECRET = "registered-key-must-never-escape"
ANONYMOUS_SECRET = "anonymous-key-must-never-escape"


class OperationMode(Enum):
    NORMAL = "normal"
    ONLINE = "online"
    COMPETITION = "competition"


class GameState(Enum):
    IN_PROGRESS = 1
    NOT_FINISHED = 2
    GAME_OVER = 3
    WIN = 4


@dataclass(frozen=True)
class FakeAction:
    name: str
    simple: bool = True

    def is_simple(self) -> bool:
        return self.simple


RESET = FakeAction("RESET")
ACTION1 = FakeAction("ACTION1")
CLICK = FakeAction("CLICK", simple=False)


def _frame(
    game_id: str,
    *,
    state: GameState = GameState.IN_PROGRESS,
    complex_action: bool = False,
) -> Any:
    grid = [[0, 0, 0], [0, 7 if complex_action else 0, 0], [0, 0, 0]]
    actions = [RESET, CLICK] if complex_action else [RESET, ACTION1]
    return SimpleNamespace(
        game_id=game_id,
        state=state,
        levels_completed=int(state is GameState.WIN),
        win_levels=1,
        full_reset=False,
        available_actions=actions,
        frame=[grid],
    )


class FakeEnvironment:
    def __init__(self, owner: "FakeArcade", game_id: str, behavior: str) -> None:
        self.owner = owner
        self.game_id = game_id
        self.behavior = behavior
        self.complex_action = behavior == "complex"
        frame_game_id = (
            EXPECTED_GAME_IDS[-1] if behavior == "wrong_game_id" else game_id
        )
        self.observation_space = _frame(
            frame_game_id, complex_action=self.complex_action
        )
        self.action_space = [RESET, CLICK] if self.complex_action else [RESET, ACTION1]
        self.step_calls: list[tuple[str, dict[str, int]]] = []
        self.reset_calls = 0

    def step(self, action: FakeAction, data=None):
        payload = dict(data or {})
        self.step_calls.append((action.name, payload))
        if self.behavior == "fail":
            raise RuntimeError(f"{REGISTERED_SECRET} {CARD_SECRET}")
        if self.behavior == "loop":
            return self.observation_space
        if self.behavior == "sticky_game_over":
            if self.reset_calls == 0:
                self.observation_space = _frame(
                    self.game_id, state=GameState.GAME_OVER
                )
            return self.observation_space
        if self.behavior == "reset_then_win" and len(self.step_calls) == 1:
            self.observation_space = _frame(
                self.game_id, state=GameState.GAME_OVER
            )
            return self.observation_space
        if self.complex_action:
            assert action == CLICK
            assert payload == {"x": 1, "y": 1}
        else:
            assert action == ACTION1
            assert payload == {}
        self.observation_space = _frame(
            self.game_id,
            state=GameState.WIN,
            complex_action=self.complex_action,
        )
        return self.observation_space

    def reset(self):
        self.reset_calls += 1
        if self.behavior in {"reset_then_win", "sticky_game_over"}:
            state = (
                GameState.NOT_FINISHED
                if self.behavior == "sticky_game_over"
                else GameState.IN_PROGRESS
            )
            self.observation_space = _frame(self.game_id, state=state)
        return self.observation_space


class RawScorecard:
    def __init__(self, owner: "FakeArcade") -> None:
        self.owner = owner

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        assert mode == "python"
        environments: list[dict[str, Any]] = []
        for game_id, environment in self.owner.environments.items():
            state = environment.observation_space.state.name
            if (
                environment.behavior == "sticky_game_over"
                and environment.reset_calls > 0
                and state == "NOT_FINISHED"
            ):
                # Exact arc-agi 0.9.9 Card regression: reset increments
                # counters, but NOT_FINISHED does not clear GAME_OVER.
                state = "GAME_OVER"
            completed = state == "WIN"
            actions = len(environment.step_calls) + environment.reset_calls
            resets = environment.reset_calls
            levels_completed = environment.observation_space.levels_completed
            run_score = 12.5 if completed else 0.0
            run = {
                "id": game_id,
                "guid": "guid-must-never-escape",
                "score": run_score,
                "levels_completed": levels_completed,
                "actions": actions,
                "resets": resets,
                "state": state,
                "completed": completed,
                "level_scores": [run_score],
                "level_actions": [actions],
                "level_baseline_actions": [1],
                "number_of_levels": None,
                "number_of_environments": 0,
                "message": REGISTERED_SECRET,
            }
            environments.append(
                {
                    "id": game_id,
                    "runs": [run],
                    "score": run_score,
                    "actions": actions,
                    "levels_completed": levels_completed,
                    "completed": completed,
                    "level_count": 1,
                    "resets": resets,
                }
            )
        score = (
            sum(float(environment["score"]) for environment in environments)
            / len(environments)
            if environments
            else 0.0
        )
        payload = {
            "score": score,
            "environments": environments,
            "competition_mode": self.owner.scorecard_competition_mode,
            "total_environments": len(environments),
            "total_environments_completed": sum(
                int(environment["completed"]) for environment in environments
            ),
            "total_levels_completed": sum(
                int(environment["levels_completed"]) for environment in environments
            ),
            "total_levels": len(environments),
            "total_actions": sum(
                int(environment["actions"]) for environment in environments
            ),
            "api_key": ANONYMOUS_SECRET,
            "hash": "raw-hash-must-never-escape",
            "card_id": CARD_SECRET,
            "guid": "guid-must-never-escape",
            "raw_scorecard": {"payload": REGISTERED_SECRET},
            "opaque": REGISTERED_SECRET,
        }
        if self.owner.scorecard_mutator is not None:
            self.owner.scorecard_mutator(payload)
        return payload


class FakeArcade:
    def __init__(
        self,
        *,
        environment_ids: tuple[str, ...],
        behaviors: dict[str, str] | None,
        close_failure: bool,
        scorecard_competition_mode: bool | None,
        scorecard_mutator: Callable[[dict[str, Any]], None] | None,
        operation_mode: OperationMode,
        arc_api_key: str,
        arc_base_url: str,
        environments_dir: str,
        recordings_dir: str,
        logger: Any,
    ) -> None:
        # Deliberately emit values that must be swallowed by the driver.
        print(ANONYMOUS_SECRET)
        logger.critical(REGISTERED_SECRET)
        self.operation_mode = operation_mode
        self.environment_ids = environment_ids
        self.behaviors = behaviors or {}
        self.close_failure = close_failure
        self.scorecard_competition_mode = scorecard_competition_mode
        self.scorecard_mutator = scorecard_mutator
        self.constructor = {
            "operation_mode": operation_mode,
            "arc_api_key": arc_api_key,
            "arc_base_url": arc_base_url,
            "environments_dir": environments_dir,
            "recordings_dir": recordings_dir,
            "env_operation_mode": os.environ.get("OPERATION_MODE"),
            "env_arc_api_key": os.environ.get("ARC_API_KEY"),
            "env_arc_base_url": os.environ.get("ARC_BASE_URL"),
            "env_environments_dir": os.environ.get("ENVIRONMENTS_DIR"),
            "env_recordings_dir": os.environ.get("RECORDINGS_DIR"),
            "competition_rerun_present": "KAGGLE_IS_COMPETITION_RERUN" in os.environ,
            "only_reset_levels_present": "ONLY_RESET_LEVELS" in os.environ,
        }
        Path(environments_dir, "sdk-environment-marker").write_text("isolated")
        Path(recordings_dir, "sdk-recording-marker").write_text("isolated")
        self.open_calls = 0
        self.close_calls = 0
        self.get_scorecard_calls = 0
        self.make_calls: list[tuple[str, dict[str, Any]]] = []
        self.environments: dict[str, FakeEnvironment] = {}

    def get_environments(self):
        return [SimpleNamespace(game_id=game_id) for game_id in self.environment_ids]

    def open_scorecard(self, **kwargs):
        self.open_calls += 1
        assert kwargs == {"tags": ["official_public_score_v1"]}
        return CARD_SECRET

    def get_scorecard(self, *_args, **_kwargs):
        self.get_scorecard_calls += 1
        raise AssertionError("in-flight scorecard access is forbidden")

    def make(self, game_id: str, **kwargs):
        self.make_calls.append((game_id, dict(kwargs)))
        assert kwargs == {
            "seed": 0,
            "scorecard_id": CARD_SECRET,
            "save_recording": False,
            "include_frame_data": True,
            "render_mode": None,
        }
        environment = FakeEnvironment(
            self, game_id, self.behaviors.get(game_id, "win")
        )
        self.environments[game_id] = environment
        return environment

    def close_scorecard(self, scorecard_id: str):
        self.close_calls += 1
        assert scorecard_id == CARD_SECRET
        if self.close_failure:
            raise RuntimeError(f"close failed: {CARD_SECRET} {ANONYMOUS_SECRET}")
        return RawScorecard(self)


def _fake_sdk(
    environment_ids: tuple[str, ...] = tuple(reversed(EXPECTED_GAME_IDS)),
    *,
    behaviors: dict[str, str] | None = None,
    close_failure: bool = False,
    scorecard_competition_mode: bool | None = None,
    scorecard_mutator: Callable[[dict[str, Any]], None] | None = None,
):
    holder: dict[str, FakeArcade] = {}

    def arcade_factory(
        *,
        operation_mode,
        arc_api_key,
        arc_base_url,
        environments_dir,
        recordings_dir,
        logger,
    ):
        arcade = FakeArcade(
            environment_ids=environment_ids,
            behaviors=behaviors,
            close_failure=close_failure,
            scorecard_competition_mode=scorecard_competition_mode,
            scorecard_mutator=scorecard_mutator,
            operation_mode=operation_mode,
            arc_api_key=arc_api_key,
            arc_base_url=arc_base_url,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            logger=logger,
        )
        holder["arcade"] = arcade
        return arcade

    sdk = SimpleNamespace(
        __version__=EXPECTED_ARC_AGI_VERSION,
        OperationMode=OperationMode,
        Arcade=arcade_factory,
    )
    return sdk, holder


class RecordingProvider:
    def __init__(self, instances: list["RecordingProvider"]) -> None:
        instances.append(self)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def observe(self, action_name, before_grid, after_grid, status):
        del action_name, before_grid, after_grid, status
        return {"api_key": ANONYMOUS_SECRET, "observed": True}

    def direction_scores(self, action_name):
        del action_name
        return {"opaque": 1.0}


class FailingProvider(RecordingProvider):
    def reset(self) -> None:
        self.reset_calls += 1
        raise RuntimeError(REGISTERED_SECRET)


def _json(receipt: Any) -> str:
    return json.dumps(receipt, sort_keys=True, allow_nan=False)


def _all_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_checked_in_manifest_is_byte_and_content_exact() -> None:
    manifest = load_public_manifest()

    assert manifest.raw_sha256 == EXPECTED_MANIFEST_SHA256
    assert manifest.canonical_sha256 == EXPECTED_MANIFEST_CANONICAL_SHA256
    assert manifest.game_ids == EXPECTED_GAME_IDS
    assert len(manifest.game_ids) == len(set(manifest.game_ids)) == 25
    assert manifest.max_actions_per_game == 160
    assert manifest.max_resets_per_game == 3


def test_manifest_byte_drift_is_rejected(tmp_path: Path) -> None:
    original = Path(__file__).parents[1] / "official_public_manifest.json"
    changed = tmp_path / "manifest.json"
    changed.write_bytes(original.read_bytes() + b"\n")

    with pytest.raises(OfficialPublicScoreError, match="frozen digest") as error:
        load_public_manifest(changed)
    assert error.value.code == "manifest_digest_mismatch"


@pytest.mark.parametrize(
    ("environment_ids", "code"),
    [
        (EXPECTED_GAME_IDS[:-1], "environment_count_mismatch"),
        ((*EXPECTED_GAME_IDS, "zz99-deadbeef"), "environment_count_mismatch"),
        ((*EXPECTED_GAME_IDS[:-1], EXPECTED_GAME_IDS[0]), "duplicate_environment_id"),
        (("bp35-deadbeef", *EXPECTED_GAME_IDS[1:]), "environment_set_mismatch"),
    ],
)
def test_24_26_duplicate_and_version_drift_fail_before_scorecard(
    environment_ids: tuple[str, ...], code: str
) -> None:
    sdk, holder = _fake_sdk(environment_ids)
    receipt = run_official_public_score(sdk_module=sdk)
    arcade = holder["arcade"]

    assert receipt["aggregate"]["status"] == "partial"
    assert receipt["aggregate"]["valid"] is False
    assert receipt["errors"] == [{"phase": "runtime_preflight", "code": code}]
    assert receipt["counts"]["attempted_games"] == 0
    assert arcade.open_calls == 0
    assert arcade.make_calls == []
    assert arcade.close_calls == 0


def test_validators_are_strict_and_accept_discovery_order_variance() -> None:
    discovered = validate_environment_ids(
        [SimpleNamespace(game_id=value) for value in reversed(EXPECTED_GAME_IDS)]
    )
    assert discovered == tuple(reversed(EXPECTED_GAME_IDS))
    validate_sdk_version(EXPECTED_ARC_AGI_VERSION)

    with pytest.raises(OfficialPublicScoreError) as error:
        validate_sdk_version("0.9.8")
    assert error.value.code == "sdk_version_mismatch"


def test_sdk_version_drift_is_partial_without_constructing_arcade() -> None:
    sdk, holder = _fake_sdk()
    receipt = run_official_public_score(sdk_module=sdk, sdk_version="0.9.8")

    assert receipt["aggregate"] == {
        "valid": False,
        "status": "partial",
        "official_score": None,
        "observed_score": None,
        "observed_score_label": "unavailable",
        "reasons": ["sdk_version_mismatch"],
    }
    assert holder == {}


def test_full_run_is_normal_one_scorecard_manifest_order_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPERATION_MODE", "COMPETITION")
    monkeypatch.setenv("ARC_API_KEY", REGISTERED_SECRET)
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    monkeypatch.setenv("ONLY_RESET_LEVELS", "1")
    monkeypatch.setenv("ARC_BASE_URL", "https://attacker.invalid")
    monkeypatch.setenv("ENVIRONMENTS_DIR", str(tmp_path / "ambient-environments"))
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path / "ambient-recordings"))
    behaviors = {EXPECTED_GAME_IDS[0]: "complex"}
    sdk, holder = _fake_sdk(behaviors=behaviors)
    providers: list[RecordingProvider] = []

    receipt = run_official_public_score(
        sdk_module=sdk,
        provider_factory=lambda: RecordingProvider(providers),
        temp_root=tmp_path,
    )
    arcade = holder["arcade"]
    serialized = _json(receipt)

    assert receipt["aggregate"] == {
        "valid": True,
        "status": "valid",
        "official_score": 12.5,
        "observed_score": 12.5,
        "observed_score_label": "official_valid",
        "reasons": [],
    }
    assert receipt["counts"] == {
        "expected_games": 25,
        "attempted_games": 25,
        "completed_games": 25,
        "failed_games": 0,
        "provider_factory_failures": 0,
        "provider_runtime_failures": 0,
    }
    assert [game_id for game_id, _ in arcade.make_calls] == list(EXPECTED_GAME_IDS)
    assert [game["game_id"] for game in receipt["games"]] == list(EXPECTED_GAME_IDS)
    assert all(game["make_calls"] == 1 for game in receipt["games"])
    assert arcade.open_calls == arcade.close_calls == 1
    assert arcade.get_scorecard_calls == 0
    assert arcade.constructor == {
        "operation_mode": OperationMode.NORMAL,
        "arc_api_key": "",
        "arc_base_url": "https://three.arcprize.org",
        "environments_dir": str(tmp_path / "official-public-environments"),
        "recordings_dir": str(tmp_path / "official-public-recordings"),
        "env_operation_mode": "NORMAL",
        "env_arc_api_key": "",
        "env_arc_base_url": "https://three.arcprize.org",
        "env_environments_dir": str(tmp_path / "official-public-environments"),
        "env_recordings_dir": str(tmp_path / "official-public-recordings"),
        "competition_rerun_present": False,
        "only_reset_levels_present": False,
    }
    # The process environment is restored after the protected SDK section.
    assert os.environ["OPERATION_MODE"] == "COMPETITION"
    assert os.environ["ARC_API_KEY"] == REGISTERED_SECRET
    assert os.environ["KAGGLE_IS_COMPETITION_RERUN"] == "1"
    assert os.environ["ONLY_RESET_LEVELS"] == "1"
    assert os.environ["ARC_BASE_URL"] == "https://attacker.invalid"
    assert os.environ["ENVIRONMENTS_DIR"] == str(tmp_path / "ambient-environments")
    assert os.environ["RECORDINGS_DIR"] == str(tmp_path / "ambient-recordings")
    assert (tmp_path / "official-public-environments" / "sdk-environment-marker").is_file()
    assert (tmp_path / "official-public-recordings" / "sdk-recording-marker").is_file()

    complex_env = arcade.environments[EXPECTED_GAME_IDS[0]]
    assert complex_env.step_calls == [("CLICK", {"x": 1, "y": 1})]
    assert len(providers) == 25
    assert all(provider.reset_calls == 1 for provider in providers)
    assert receipt["games"][0]["provider_assignment"] == {
        "source": "factory",
        "factory_call_index": 1,
        "assigned": True,
    }
    assert receipt["games"][-1]["provider_assignment"]["factory_call_index"] == 25
    assert receipt["games"][0]["provider"] == {
        "enabled": True,
        "reset_calls": 1,
        "failures": [],
        "fallback_used": False,
    }
    assert receipt["execution"] == {
        "mode": "sequential",
        "manifest_order_preserved": True,
        "shared_scorecard_concurrency_validated": False,
        "run_one_game_primitive_available": True,
        "isolated_runtime_directories": True,
        "temp_root_provided": True,
    }

    sanitized = receipt["scorecard"]["sanitized"]
    assert sanitized["score"] == 12.5
    assert sanitized["competition_mode"] is False
    assert sanitized["competition_mode_sdk_raw_was_null"] is True
    assert sanitized["environments"][0]["runs"][0]["actions"] == 1
    assert receipt["scorecard"]["validation"]["valid"] is True
    assert receipt["scorecard"]["validation"]["environment_ids_exact"] is True
    assert receipt["scorecard"]["validation"]["one_run_each"] is True
    assert receipt["scorecard"]["validation"]["trace_scorecard_consistent"] is True
    assert receipt["scorecard"]["validation"][
        "competition_mode_sdk_raw_was_null"
    ] is True
    assert receipt["authentication"] == {
        "mode": "anonymous_public",
        "ambient_credential_ignored": True,
        "credential_material_persisted": False,
    }
    assert all(
        forbidden not in serialized
        for forbidden in (
            CARD_SECRET,
            REGISTERED_SECRET,
            ANONYMOUS_SECRET,
            "credential-hash-must-never-escape",
            "raw-hash-must-never-escape",
            "guid-must-never-escape",
        )
    )
    assert not {
        "api_key",
        "card_id",
        "guid",
        "key",
        "key_hash",
        "raw_scorecard",
        "scorecard_id",
    }.intersection(_all_keys(receipt))
    captured = capsys.readouterr()
    assert ANONYMOUS_SECRET not in captured.out + captured.err
    assert REGISTERED_SECRET not in captured.out + captured.err


def test_provider_failure_is_exposed_and_invalidates_intended_model_score() -> None:
    sdk, _holder = _fake_sdk()
    providers: list[RecordingProvider] = []

    receipt = run_official_public_score(
        sdk_module=sdk,
        provider_factory=lambda: FailingProvider(providers),
    )

    assert receipt["counts"]["completed_games"] == 25
    assert receipt["counts"]["provider_runtime_failures"] == 25
    assert receipt["checks"]["provider_runtime_failure_free"] is False
    provider = receipt["games"][0]["provider"]
    assert provider["enabled"] is True
    assert provider["reset_calls"] == 1
    assert provider["fallback_used"] is True
    assert len(provider["failures"]) == 1
    assert provider["failures"][0]["phase"] == "reset"
    assert provider["failures"][0]["error_type"] == "RuntimeError"
    assert provider["failures"][0]["fallback"] == "exact_state_frontier"
    assert "error" not in provider["failures"][0]
    assert receipt["aggregate"]["valid"] is False
    assert receipt["aggregate"]["official_score"] is None
    assert receipt["aggregate"]["observed_score"] == 12.5
    assert receipt["aggregate"]["observed_score_label"] == "partial_not_promoted"
    assert "provider_runtime_failure" in receipt["aggregate"]["reasons"]
    assert REGISTERED_SECRET not in _json(receipt)


def test_factory_returning_none_is_an_explicit_assignment_fallback() -> None:
    sdk, _holder = _fake_sdk()

    receipt = run_official_public_score(
        sdk_module=sdk,
        provider_factory=lambda: None,
    )

    assert receipt["counts"]["provider_factory_failures"] == 25
    assert receipt["games"][0]["provider_assignment"] == {
        "source": "factory",
        "factory_call_index": 1,
        "assigned": False,
    }
    assert receipt["games"][0]["provider_factory"] == {
        "status": "fallback",
        "error_code": "provider_factory_returned_none",
    }
    assert receipt["games"][0]["provider"]["enabled"] is False
    assert receipt["aggregate"]["valid"] is False
    assert "provider_factory_failure" in receipt["aggregate"]["reasons"]


def test_nonterminating_game_is_bounded_at_exactly_160_actions() -> None:
    loop_game = EXPECTED_GAME_IDS[0]
    sdk, holder = _fake_sdk(behaviors={loop_game: "loop"})

    receipt = run_official_public_score(sdk_module=sdk)
    environment = holder["arcade"].environments[loop_game]
    game = receipt["games"][0]

    assert len(environment.step_calls) == MAX_ACTIONS_PER_GAME == 160
    assert game["status"] == "completed"
    assert game["receipt"]["actions_taken"] == 160
    assert game["receipt"]["stop_reason"] == "action_budget_exhausted"
    # A bounded, completed playthrough need not win for the scorecard itself to
    # be a valid measurement (zero is explicitly valid in the protocol).
    assert receipt["aggregate"]["valid"] is True


def test_game_failure_still_closes_once_and_produces_explicit_partial() -> None:
    failed_game = EXPECTED_GAME_IDS[0]
    sdk, holder = _fake_sdk(behaviors={failed_game: "fail"})

    receipt = run_official_public_score(sdk_module=sdk)
    arcade = holder["arcade"]
    serialized = _json(receipt)

    assert len(arcade.make_calls) == 25
    assert arcade.open_calls == arcade.close_calls == 1
    assert receipt["counts"]["completed_games"] == 24
    assert receipt["games"][0]["error"] == {
        "phase": "game_run",
        "code": "game_run_failed",
    }
    assert receipt["scorecard"]["close_succeeded"] is True
    assert receipt["aggregate"]["status"] == "partial"
    assert receipt["aggregate"]["official_score"] is None
    assert "game_runs_incomplete" in receipt["aggregate"]["reasons"]
    assert REGISTERED_SECRET not in serialized
    assert CARD_SECRET not in serialized


def test_close_failure_invalidates_otherwise_complete_score() -> None:
    sdk, holder = _fake_sdk(close_failure=True)

    receipt = run_official_public_score(sdk_module=sdk)

    assert holder["arcade"].close_calls == 1
    assert receipt["counts"]["completed_games"] == 25
    assert receipt["scorecard"]["opened"] is True
    assert receipt["scorecard"]["close_attempted"] is True
    assert receipt["scorecard"]["close_succeeded"] is False
    assert receipt["scorecard"]["sanitized"] == {}
    assert receipt["scorecard"]["validation"]["valid"] is False
    assert receipt["aggregate"]["valid"] is False
    assert receipt["aggregate"]["status"] == "partial"
    assert receipt["aggregate"]["official_score"] is None
    assert "scorecard_close_failed" in receipt["aggregate"]["reasons"]


def test_competition_scorecard_is_never_promoted() -> None:
    sdk, _holder = _fake_sdk(scorecard_competition_mode=True)

    receipt = run_official_public_score(sdk_module=sdk)

    validation = receipt["scorecard"]["validation"]
    assert validation["schema_valid"] is True
    assert validation["trace_scorecard_consistent"] is True
    assert validation["competition_mode_normal"] is False
    assert validation["valid"] is False
    assert receipt["aggregate"]["valid"] is False
    assert "scorecard_competition_mode_invalid" in receipt["aggregate"]["reasons"]


def _drop_last_scorecard_environment(payload: dict[str, Any]) -> None:
    payload["environments"].pop()


def _duplicate_first_scorecard_run(payload: dict[str, Any]) -> None:
    run = dict(payload["environments"][0]["runs"][0])
    payload["environments"][0]["runs"].append(run)


def _change_first_scorecard_action_count(payload: dict[str, Any]) -> None:
    environment = payload["environments"][0]
    environment["runs"][0]["actions"] += 1
    environment["runs"][0]["level_actions"] = [
        environment["runs"][0]["actions"]
    ]
    environment["actions"] += 1
    payload["total_actions"] += 1


def _change_aggregate_mean(payload: dict[str, Any]) -> None:
    payload["score"] += 1.0


def _make_aggregate_out_of_range(payload: dict[str, Any]) -> None:
    payload["score"] = 101.0


def _remove_optional_run_ids(payload: dict[str, Any]) -> None:
    for environment in payload["environments"]:
        environment["runs"][0].pop("id")


@pytest.mark.parametrize(
    ("mutator", "failed_check", "reason"),
    [
        (
            _drop_last_scorecard_environment,
            "environment_ids_exact",
            "scorecard_environment_coverage_invalid",
        ),
        (
            _duplicate_first_scorecard_run,
            "one_run_each",
            "scorecard_run_cardinality_invalid",
        ),
        (
            _change_first_scorecard_action_count,
            "trace_scorecard_consistent",
            "scorecard_trace_mismatch",
        ),
        (
            _change_aggregate_mean,
            "official_score_mean_matches",
            "scorecard_schema_invalid",
        ),
        (
            _make_aggregate_out_of_range,
            "official_score_valid",
            "official_score_invalid",
        ),
    ],
)
def test_malformed_or_trace_inconsistent_scorecard_is_partial(
    mutator: Callable[[dict[str, Any]], None], failed_check: str, reason: str
) -> None:
    sdk, _holder = _fake_sdk(scorecard_mutator=mutator)

    receipt = run_official_public_score(sdk_module=sdk)
    validation = receipt["scorecard"]["validation"]

    assert validation[failed_check] is False
    assert validation["valid"] is False
    assert receipt["aggregate"]["valid"] is False
    assert receipt["aggregate"]["official_score"] is None
    assert reason in receipt["aggregate"]["reasons"]


def test_missing_optional_run_ids_are_accepted_under_environment_ids() -> None:
    sdk, _holder = _fake_sdk(scorecard_mutator=_remove_optional_run_ids)

    receipt = run_official_public_score(sdk_module=sdk)

    assert receipt["scorecard"]["validation"]["environment_ids_exact"] is True
    assert receipt["scorecard"]["validation"]["one_run_each"] is True
    assert receipt["scorecard"]["validation"]["valid"] is True
    assert receipt["aggregate"]["valid"] is True


def test_reset_counters_match_scorecard_and_action_identity() -> None:
    game_id = EXPECTED_GAME_IDS[0]
    sdk, _holder = _fake_sdk(behaviors={game_id: "reset_then_win"})

    receipt = run_official_public_score(sdk_module=sdk)
    game = receipt["games"][0]["receipt"]
    scorecard_run = receipt["scorecard"]["sanitized"]["environments"][0]["runs"][0]

    assert game["step_actions"] == 2
    assert game["resets"] == 1
    assert game["actions_taken"] == game["step_actions"] + game["resets"] == 3
    assert scorecard_run["actions"] == game["actions_taken"]
    assert scorecard_run["resets"] == game["resets"]
    assert receipt["scorecard"]["validation"]["trace_scorecard_consistent"] is True
    assert receipt["aggregate"]["valid"] is True


def test_sdk_099_sticky_game_over_after_level_reset_reconciles_exactly() -> None:
    game_id = EXPECTED_GAME_IDS[0]
    sdk, _holder = _fake_sdk(behaviors={game_id: "sticky_game_over"})

    receipt = run_official_public_score(sdk_module=sdk)
    trace = receipt["games"][0]["receipt"]
    scorecard_run = receipt["scorecard"]["sanitized"]["environments"][0]["runs"][0]
    validation = receipt["scorecard"]["validation"]

    assert trace["final_observation"]["environment_state"] == "NOT_FINISHED"
    assert scorecard_run["state"] == "GAME_OVER"
    assert trace["status"] == "stopped"
    assert trace["stop_reason"] == "action_budget_exhausted"
    assert trace["actions_taken"] == scorecard_run["actions"] == 160
    assert trace["step_actions"] + trace["resets"] == trace["actions_taken"]
    assert trace["resets"] == scorecard_run["resets"] == 1
    assert trace["final_observation"]["levels_completed"] == scorecard_run[
        "levels_completed"
    ] == 0
    assert scorecard_run["completed"] is False
    assert validation["trace_scorecard_consistent"] is True
    assert validation["sticky_game_over_reset_state_accepted"] is True
    assert validation["sticky_game_over_reset_state_accepted_count"] == 1
    assert validation["sticky_game_over_reset_state_accepted_game_ids"] == [game_id]
    assert validation["environment_totals_consistent"] is True
    assert validation["level_arrays_valid"] is True
    assert receipt["checks"]["provider_runtime_failure_free"] is True
    assert receipt["aggregate"]["valid"] is True


def test_sticky_game_over_exception_requires_adjacent_reset_trace_evidence() -> None:
    game_id = EXPECTED_GAME_IDS[0]
    sdk, _holder = _fake_sdk(behaviors={game_id: "sticky_game_over"})
    receipt = run_official_public_score(sdk_module=sdk)
    games = copy.deepcopy(receipt["games"])
    scorecard = receipt["scorecard"]["sanitized"]
    events = games[0]["receipt"]["actions"]
    games[0]["receipt"]["actions"] = [
        event for event in events if event["kind"] != "reset"
    ]

    validation = validate_scorecard_consistency(scorecard, games)

    assert validation["sticky_game_over_reset_state_accepted"] is False
    assert validation["trace_scorecard_consistent"] is False
    assert validation["mismatched_game_ids"] == [game_id]


@pytest.mark.parametrize("counter", ["actions", "resets", "levels_completed"])
def test_sticky_game_over_exception_does_not_relax_counter_gates(counter: str) -> None:
    game_id = EXPECTED_GAME_IDS[0]
    sdk, _holder = _fake_sdk(behaviors={game_id: "sticky_game_over"})
    receipt = run_official_public_score(sdk_module=sdk)
    scorecard = copy.deepcopy(receipt["scorecard"]["sanitized"])
    environment = scorecard["environments"][0]
    run = environment["runs"][0]
    run[counter] += 1
    environment[counter] += 1
    if counter == "actions":
        run["level_actions"][0] += 1
        scorecard["total_actions"] += 1
    elif counter == "levels_completed":
        scorecard["total_levels_completed"] += 1

    validation = validate_scorecard_consistency(scorecard, receipt["games"])

    assert validation["trace_scorecard_consistent"] is False
    assert validation["mismatched_game_ids"] == [game_id]


def test_reported_game_id_must_match_requested_scorecard_environment() -> None:
    game_id = EXPECTED_GAME_IDS[0]
    sdk, _holder = _fake_sdk(behaviors={game_id: "wrong_game_id"})

    receipt = run_official_public_score(sdk_module=sdk)

    assert receipt["games"][0]["receipt"]["reported_game_id"] != game_id
    assert receipt["scorecard"]["validation"]["trace_scorecard_consistent"] is False
    assert receipt["aggregate"]["valid"] is False
    assert "scorecard_trace_mismatch" in receipt["aggregate"]["reasons"]


def test_run_one_game_checks_monotonic_deadline_before_action() -> None:
    environment = FakeEnvironment(
        SimpleNamespace(), EXPECTED_GAME_IDS[0], behavior="win"
    )
    times = iter((0.0, 2.0))

    with pytest.raises(OfficialPublicScoreError) as error:
        run_one_game(
            environment,
            monotonic_deadline=1.0,
            monotonic_clock=lambda: next(times),
        )

    assert error.value.code == "runtime_deadline"
    assert environment.step_calls == []


def test_deadline_between_games_closes_scorecard_and_returns_partial() -> None:
    sdk, holder = _fake_sdk()
    checks = 0

    def abort() -> bool:
        nonlocal checks
        checks += 1
        # Driver preflight, game boundary, run start, first action, then the
        # second game boundary.
        return checks >= 5

    receipt = run_official_public_score(sdk_module=sdk, abort_callback=abort)
    arcade = holder["arcade"]

    assert checks == 5
    assert len(arcade.make_calls) == 1
    assert arcade.open_calls == arcade.close_calls == 1
    assert receipt["scorecard"]["close_succeeded"] is True
    assert receipt["counts"]["attempted_games"] == 1
    assert receipt["counts"]["completed_games"] == 1
    assert receipt["errors"] == [
        {"phase": "runtime_deadline", "code": "runtime_deadline"}
    ]
    assert receipt["checks"]["runtime_deadline_clear"] is False
    assert receipt["aggregate"]["status"] == "partial"
    assert "runtime_deadline" in receipt["aggregate"]["reasons"]


def test_default_sdk_directories_are_ephemeral() -> None:
    sdk, holder = _fake_sdk()

    receipt = run_official_public_score(sdk_module=sdk)
    constructor = holder["arcade"].constructor

    assert receipt["aggregate"]["valid"] is True
    assert receipt["execution"]["temp_root_provided"] is False
    assert not Path(constructor["environments_dir"]).exists()
    assert not Path(constructor["recordings_dir"]).exists()


def test_scorecard_allowlist_rejects_identifiers_and_nonfinite_scores() -> None:
    clean = sanitize_scorecard(
        {
            "score": float("nan"),
            "competition_mode": False,
            "total_environments": 1,
            "total_environments_completed": 0,
            "total_levels_completed": 0,
            "total_levels": 1,
            "total_actions": 1,
            "card_id": CARD_SECRET,
            "api_key": ANONYMOUS_SECRET,
            "guid": "guid-must-never-escape",
            "environments": [
                {
                    "id": EXPECTED_GAME_IDS[0],
                    "runs": [
                        {
                            "id": EXPECTED_GAME_IDS[0],
                            "guid": "guid-must-never-escape",
                            "score": 0.0,
                            "levels_completed": 0,
                            "actions": 1,
                            "resets": 0,
                            "state": "NOT_FINISHED",
                            "completed": False,
                            "level_scores": [0.0],
                            "level_actions": [1],
                            "level_baseline_actions": [1],
                            "message": REGISTERED_SECRET,
                        }
                    ],
                    "score": 0.0,
                    "actions": 1,
                    "levels_completed": 0,
                    "completed": False,
                    "level_count": 1,
                    "resets": 0,
                }
            ],
        }
    )

    assert clean == {
        "competition_mode": False,
        "competition_mode_sdk_raw_was_null": False,
        "environments": [
            {
                "actions": 1,
                "completed": False,
                "id": EXPECTED_GAME_IDS[0],
                "level_count": 1,
                "levels_completed": 0,
                "resets": 0,
                "runs": [
                    {
                        "actions": 1,
                        "completed": False,
                        "id": EXPECTED_GAME_IDS[0],
                        "level_actions": [1],
                        "level_baseline_actions": [1],
                        "level_scores": [0.0],
                        "levels_completed": 0,
                        "resets": 0,
                        "score": 0.0,
                        "state": "NOT_FINISHED",
                    }
                ],
                "score": 0.0,
            }
        ],
        "total_actions": 1,
        "total_environments": 1,
        "total_environments_completed": 0,
        "total_levels": 1,
        "total_levels_completed": 0,
    }
    _json(clean)


def test_import_boundary_loads_neither_arc_agi_nor_torch() -> None:
    root = Path(__file__).parents[2]
    script = """
import json
import sys
import arcgpt2.official_public_score
print(json.dumps({"arc_agi": "arc_agi" in sys.modules, "torch": "torch" in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"arc_agi": False, "torch": False}
