"""Strict CPU driver for the frozen ARC-AGI-3 public-score protocol.

The module intentionally has no import-time dependency on ``arc_agi`` (or on
``torch``).  The official SDK is loaded only by the runtime entry point and is
otherwise duck-typed so the complete scorecard lifecycle can be tested with a
small fake.

This is a public-development measurement, not a competition or leaderboard
submission path.  It owns exactly one local scorecard, makes every frozen game
exactly once in manifest order, and always closes an opened scorecard in a
``finally`` block.  SDK output is suppressed because NORMAL mode may acquire
and log an anonymous API key; neither that key nor any scorecard identifier is
part of the evidence surface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, TypeAlias

from .official_score_agent import (
    AgentConfig,
    DirectionScoreProvider,
    run_official_game,
)


PROTOCOL = "official_public_score_v1"
RECEIPT_SCHEMA_VERSION = 1
EXPECTED_ARC_AGI_VERSION = "0.9.9"
EXPECTED_MANIFEST_SHA256 = (
    "77b441ebbba044653c6abe64d17ab3023e54fd1ed6fec732bf0c525614e2e7dc"
)
EXPECTED_MANIFEST_CANONICAL_SHA256 = (
    "d050ea26ba2df1f49d620744517560b254d6ec80a20013a28267983d833bf065"
)
EXPECTED_GAME_IDS = (
    "bp35-0a0ad940",
    "ar25-0c556536",
    "sc25-635fd71a",
    "sk48-d8078629",
    "tu93-0768757b",
    "su15-1944f8ab",
    "ls20-9607627b",
    "tr87-cd924810",
    "m0r0-492f87ba",
    "ft09-0d8bbf25",
    "s5i5-18d95033",
    "dc22-fdcac232",
    "lf52-271a04aa",
    "vc33-5430563c",
    "lp85-305b61c3",
    "sp80-589a99af",
    "sb26-7fbdac44",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "g50t-5849a774",
    "re86-8af5384d",
    "wa30-ee6fef47",
    "ka59-38d34dbb",
    "r11l-495a7899",
    "tn36-ef4dde99",
)

MAX_ACTIONS_PER_GAME = 160
MAX_RESETS_PER_GAME = 3
SEED = 0
OFFICIAL_ARC_BASE_URL = "https://three.arcprize.org"

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
ProviderFactory: TypeAlias = Callable[[], DirectionScoreProvider | None]
AbortCallback: TypeAlias = Callable[[], bool]
MonotonicClock: TypeAlias = Callable[[], float]

_VERSIONED_GAME_ID = re.compile(r"^[a-z0-9]{4}-[0-9a-f]{8}$")
_SAFE_ENV_SET = {
    # Constructor arguments take precedence in the pinned SDK, but pinning the
    # environment too prevents dotenv or wrapper code from selecting ONLINE or
    # COMPETITION before Arcade applies its explicit argument.
    "OPERATION_MODE": "NORMAL",
    # Never inherit a user's registered key.  NORMAL may acquire an anonymous
    # key; its logger/stdout are suppressed below and it is never inspected.
    "ARC_API_KEY": "",
    "ARC_BASE_URL": OFFICIAL_ARC_BASE_URL,
}
_SAFE_ENV_REMOVE = (
    "ARC_COMPETITION_MODE",
    "ARC_ONLINE_MODE",
    "COMPETITION_MODE",
    "KAGGLE_IS_COMPETITION_RERUN",
    "KAGGLE_KERNEL_RUN_TYPE",
    "ONLINE_MODE",
    "ONLY_RESET_LEVELS",
)

_FORBIDDEN_RECEIPT_KEYS = {
    "api_key",
    "apikey",
    "arc_api_key",
    "access_key",
    "anonymous_key",
    "card_id",
    "credential",
    "credentials",
    "exception",
    "guid",
    "key",
    "key_hash",
    "password",
    "raw_scorecard",
    "response_body",
    "scorecard_id",
    "secret",
    "token",
    "traceback",
    "message",
}

_SAFE_STATUS_VALUES = {
    "GAME_OVER",
    "IN_PROGRESS",
    "NOT_FINISHED",
    "NOT_PLAYED",
    "NOT_STARTED",
    "WIN",
    "WON",
}


class OfficialPublicScoreError(ValueError):
    """A protocol violation with a stable, non-secret error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PublicManifest:
    """Validated, byte-exact public manifest."""

    path: Path
    raw_sha256: str
    canonical_sha256: str
    game_ids: tuple[str, ...]
    seed: int
    max_actions_per_game: int
    max_resets_per_game: int

    def receipt(self) -> dict[str, JSONValue]:
        return {
            "raw_sha256": self.raw_sha256,
            "canonical_sha256": self.canonical_sha256,
            "game_count": len(self.game_ids),
            "game_ids": list(self.game_ids),
            "seed": self.seed,
            "max_actions_per_game": self.max_actions_per_game,
            "max_resets_per_game": self.max_resets_per_game,
        }


class _DiscardTextIO(io.TextIOBase):
    """Writable sink used to prevent SDK credential messages escaping."""

    @property
    def encoding(self) -> str:  # pragma: no cover - compatibility property
        return "utf-8"

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_public_manifest(path: str | Path | None = None) -> PublicManifest:
    """Load and validate the one byte-exact manifest permitted by V1."""

    resolved = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("official_public_manifest.json")
    )
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise OfficialPublicScoreError(
            "manifest_unreadable", "the frozen public manifest is unreadable"
        ) from exc

    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != EXPECTED_MANIFEST_SHA256:
        raise OfficialPublicScoreError(
            "manifest_digest_mismatch",
            "the public manifest bytes do not match the frozen digest",
        )
    try:
        payload = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialPublicScoreError(
            "manifest_json_invalid", "the frozen public manifest is not strict JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OfficialPublicScoreError(
            "manifest_schema_invalid", "the frozen public manifest must be an object"
        )

    canonical_sha256 = _canonical_sha256(payload)
    if canonical_sha256 != EXPECTED_MANIFEST_CANONICAL_SHA256:
        raise OfficialPublicScoreError(
            "manifest_canonical_digest_mismatch",
            "the public manifest content does not match the frozen digest",
        )

    required_keys = {
        "schema_version",
        "protocol",
        "source",
        "claim_boundary",
        "seed",
        "max_actions_per_game",
        "max_resets_per_game",
        "game_ids",
    }
    if set(payload) != required_keys:
        raise OfficialPublicScoreError(
            "manifest_schema_invalid", "the public manifest fields are not exact"
        )
    raw_ids = payload.get("game_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(game_id, str) for game_id in raw_ids
    ):
        raise OfficialPublicScoreError(
            "manifest_schema_invalid", "manifest game_ids must be a string list"
        )
    game_ids = tuple(raw_ids)
    if game_ids != EXPECTED_GAME_IDS or len(set(game_ids)) != 25:
        raise OfficialPublicScoreError(
            "manifest_game_ids_mismatch", "manifest game_ids are not the frozen 25"
        )
    if not all(_VERSIONED_GAME_ID.fullmatch(game_id) for game_id in game_ids):
        raise OfficialPublicScoreError(
            "manifest_game_id_invalid", "manifest game IDs must be versioned"
        )
    exact_scalars = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "max_actions_per_game": MAX_ACTIONS_PER_GAME,
        "max_resets_per_game": MAX_RESETS_PER_GAME,
    }
    if any(payload.get(key) != value for key, value in exact_scalars.items()):
        raise OfficialPublicScoreError(
            "manifest_protocol_mismatch", "manifest protocol constants are not frozen"
        )
    if not isinstance(payload.get("source"), Mapping) or not isinstance(
        payload.get("claim_boundary"), str
    ):
        raise OfficialPublicScoreError(
            "manifest_schema_invalid", "manifest provenance fields are malformed"
        )
    return PublicManifest(
        path=resolved,
        raw_sha256=raw_sha256,
        canonical_sha256=canonical_sha256,
        game_ids=game_ids,
        seed=SEED,
        max_actions_per_game=MAX_ACTIONS_PER_GAME,
        max_resets_per_game=MAX_RESETS_PER_GAME,
    )


def validate_sdk_version(version: str) -> None:
    """Reject toolkit version drift before constructing ``Arcade``."""

    if version != EXPECTED_ARC_AGI_VERSION:
        raise OfficialPublicScoreError(
            "sdk_version_mismatch",
            f"arc-agi must be exactly {EXPECTED_ARC_AGI_VERSION}",
        )


def _environment_id(info: Any) -> str:
    value = info.get("game_id") if isinstance(info, Mapping) else getattr(info, "game_id", None)
    if not isinstance(value, str) or not _VERSIONED_GAME_ID.fullmatch(value):
        raise OfficialPublicScoreError(
            "environment_id_invalid", "SDK returned an unversioned environment ID"
        )
    return value


def validate_environment_ids(
    environment_infos: Iterable[Any],
    expected: tuple[str, ...] = EXPECTED_GAME_IDS,
) -> tuple[str, ...]:
    """Validate exact set equality while retaining deterministic discovery data."""

    discovered = tuple(_environment_id(info) for info in environment_infos)
    if len(discovered) != len(expected):
        raise OfficialPublicScoreError(
            "environment_count_mismatch",
            f"expected {len(expected)} public environments, found {len(discovered)}",
        )
    if len(set(discovered)) != len(discovered):
        raise OfficialPublicScoreError(
            "duplicate_environment_id", "SDK returned duplicate environment IDs"
        )
    if set(discovered) != set(expected):
        raise OfficialPublicScoreError(
            "environment_set_mismatch",
            "SDK environment versions do not match the frozen manifest",
        )
    return discovered


@contextmanager
def _safe_sdk_environment(environments_dir: Path, recordings_dir: Path):
    """Pin NORMAL mode and prevent ambient credential/competition overrides."""

    pinned = {
        **_SAFE_ENV_SET,
        "ENVIRONMENTS_DIR": str(environments_dir),
        "RECORDINGS_DIR": str(recordings_dir),
    }
    names = set(pinned) | set(_SAFE_ENV_REMOVE)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _SAFE_ENV_REMOVE:
            os.environ.pop(name, None)
        os.environ.update(pinned)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _isolated_runtime_directories(temp_root: str | Path | None):
    """Yield explicit SDK directories without ever defaulting to the repo CWD."""

    if temp_root is None:
        with tempfile.TemporaryDirectory(prefix="arcgpt2-official-public-") as raw:
            root = Path(raw)
            environments = root / "environments"
            recordings = root / "recordings"
            environments.mkdir()
            recordings.mkdir()
            yield environments, recordings
        return

    root = Path(temp_root).expanduser().resolve()
    environments = root / "official-public-environments"
    recordings = root / "official-public-recordings"
    environments.mkdir(parents=True, exist_ok=True)
    recordings.mkdir(parents=True, exist_ok=True)
    yield environments, recordings


@contextmanager
def _suppress_sdk_output():
    sink = _DiscardTextIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


def _sdk_logger() -> logging.Logger:
    logger = logging.Logger("arcgpt2.official_public_score.sdk", level=logging.CRITICAL)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _load_runtime_sdk() -> tuple[Any, str]:
    """Import the pinned SDK only when a score run is explicitly requested."""

    import importlib
    import importlib.metadata

    sdk = importlib.import_module("arc_agi")
    version = importlib.metadata.version("arc-agi")
    return sdk, version


def _mode_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).rsplit(".", 1)[-1].upper()


def _normal_mode(sdk: Any) -> Any:
    operation_mode = getattr(sdk, "OperationMode", None)
    normal = getattr(operation_mode, "NORMAL", None)
    if normal is None or _mode_name(normal) != "NORMAL":
        raise OfficialPublicScoreError(
            "normal_mode_unavailable", "SDK does not expose OperationMode.NORMAL"
        )
    return normal


def _scorecard_mapping(value: Any, allowed_fields: set[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                candidate = method(mode="python") if method_name == "model_dump" else method()
            except TypeError:
                candidate = method()
            if isinstance(candidate, Mapping):
                return candidate
    # Avoid ``vars(value)``: only exact documented 0.9.9 score fields are read,
    # so card/auth/GUID data cannot enter through object fallbacks.
    output: dict[str, Any] = {}
    for field in sorted(allowed_fields):
        try:
            output[field] = getattr(value, field)
        except AttributeError:
            continue
    return output


_REJECTED = object()


def _finite_number(value: Any) -> int | float | object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _REJECTED
    if not math.isfinite(float(value)):
        return _REJECTED
    return value


def _integer(value: Any) -> int | object:
    if isinstance(value, bool) or not isinstance(value, int):
        return _REJECTED
    return value


def _versioned_id(value: Any) -> str | object:
    if isinstance(value, str) and _VERSIONED_GAME_ID.fullmatch(value):
        return value
    return _REJECTED


def _state_name(value: Any) -> str | object:
    token = _mode_name(value)
    return token if token in _SAFE_STATUS_VALUES else _REJECTED


def _number_list(value: Any) -> list[int | float] | object:
    if not isinstance(value, (list, tuple)):
        return _REJECTED
    clean = [_finite_number(item) for item in value]
    if any(item is _REJECTED for item in clean):
        return _REJECTED
    return [item for item in clean if item is not _REJECTED]  # type: ignore[misc]


def _integer_list(value: Any) -> list[int] | object:
    if not isinstance(value, (list, tuple)):
        return _REJECTED
    clean = [_integer(item) for item in value]
    if any(item is _REJECTED for item in clean):
        return _REJECTED
    return [item for item in clean if item is not _REJECTED]  # type: ignore[misc]


_RUN_FIELDS = {
    "actions",
    "completed",
    "id",
    "level_actions",
    "level_baseline_actions",
    "level_scores",
    "levels_completed",
    "number_of_environments",
    "number_of_levels",
    "resets",
    "score",
    "state",
}
_ENVIRONMENT_FIELDS = {
    "actions",
    "completed",
    "id",
    "level_count",
    "levels_completed",
    "resets",
    "runs",
    "score",
}
_SCORECARD_FIELDS = {
    "competition_mode",
    "environments",
    "score",
    "total_actions",
    "total_environments",
    "total_environments_completed",
    "total_levels",
    "total_levels_completed",
}


def _sanitize_run(value: Any) -> dict[str, JSONValue]:
    raw = _scorecard_mapping(value, _RUN_FIELDS)
    output: dict[str, JSONValue] = {}
    converters: dict[str, Callable[[Any], Any]] = {
        "id": _versioned_id,
        "score": _finite_number,
        "levels_completed": _integer,
        "actions": _integer,
        "resets": _integer,
        "state": _state_name,
        "level_scores": _number_list,
        "level_actions": _integer_list,
        "level_baseline_actions": _integer_list,
        "number_of_levels": _integer,
        "number_of_environments": _integer,
    }
    for field, converter in converters.items():
        if field not in raw or raw[field] is None:
            continue
        clean = converter(raw[field])
        if clean is not _REJECTED:
            output[field] = clean
    if isinstance(raw.get("completed"), bool):
        output["completed"] = raw["completed"]
    return output


def _sanitize_environment_score(value: Any) -> dict[str, JSONValue]:
    raw = _scorecard_mapping(value, _ENVIRONMENT_FIELDS)
    output: dict[str, JSONValue] = {}
    converters: dict[str, Callable[[Any], Any]] = {
        "id": _versioned_id,
        "score": _finite_number,
        "actions": _integer,
        "levels_completed": _integer,
        "level_count": _integer,
        "resets": _integer,
    }
    for field, converter in converters.items():
        if field not in raw or raw[field] is None:
            continue
        clean = converter(raw[field])
        if clean is not _REJECTED:
            output[field] = clean
    if isinstance(raw.get("completed"), bool):
        output["completed"] = raw["completed"]
    runs = raw.get("runs")
    if isinstance(runs, (list, tuple)):
        output["runs"] = [_sanitize_run(run) for run in runs]
    return output


def sanitize_scorecard(scorecard: Any) -> dict[str, JSONValue]:
    """Project the documented 0.9.9 scorecard hierarchy onto an allowlist.

    Kept fields are exactly the public score, non-competition flag,
    environment/run IDs and numeric scoring/counter arrays.  Source metadata,
    tags, messages, opaque payloads, card/auth data, GUIDs, and timestamps are
    never returned.
    """

    raw = _scorecard_mapping(scorecard, _SCORECARD_FIELDS)
    output: dict[str, JSONValue] = {}
    score = _finite_number(raw.get("score"))
    if score is not _REJECTED:
        output["score"] = score
    if raw.get("competition_mode") is None and "competition_mode" in raw:
        # NORMAL local cards are created with ``None`` in 0.9.9.  Preserve that
        # fact explicitly while normalizing the public non-competition claim.
        output["competition_mode"] = False
        output["competition_mode_sdk_raw_was_null"] = True
    elif isinstance(raw.get("competition_mode"), bool):
        output["competition_mode"] = raw["competition_mode"]
        output["competition_mode_sdk_raw_was_null"] = False
    for field in (
        "total_actions",
        "total_environments",
        "total_environments_completed",
        "total_levels",
        "total_levels_completed",
    ):
        clean = _integer(raw.get(field))
        if clean is not _REJECTED:
            output[field] = clean
    environments = raw.get("environments")
    if isinstance(environments, (list, tuple)):
        output["environments"] = [
            _sanitize_environment_score(environment) for environment in environments
        ]
    json.dumps(output, sort_keys=True, allow_nan=False)
    return output


def _in_range(value: Any, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _trace_state(game: Mapping[str, Any]) -> str | None:
    receipt = game.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    final = receipt.get("final_observation")
    if not isinstance(final, Mapping):
        return None
    state = final.get("environment_state")
    return _mode_name(state) if isinstance(state, str) else None


def _sticky_game_over_reset_evidence(receipt: Mapping[str, Any]) -> bool:
    """Recognize the exact arc-agi 0.9.9 sticky-state regression.

    ``Card.inc_reset_count`` updates reset/action counters but a subsequent
    ``NOT_FINISHED`` frame does not overwrite a prior ``GAME_OVER`` Card.state.
    Accept that one state disagreement only when the exact trace proves an
    immediately adjacent GAME_OVER step followed by a successful level reset.
    """

    events = receipt.get("actions")
    if not isinstance(events, list):
        return False
    for before_reset, reset in zip(events, events[1:]):
        if not isinstance(before_reset, Mapping) or not isinstance(reset, Mapping):
            continue
        game_over_after = before_reset.get("after")
        reset_before = reset.get("before")
        reset_after = reset.get("after")
        if not all(
            isinstance(value, Mapping)
            for value in (game_over_after, reset_before, reset_after)
        ):
            continue
        if (
            before_reset.get("kind") == "step"
            and before_reset.get("terminal") == "game_over"
            and _mode_name(game_over_after.get("environment_state")) == "GAME_OVER"
            and reset.get("kind") == "reset"
            and _mode_name(reset_before.get("environment_state")) == "GAME_OVER"
            and _mode_name(reset_after.get("environment_state")) == "NOT_FINISHED"
        ):
            return True
    return False


def validate_scorecard_consistency(
    scorecard: Mapping[str, Any],
    games: Iterable[Mapping[str, Any]],
    expected: tuple[str, ...] = EXPECTED_GAME_IDS,
) -> dict[str, JSONValue]:
    """Cross-check sanitized official scores against all exact game receipts."""

    game_list = list(games)
    trace_by_id = {
        game.get("game_id"): game
        for game in game_list
        if isinstance(game.get("game_id"), str)
    }
    environments = scorecard.get("environments")
    environment_list = environments if isinstance(environments, list) else []
    ids = [
        environment.get("id")
        for environment in environment_list
        if isinstance(environment, Mapping)
    ]
    ids_exact = (
        len(ids) == len(expected)
        and len(set(ids)) == len(expected)
        and set(ids) == set(expected)
    )
    one_run_each = ids_exact
    trace_consistent = ids_exact
    environment_totals_consistent = ids_exact
    level_arrays_valid = ids_exact
    run_scores_valid = ids_exact
    mismatched_ids: list[str] = []
    sticky_game_over_reset_ids: list[str] = []

    for environment in environment_list:
        if not isinstance(environment, Mapping):
            one_run_each = trace_consistent = False
            continue
        game_id = environment.get("id")
        runs = environment.get("runs")
        if (
            not isinstance(game_id, str)
            or not isinstance(runs, list)
            or len(runs) != 1
            or not isinstance(runs[0], Mapping)
        ):
            one_run_each = trace_consistent = False
            if isinstance(game_id, str):
                mismatched_ids.append(game_id)
            continue
        run = runs[0]
        # Baseline-backed 0.9.9 runs may omit/null ``id``; the enclosing
        # EnvironmentScoreList.id remains authoritative.  If a run ID is
        # present it must still agree exactly.
        if "id" in run and run.get("id") != game_id:
            one_run_each = False
        score = run.get("score")
        level_scores = run.get("level_scores")
        level_actions = run.get("level_actions")
        baseline_actions = run.get("level_baseline_actions")
        actions = run.get("actions")
        resets = run.get("resets")
        levels_completed = run.get("levels_completed")
        state = run.get("state")
        completed = run.get("completed")
        run_scores_valid &= _in_range(score, 0.0, 100.0)
        arrays_are_lists = all(
            isinstance(value, list)
            for value in (level_scores, level_actions, baseline_actions)
        )
        arrays_same_length = arrays_are_lists and len(level_scores) == len(
            level_actions
        ) == len(baseline_actions)
        arrays_values_valid = bool(
            arrays_same_length
            and all(_in_range(value, 0.0, 115.0) for value in level_scores)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in level_actions
            )
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= -1
                for value in baseline_actions
            )
        )
        level_arrays_valid &= arrays_values_valid
        counters_valid = (
            isinstance(actions, int)
            and not isinstance(actions, bool)
            and 0 <= actions <= MAX_ACTIONS_PER_GAME
            and isinstance(resets, int)
            and not isinstance(resets, bool)
            and 0 <= resets <= MAX_RESETS_PER_GAME
            and isinstance(levels_completed, int)
            and not isinstance(levels_completed, bool)
            and levels_completed >= 0
            and arrays_values_valid
            and levels_completed <= len(level_scores)
            and sum(level_actions) == actions
            and state in _SAFE_STATUS_VALUES
            and isinstance(completed, bool)
            and completed == (state in {"WIN", "WON"})
        )
        level_arrays_valid &= counters_valid

        environment_totals_consistent &= (
            environment.get("score") == score
            and environment.get("actions") == actions
            and environment.get("resets") == resets
            and environment.get("levels_completed") == levels_completed
            and environment.get("completed") == completed
            and environment.get("level_count") == len(level_scores)
        )
        trace = trace_by_id.get(game_id)
        receipt = trace.get("receipt") if isinstance(trace, Mapping) else None
        final = receipt.get("final_observation") if isinstance(receipt, Mapping) else None
        trace_actions = receipt.get("actions_taken") if isinstance(receipt, Mapping) else None
        trace_steps = receipt.get("step_actions") if isinstance(receipt, Mapping) else None
        trace_resets = receipt.get("resets") if isinstance(receipt, Mapping) else None
        trace_counter_identity = (
            isinstance(trace_actions, int)
            and not isinstance(trace_actions, bool)
            and isinstance(trace_steps, int)
            and not isinstance(trace_steps, bool)
            and isinstance(trace_resets, int)
            and not isinstance(trace_resets, bool)
            and trace_actions == trace_steps + trace_resets
        )
        trace_state = _trace_state(trace) if isinstance(trace, Mapping) else None
        exact_state_match = trace_state == state
        sticky_game_over_reset_match = bool(
            isinstance(receipt, Mapping)
            and state == "GAME_OVER"
            and trace_state == "NOT_FINISHED"
            and completed is False
            and receipt.get("status") == "stopped"
            and isinstance(trace_resets, int)
            and trace_resets > 0
            and _sticky_game_over_reset_evidence(receipt)
        )
        current_trace_consistent = (
            isinstance(trace, Mapping)
            and trace.get("status") == "completed"
            and isinstance(receipt, Mapping)
            and isinstance(final, Mapping)
            and receipt.get("reported_game_id") == game_id
            and trace_counter_identity
            and trace_actions == actions
            and trace_resets == resets
            and final.get("levels_completed") == levels_completed
            and (exact_state_match or sticky_game_over_reset_match)
            and (receipt.get("status") == "won") == completed
        )
        trace_consistent &= current_trace_consistent
        if current_trace_consistent and sticky_game_over_reset_match:
            sticky_game_over_reset_ids.append(game_id)
        if not current_trace_consistent and isinstance(game_id, str):
            mismatched_ids.append(game_id)

    aggregate_score = scorecard.get("score")
    official_score_valid = _in_range(aggregate_score, 0.0, 100.0)
    environment_scores = [
        environment.get("score")
        for environment in environment_list
        if isinstance(environment, Mapping)
    ]
    mean_matches = bool(
        ids_exact
        and all(_in_range(score, 0.0, 100.0) for score in environment_scores)
        and math.isclose(
            float(aggregate_score),
            sum(float(score) for score in environment_scores) / len(expected),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ) if official_score_valid else False
    environment_counter_fields_valid = bool(
        ids_exact
        and all(
            isinstance(environment.get(field), int)
            and not isinstance(environment.get(field), bool)
            and int(environment[field]) >= 0
            for environment in environment_list
            if isinstance(environment, Mapping)
            for field in ("levels_completed", "level_count", "actions")
        )
        and all(
            isinstance(environment.get("completed"), bool)
            for environment in environment_list
            if isinstance(environment, Mapping)
        )
    )
    totals_consistent = bool(
        environment_counter_fields_valid
        and scorecard.get("total_environments") == len(expected)
        and scorecard.get("total_environments_completed")
        == sum(bool(environment["completed"]) for environment in environment_list)
        and scorecard.get("total_levels_completed")
        == sum(int(environment["levels_completed"]) for environment in environment_list)
        and scorecard.get("total_levels")
        == sum(int(environment["level_count"]) for environment in environment_list)
        and scorecard.get("total_actions")
        == sum(int(environment["actions"]) for environment in environment_list)
    )
    competition_mode_normal = (
        scorecard.get("competition_mode") is False
        and isinstance(scorecard.get("competition_mode_sdk_raw_was_null"), bool)
    )
    schema_valid = bool(
        official_score_valid
        and ids_exact
        and one_run_each
        and run_scores_valid
        and level_arrays_valid
        and environment_totals_consistent
        and totals_consistent
        and mean_matches
    )
    valid = schema_valid and trace_consistent and competition_mode_normal
    return {
        "valid": valid,
        "schema_valid": schema_valid,
        "competition_mode_normal": competition_mode_normal,
        "competition_mode_sdk_raw_was_null": scorecard.get(
            "competition_mode_sdk_raw_was_null"
        )
        is True,
        "official_score_valid": official_score_valid,
        "official_score_mean_matches": mean_matches,
        "environment_ids_exact": ids_exact,
        "one_run_each": one_run_each,
        "run_scores_valid": run_scores_valid,
        "level_arrays_valid": level_arrays_valid,
        "environment_totals_consistent": environment_totals_consistent,
        "scorecard_totals_consistent": totals_consistent,
        "trace_scorecard_consistent": trace_consistent,
        "sticky_game_over_reset_state_accepted": bool(sticky_game_over_reset_ids),
        "sticky_game_over_reset_state_accepted_count": len(
            sticky_game_over_reset_ids
        ),
        "sticky_game_over_reset_state_accepted_game_ids": sorted(
            sticky_game_over_reset_ids
        ),
        "mismatched_game_ids": sorted(set(mismatched_ids)),
    }


def _forbidden_receipt_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _FORBIDDEN_RECEIPT_KEYS or normalized.endswith(
        ("_card_id", "_guid", "_token", "_password", "_secret")
    )


def _scrub_receipt(value: Any, *, secrets: tuple[str, ...] = ()) -> JSONValue:
    """Defensively remove credential-shaped provider/SDK receipt fields."""

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OfficialPublicScoreError(
                "receipt_not_strict_json", "receipt contains a non-finite number"
            )
        return value
    if isinstance(value, str):
        clean = value
        for secret in secrets:
            if secret:
                clean = clean.replace(secret, "[REDACTED]")
        return clean
    if isinstance(value, Mapping):
        output: dict[str, JSONValue] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if key.lower() == "error" and not isinstance(value[raw_key], Mapping):
                # Keep our own fixed structured error receipts, but discard
                # free-form SDK/provider messages which may embed credentials.
                continue
            if _forbidden_receipt_key(key):
                continue
            output[key] = _scrub_receipt(value[raw_key], secrets=secrets)
        return output
    if isinstance(value, (list, tuple)):
        return [_scrub_receipt(item, secrets=secrets) for item in value]
    raise OfficialPublicScoreError(
        "receipt_not_json_safe", "receipt contains a non-JSON object"
    )


def _check_runtime_abort(
    *,
    monotonic_deadline: float | None,
    abort_callback: AbortCallback | None,
    monotonic_clock: MonotonicClock,
) -> None:
    if monotonic_deadline is not None:
        now = monotonic_clock()
        if not math.isfinite(now) or not math.isfinite(monotonic_deadline):
            raise OfficialPublicScoreError(
                "runtime_clock_invalid", "runtime deadline/clock must be finite"
            )
        if now >= monotonic_deadline:
            raise OfficialPublicScoreError(
                "runtime_deadline", "the monotonic runtime deadline was reached"
            )
    if abort_callback is not None:
        try:
            requested = abort_callback()
        except Exception as exc:
            raise OfficialPublicScoreError(
                "abort_callback_failed", "the runtime abort callback failed"
            ) from exc
        if not isinstance(requested, bool):
            raise OfficialPublicScoreError(
                "abort_callback_invalid", "the runtime abort callback must return bool"
            )
        if requested:
            raise OfficialPublicScoreError(
                "runtime_deadline", "the runtime abort callback requested a stop"
            )


class _AbortCheckingEnvironment:
    """Duck-typed SDK wrapper that checks immediately before every action."""

    def __init__(
        self,
        environment: Any,
        *,
        monotonic_deadline: float | None,
        abort_callback: AbortCallback | None,
        monotonic_clock: MonotonicClock,
    ) -> None:
        self._environment = environment
        self._monotonic_deadline = monotonic_deadline
        self._abort_callback = abort_callback
        self._monotonic_clock = monotonic_clock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def _check(self) -> None:
        _check_runtime_abort(
            monotonic_deadline=self._monotonic_deadline,
            abort_callback=self._abort_callback,
            monotonic_clock=self._monotonic_clock,
        )

    def step(self, action: Any, data=None):
        self._check()
        return self._environment.step(action, data=data)

    def reset(self):
        self._check()
        return self._environment.reset()


def run_one_game(
    environment: Any,
    *,
    direction_score_provider: DirectionScoreProvider | None = None,
    monotonic_deadline: float | None = None,
    abort_callback: AbortCallback | None = None,
    monotonic_clock: MonotonicClock | None = None,
) -> dict[str, JSONValue]:
    """Run one already-created environment under the exact V1 action bounds.

    This pure primitive owns no SDK or scorecard state.  A higher-level runner
    may use it in separately validated lanes; this module does not claim the
    official SDK's shared-scorecard object is thread-safe.
    """

    clock = monotonic_clock or time.monotonic
    _check_runtime_abort(
        monotonic_deadline=monotonic_deadline,
        abort_callback=abort_callback,
        monotonic_clock=clock,
    )
    protected_environment = _AbortCheckingEnvironment(
        environment,
        monotonic_deadline=monotonic_deadline,
        abort_callback=abort_callback,
        monotonic_clock=clock,
    )
    receipt = run_official_game(
        protected_environment,
        config=AgentConfig(MAX_ACTIONS_PER_GAME, MAX_RESETS_PER_GAME),
        direction_score_provider=direction_score_provider,
    )
    if receipt.get("actions_taken", MAX_ACTIONS_PER_GAME + 1) > MAX_ACTIONS_PER_GAME:
        raise OfficialPublicScoreError(
            "action_budget_exceeded", "game receipt exceeded the action budget"
        )
    if receipt.get("resets", MAX_RESETS_PER_GAME + 1) > MAX_RESETS_PER_GAME:
        raise OfficialPublicScoreError(
            "reset_budget_exceeded", "game receipt exceeded the reset budget"
        )
    clean = _scrub_receipt(receipt)
    if not isinstance(clean, dict):  # pragma: no cover - mapping above guarantees
        raise OfficialPublicScoreError("receipt_invalid", "game receipt is not an object")
    json.dumps(clean, sort_keys=True, allow_nan=False)
    return clean


def _base_receipt(manifest: PublicManifest, sdk_version: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": (
            "Official ARC-AGI-3 public development environments only; not locked "
            "transfer, online leaderboard, or competition evidence."
        ),
        "operation_mode": "NORMAL",
        "sdk": {"distribution": "arc-agi", "version": sdk_version},
        "manifest": manifest.receipt(),
        "config": {
            "seed": SEED,
            "max_actions_per_game": MAX_ACTIONS_PER_GAME,
            "max_resets_per_game": MAX_RESETS_PER_GAME,
            "scorecards_opened_bound": 1,
            "runs_per_game_bound": 1,
        },
        "execution": {
            "mode": "sequential",
            "manifest_order_preserved": True,
            "shared_scorecard_concurrency_validated": False,
            "run_one_game_primitive_available": True,
        },
        "competition_submission_performed": False,
        "online_only_mode_used": False,
        "authentication": {
            "mode": "anonymous_public",
            "ambient_credential_ignored": True,
            "credential_material_persisted": False,
        },
        "games": [],
        "errors": [],
    }


def _partial_preflight(
    manifest: PublicManifest,
    sdk_version: str,
    *,
    phase: str,
    code: str,
) -> dict[str, JSONValue]:
    receipt = _base_receipt(manifest, sdk_version)
    receipt["errors"] = [{"phase": phase, "code": code}]
    receipt["scorecard"] = {
        "opened": False,
        "close_attempted": False,
        "close_succeeded": False,
        "sanitized": {},
    }
    receipt["counts"] = {
        "expected_games": 25,
        "attempted_games": 0,
        "completed_games": 0,
        "failed_games": 0,
    }
    receipt["checks"] = {
        "sdk_version_exact": code != "sdk_version_mismatch",
        "environment_manifest_exact": False,
        "one_scorecard": True,
        "one_run_per_game": True,
        "all_games_completed": False,
        "scorecard_close_succeeded": False,
        "scorecard_schema_valid": False,
        "scorecard_trace_consistent": False,
        "runtime_deadline_clear": code != "runtime_deadline",
    }
    receipt["aggregate"] = {
        "valid": False,
        "status": "partial",
        "official_score": None,
        "observed_score": None,
        "observed_score_label": "unavailable",
        "reasons": [code],
    }
    clean = _scrub_receipt(receipt)
    assert isinstance(clean, dict)
    return clean


def run_official_public_score(
    *,
    manifest_path: str | Path | None = None,
    direction_score_provider: DirectionScoreProvider | None = None,
    provider_factory: ProviderFactory | None = None,
    sdk_module: Any | None = None,
    sdk_version: str | None = None,
    temp_root: str | Path | None = None,
    monotonic_deadline: float | None = None,
    abort_callback: AbortCallback | None = None,
    monotonic_clock: MonotonicClock | None = None,
) -> dict[str, JSONValue]:
    """Run the exact 25-game NORMAL-mode public scorecard once.

    ``sdk_module``/``sdk_version`` are dependency-injection seams for strict
    fakes; production callers omit them.  ``provider_factory`` is invoked with
    no game ID exactly once per game so it cannot become a game-specific policy
    branch.  The scorecard-owning loop stays sequential because SDK
    shared-scorecard thread safety has not been established.
    """

    if direction_score_provider is not None and provider_factory is not None:
        raise OfficialPublicScoreError(
            "provider_configuration_invalid",
            "pass a provider or a provider_factory, not both",
        )
    manifest = load_public_manifest(manifest_path)
    if sdk_module is None:
        sdk, detected_version = _load_runtime_sdk()
    else:
        sdk = sdk_module
        detected_version = sdk_version or getattr(sdk_module, "__version__", "")
    resolved_version = sdk_version or detected_version
    try:
        validate_sdk_version(resolved_version)
        normal_mode = _normal_mode(sdk)
    except OfficialPublicScoreError as exc:
        return _partial_preflight(
            manifest,
            resolved_version,
            phase="sdk_preflight",
            code=exc.code,
        )

    receipt = _base_receipt(manifest, resolved_version)
    arcade: Any | None = None
    scorecard_id: Any | None = None
    scorecard_opened = False
    close_attempted = False
    close_succeeded = False
    sanitized_scorecard: dict[str, JSONValue] = {}
    discovered: tuple[str, ...] = ()
    provider_factory_failures = 0
    runtime_deadline_triggered = False
    clock = monotonic_clock or time.monotonic
    receipt["execution"]["isolated_runtime_directories"] = True
    receipt["execution"]["temp_root_provided"] = temp_root is not None

    with (
        _isolated_runtime_directories(temp_root) as runtime_directories,
        _safe_sdk_environment(*runtime_directories),
        _suppress_sdk_output(),
    ):
        environments_dir, recordings_dir = runtime_directories
        try:
            _check_runtime_abort(
                monotonic_deadline=monotonic_deadline,
                abort_callback=abort_callback,
                monotonic_clock=clock,
            )
            arcade_type = getattr(sdk, "Arcade", None)
            if not callable(arcade_type):
                raise OfficialPublicScoreError(
                    "arcade_unavailable", "SDK does not expose Arcade"
                )
            arcade = arcade_type(
                operation_mode=normal_mode,
                arc_api_key="",
                arc_base_url=OFFICIAL_ARC_BASE_URL,
                environments_dir=str(environments_dir),
                recordings_dir=str(recordings_dir),
                logger=_sdk_logger(),
            )
            actual_mode = getattr(arcade, "operation_mode", normal_mode)
            if _mode_name(actual_mode) != "NORMAL":
                raise OfficialPublicScoreError(
                    "runtime_mode_not_normal", "Arcade did not retain NORMAL mode"
                )
            discovered = validate_environment_ids(
                arcade.get_environments(), manifest.game_ids
            )
            receipt["environment_preflight"] = {
                "valid": True,
                "discovered_count": len(discovered),
                "discovered_ids_sorted": sorted(discovered),
                "manifest_set_equal": True,
                "validated_before_scorecard": True,
            }

            scorecard_id = arcade.open_scorecard(tags=[PROTOCOL])
            if not isinstance(scorecard_id, str) or not scorecard_id:
                raise OfficialPublicScoreError(
                    "scorecard_open_failed", "SDK returned no scorecard identifier"
                )
            scorecard_opened = True

            for index, game_id in enumerate(manifest.game_ids):
                try:
                    _check_runtime_abort(
                        monotonic_deadline=monotonic_deadline,
                        abort_callback=abort_callback,
                        monotonic_clock=clock,
                    )
                except OfficialPublicScoreError as exc:
                    runtime_deadline_triggered |= exc.code == "runtime_deadline"
                    receipt["errors"].append(
                        {"phase": "runtime_deadline", "code": exc.code}
                    )
                    break
                game_result: dict[str, Any] = {
                    "index": index,
                    "game_id": game_id,
                    "status": "failed",
                    "make_calls": 0,
                    "provider_assignment": {
                        "source": (
                            "factory"
                            if provider_factory is not None
                            else (
                                "shared"
                                if direction_score_provider is not None
                                else "none"
                            )
                        ),
                        "factory_call_index": (
                            index + 1 if provider_factory is not None else None
                        ),
                        "assigned": direction_score_provider is not None,
                    },
                }
                provider = direction_score_provider
                if provider_factory is not None:
                    try:
                        provider = provider_factory()
                        game_result["provider_assignment"]["assigned"] = (
                            provider is not None
                        )
                        if provider is None:
                            provider_factory_failures += 1
                            game_result["provider_factory"] = {
                                "status": "fallback",
                                "error_code": "provider_factory_returned_none",
                            }
                    except Exception:
                        provider_factory_failures += 1
                        provider = None
                        game_result["provider_assignment"]["assigned"] = False
                        game_result["provider_factory"] = {
                            "status": "fallback",
                            "error_code": "provider_factory_failed",
                        }
                try:
                    environment = arcade.make(
                        game_id,
                        seed=manifest.seed,
                        scorecard_id=scorecard_id,
                        save_recording=False,
                        include_frame_data=True,
                        render_mode=None,
                    )
                    game_result["make_calls"] = 1
                    if environment is None:
                        raise OfficialPublicScoreError(
                            "environment_make_failed", "SDK returned no environment"
                        )
                    game_receipt = run_one_game(
                        environment,
                        direction_score_provider=provider,
                        monotonic_deadline=monotonic_deadline,
                        abort_callback=abort_callback,
                        monotonic_clock=clock,
                    )
                    game_result["status"] = "completed"
                    game_result["receipt"] = game_receipt
                    provider_receipt = game_receipt.get("provider", {})
                    if isinstance(provider_receipt, Mapping):
                        game_result["provider"] = {
                            "enabled": provider_receipt.get("enabled") is True,
                            "reset_calls": provider_receipt.get("reset_calls", 0),
                            "failures": provider_receipt.get("failures", []),
                            "fallback_used": provider_receipt.get("fallback_used")
                            is True,
                        }
                except Exception as exc:
                    error_code = (
                        exc.code
                        if isinstance(exc, OfficialPublicScoreError)
                        else "game_run_failed"
                    )
                    game_result["error"] = {
                        "phase": "game_run",
                        "code": error_code,
                        # Exception messages and class names are deliberately
                        # suppressed: SDK 0.9.9 may include response bodies or
                        # generated identifiers in either surface.
                    }
                receipt["games"].append(game_result)
                if game_result.get("error", {}).get("code") == "runtime_deadline":
                    runtime_deadline_triggered = True
                    receipt["errors"].append(
                        {"phase": "runtime_deadline", "code": "runtime_deadline"}
                    )
                    break
        except OfficialPublicScoreError as exc:
            runtime_deadline_triggered |= exc.code == "runtime_deadline"
            receipt["errors"].append(
                {
                    "phase": (
                        "runtime_deadline"
                        if exc.code == "runtime_deadline"
                        else "runtime_preflight"
                    ),
                    "code": exc.code,
                }
            )
        except Exception:
            receipt["errors"].append(
                {
                    "phase": "runtime_preflight",
                    "code": "runtime_preflight_failed",
                }
            )
        finally:
            if scorecard_opened:
                close_attempted = True
                try:
                    raw_scorecard = arcade.close_scorecard(scorecard_id)
                    if raw_scorecard is None:
                        raise OfficialPublicScoreError(
                            "scorecard_close_empty", "SDK returned no final scorecard"
                        )
                    sanitized_scorecard = sanitize_scorecard(raw_scorecard)
                    close_succeeded = True
                except Exception as exc:
                    receipt["errors"].append(
                        {
                            "phase": "scorecard_close",
                            "code": (
                                exc.code
                                if isinstance(exc, OfficialPublicScoreError)
                                else "scorecard_close_failed"
                            ),
                        }
                    )

    games = receipt["games"]
    attempted = sum(int(game.get("make_calls") == 1) for game in games)
    completed = sum(int(game.get("status") == "completed") for game in games)
    failed = len(games) - completed
    provider_runtime_failures = sum(
        len(game.get("receipt", {}).get("provider", {}).get("failures", []))
        for game in games
        if isinstance(game.get("receipt"), Mapping)
        and isinstance(game.get("receipt", {}).get("provider"), Mapping)
        and isinstance(
            game.get("receipt", {}).get("provider", {}).get("failures"), list
        )
    )
    score = sanitized_scorecard.get("score")
    scorecard_validation = validate_scorecard_consistency(
        sanitized_scorecard, games, manifest.game_ids
    )
    scorecard_schema_valid = bool(scorecard_validation["schema_valid"])
    one_run_per_game = (
        len(games) == len(manifest.game_ids)
        and [game.get("game_id") for game in games] == list(manifest.game_ids)
        and all(game.get("make_calls") == 1 for game in games)
    )
    all_completed = completed == len(manifest.game_ids)
    aggregate_valid = (
        all_completed
        and close_succeeded
        and scorecard_validation["valid"] is True
        and provider_factory_failures == 0
        and provider_runtime_failures == 0
        and not runtime_deadline_triggered
        and not receipt["errors"]
    )
    reasons: list[str] = []
    if not all_completed:
        reasons.append("game_runs_incomplete")
    if not close_succeeded:
        reasons.append("scorecard_close_failed")
    if not scorecard_schema_valid:
        reasons.append("scorecard_schema_invalid")
    if not scorecard_validation["competition_mode_normal"]:
        reasons.append("scorecard_competition_mode_invalid")
    if not scorecard_validation["official_score_valid"]:
        reasons.append("official_score_invalid")
    if not scorecard_validation["environment_ids_exact"]:
        reasons.append("scorecard_environment_coverage_invalid")
    if not scorecard_validation["one_run_each"]:
        reasons.append("scorecard_run_cardinality_invalid")
    if not scorecard_validation["trace_scorecard_consistent"]:
        reasons.append("scorecard_trace_mismatch")
    if provider_factory_failures:
        reasons.append("provider_factory_failure")
    if provider_runtime_failures:
        reasons.append("provider_runtime_failure")
    if runtime_deadline_triggered:
        reasons.append("runtime_deadline")
    if receipt["errors"]:
        reasons.append("driver_error")

    receipt["scorecard"] = {
        "opened": scorecard_opened,
        "close_attempted": close_attempted,
        "close_succeeded": close_succeeded,
        "sanitized": sanitized_scorecard,
        "validation": scorecard_validation,
    }
    receipt["counts"] = {
        "expected_games": len(manifest.game_ids),
        "attempted_games": attempted,
        "completed_games": completed,
        "failed_games": failed,
        "provider_factory_failures": provider_factory_failures,
        "provider_runtime_failures": provider_runtime_failures,
    }
    receipt["checks"] = {
        "sdk_version_exact": True,
        "environment_manifest_exact": bool(discovered),
        "environment_validation_before_scorecard": bool(discovered),
        "one_scorecard": scorecard_opened,
        "one_run_per_game": one_run_per_game,
        "all_games_completed": all_completed,
        "scorecard_close_succeeded": close_succeeded,
        "scorecard_schema_valid": scorecard_schema_valid,
        "scorecard_trace_consistent": scorecard_validation[
            "trace_scorecard_consistent"
        ],
        "scorecard_competition_mode_normal": scorecard_validation[
            "competition_mode_normal"
        ],
        "official_score_valid": scorecard_validation["official_score_valid"],
        "provider_factory_failure_free": provider_factory_failures == 0,
        "provider_runtime_failure_free": provider_runtime_failures == 0,
        "runtime_deadline_clear": not runtime_deadline_triggered,
    }
    receipt["aggregate"] = {
        "valid": aggregate_valid,
        "status": "valid" if aggregate_valid else "partial",
        "official_score": float(score) if aggregate_valid else None,
        "observed_score": (
            float(score)
            if close_succeeded and scorecard_validation["official_score_valid"]
            else None
        ),
        "observed_score_label": (
            "official_valid"
            if aggregate_valid
            else (
                "partial_not_promoted"
                if close_succeeded and scorecard_validation["official_score_valid"]
                else "unavailable"
            )
        ),
        "reasons": reasons,
    }

    secret_strings = (str(scorecard_id),) if scorecard_id is not None else ()
    clean = _scrub_receipt(receipt, secrets=secret_strings)
    if not isinstance(clean, dict):  # pragma: no cover - base receipt is a dict
        raise OfficialPublicScoreError("receipt_invalid", "driver receipt is not an object")
    serialized = json.dumps(clean, sort_keys=True, allow_nan=False)
    if any(secret and secret in serialized for secret in secret_strings):
        raise OfficialPublicScoreError(
            "scorecard_identifier_leaked", "scorecard identifier escaped redaction"
        )
    return clean


# Compact compatibility alias for runners that use a shorter verb.
run_public_score = run_official_public_score


def main() -> int:
    receipt = run_official_public_score()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if receipt["aggregate"]["valid"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised by the runtime job
    raise SystemExit(main())


__all__ = [
    "AbortCallback",
    "EXPECTED_ARC_AGI_VERSION",
    "EXPECTED_GAME_IDS",
    "EXPECTED_MANIFEST_CANONICAL_SHA256",
    "EXPECTED_MANIFEST_SHA256",
    "MAX_ACTIONS_PER_GAME",
    "MAX_RESETS_PER_GAME",
    "MonotonicClock",
    "OFFICIAL_ARC_BASE_URL",
    "OfficialPublicScoreError",
    "PROTOCOL",
    "PublicManifest",
    "ProviderFactory",
    "load_public_manifest",
    "run_official_public_score",
    "run_one_game",
    "run_public_score",
    "sanitize_scorecard",
    "validate_environment_ids",
    "validate_scorecard_consistency",
    "validate_sdk_version",
]
