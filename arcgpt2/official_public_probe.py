"""Bounded probe of the official ARC-AGI-3 public environment interface.

This module is not an agent benchmark. It verifies that the research branch can
open current public environments through the official ``arc-agi`` toolkit,
record exact frames and legal actions, and execute a tiny bounded interaction
without relying on Kaggle or an ARC account secret. The official client may use
an anonymous development key when no ``ARC_API_KEY`` is configured.

No competition-mode submission is created and no private environment is
requested.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ProbeConfig:
    game_ids: tuple[str, ...]
    actions_per_game: int = 2
    seed: int = 0


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def frame_receipt(frame: Any) -> dict[str, Any]:
    """Serialize only public, non-secret observation fields."""

    if frame is None:
        return {"present": False}
    raw_frame = getattr(frame, "frame", None)
    frame_lists = _jsonable(raw_frame)
    shapes: list[list[int]] = []
    colors: set[int] = set()
    if isinstance(frame_lists, list):
        for layer in frame_lists:
            if isinstance(layer, list) and layer and isinstance(layer[0], list):
                shapes.append([len(layer), len(layer[0])])
                for row in layer:
                    for value in row:
                        if isinstance(value, int):
                            colors.add(value)
    available = getattr(frame, "available_actions", None)
    return {
        "present": True,
        "game_id": getattr(frame, "game_id", None),
        "state": _enum_name(getattr(frame, "state", None)),
        "levels_completed": getattr(frame, "levels_completed", None),
        "win_levels": getattr(frame, "win_levels", None),
        "full_reset": getattr(frame, "full_reset", None),
        "available_actions": _jsonable(available),
        "frame_shapes": shapes,
        "colors": sorted(colors),
        "frame": frame_lists,
    }


def environment_receipt(info: Any) -> dict[str, Any]:
    """Keep a small stable subset of environment-discovery metadata."""

    raw = _jsonable(info)
    if isinstance(raw, dict):
        return {
            key: raw.get(key)
            for key in (
                "game_id",
                "id",
                "name",
                "version",
                "source",
                "local",
                "remote",
            )
            if key in raw
        } or raw
    return {"value": raw}


def _simple_actions(action_space: Iterable[Any]) -> list[Any]:
    actions: list[Any] = []
    for action in action_space:
        if _enum_name(action) == "RESET":
            continue
        is_simple = getattr(action, "is_simple", None)
        if callable(is_simple) and not is_simple():
            continue
        actions.append(action)
    return actions


def probe(config: ProbeConfig) -> dict[str, Any]:
    import arc_agi

    arcade = arc_agi.Arcade()
    environments = list(arcade.get_environments() or [])
    result: dict[str, Any] = {
        "status": "completed",
        "scope": (
            "Official public-environment interface/runtime probe only; not an "
            "ARC-AGI-3 score, submission, or capability claim."
        ),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "environment_count": len(environments),
        "environments": [environment_receipt(info) for info in environments],
        "games": [],
    }

    for game_id in config.game_ids:
        game_result: dict[str, Any] = {
            "game_id": game_id,
            "opened": False,
            "error_type": None,
            "error_message": None,
            "action_space": [],
            "observations": [],
            "actions": [],
        }
        try:
            env = arcade.make(
                game_id,
                seed=config.seed,
                save_recording=False,
                include_frame_data=True,
                render_mode=None,
            )
            if env is None:
                raise RuntimeError("arcade.make returned None")
            game_result["opened"] = True
            game_result["action_space"] = [
                _enum_name(action) for action in list(env.action_space)
            ]

            initial = getattr(env, "observation_space", None)
            game_result["observations"].append(
                {"phase": "initial", **frame_receipt(initial)}
            )
            if initial is None or _enum_name(getattr(initial, "state", None)) in {
                "NOT_PLAYED",
                "GAME_OVER",
            }:
                reset = env.reset()
                game_result["observations"].append(
                    {"phase": "reset", **frame_receipt(reset)}
                )

            candidates = _simple_actions(env.action_space)
            for index, action in enumerate(candidates[: config.actions_per_game], start=1):
                observation = env.step(action, data={})
                game_result["actions"].append(
                    {"index": index, "action": _enum_name(action)}
                )
                game_result["observations"].append(
                    {
                        "phase": f"after_action_{index}",
                        **frame_receipt(observation),
                    }
                )
        except Exception as exc:  # durable diagnostic, not a hidden retry loop
            game_result["error_type"] = type(exc).__name__
            game_result["error_message"] = str(exc)[:1000]
        result["games"].append(game_result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", action="append", dest="games")
    parser.add_argument("--actions-per-game", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/official_public_probe/result.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actions_per_game < 0 or args.actions_per_game > 8:
        raise ValueError("actions-per-game must lie in 0..8")
    game_ids = tuple(args.games or ("ls20", "ft09"))
    result = probe(
        ProbeConfig(
            game_ids=game_ids,
            actions_per_game=args.actions_per_game,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
