"""Bounded public ARC-AGI-3 probe for the deterministic controller shell.

This is a development evaluation, not competition mode and not a submission.
Each requested game is created once, receives at most the configured action
budget, and writes exact controller receipts. Failures remain in the report so
one game cannot erase evidence from another.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from .controller import ClosedLoopController
from .toolkit_runner import ToolkitControllerRunner


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _serialize(model_dump())
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _serialize(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-40:],
    }


def run_probe(games: list[str], *, max_actions: int) -> dict[str, Any]:
    try:
        import arc_agi
        from arc_agi import OperationMode
    except ImportError as exc:
        raise RuntimeError("public probe requires the official arc-agi package") from exc

    started = datetime.now(timezone.utc)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)
    game_reports: list[dict[str, Any]] = []
    for game_id in games:
        entry: dict[str, Any] = {
            "game_id": game_id,
            "status": "running",
            "result": None,
            "error": None,
        }
        try:
            environment = arcade.make(
                game_id,
                seed=0,
                save_recording=False,
                include_frame_data=True,
            )
            if environment is None:
                raise RuntimeError("Arcade.make returned None")
            result = ToolkitControllerRunner(
                environment=environment,
                controller=ClosedLoopController(),
                max_actions=max_actions,
            ).run()
            entry["status"] = "success"
            entry["result"] = asdict(result)
        except Exception as exc:
            entry["status"] = "failure"
            entry["error"] = _error(exc)
        game_reports.append(entry)

    scorecard = None
    try:
        scorecard = arcade.close_scorecard()
    except Exception as exc:
        scorecard = {"error": _error(exc)}
    finished = datetime.now(timezone.utc)
    completed_levels = sum(
        int(entry["result"].get("levels_completed") or 0)
        for entry in game_reports
        if isinstance(entry.get("result"), dict)
    )
    return {
        "schema": "instella_arc.public_shell_probe.v1",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "operation_mode": "ONLINE",
        "competition_mode": False,
        "anonymous_or_environment_key": True,
        "games_requested": games,
        "max_actions_per_game": max_actions,
        "games": game_reports,
        "completed_levels_total": completed_levels,
        "scorecard": _serialize(scorecard),
        "scope": (
            "Public development probe of deterministic evidence/navigation shell. "
            "No competition submission and no Instella model capability claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["ft09", "ls20"])
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_probe(args.games, max_actions=args.max_actions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output),
        "sha256": digest,
        "completed_levels_total": report["completed_levels_total"],
        "games": [
            {
                "game_id": entry["game_id"],
                "status": entry["status"],
                "levels_completed": (
                    entry["result"].get("levels_completed")
                    if isinstance(entry.get("result"), dict)
                    else None
                ),
                "actions": (
                    entry["result"].get("actions")
                    if isinstance(entry.get("result"), dict)
                    else None
                ),
                "error": entry["error"],
            }
            for entry in report["games"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
