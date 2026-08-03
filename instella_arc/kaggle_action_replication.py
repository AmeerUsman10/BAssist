"""Kaggle entry point for the 64-world frozen action-binding replication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from .action_replication import run_replication
from .backend import TransformersBackend
from .catalog import CHECKPOINTS
from .kaggle_runner import gpu_inventory, load_with_fallbacks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-40:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), default="think")
    parser.add_argument(
        "--quantizations",
        nargs="+",
        choices=("int4", "int8", "none"),
        default=["int4", "int8"],
    )
    parser.add_argument("--seed-base", type=int, default=940_000)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=6144)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/instella_action_replication"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "action_replication.json"
    status_path = args.output_dir / "replication_status.json"
    checkpoint = CHECKPOINTS[args.checkpoint]
    status: dict[str, Any] = {
        "schema": "instella_arc.kaggle_action_replication.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "arguments": {
            "checkpoint": args.checkpoint,
            "quantizations": list(args.quantizations),
            "seed_base": args.seed_base,
            "games": args.games,
            "max_context_tokens": args.max_context_tokens,
            "output_dir": str(args.output_dir),
        },
        "checkpoint_spec": {
            "repository_id": checkpoint.repository_id,
            "revision": checkpoint.revision,
        },
        "gpu_inventory": gpu_inventory(),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        backend, attempts = load_with_fallbacks(
            checkpoint=args.checkpoint,
            quantizations=list(args.quantizations),
            max_context_tokens=args.max_context_tokens,
            allow_fp16_offload=False,
        )
        status["load_attempts"] = attempts
        report = run_replication(
            backend,
            seed_base=args.seed_base,
            games=args.games,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        status.update(
            status="success",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            report=str(report_path),
            report_sha256=_sha256(report_path),
            summary=report["summary"],
        )
    except Exception as exc:
        status.update(
            status="failure",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=_error(exc),
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
