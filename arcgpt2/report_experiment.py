"""Write a durable GitHub Actions experiment receipt.

This utility contains no model or policy logic.  It only copies existing JSON
outputs and a bounded log tail into a versioned reports directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--outcome-env", default="EXPERIMENT_OUTCOME")
    parser.add_argument("--log-lines", type=int, default=800)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = None
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        shutil.copy2(args.summary, args.report_dir / "latest-summary.json")
    if args.manifest and args.manifest.exists():
        shutil.copy2(args.manifest, args.report_dir / "latest-data-manifest.json")
    if args.log and args.log.exists():
        lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
        (args.report_dir / "latest-log-tail.txt").write_text(
            "\n".join(lines[-args.log_lines :]) + "\n",
            encoding="utf-8",
        )

    receipt = {
        "execution_status": os.environ.get(args.outcome_env, "unknown"),
        "capability_gate": (
            "passed"
            if summary and summary.get("gates", {}).get("stage02_gate_passed")
            else "not_yet_passed" if summary else "not_evaluated"
        ),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "branch": os.environ.get("GITHUB_REF_NAME"),
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
    }
    (args.report_dir / "latest-status.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
