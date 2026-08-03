"""Recover bounded Instella ARC evidence from the authenticated Kaggle account.

This program is intentionally conservative:

* it discovers the actual owner-qualified kernel ref from ``kernels list --mine``;
* it downloads only ``status.json`` and ``frozen_benchmark.json``;
* it records a redacted log tail and checksums;
* it cancels queued duplicate GitHub dispatches;
* it cancels the stale in-progress poller only after the Kaggle run is terminal.

No Kaggle kernel is created, updated, deleted, or resubmitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TARGET_SLUG = "instella-arc-frozen-probe"
DISPATCH_WORKFLOW = "instella-arc-kaggle-dispatch.yml"


def run_command(arguments: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def resolve_owned_ref(listing: str, slug: str = TARGET_SLUG) -> list[str]:
    return sorted(
        set(
            re.findall(
                rf"([A-Za-z0-9_-]+/{re.escape(slug)})",
                listing,
                flags=re.IGNORECASE,
            )
        )
    )


def classify_kernel_status(returncode: int, output: str) -> str:
    if returncode != 0:
        return "query_failure"
    lowered = output.lower()
    if any(word in lowered for word in ("complete", "success")):
        return "complete"
    if any(word in lowered for word in ("running", "queued", "pending")):
        return "running"
    if any(word in lowered for word in ("error", "failed", "cancelled")):
        return "failure"
    return "unknown"


def sanitize_owner(text: str, actual_ref: str) -> str:
    owner, slug = actual_ref.split("/", 1)
    sanitized = text.replace(actual_ref, f"<redacted>/{slug}")
    return re.sub(
        rf"(?<![A-Za-z0-9_-]){re.escape(owner)}(?![A-Za-z0-9_-])",
        "<redacted>",
        sanitized,
        flags=re.IGNORECASE,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_request(url: str, token: str, *, method: str = "GET") -> Any:
    request = Request(
        url,
        data=b"" if method == "POST" else None,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "BAssist-Instella-Recovery",
        },
    )
    with urlopen(request, timeout=60) as response:
        if response.status == 204:
            return None
        body = response.read()
        return json.loads(body) if body else None


def cancel_dispatches(
    *,
    repository: str,
    token: str,
    terminal_kernel: bool,
) -> list[dict[str, Any]]:
    runs_url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{DISPATCH_WORKFLOW}/runs?branch=instella-arc&per_page=20"
    )
    payload = github_request(runs_url, token)
    cancellations: list[dict[str, Any]] = []
    for run in payload.get("workflow_runs", []):
        run_status = str(run.get("status"))
        should_cancel = run_status in {"queued", "pending"}
        if terminal_kernel:
            should_cancel = should_cancel or run_status == "in_progress"
        if not should_cancel:
            continue
        run_id = int(run["id"])
        entry = {
            "run_id": run_id,
            "previous_status": run_status,
            "requested": True,
            "accepted": False,
            "error": None,
        }
        try:
            github_request(
                f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/cancel",
                token,
                method="POST",
            )
            entry["accepted"] = True
        except HTTPError as exc:
            entry["error"] = f"HTTP {exc.code}: {exc.reason}"
        cancellations.append(entry)
    return cancellations


def locate_unique(files: list[Path], name: str) -> Path | None:
    matches = [path for path in files if path.name == name]
    return matches[0] if len(matches) == 1 else None


def recover(output_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "instella_arc.kaggle_recovery.v2",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_ref_resolved": False,
        "kernel_status": "unknown",
        "status_query_exit_code": None,
        "output_download_exit_code": None,
        "output_files": [],
        "output_sha256": {},
        "runner_status": None,
        "benchmark_summary": None,
        "sanitized_kernel_log_tail": "",
        "cancellations": [],
        "status": "failure",
        "reason": None,
        "mutated_kaggle": False,
    }

    listing = run_command(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "100", "-v"],
        timeout=180,
    )
    listing_text = (listing.stdout + "\n" + listing.stderr).strip()
    refs = resolve_owned_ref(listing_text)
    if listing.returncode != 0:
        report["reason"] = "Unable to list kernels owned by the authenticated Kaggle account."
    elif len(refs) != 1:
        report["reason"] = f"Expected one owned target kernel, found {len(refs)}."
    else:
        actual_ref = refs[0]
        report["kernel_ref_resolved"] = True
        status_process = run_command(
            ["kaggle", "kernels", "status", actual_ref], timeout=180
        )
        status_text = (status_process.stdout + "\n" + status_process.stderr).strip()
        report["status_query_exit_code"] = status_process.returncode
        report["kernel_status"] = classify_kernel_status(
            status_process.returncode, status_text
        )

        output_root.mkdir(parents=True, exist_ok=True)
        # Deliberately broad substring pattern: Kaggle may preserve nested output
        # paths, but these two file names are unique in the bounded runner.
        download = run_command(
            [
                "kaggle",
                "kernels",
                "output",
                actual_ref,
                "--path",
                str(output_root),
                "--force",
                "--quiet",
                "--file-pattern",
                "status.json|frozen_benchmark.json",
            ],
            timeout=600,
        )
        report["output_download_exit_code"] = download.returncode

        logs = run_command(["kaggle", "kernels", "logs", actual_ref], timeout=180)
        raw_log = (logs.stdout + "\n" + logs.stderr).strip()
        report["sanitized_kernel_log_tail"] = sanitize_owner(raw_log, actual_ref)[-16000:]

        files = sorted(path for path in output_root.rglob("*") if path.is_file())
        report["output_files"] = [str(path.relative_to(output_root)) for path in files]
        report["output_sha256"] = {
            str(path.relative_to(output_root)): sha256(path) for path in files
        }

        status_path = locate_unique(files, "status.json")
        benchmark_path = locate_unique(files, "frozen_benchmark.json")
        if status_path is not None:
            try:
                report["runner_status"] = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - durable diagnostic path
                report["runner_status"] = {
                    "parse_error": f"{type(exc).__name__}: {exc}"
                }
        if benchmark_path is not None:
            try:
                benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
                report["benchmark_summary"] = {
                    "schema": benchmark.get("schema"),
                    "profile": benchmark.get("profile"),
                    "backend": benchmark.get("backend"),
                    "tasks": benchmark.get("tasks"),
                    "seed_base": benchmark.get("seed_base"),
                    "case_count": benchmark.get("case_count"),
                    "selected_case_ids": benchmark.get("selected_case_ids"),
                    "summary": benchmark.get("summary"),
                }
            except Exception as exc:  # pragma: no cover - durable diagnostic path
                report["benchmark_summary"] = {
                    "parse_error": f"{type(exc).__name__}: {exc}"
                }

        runner_success = (
            isinstance(report["runner_status"], dict)
            and report["runner_status"].get("status") == "success"
        )
        benchmark_present = (
            isinstance(report["benchmark_summary"], dict)
            and "summary" in report["benchmark_summary"]
        )
        if report["kernel_status"] == "complete" and runner_success and benchmark_present:
            report["status"] = "success"
            report["reason"] = None
        elif report["kernel_status"] == "complete":
            report["status"] = "recovered_failure"
            report["reason"] = (
                "Kaggle completed, but the bounded runner did not produce a "
                "successful status and benchmark pair."
            )
        else:
            report["status"] = "not_ready"
            report["reason"] = "The target Kaggle run is not complete."

    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_ACTIONS_TOKEN", "")
    if repository and token:
        try:
            report["cancellations"] = cancel_dispatches(
                repository=repository,
                token=token,
                terminal_kernel=report["kernel_status"] in {"complete", "failure"},
            )
        except Exception as exc:  # pragma: no cover - external diagnostic path
            report["cancellations"] = [
                {
                    "run_id": None,
                    "requested": False,
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = recover(args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "kernel_status": report["kernel_status"],
                "output_files": report["output_files"],
                "runner_status": (
                    report["runner_status"].get("status")
                    if isinstance(report["runner_status"], dict)
                    else None
                ),
                "cancellations": report["cancellations"],
                "reason": report["reason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
