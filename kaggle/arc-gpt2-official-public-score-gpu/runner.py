"""Private Kaggle driver for ``official_public_score_v1``.

This is the only executable that may open the frozen ARC-AGI-3 public
scorecard.  It trains the preregistered positive control once on ``cuda:0``,
verifies an exact in-memory frozen clone on ``cuda:1``, and scores the manifest
in deterministic order as contiguous 13/12 GPU shards.  Shards are deliberately
sequential: the official local scorecard manager is not documented as a
concurrent writer, so parallel environment calls would make the receipt order
and action counters race.

Only JSON evidence and redacted text logs are written below ``RUN_ROOT``.
Model weights, prefixes, recordings, API credentials, and competition
submissions are never persisted.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import string
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROTOCOL = "official_public_score_v1"
RUNNER_SCHEMA_VERSION = 1
ARC_AGI_VERSION = "0.9.9"
ARCENGINE_VERSION = "0.9.3"
PYTHON_VERSION = (3, 12)
TORCH_VERSION = "2.10.0+cu128"
CUDA_RUNTIME_VERSION = "12.8"
LOCK_SHA256 = "0b6ba06bbb5104c49608dcfc25e54cdb54202844057b519c65395765ca0c3bec"
PUBLIC_MANIFEST_SHA256 = (
    "77b441ebbba044653c6abe64d17ab3023e54fd1ed6fec732bf0c525614e2e7dc"
)
MAX_RUNTIME_SECONDS = 10_800.0
OUTPUT_MARGIN_SECONDS = 900.0
MAX_ACTIONS_PER_GAME = 160
MAX_RESETS_PER_GAME = 3
PUBLIC_GAME_COUNT = 25

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_official_public_score_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_official_public_score_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
HF_HOME = TEMP_ROOT / "huggingface"
OUTPUT_DIR = RUN_ROOT / "outputs"
TRACE_DIR = RUN_ROOT / "traces"
LOG_DIR = RUN_ROOT / "logs"

ROOT_ALLOWED_FILES = {
    "artifact-manifest.json",
    "environment.json",
    "failure.json",
    "model-manifest.json",
    "official-driver-receipt.json",
    "official-public-score-result.json",
    "receipt.json",
    "sanitized-scorecard.json",
    "source-manifest.json",
    "summary.json",
    "training-canary.json",
}
LOG_ALLOWED_FILES = {"cpu-preflight-tests.log", "dependency-install.log"}
MODEL_PAYLOAD_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "arc_api_key",
    "authorization",
    "card_id",
    "cookie",
    "guid",
    "password",
    "secret",
    "token",
}
TOKEN_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(?:bearer|x-api-key)\s*[:=]?\s*[A-Za-z0-9._~+/-]{12,}"),
)

_ACTIVE_SECRETS: set[str] = set()


class ScoreLane:
    """Lightweight lane holder that keeps runner import dependency-free."""

    __slots__ = ("name", "device", "model", "prefix")

    def __init__(self, name: str, device: Any, model: Any, prefix: Any) -> None:
        self.name = name
        self.device = device
        self.model = model
        self.prefix = prefix


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_source_sha(value: str) -> None:
    if len(value) != 40 or any(character not in string.hexdigits for character in value):
        raise RuntimeError("runner source SHA was not injected as a full commit SHA")


def sanitize_text(value: str) -> str:
    result = str(value)
    for secret in sorted(_ACTIVE_SECRETS, key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED_SECRET]")
    for pattern in TOKEN_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def sanitize_evidence(value: Any) -> Any:
    """Remove SDK identifiers/credentials from a JSON evidence tree."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot enter durable evidence")
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_evidence(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_evidence(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return sanitize_evidence(model_dump(mode="json", exclude_none=True))
    raise TypeError(f"unsupported evidence value: {type(value)!r}")


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> None:
    """Run a non-secret command and retain a redacted line-oriented log."""

    safe_command = [sanitize_text(part) for part in command]
    print(
        json.dumps({"event": "command_start", "command": safe_command, "utc": utc_now()}),
        flush=True,
    )
    handle = log.open("w", encoding="utf-8") if log else None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = sanitize_text(line)
            print(safe_line, end="", flush=True)
            if handle:
                handle.write(safe_line)
        return_code = process.wait()
    finally:
        if handle:
            handle.close()
    if return_code:
        raise subprocess.CalledProcessError(return_code, safe_command)


def load_public_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PUBLIC_MANIFEST_SHA256:
        raise RuntimeError("official public manifest hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("official public manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": 0,
        "max_actions_per_game": MAX_ACTIONS_PER_GAME,
        "max_resets_per_game": MAX_RESETS_PER_GAME,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"official public manifest field {key!r} drifted")
    game_ids = value.get("game_ids")
    if (
        not isinstance(game_ids, list)
        or len(game_ids) != PUBLIC_GAME_COUNT
        or len(set(game_ids)) != PUBLIC_GAME_COUNT
        or not all(isinstance(item, str) and re.fullmatch(r"[a-z0-9]{4}-[a-f0-9]{8}", item) for item in game_ids)
    ):
        raise RuntimeError("official public manifest must contain 25 unique versioned IDs")
    return value


def score_shards(game_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(game_ids) != PUBLIC_GAME_COUNT:
        raise ValueError("official score sharding requires exactly 25 games")
    first = tuple(game_ids[:13])
    second = tuple(game_ids[13:])
    if len(first) != 13 or len(second) != 12 or (*first, *second) != tuple(game_ids):
        raise RuntimeError("deterministic 13/12 shard partition failed")
    return (
        {"lane": "cuda:0", "manifest_start": 0, "manifest_stop": 13, "game_ids": list(first)},
        {"lane": "cuda:1", "manifest_start": 13, "manifest_stop": 25, "game_ids": list(second)},
    )


class LaneProviderFactory:
    """Bind the driver's sequential provider seam to deterministic GPU shards."""

    __slots__ = ("lanes", "memory_factory", "assignments")

    def __init__(
        self,
        lanes: Sequence[ScoreLane],
        memory_factory: Callable[[ScoreLane], Any],
    ) -> None:
        if len(lanes) != 2 or [lane.name for lane in lanes] != ["cuda:0", "cuda:1"]:
            raise RuntimeError("official provider factory requires cuda:0/cuda:1 lanes")
        self.lanes = tuple(lanes)
        self.memory_factory = memory_factory
        self.assignments: list[str] = []

    def __call__(self) -> Any:
        index = len(self.assignments)
        if index >= PUBLIC_GAME_COUNT:
            raise RuntimeError("official provider factory exceeded the 25-game manifest")
        lane = self.lanes[0 if index < 13 else 1]
        provider = self.memory_factory(lane)
        self.assignments.append(lane.name)
        return provider

    def receipt(self, game_ids: Sequence[str]) -> dict[str, Any]:
        expected = ["cuda:0"] * 13 + ["cuda:1"] * 12
        if self.assignments != expected:
            raise RuntimeError("driver did not exercise the deterministic 13/12 lanes")
        return {
            "execution_strategy": "sequential_manifest_order",
            "parallel_score_execution": False,
            "parallel_score_execution_reason": (
                "arc_agi_0_9_9_shared_scorecard_thread_safety_not_proven"
            ),
            "score_speedup_claimed": False,
            "concurrent_scorecard_writers": 0,
            "one_scorecard": True,
            "one_run_per_game": True,
            "shards": list(score_shards(game_ids)),
            "lane_game_counts": {"cuda:0": 13, "cuda:1": 12},
            "assignments": list(self.assignments),
        }


def validate_provider_receipt(value: Any) -> None:
    """Reject any learned-provider fallback before reporting a model score."""

    if (
        not isinstance(value, Mapping)
        or value.get("enabled") is not True
        or value.get("failures") != []
        or value.get("fallback_used") is not False
        or value.get("reset_calls") != 1
    ):
        raise RuntimeError("official exact-gradient provider receipt is not clean")


def driver_receipt_for_output(
    value: Mapping[str, Any],
    *,
    model_score_eligible: bool,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Make the runner's model-claim decision explicit in driver evidence."""

    output = copy.deepcopy(dict(value))
    output["runner_model_score_contract"] = {
        "eligible": model_score_eligible,
        "failure_reason": failure_reason,
    }
    if not model_score_eligible:
        aggregate = output.get("aggregate")
        if isinstance(aggregate, dict):
            aggregate["valid"] = False
            aggregate["status"] = "partial"
            aggregate["official_score"] = None
            reasons = aggregate.get("reasons")
            if not isinstance(reasons, list):
                reasons = []
                aggregate["reasons"] = reasons
            if failure_reason and failure_reason not in reasons:
                reasons.append(failure_reason)
    sanitized = sanitize_evidence(output)
    if not isinstance(sanitized, dict):
        raise RuntimeError("official driver evidence did not sanitize to an object")
    return sanitized


def output_boundary_violations(root: Path) -> dict[str, list[str]]:
    unexpected: list[str] = []
    model_payloads: list[str] = []
    secret_leaks: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_text = str(relative)
        allowed = (
            (len(relative.parts) == 1 and relative.name in ROOT_ALLOWED_FILES)
            or (
                len(relative.parts) == 2
                and relative.parts[0] == "logs"
                and relative.name in LOG_ALLOWED_FILES
            )
            or (
                len(relative.parts) == 2
                and relative.parts[0] == "traces"
                and relative.suffix == ".json"
                and re.fullmatch(r"\d{2}-[a-z0-9]{4}-[a-f0-9]{8}\.json", relative.name)
            )
        )
        if not allowed:
            unexpected.append(relative_text)
        lower_name = path.name.lower()
        text_model_manifest = lower_name.endswith("-manifest.json") or lower_name.endswith(
            ".index.json"
        )
        if not text_model_manifest and (
            path.suffix.lower() in MODEL_PAYLOAD_SUFFIXES
            or lower_name.startswith(("checkpoint-", "pytorch_model", "model-"))
        ):
            model_payloads.append(relative_text)
        if path.suffix.lower() in {".json", ".log", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(secret and secret in text for secret in _ACTIVE_SECRETS):
                secret_leaks.append(relative_text)
            if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
                secret_leaks.append(relative_text)
    return {
        "unexpected_files": sorted(set(unexpected)),
        "model_payloads": sorted(set(model_payloads)),
        "secret_leaks": sorted(set(secret_leaks)),
    }


def artifact_manifest(root: Path) -> dict[str, Any]:
    excluded = {"artifact-manifest.json"}
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_sha": SOURCE_SHA,
        "created_at_utc": utc_now(),
        "files": files,
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), *args], text=True
    ).strip()


def _source_manifest(runner_path: Path, manifest_path: Path, lock_path: Path) -> dict[str, Any]:
    required = tuple(
        name
        for name in _git_output(
            "ls-files",
            "--",
            "arcgpt2",
            "kaggle/arc-gpt2-official-public-score-gpu/runner.py",
        ).splitlines()
        if name
    )
    if not required:
        raise RuntimeError("tracked runtime source list is empty")
    files = {}
    for name in required:
        path = SOURCE_DIR / name
        if not path.is_file():
            raise RuntimeError(f"required source file is absent: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_sha": SOURCE_SHA,
        "resolved_source_sha": _git_output("rev-parse", "HEAD"),
        "source_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "runner_source_sha256": sha256_file(runner_path),
        "public_manifest_sha256": sha256_file(manifest_path),
        "requirements_lock_sha256": sha256_file(lock_path),
        "files": files,
    }


def verify_runtime_source_integrity(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the tested tracked tree and every source-manifest file."""

    diff = subprocess.run(
        ["git", "-C", str(SOURCE_DIR), "diff", "--exit-code", "--", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError("tracked source changed during dependency install or pytest")
    recorded = source_receipt.get("files")
    if not isinstance(recorded, Mapping) or not recorded:
        raise RuntimeError("source manifest has no runtime files")
    mismatches: list[str] = []
    for name, metadata in recorded.items():
        path = SOURCE_DIR / str(name)
        expected = metadata.get("sha256") if isinstance(metadata, Mapping) else None
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(name))
    if mismatches:
        raise RuntimeError(f"runtime source rehash mismatch: {mismatches}")
    return {
        "tracked_diff_clean": True,
        "files_rehashed": len(recorded),
        "all_manifest_hashes_exact": True,
        "untracked_test_caches_ignored": True,
    }


def _model_manifest(snapshot: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(snapshot)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    if not files or not any(name.endswith("model.safetensors") for name in files):
        raise RuntimeError("pinned GPT-2 snapshot is incomplete")
    return {
        "schema_version": 1,
        "repository": GPT2_REPOSITORY,
        "revision": GPT2_REVISION,
        "files": files,
        "payload_location": "temporary_huggingface_cache_not_output",
        "model_payload_persisted": False,
    }


def main() -> None:
    validate_source_sha(SOURCE_SHA)
    if sys.version_info[:2] != PYTHON_VERSION:
        raise RuntimeError("official public score requires exactly Python 3.12")
    started = time.perf_counter()
    runner_path = Path(__file__).resolve()
    for root in (RUN_ROOT, TEMP_ROOT):
        if root.exists():
            shutil.rmtree(root)
    OUTPUT_DIR.mkdir(parents=True)
    TRACE_DIR.mkdir(parents=True)
    LOG_DIR.mkdir(parents=True)

    run_command(["git", "init", str(SOURCE_DIR)])
    run_command(["git", "-C", str(SOURCE_DIR), "remote", "add", "origin", SOURCE_REPOSITORY])
    run_command(["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", SOURCE_SHA])
    run_command(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"])
    if _git_output("rev-parse", "HEAD") != SOURCE_SHA:
        raise RuntimeError("cloned source SHA disagrees with injected source SHA")

    manifest_path = SOURCE_DIR / "arcgpt2/official_public_manifest.json"
    lock_path = SOURCE_DIR / "arcgpt2/requirements-arc3.lock"
    manifest = load_public_manifest(manifest_path)
    if sha256_file(lock_path) != LOCK_SHA256:
        raise RuntimeError("hashed official SDK dependency lock drifted")
    source_receipt = _source_manifest(runner_path, manifest_path, lock_path)
    if source_receipt["resolved_source_sha"] != SOURCE_SHA:
        raise RuntimeError("source receipt SHA chain is inconsistent")
    write_json(RUN_ROOT / "source-manifest.json", source_receipt)

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--require-hashes",
            "-r",
            str(lock_path),
        ],
        log=LOG_DIR / "dependency-install.log",
    )
    if package_version("arc-agi") != ARC_AGI_VERSION:
        raise RuntimeError("arc-agi version does not match the hashed lock")
    if package_version("arcengine") != ARCENGINE_VERSION:
        raise RuntimeError("arcengine version does not match the hashed lock")

    preflight_env = os.environ.copy()
    preflight_env.update(
        {
            "ARC_API_KEY": "",
            "CUDA_VISIBLE_DEVICES": "",
            "OPERATION_MODE": "normal",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(SOURCE_DIR),
        }
    )
    run_command(
        [sys.executable, "-m", "pytest", "-q", "arcgpt2/tests"],
        cwd=SOURCE_DIR,
        env=preflight_env,
        log=LOG_DIR / "cpu-preflight-tests.log",
    )
    source_integrity = verify_runtime_source_integrity(source_receipt)

    os.environ.update(
        {
            "ARC_API_KEY": "",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HOME": str(HF_HOME),
            "OPERATION_MODE": "normal",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if os.environ.get("OPERATION_MODE") != "normal" or os.environ.get("ARC_API_KEY"):
        raise RuntimeError("NORMAL anonymous operation boundary could not be established")

    import torch
    from huggingface_hub import snapshot_download

    if str(torch.__version__) != TORCH_VERSION or str(torch.version.cuda) != CUDA_RUNTIME_VERSION:
        raise RuntimeError(
            "Kaggle torch/CUDA base drifted from the frozen 2.10.0+cu128 / 12.8 runtime"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("official public score requires exactly two visible CUDA devices")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != (7, 5):
            raise RuntimeError(f"cuda:{index} is not a Tesla T4 with capability 7.5")
        probe = torch.ones(256, device=f"cuda:{index}")
        if float((probe @ probe).item()) != 256.0:
            raise RuntimeError(f"CUDA compute probe failed on cuda:{index}")
        devices.append(
            {"index": index, "name": name, "compute_capability": list(capability)}
        )
    del probe
    torch.cuda.empty_cache()

    snapshot = Path(
        snapshot_download(
            repo_id=GPT2_REPOSITORY,
            revision=GPT2_REVISION,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
    )
    write_json(RUN_ROOT / "model-manifest.json", _model_manifest(snapshot))

    sys.path.insert(0, str(SOURCE_DIR))
    from arcgpt2.official_mapping_memory import OfficialGPT2MappingMemory
    from arcgpt2.official_public_score import run_official_public_score
    from arcgpt2.official_score_training import (
        Config as TrainingConfig,
        frozen_state_hashes,
        prepare_training_canary,
        verify_in_memory_lane_clone,
    )

    prepared = prepare_training_canary(
        TrainingConfig(
            source_sha=SOURCE_SHA,
            output_dir=str(TEMP_ROOT / "training-receipt-only"),
            require_cuda=True,
        )
    )
    training = prepared.receipt
    write_json(RUN_ROOT / "training-canary.json", training)
    if (
        training.get("eligible_for_official_public_score") is not True
        or training.get("gate", {}).get("passed") is not True
        or training.get("runtime_projection", {}).get("within_runtime_ceiling") is not True
    ):
        raise RuntimeError("training/canary/runtime gate failed before public actions")
    if time.perf_counter() - started >= MAX_RUNTIME_SECONDS:
        raise RuntimeError("wall clock reached the three-hour gate before public actions")

    source_device = next(prepared.model.parameters()).device
    if str(source_device) != "cuda:0":
        raise RuntimeError("positive-control training did not run on cuda:0")
    prepared.model.eval()
    for parameter in prepared.model.parameters():
        parameter.requires_grad_(False)
    lane_one_model = copy.deepcopy(prepared.model).to(torch.device("cuda:1"))
    lane_one_prefix = prepared.initial_prefix.detach().clone().to(torch.device("cuda:1"))
    lane_one_model.eval()
    for parameter in lane_one_model.parameters():
        parameter.requires_grad_(False)
    clone_receipt = verify_in_memory_lane_clone(
        prepared.model,
        prepared.initial_prefix,
        lane_one_model,
        lane_one_prefix,
        expected_device="cuda:1",
    )
    if clone_receipt.get("passed") is not True:
        raise RuntimeError("cuda:1 frozen-state clone failed exact hash verification")
    lanes = (
        ScoreLane("cuda:0", torch.device("cuda:0"), prepared.model, prepared.initial_prefix),
        ScoreLane("cuda:1", torch.device("cuda:1"), lane_one_model, lane_one_prefix),
    )
    before_score_hashes = {
        lane.name: frozen_state_hashes(lane.model, lane.prefix) for lane in lanes
    }

    def memory_factory(lane: ScoreLane) -> Any:
        torch.cuda.set_device(lane.device)
        return OfficialGPT2MappingMemory(
            lane.model,
            prepared.tokenizer,
            lane.prefix,
            device=lane.device,
        )

    projected_public_seconds = training.get("runtime_projection", {}).get(
        "projected_public_inference_seconds"
    )
    setup_elapsed_seconds = time.perf_counter() - started
    if (
        not isinstance(projected_public_seconds, (int, float))
        or isinstance(projected_public_seconds, bool)
        or not math.isfinite(float(projected_public_seconds))
        or setup_elapsed_seconds
        + float(projected_public_seconds)
        + OUTPUT_MARGIN_SECONDS
        > MAX_RUNTIME_SECONDS
    ):
        raise RuntimeError(
            "setup elapsed plus projected public runtime and output margin exceeds three hours"
        )
    score_deadline = started + MAX_RUNTIME_SECONDS - OUTPUT_MARGIN_SECONDS
    lane_provider_factory = LaneProviderFactory(lanes, memory_factory)
    official_driver_receipt = run_official_public_score(
        manifest_path=manifest_path,
        provider_factory=lane_provider_factory,
        temp_root=TEMP_ROOT / "arc-sdk-runtime",
        monotonic_deadline=score_deadline,
    )
    aggregate = official_driver_receipt.get("aggregate")
    driver_checks = official_driver_receipt.get("checks")
    partial_scorecard_container = official_driver_receipt.get("scorecard")
    partial_scorecard = (
        partial_scorecard_container.get("sanitized")
        if isinstance(partial_scorecard_container, Mapping)
        else None
    )
    if isinstance(partial_scorecard, Mapping):
        write_json(
            RUN_ROOT / "sanitized-scorecard.json",
            sanitize_evidence(partial_scorecard),
        )
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("valid") is not True
        or aggregate.get("status") != "valid"
        or not isinstance(driver_checks, Mapping)
        or not driver_checks
        or not all(driver_checks.values())
    ):
        write_json(
            RUN_ROOT / "official-driver-receipt.json",
            driver_receipt_for_output(
                official_driver_receipt,
                model_score_eligible=False,
                failure_reason="official_driver_partial",
            ),
        )
        raise RuntimeError("official scorecard driver returned partial or invalid evidence")
    games = official_driver_receipt.get("games")
    if not isinstance(games, list) or len(games) != PUBLIC_GAME_COUNT:
        write_json(
            RUN_ROOT / "official-driver-receipt.json",
            driver_receipt_for_output(
                official_driver_receipt,
                model_score_eligible=False,
                failure_reason="runner_game_receipt_cardinality_failure",
            ),
        )
        raise RuntimeError("official driver did not return 25 game receipts")
    try:
        driver_execution = lane_provider_factory.receipt(manifest["game_ids"])
        for index, game in enumerate(games):
            if not isinstance(game, Mapping) or game.get("status") != "completed":
                raise RuntimeError("official driver returned an incomplete game receipt")
            assignment = game.get("provider_assignment")
            if (
                not isinstance(assignment, Mapping)
                or assignment.get("source") != "factory"
                or assignment.get("factory_call_index") != index + 1
                or assignment.get("assigned") is not True
            ):
                raise RuntimeError("official driver provider assignment drifted")
            receipt = game.get("receipt")
            if not isinstance(receipt, Mapping):
                raise RuntimeError("official driver game receipt is absent")
            validate_provider_receipt(receipt.get("provider"))
    except RuntimeError:
        write_json(
            RUN_ROOT / "official-driver-receipt.json",
            driver_receipt_for_output(
                official_driver_receipt,
                model_score_eligible=False,
                failure_reason="runner_provider_contract_failure",
            ),
        )
        raise
    write_json(
        RUN_ROOT / "official-driver-receipt.json",
        driver_receipt_for_output(
            official_driver_receipt,
            model_score_eligible=True,
        ),
    )
    game_summaries: list[dict[str, Any]] = []
    for index, game in enumerate(games):
        if not isinstance(game, Mapping) or game.get("status") != "completed":
            raise RuntimeError("official driver returned an incomplete game receipt")
        game_id = manifest["game_ids"][index]
        if game.get("index") != index or game.get("game_id") != game_id:
            raise RuntimeError("official driver game order differs from the manifest")
        receipt = game.get("receipt")
        if not isinstance(receipt, Mapping) or receipt.get("reported_game_id") != game_id:
            raise RuntimeError("official driver trace provenance is inconsistent")
        receipt_checks = receipt.get("checks")
        provider_receipt = receipt.get("provider")
        if (
            not isinstance(receipt_checks, Mapping)
            or not receipt_checks
            or not all(receipt_checks.values())
        ):
            raise RuntimeError("official agent execution checks failed")
        validate_provider_receipt(provider_receipt)
        lane_name = lane_provider_factory.assignments[index]
        trace = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_sha": SOURCE_SHA,
            "manifest_index": index,
            "game_id": game_id,
            "seed": 0,
            "score_lane": lane_name,
            "execution_strategy": "sequential_manifest_order",
            "receipt": receipt,
        }
        trace_path = TRACE_DIR / f"{index:02d}-{game_id}.json"
        write_json(trace_path, sanitize_evidence(trace))
        game_summaries.append(
            {
                "manifest_index": index,
                "game_id": game_id,
                "score_lane": lane_name,
                "actions_taken": receipt.get("actions_taken"),
                "resets": receipt.get("resets"),
                "status": receipt.get("status"),
                "stop_reason": receipt.get("stop_reason"),
                "provider_score_calls": provider_receipt.get("score_calls"),
                "trace": str(trace_path.relative_to(RUN_ROOT)),
                "trace_sha256": sha256_file(trace_path),
            }
        )
    scorecard_container = official_driver_receipt.get("scorecard")
    scorecard = (
        scorecard_container.get("sanitized")
        if isinstance(scorecard_container, Mapping)
        else None
    )
    if not isinstance(scorecard, Mapping):
        raise RuntimeError("official driver did not return a sanitized scorecard")
    write_json(RUN_ROOT / "sanitized-scorecard.json", sanitize_evidence(scorecard))
    scorecard_summary = {
        "score": aggregate.get("official_score"),
        "total_environments": scorecard.get("total_environments"),
        "total_environments_completed": scorecard.get(
            "total_environments_completed"
        ),
        "total_levels": scorecard.get("total_levels"),
        "total_levels_completed": scorecard.get("total_levels_completed"),
        "total_actions": scorecard.get("total_actions"),
        "competition_mode": scorecard.get("competition_mode"),
    }
    after_score_hashes = {
        lane.name: frozen_state_hashes(lane.model, lane.prefix) for lane in lanes
    }
    frozen_unchanged = before_score_hashes == after_score_hashes
    if not frozen_unchanged:
        raise RuntimeError("frozen model or selected initial prefix changed during scoring")

    elapsed_seconds = time.perf_counter() - started
    if elapsed_seconds > MAX_RUNTIME_SECONDS:
        raise RuntimeError("official public score exceeded the frozen three-hour ceiling")
    environment_receipt = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "python": sys.version,
        "python_version_gate": {
            "expected": list(PYTHON_VERSION),
            "observed": list(sys.version_info[:2]),
            "passed": sys.version_info[:2] == PYTHON_VERSION,
        },
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": devices,
        "arc_agi": package_version("arc-agi"),
        "arcengine": package_version("arcengine"),
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "operation_mode": "normal",
        "anonymous_access": True,
        "arc_account_secret_provided": False,
        "competition_submission_performed": False,
        "competition_source_attached": False,
        "recordings_persisted": False,
        "model_weights_persisted": False,
        "inventory": official_driver_receipt.get("environment_preflight"),
        "gpu_assignment": {
            "training_and_canary": "cuda:0",
            "public_score_shards": driver_execution["shards"],
        },
        "clone_verification": clone_receipt,
        "runtime_source_integrity": source_integrity,
    }
    write_json(RUN_ROOT / "environment.json", sanitize_evidence(environment_receipt))
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_sha": SOURCE_SHA,
        "scope": "One official local NORMAL public-development scorecard; no competition submission.",
        "official_local_score": scorecard_summary.get("score"),
        "scorecard": scorecard_summary,
        "games": game_summaries,
        "game_count": len(game_summaries),
        "manifest_order_sha256": canonical_json_sha256(manifest["game_ids"]),
        "driver": driver_execution,
        "official_driver_receipt_sha256": sha256_file(
            RUN_ROOT / "official-driver-receipt.json"
        ),
        "runtime_gate": {
            "setup_elapsed_seconds": setup_elapsed_seconds,
            "projected_public_inference_seconds": projected_public_seconds,
            "output_margin_seconds": OUTPUT_MARGIN_SECONDS,
            "ceiling_seconds": MAX_RUNTIME_SECONDS,
            "passed": True,
        },
        "training_gate_passed": True,
        "runtime_projection_within_three_hours": True,
        "elapsed_seconds": elapsed_seconds,
        "frozen_state_unchanged_during_score": frozen_unchanged,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
    }
    write_json(RUN_ROOT / "summary.json", summary)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_sha": SOURCE_SHA,
        "status": "completed",
        "decision": "official_public_score_completed",
        "score_kind": "official_local_normal_public_development",
        "operation_mode": "normal",
        "scope": (
            "One bounded official ARC-AGI-3 public-development scorecard; "
            "not locked transfer, online, leaderboard, or competition evidence."
        ),
        "score": scorecard_summary.get("score"),
        "game_count": len(game_summaries),
        "manifest_sha256": PUBLIC_MANIFEST_SHA256,
        "requirements_lock_sha256": LOCK_SHA256,
        "model_repository": GPT2_REPOSITORY,
        "model_revision": GPT2_REVISION,
        "training_canary_sha256": sha256_file(RUN_ROOT / "training-canary.json"),
        "official_driver_receipt_sha256": sha256_file(
            RUN_ROOT / "official-driver-receipt.json"
        ),
        "scorecard_sha256": sha256_file(RUN_ROOT / "sanitized-scorecard.json"),
        "summary_sha256": sha256_file(RUN_ROOT / "summary.json"),
        "trace_hashes": {
            item["trace"]: item["trace_sha256"] for item in game_summaries
        },
        "execution_checks": {
            "normal_local_mode_only": True,
            "python_3_12_exact": sys.version_info[:2] == PYTHON_VERSION,
            "torch_2_10_0_cu128_exact": str(torch.__version__) == TORCH_VERSION,
            "cuda_12_8_exact": str(torch.version.cuda) == CUDA_RUNTIME_VERSION,
            "tested_runtime_source_rehashed_exact": all(
                value is True
                for key, value in source_integrity.items()
                if key != "files_rehashed"
            ),
            "anonymous_access_without_account_secret": True,
            "exactly_one_scorecard": driver_execution["one_scorecard"],
            "exactly_one_run_per_game": driver_execution["one_run_per_game"],
            "all_25_manifest_games_scored": len(game_summaries) == PUBLIC_GAME_COUNT,
            "manifest_order_preserved": [item["game_id"] for item in game_summaries]
            == manifest["game_ids"],
            "dual_t4_exact_clone_verified": clone_receipt["passed"],
            "deterministic_13_12_shards_exercised": driver_execution["lane_game_counts"]
            == {"cuda:0": 13, "cuda:1": 12},
            "no_concurrent_scorecard_writers": driver_execution["concurrent_scorecard_writers"]
            == 0,
            "parallel_score_execution_disabled": driver_execution[
                "parallel_score_execution"
            ]
            is False,
            "shared_scorecard_thread_safety_not_assumed": driver_execution[
                "parallel_score_execution_reason"
            ]
            == "arc_agi_0_9_9_shared_scorecard_thread_safety_not_proven",
            "no_parallel_speedup_claim": driver_execution["score_speedup_claimed"]
            is False,
            "frozen_state_unchanged": frozen_unchanged,
            "runtime_within_three_hours": elapsed_seconds <= MAX_RUNTIME_SECONDS,
            "training_gate_passed": training["gate"]["passed"] is True,
            "no_competition_submission": True,
            "no_model_weights_persisted": True,
        },
        "competition_submission_performed": False,
        "locked_transfer_evaluated": False,
        "online_mode_used": False,
        "leaderboard_submission_performed": False,
        "model_weights_persisted": False,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": elapsed_seconds,
    }
    if not all(result["execution_checks"].values()):
        raise RuntimeError("one or more official score execution checks failed")
    write_json(RUN_ROOT / "official-public-score-result.json", result)

    receipt = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "completed",
        "protocol": PROTOCOL,
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(runner_path),
        "source_manifest_sha256": sha256_file(RUN_ROOT / "source-manifest.json"),
        "result_sha256": sha256_file(RUN_ROOT / "official-public-score-result.json"),
        "dual_t4_verified": len(devices) == 2,
        "score_shards": driver_execution["lane_game_counts"],
        "one_local_scorecard": True,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(RUN_ROOT / "receipt.json", receipt)
    boundary = output_boundary_violations(RUN_ROOT)
    if any(boundary.values()):
        raise RuntimeError(f"output security boundary failed: {boundary}")
    write_json(RUN_ROOT / "artifact-manifest.json", artifact_manifest(RUN_ROOT))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        safe_error = sanitize_text(str(exc))[:2000]
        write_json(
            RUN_ROOT / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "protocol": PROTOCOL,
                "source_sha": SOURCE_SHA,
                "error_type": type(exc).__name__,
                "error": safe_error,
                "competition_submission_performed": False,
                "model_weights_persisted": False,
                "failed_at_utc": utc_now(),
            },
        )
        boundary = output_boundary_violations(RUN_ROOT)
        write_json(
            RUN_ROOT / "artifact-manifest.json",
            {**artifact_manifest(RUN_ROOT), "boundary_at_failure": boundary},
        )
        raise RuntimeError(f"official public score runner failed: {safe_error}") from None
