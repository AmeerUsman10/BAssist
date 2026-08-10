"""Private Kaggle execution driver for Muse Glimmer utility gate V1.

The driver is intentionally one-shot and evidence-first.  It clones one exact
source commit, builds one exact llama.cpp commit, downloads one exact official
quant/projector revision, runs deployment and synthetic gates, then promotes to
matched public-development and local evidence/tool comparisons only when the
pre-registered gates pass.

No model weights, API credentials, hidden reasoning, raw ARC recordings, or
competition artifacts are persisted under the output root.
"""

from __future__ import annotations

from contextlib import contextmanager
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
import statistics
import string
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
PROTOCOL = "muse_glimmer_arc_gate_v1"
RUNNER_SCHEMA_VERSION = 1
MAX_RUNTIME_SECONDS = 10_800.0
OUTPUT_MARGIN_SECONDS = 600.0

MODEL_REPOSITORY = "meta-models/Muse-Glimmer-30B-GGUF"
MODEL_REVISION = "93769bc7ab5ad1e9cd22d857e3138cf5d977ae81"
MODEL_FILENAME = "muse-glimmer-30B-kquant-17gb.gguf"
MMPROJ_FILENAME = "mmproj-kquant.gguf"
DFLASH_ENABLED = False
LLAMA_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT = "dd1ea524333b1e697489067d7a4c39c60d32beee"

EXPECTED_PUBLIC_MANIFEST_SHA256 = (
    "77b441ebbba044653c6abe64d17ab3023e54fd1ed6fec732bf0c525614e2e7dc"
)
MODEL_CALLS_PER_GAME = 8
ACTIONS_PER_GAME = 32
RESETS_PER_GAME = 2
OFFICIAL_GAME_COUNT = 25
PROJECTED_OFFICIAL_CALLS = MODEL_CALLS_PER_GAME * OFFICIAL_GAME_COUNT * 2
PROJECTED_RUNTIME_CEILING_SECONDS = 7_200.0

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "muse_glimmer_arc_gate_v1"
TEMP_ROOT = Path("/kaggle/temp/muse_glimmer_arc_gate_v1")
SOURCE_DIR = TEMP_ROOT / "source"
LLAMA_DIR = TEMP_ROOT / "llama.cpp"
MODEL_CACHE = TEMP_ROOT / "huggingface"
MODEL_DIR = TEMP_ROOT / "model"
LOG_DIR = RUN_ROOT / "logs"
CALL_DIR = RUN_ROOT / "calls"

OUTPUT_ALLOWLIST = {
    "artifact-manifest.json",
    "backend-manifest.json",
    "deployment.json",
    "environment.json",
    "evidence-hashes.json",
    "failure.json",
    "final-decision.json",
    "hardware-receipt.json",
    "model-manifest.json",
    "official-cfs-lite.json",
    "official-cfs-lite-metrics.json",
    "official-native.json",
    "official-native-metrics.json",
    "preregistration.json",
    "runtime-summary.json",
    "source-manifest.json",
    "synthetic-summary.json",
    "transfer-summary.json",
}
LOG_ALLOWLIST = {
    "arc-dependencies.log",
    "cpu-tests.log",
    "llama-build.log",
    "llama-configure.log",
    "server-cfs-lite.log",
    "server-initial.log",
    "server-native.log",
}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "arc_api_key",
    "authorization",
    "cookie",
    "credential",
    "guid",
    "password",
    "secret",
    "token",
}
TOKEN_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(?:bearer|x-api-key)\s*[:=]?\s*[A-Za-z0-9._~+/-]{12,}"),
)
_ACTIVE_SECRETS: set[str] = set()


class GateFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_source_sha(value: str) -> None:
    if len(value) != 40 or any(character not in string.hexdigits for character in value):
        raise GateFailure("source_sha_not_injected")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sanitize_text(value: str) -> str:
    result = str(value)
    for secret in sorted(_ACTIVE_SECRETS, key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED_SECRET]")
    for pattern in TOKEN_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GateFailure("non_finite_evidence")
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    raise GateFailure("non_json_evidence")


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    safe = [sanitize_text(str(part)) for part in command]
    print(json.dumps({"event": "command_start", "command": safe, "utc": utc_now()}), flush=True)
    process = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = sanitize_text(process.stdout or "")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
    if output:
        print(output[-8000:], flush=True)
    if process.returncode:
        raise GateFailure("command_failed")
    return subprocess.CompletedProcess(process.args, process.returncode, output, None)


def command_output(command: Sequence[str], *, timeout: float = 30.0) -> str:
    process = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        raise GateFailure("command_probe_failed")
    return sanitize_text(process.stdout or "").strip()


def git_output(directory: Path, *args: str) -> str:
    return command_output(["git", "-C", str(directory), *args], timeout=60.0)


def disk_receipt(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def gpu_rows() -> list[dict[str, Any]]:
    query = "index,name,memory.total,memory.used,driver_version,compute_cap"
    output = command_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_used_mib": int(float(parts[3])),
                "driver_version": parts[4],
                "compute_capability": parts[5],
            }
        )
    return rows


def hardware_receipt() -> dict[str, Any]:
    rows = gpu_rows()
    if len(rows) < 2:
        raise GateFailure("two_gpus_required")
    memory_total = sum(row["memory_total_mib"] for row in rows[:2])
    if memory_total < 30_000:
        raise GateFailure("aggregate_vram_below_30gb")
    return {
        "recorded_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpus": rows,
        "first_two_total_vram_mib": memory_total,
        "working_disk": disk_receipt(WORKING),
        "temp_disk": disk_receipt(TEMP_ROOT.parent),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def clone_exact(repository: str, commit: str, target: Path) -> None:
    run_command(["git", "init", str(target)])
    run_command(["git", "-C", str(target), "remote", "add", "origin", repository])
    run_command(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit], timeout=900)
    run_command(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"])
    if git_output(target, "rev-parse", "HEAD") != commit:
        raise GateFailure("git_commit_mismatch")


def source_manifest() -> dict[str, Any]:
    required_roots = ("glimmer_arc", "arcgpt2", "kaggle/muse-glimmer-arc-gate-v1")
    tracked = [
        name
        for name in git_output(SOURCE_DIR, "ls-files", "--", *required_roots).splitlines()
        if name
    ]
    if not tracked:
        raise GateFailure("source_manifest_empty")
    files: dict[str, Any] = {}
    for name in tracked:
        path = SOURCE_DIR / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_sha": SOURCE_SHA,
        "source_tree": git_output(SOURCE_DIR, "rev-parse", "HEAD^{tree}"),
        "files": files,
    }


def install_runtime_dependencies() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "huggingface_hub==0.36.0",
            "Pillow==11.3.0",
        ],
        timeout=900,
    )


def build_llama(hardware: Mapping[str, Any]) -> dict[str, Any]:
    clone_exact(LLAMA_REPOSITORY, LLAMA_COMMIT, LLAMA_DIR)
    names = " ".join(str(row.get("name", "")) for row in hardware["gpus"][:2]).lower()
    architecture = "75" if "t4" in names else "native"
    configure = [
        "cmake",
        "-S",
        str(LLAMA_DIR),
        "-B",
        str(LLAMA_DIR / "build"),
        "-DGGML_CUDA=ON",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
    ]
    run_command(configure, log=LOG_DIR / "llama-configure.log", timeout=1200)
    run_command(
        [
            "cmake",
            "--build",
            str(LLAMA_DIR / "build"),
            "--config",
            "Release",
            "-j",
            "2",
            "--target",
            "llama-server",
            "llama-cli",
        ],
        log=LOG_DIR / "llama-build.log",
        timeout=2400,
    )
    server = LLAMA_DIR / "build/bin/llama-server"
    cli = LLAMA_DIR / "build/bin/llama-cli"
    if not server.is_file() or not cli.is_file():
        raise GateFailure("llama_binaries_missing")
    version = command_output([str(server), "--version"], timeout=30)
    return {
        "schema_version": 1,
        "repository": LLAMA_REPOSITORY,
        "commit": LLAMA_COMMIT,
        "source_tree": git_output(LLAMA_DIR, "rev-parse", "HEAD^{tree}"),
        "server_sha256": sha256_file(server),
        "cli_sha256": sha256_file(cli),
        "version_output": version,
        "cuda_architectures": architecture,
        "cmake_version": command_output(["cmake", "--version"]).splitlines()[0],
        "compiler": command_output(["c++", "--version"]).splitlines()[0],
        "dflash_enabled": DFLASH_ENABLED,
    }


def download_model() -> tuple[Path, Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=str(MODEL_CACHE),
            local_dir=str(MODEL_DIR),
        )
    )
    mmproj = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MMPROJ_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=str(MODEL_CACHE),
            local_dir=str(MODEL_DIR),
        )
    )
    if not model.is_file() or model.stat().st_size < 15_000_000_000:
        raise GateFailure("model_download_incomplete")
    if not mmproj.is_file() or mmproj.stat().st_size < 1_000_000_000:
        raise GateFailure("mmproj_download_incomplete")
    manifest = {
        "schema_version": 1,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "license": "apache-2.0",
        "files": {
            MODEL_FILENAME: {"bytes": model.stat().st_size, "sha256": sha256_file(model)},
            MMPROJ_FILENAME: {"bytes": mmproj.stat().st_size, "sha256": sha256_file(mmproj)},
        },
        "quantization": "official_calibrated_17gb",
        "target_hardware_class": "24gb_or_multi_gpu",
        "dflash_enabled": False,
        "payload_location": "temporary_kaggle_storage_not_output",
        "model_payload_persisted": False,
    }
    return model, mmproj, manifest


def server_command(model: Path, mmproj: Path, port: int) -> list[str]:
    server = LLAMA_DIR / "build/bin/llama-server"
    help_text = command_output([str(server), "--help"], timeout=60)
    command = [
        str(server),
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--gpu-layers",
        "999",
        "--predict",
        "1024",
    ]
    if "--split-mode" in help_text:
        command.extend(["--split-mode", "layer"])
    if "--tensor-split" in help_text:
        command.extend(["--tensor-split", "1,1"])
    if "--flash-attn" in help_text:
        command.extend(["--flash-attn", "on"])
    if "--jinja" in help_text:
        command.append("--jinja")
    if "--alias" in help_text:
        command.extend(["--alias", "muse-glimmer-30b-kquant-17gb"])
    return command


class ServerHandle:
    def __init__(self, process: subprocess.Popen[str], log_handle: Any, command: Sequence[str], port: int) -> None:
        self.process = process
        self.log_handle = log_handle
        self.command = list(command)
        self.port = port

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.log_handle.close()


@contextmanager
def running_server(model: Path, mmproj: Path, *, port: int, log_name: str):
    from glimmer_arc.client import LlamaClient

    command = server_command(model, mmproj, port)
    log_path = LOG_DIR / log_name
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    handle = ServerHandle(process, log_handle, command, port)
    client = LlamaClient(f"http://127.0.0.1:{port}/v1", request_timeout=150.0, seed=0)
    deadline = time.monotonic() + 900.0
    last_code: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            handle.stop()
            raise GateFailure("llama_server_exited_during_load")
        try:
            client.health()
            client.discover_model()
            break
        except Exception as exc:
            last_code = type(exc).__name__
            time.sleep(5)
    else:
        handle.stop()
        raise GateFailure("llama_server_load_timeout")
    try:
        yield handle, client
    finally:
        handle.stop()


def create_smoke_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), "red")
    draw = ImageDraw.Draw(image)
    draw.rectangle((96, 96, 160, 160), fill="blue")
    image.save(path)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return None
    index = min(len(cleaned) - 1, max(0, math.ceil(fraction * len(cleaned)) - 1))
    return cleaned[index]


def deployment_gate(client: Any, image_path: Path) -> dict[str, Any]:
    structured_rows: list[dict[str, Any]] = []
    for index in range(20):
        expected = index + 1
        valid = False
        correct = False
        try:
            value = client.complete_json(
                system="Return one strict JSON object and no prose.",
                user=json.dumps(
                    {
                        "task": f"Add one to {index}.",
                        "schema": {"answer": "INTEGER"},
                    },
                    sort_keys=True,
                ),
                purpose="deployment_structured_json",
                max_tokens=64,
                temperature=0.0,
            )
            valid = isinstance(value.get("answer"), int) and not isinstance(value.get("answer"), bool)
            correct = valid and value["answer"] == expected
        except Exception:
            pass
        structured_rows.append({"index": index, "valid": valid, "correct": correct})

    deterministic_hashes: list[str] = []
    for _ in range(3):
        try:
            value = client.complete_json(
                system="Return strict JSON only.",
                user='Return exactly {"status":"muse-ready","value":7}.',
                purpose="deployment_determinism",
                max_tokens=64,
                temperature=0.0,
            )
            deterministic_hashes.append(canonical_sha256(value))
        except Exception:
            deterministic_hashes.append("ERROR")

    image_valid = image_correct = False
    try:
        image_value = client.complete_json(
            system="Inspect the image and return strict JSON only.",
            user='Return {"dominant_color":"red|blue|other"} for the image.',
            purpose="deployment_image",
            max_tokens=80,
            temperature=0.0,
            image_path=image_path,
        )
        image_valid = image_value.get("dominant_color") in {"red", "blue", "other"}
        image_correct = image_value.get("dominant_color") == "red"
    except Exception:
        image_value = None

    action_valid = action_correct = False
    try:
        action_value = client.complete_json(
            system="Choose only an exact legal signature and return strict JSON.",
            user=json.dumps(
                {
                    "history": [
                        {"action": "ACTION1", "outcome": "NO_CHANGE"},
                        {"action": "ACTION2", "outcome": "PROGRESS"},
                    ],
                    "candidates": ["ACTION1", "ACTION2"],
                    "schema": {"signature": "EXACT_CANDIDATE"},
                },
                sort_keys=True,
            ),
            purpose="deployment_legal_action",
            max_tokens=80,
            temperature=0.0,
        )
        action_valid = action_value.get("signature") in {"ACTION1", "ACTION2"}
        action_correct = action_value.get("signature") == "ACTION2"
    except Exception:
        action_value = None

    latencies = [receipt.latency_seconds for receipt in client.receipts]
    structured_validity = sum(row["valid"] for row in structured_rows) / len(structured_rows)
    structured_accuracy = sum(row["correct"] for row in structured_rows) / len(structured_rows)
    deterministic = len(set(deterministic_hashes)) == 1 and deterministic_hashes[0] != "ERROR"
    passed = (
        structured_validity >= 0.95
        and structured_accuracy >= 0.90
        and deterministic
        and image_valid
        and image_correct
        and action_valid
        and action_correct
    )
    return {
        "schema_version": 1,
        "structured_rows": structured_rows,
        "structured_validity": structured_validity,
        "structured_accuracy": structured_accuracy,
        "deterministic_hashes": deterministic_hashes,
        "deterministic": deterministic,
        "image_valid": image_valid,
        "image_correct": image_correct,
        "action_valid": action_valid,
        "action_correct": action_correct,
        "latency_seconds": {
            "count": len(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "passed": passed,
    }


def persist_client_calls(client: Any, name: str) -> None:
    append_jsonl(CALL_DIR / f"{name}.jsonl", client.receipt_slice(0))


def install_arc_dependencies_and_test() -> None:
    lock = SOURCE_DIR / "arcgpt2/requirements-arc3.lock"
    if sha256_file(SOURCE_DIR / "arcgpt2/official_public_manifest.json") != EXPECTED_PUBLIC_MANIFEST_SHA256:
        raise GateFailure("official_manifest_hash_mismatch")
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
            str(lock),
        ],
        log=LOG_DIR / "arc-dependencies.log",
        timeout=1200,
    )
    env = os.environ.copy()
    env.update(
        {
            "ARC_API_KEY": "",
            "CUDA_VISIBLE_DEVICES": "",
            "OPERATION_MODE": "normal",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(SOURCE_DIR),
        }
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "glimmer_arc/tests",
            "arcgpt2/tests/test_official_observation.py",
            "arcgpt2/tests/test_official_score_agent.py",
            "arcgpt2/tests/test_official_public_score.py",
        ],
        cwd=SOURCE_DIR,
        env=env,
        log=LOG_DIR / "cpu-tests.log",
        timeout=1200,
    )


def summarize_official(receipt: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    scorecard = receipt.get("scorecard") if isinstance(receipt.get("scorecard"), Mapping) else {}
    sanitized_scorecard = scorecard.get("sanitized") if isinstance(scorecard, Mapping) else {}
    aggregate = receipt.get("aggregate") if isinstance(receipt.get("aggregate"), Mapping) else {}
    return {
        "valid": aggregate.get("valid") is True,
        "official_score": aggregate.get("official_score"),
        "observed_score": aggregate.get("observed_score"),
        "levels_completed": (
            sanitized_scorecard.get("total_levels_completed")
            if isinstance(sanitized_scorecard, Mapping)
            else None
        ),
        "environments_completed": (
            sanitized_scorecard.get("total_environments_completed")
            if isinstance(sanitized_scorecard, Mapping)
            else None
        ),
        "actions": (
            sanitized_scorecard.get("total_actions")
            if isinstance(sanitized_scorecard, Mapping)
            else None
        ),
        "model_calls": metrics.get("model_calls"),
        "fallback_rate": metrics.get("fallback_rate"),
        "reasons": aggregate.get("reasons", []),
    }


def final_disposition(
    *,
    deployment: Mapping[str, Any],
    synthetic: Mapping[str, Any] | None,
    transfer: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None,
    cfs: Mapping[str, Any] | None,
    official_skipped_reason: str | None,
) -> dict[str, Any]:
    if deployment.get("passed") is not True:
        return {
            "disposition": "KILL",
            "reason": "The pinned quantized deployment failed the correctness gate.",
            "next_action": "Do not integrate Muse Glimmer into the permanent stack; retain artifacts as negative evidence.",
        }
    if synthetic is None or synthetic.get("gate", {}).get("passed") is not True:
        return {
            "disposition": "KILL",
            "reason": "Deployment worked, but the model/action protocol failed the preregistered synthetic qualification gate.",
            "next_action": "Stop this Glimmer agent track rather than spending official ARC actions on an unqualified policy.",
        }

    transfer_useful = bool(transfer and transfer.get("any_useful"))
    transfer_controller = bool(transfer and transfer.get("controller_positive"))
    native_levels = int(native.get("levels_completed") or 0) if native else 0
    cfs_levels = int(cfs.get("levels_completed") or 0) if cfs else 0
    official_any = max(native_levels, cfs_levels) > 0
    controller_positive = False
    if native and cfs and native.get("valid") and cfs.get("valid"):
        if cfs_levels > native_levels:
            controller_positive = True
        elif cfs_levels == native_levels and cfs_levels > 0:
            native_calls = int(native.get("model_calls") or 0)
            cfs_calls = int(cfs.get("model_calls") or 0)
            native_actions = int(native.get("actions") or 0)
            cfs_actions = int(cfs.get("actions") or 0)
            controller_positive = (
                (native_calls > 0 and cfs_calls <= 0.8 * native_calls)
                or (native_actions > 0 and cfs_actions <= 0.8 * native_actions)
            ) and float(cfs.get("fallback_rate") or 0.0) <= float(native.get("fallback_rate") or 0.0) + 0.01

    if official_any and controller_positive and transfer_useful and transfer_controller:
        return {
            "disposition": "ADOPT",
            "reason": "Glimmer produced nonzero official public-development capability, CFS-Lite improved the matched arm, and the controller transferred to evidence/tool choice.",
            "next_action": "Adopt the pinned Glimmer+CFS configuration as a controlled local worker; keep the exact manifests and rerun only under a separately versioned protocol.",
        }
    if official_any:
        winning = "cfs_lite" if cfs_levels > native_levels else "native"
        return {
            "disposition": "SPECIALIZE",
            "reason": f"Muse Glimmer showed bounded official ARC utility, but cross-domain/controller evidence was insufficient. The better arm was {winning}.",
            "next_action": f"Use only the frozen {winning} configuration for unknown-world research; do not generalize it to trading or autonomous business decisions.",
        }
    if transfer_useful:
        preferred = (
            "cfs_lite"
            if transfer and transfer["arms"]["cfs_lite"]["answer_accuracy"] >= transfer["arms"]["native"]["answer_accuracy"]
            else "native"
        )
        return {
            "disposition": "SPECIALIZE",
            "reason": "The model did not complete an official public-development level, but it passed the local evidence/tool benchmark.",
            "next_action": f"Use Muse Glimmer only as a local evidence/tool worker in {preferred} mode; kill the ARC integration and do not treat synthetic CFS performance as ARC evidence.",
        }
    if official_skipped_reason:
        return {
            "disposition": "CONTINUE_EXPERIMENTALLY",
            "reason": f"Deployment and synthetic gates passed, but the official comparison was not validly completed: {official_skipped_reason}.",
            "next_action": "The only justified continuation is a new, separately frozen hardware-fit run; no product integration is authorized from this evidence.",
        }
    return {
        "disposition": "KILL",
        "reason": "The model produced no official level completion and no useful transfer result under the frozen budgets.",
        "next_action": "Stop this Muse Glimmer agent track and retain the negative result; do not spend more Kaggle time on prompt retuning against the same public environments.",
    }


def boundary_violations(root: Path) -> dict[str, list[str]]:
    unexpected: list[str] = []
    payloads: list[str] = []
    secrets: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        allowed = (
            (len(relative.parts) == 1 and relative.name in OUTPUT_ALLOWLIST)
            or (len(relative.parts) == 2 and relative.parts[0] == "logs" and relative.name in LOG_ALLOWLIST)
            or (len(relative.parts) == 2 and relative.parts[0] == "calls" and relative.suffix == ".jsonl")
        )
        if not allowed:
            unexpected.append(str(relative))
        if path.suffix.lower() in {".gguf", ".bin", ".pt", ".pth", ".safetensors"}:
            payloads.append(str(relative))
        if path.suffix.lower() in {".json", ".jsonl", ".log", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(secret and secret in text for secret in _ACTIVE_SECRETS):
                secrets.append(str(relative))
            if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
                secrets.append(str(relative))
    return {
        "unexpected_files": sorted(set(unexpected)),
        "model_payloads": sorted(set(payloads)),
        "secret_leaks": sorted(set(secrets)),
    }


def artifact_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
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


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": "muse-glimmer-utility-gate/v1-frozen",
        "frozen_before_model_inference": True,
        "source_sha": SOURCE_SHA,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quant": MODEL_FILENAME,
            "projector": MMPROJ_FILENAME,
            "dflash": False,
            "backend_repository": LLAMA_REPOSITORY,
            "backend_commit": LLAMA_COMMIT,
        },
        "budgets": {
            "official_games_per_arm": OFFICIAL_GAME_COUNT,
            "model_calls_per_game": MODEL_CALLS_PER_GAME,
            "actions_per_game": ACTIONS_PER_GAME,
            "resets_per_game": RESETS_PER_GAME,
            "shortlist": 16,
            "plan_length": 4,
            "context_tokens": 8192,
            "runtime_seconds": MAX_RUNTIME_SECONDS,
        },
        "gates": {
            "structured_json_success_min": 0.95,
            "illegal_action_rate_max": 0.01,
            "fallback_rate_max": 0.05,
            "synthetic_task_accuracy_any_arm_min": 0.25,
            "projected_official_runtime_max_seconds": PROJECTED_RUNTIME_CEILING_SECONDS,
            "official_model_positive": "at least one valid public-development level",
            "controller_positive": "more levels, or tied nonzero with >=20% fewer calls/actions and no fallback regression",
        },
        "network_boundary": {
            "model_provider_during_scored_play": "loopback_only",
            "external_before_scored_play": ["github source", "huggingface model", "pypi dependencies"],
            "official_sdk_during_public_score": "official ARC NORMAL endpoint only",
        },
        "terminal_decisions": ["ADOPT", "SPECIALIZE", "CONTINUE_EXPERIMENTALLY", "KILL"],
        "claim_exclusions": [
            "no locked/private generalization claim",
            "no competition or leaderboard claim",
            "no synthetic-to-official claim laundering",
            "no hidden reasoning artifact",
            "no post-outcome retuning in V1",
        ],
    }


def main() -> int:
    validate_source_sha(SOURCE_SHA)
    started = time.monotonic()
    deadline = started + MAX_RUNTIME_SECONDS
    for root in (RUN_ROOT, TEMP_ROOT):
        if root.exists():
            shutil.rmtree(root)
    RUN_ROOT.mkdir(parents=True)
    LOG_DIR.mkdir()
    CALL_DIR.mkdir()
    MODEL_DIR.mkdir(parents=True)
    write_json(RUN_ROOT / "preregistration.json", preregistration())

    for name in ("KAGGLE_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ARC_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            _ACTIVE_SECRETS.add(value)

    deployment: dict[str, Any] = {"passed": False, "status": "not_started"}
    synthetic: dict[str, Any] | None = None
    transfer: dict[str, Any] | None = None
    native_summary: dict[str, Any] | None = None
    cfs_summary: dict[str, Any] | None = None
    official_skipped_reason: str | None = None
    server_commands: list[list[str]] = []

    try:
        clone_exact(SOURCE_REPOSITORY, SOURCE_SHA, SOURCE_DIR)
        sys.path.insert(0, str(SOURCE_DIR))
        write_json(RUN_ROOT / "source-manifest.json", source_manifest())

        hardware = hardware_receipt()
        write_json(RUN_ROOT / "hardware-receipt.json", hardware)
        if hardware["temp_disk"]["free_bytes"] < 35_000_000_000:
            raise GateFailure("insufficient_temp_disk")

        install_runtime_dependencies()
        backend = build_llama(hardware)
        write_json(RUN_ROOT / "backend-manifest.json", backend)
        model, mmproj, model_manifest = download_model()
        write_json(RUN_ROOT / "model-manifest.json", model_manifest)

        smoke_image = TEMP_ROOT / "smoke.png"
        create_smoke_image(smoke_image)
        with running_server(model, mmproj, port=8080, log_name="server-initial.log") as (handle, client):
            server_commands.append(handle.command)
            deployment = deployment_gate(client, smoke_image)
            deployment["server_command"] = [sanitize_text(part) for part in handle.command]
            deployment["gpu_after_load"] = gpu_rows()
            write_json(RUN_ROOT / "deployment.json", sanitize(deployment))
            persist_client_calls(client, "deployment")
            if deployment.get("passed") is not True:
                raise GateFailure("deployment_gate_failed")

            from glimmer_arc.synthetic import run_synthetic_qualification, run_transfer_benchmark

            synthetic_client_start = len(client.receipts)
            synthetic = run_synthetic_qualification(client, task_count=20)
            write_json(RUN_ROOT / "synthetic-summary.json", sanitize(synthetic))
            append_jsonl(CALL_DIR / "synthetic.jsonl", client.receipt_slice(synthetic_client_start))
            if synthetic.get("gate", {}).get("passed") is not True:
                raise GateFailure("synthetic_gate_failed")

            transfer_client_start = len(client.receipts)
            transfer = run_transfer_benchmark(client)
            write_json(RUN_ROOT / "transfer-summary.json", sanitize(transfer))
            append_jsonl(CALL_DIR / "transfer.jsonl", client.receipt_slice(transfer_client_start))

            p90 = deployment.get("latency_seconds", {}).get("p90")
            projected = (
                float(p90) * PROJECTED_OFFICIAL_CALLS + 1200.0
                if isinstance(p90, (int, float)) and not isinstance(p90, bool)
                else math.inf
            )
            deployment["official_runtime_projection_seconds"] = projected
            deployment["official_runtime_projection_pass"] = projected <= PROJECTED_RUNTIME_CEILING_SECONDS
            write_json(RUN_ROOT / "deployment.json", sanitize(deployment))

        if deployment.get("official_runtime_projection_pass") is not True:
            official_skipped_reason = "runtime_projection_failed"
        elif time.monotonic() >= deadline - 3600:
            official_skipped_reason = "insufficient_remaining_runtime_before_official_gate"
        else:
            install_arc_dependencies_and_test()
            os.environ.update(
                {
                    "ARC_API_KEY": "",
                    "OPERATION_MODE": "normal",
                    "PYTHONHASHSEED": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            from glimmer_arc.policy import PolicyConfig, run_official_arm

            manifest_path = str(SOURCE_DIR / "arcgpt2/official_public_manifest.json")
            native_budget_end = min(deadline - 2100.0, time.monotonic() + 3000.0)
            with running_server(model, mmproj, port=8081, log_name="server-native.log") as (handle, native_client):
                server_commands.append(handle.command)
                native_receipt, native_metrics = run_official_arm(
                    client=native_client,
                    config=PolicyConfig(
                        mode="native",
                        max_model_calls=MODEL_CALLS_PER_GAME,
                        max_repair_calls=1,
                        shortlist_size=16,
                        plan_length=4,
                        max_actions=ACTIONS_PER_GAME,
                        max_resets=RESETS_PER_GAME,
                        max_click_candidates_per_action=24,
                        low_discrepancy_points=16,
                    ),
                    manifest_path=manifest_path,
                    temp_root=str(TEMP_ROOT / "official-native"),
                    monotonic_deadline=native_budget_end,
                )
                write_json(RUN_ROOT / "official-native.json", sanitize(native_receipt))
                write_json(RUN_ROOT / "official-native-metrics.json", sanitize(native_metrics))
                persist_client_calls(native_client, "official-native")
                native_summary = summarize_official(native_receipt, native_metrics)

            if sha256_file(model) != model_manifest["files"][MODEL_FILENAME]["sha256"] or sha256_file(mmproj) != model_manifest["files"][MMPROJ_FILENAME]["sha256"]:
                raise GateFailure("model_hash_changed_between_arms")

            if time.monotonic() >= deadline - 1800:
                official_skipped_reason = "native_completed_but_insufficient_runtime_for_matched_cfs_arm"
            else:
                cfs_budget_end = deadline - OUTPUT_MARGIN_SECONDS
                with running_server(model, mmproj, port=8082, log_name="server-cfs-lite.log") as (handle, cfs_client):
                    server_commands.append(handle.command)
                    cfs_receipt, cfs_metrics = run_official_arm(
                        client=cfs_client,
                        config=PolicyConfig(
                            mode="cfs_lite",
                            max_model_calls=MODEL_CALLS_PER_GAME,
                            max_repair_calls=1,
                            shortlist_size=16,
                            plan_length=1,
                            max_actions=ACTIONS_PER_GAME,
                            max_resets=RESETS_PER_GAME,
                            max_click_candidates_per_action=24,
                            low_discrepancy_points=16,
                        ),
                        manifest_path=manifest_path,
                        temp_root=str(TEMP_ROOT / "official-cfs-lite"),
                        monotonic_deadline=cfs_budget_end,
                    )
                    write_json(RUN_ROOT / "official-cfs-lite.json", sanitize(cfs_receipt))
                    write_json(RUN_ROOT / "official-cfs-lite-metrics.json", sanitize(cfs_metrics))
                    persist_client_calls(cfs_client, "official-cfs-lite")
                    cfs_summary = summarize_official(cfs_receipt, cfs_metrics)

        decision = final_disposition(
            deployment=deployment,
            synthetic=synthetic,
            transfer=transfer,
            native=native_summary,
            cfs=cfs_summary,
            official_skipped_reason=official_skipped_reason,
        )
        decision.update(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "source_sha": SOURCE_SHA,
                "deployment_passed": deployment.get("passed") is True,
                "synthetic_passed": bool(synthetic and synthetic.get("gate", {}).get("passed")),
                "transfer_any_useful": bool(transfer and transfer.get("any_useful")),
                "transfer_controller_positive": bool(transfer and transfer.get("controller_positive")),
                "official_native": native_summary,
                "official_cfs_lite": cfs_summary,
                "official_skipped_reason": official_skipped_reason,
                "competition_submission_performed": False,
                "leaderboard_action_performed": False,
                "model_weights_persisted": False,
                "hidden_reasoning_persisted": False,
            }
        )
        write_json(RUN_ROOT / "final-decision.json", sanitize(decision))

    except GateFailure as exc:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_sha": SOURCE_SHA,
            "status": "failed",
            "code": exc.code,
            "phase_disposition": "KILL",
            "competition_submission_performed": False,
            "model_weights_persisted": False,
            "recorded_at_utc": utc_now(),
        }
        write_json(RUN_ROOT / "failure.json", failure)
        decision = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_sha": SOURCE_SHA,
            "disposition": "KILL",
            "reason": f"Execution stopped at preregistered failure gate: {exc.code}.",
            "next_action": "Do not integrate this configuration; retain the failure evidence and stop V1.",
            "failure_code": exc.code,
            "competition_submission_performed": False,
            "model_weights_persisted": False,
            "hidden_reasoning_persisted": False,
        }
        write_json(RUN_ROOT / "final-decision.json", decision)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_sha": SOURCE_SHA,
            "status": "failed",
            "code": "unhandled_runner_failure",
            "error_type": type(exc).__name__,
            "phase_disposition": "KILL",
            "competition_submission_performed": False,
            "model_weights_persisted": False,
            "recorded_at_utc": utc_now(),
        }
        write_json(RUN_ROOT / "failure.json", failure)
        write_json(
            RUN_ROOT / "final-decision.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "source_sha": SOURCE_SHA,
                "disposition": "KILL",
                "reason": "The one-shot runner encountered an unhandled execution failure.",
                "next_action": "Do not integrate this configuration; inspect the sanitized failure artifact only.",
                "failure_code": "unhandled_runner_failure",
                "competition_submission_performed": False,
                "model_weights_persisted": False,
                "hidden_reasoning_persisted": False,
            },
        )

    runtime = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_sha": SOURCE_SHA,
        "started_at_utc": datetime.fromtimestamp(time.time() - (time.monotonic() - started), timezone.utc).isoformat(),
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "runtime_ceiling_seconds": MAX_RUNTIME_SECONDS,
        "server_commands": [[sanitize_text(part) for part in command] for command in server_commands],
        "package_versions": {
            "huggingface_hub": package_version("huggingface-hub"),
            "Pillow": package_version("Pillow"),
            "arc-agi": package_version("arc-agi"),
            "arcengine": package_version("arcengine"),
        },
        "competition_submission_performed": False,
    }
    write_json(RUN_ROOT / "runtime-summary.json", sanitize(runtime))

    violations = boundary_violations(RUN_ROOT)
    write_json(RUN_ROOT / "evidence-hashes.json", {"boundary_violations": violations})
    if any(violations.values()):
        # Preserve the violation as evidence but make the final result unusable.
        write_json(
            RUN_ROOT / "failure.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "source_sha": SOURCE_SHA,
                "status": "failed",
                "code": "output_boundary_violation",
                "violations": violations,
                "phase_disposition": "KILL",
            },
        )
        write_json(
            RUN_ROOT / "final-decision.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "source_sha": SOURCE_SHA,
                "disposition": "KILL",
                "reason": "Evidence boundary validation failed.",
                "next_action": "Do not use or promote this run.",
                "failure_code": "output_boundary_violation",
                "competition_submission_performed": False,
                "leaderboard_action_performed": False,
                "model_weights_persisted": False,
                "hidden_reasoning_persisted": False,
            },
        )

    write_json(RUN_ROOT / "artifact-manifest.json", artifact_manifest(RUN_ROOT))
    decision = json.loads((RUN_ROOT / "final-decision.json").read_text(encoding="utf-8"))
    print(json.dumps({"event": "final_decision", **decision}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
