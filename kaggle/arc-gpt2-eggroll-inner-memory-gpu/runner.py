"""Private Kaggle runner for ``eggroll_inner_memory_v1``.

The runner pins exact source and model revisions, runs focused CPU preflight
tests, verifies the available dual-T4 environment, executes the preregistered
gate on one T4, and emits evidence only. It never submits to a competition and
never preserves trained weights.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROTOCOL = "eggroll_inner_memory_v1"
RUNNER_SCHEMA_VERSION = 1

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_eggroll_inner_memory_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_eggroll_inner_memory_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
MODEL_DIR = TEMP_ROOT / "gpt2-small-pinned"
OUTPUT_DIR = RUN_ROOT / "outputs"
LOG_DIR = RUN_ROOT / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_source_sha(value: str) -> None:
    if len(value) != 40 or any(character not in string.hexdigits for character in value):
        raise RuntimeError("runner source SHA was not injected")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> None:
    print(json.dumps({"event": "command_start", "command": command, "utc": utc_now()}), flush=True)
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
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        code = process.wait()
    finally:
        if handle:
            handle.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def artifact_manifest() -> dict[str, Any]:
    excluded = {"artifact-manifest.json", "failure.json"}
    files = {}
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files[str(path.relative_to(RUN_ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": 1,
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "files": files,
    }


def forbidden_model_artifacts() -> list[str]:
    forbidden_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
    return [
        str(path.relative_to(RUN_ROOT))
        for path in sorted(RUN_ROOT.rglob("*"))
        if path.is_file()
        and (
            path.suffix.lower() in forbidden_suffixes
            or path.name.startswith(("pytorch_model", "model-", "checkpoint-"))
        )
    ]


def main() -> None:
    validate_source_sha(SOURCE_SHA)
    started = time.time()
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
    OUTPUT_DIR.mkdir(parents=True)
    LOG_DIR.mkdir(parents=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "transformers==4.57.6",
            "safetensors==0.6.2",
            "pytest==8.4.2",
        ],
        log=LOG_DIR / "dependency-install.log",
    )
    run(["git", "init", str(SOURCE_DIR)])
    run(["git", "-C", str(SOURCE_DIR), "remote", "add", "origin", SOURCE_REPOSITORY])
    run(["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", SOURCE_SHA])
    run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"])
    resolved_sha = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved_sha != SOURCE_SHA:
        raise RuntimeError("cloned source SHA disagrees with injected source SHA")

    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arcgpt2/tests/test_eggroll_inner_memory.py",
            "arcgpt2/tests/test_meta_soft_heldout_binding.py",
            "arcgpt2/tests/test_completion_scorer.py",
        ],
        cwd=SOURCE_DIR,
        env=cpu_env,
        log=LOG_DIR / "cpu-preflight-tests.log",
    )

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("the preregistered environment requires exactly two CUDA devices")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "T4" not in name.upper() or tuple(capability) != (7, 5):
            raise RuntimeError(f"device {index} is not a Tesla T4 with capability 7.5")
        probe = torch.ones(256, device=f"cuda:{index}")
        if float((probe @ probe).item()) != 256.0:
            raise RuntimeError(f"CUDA compute probe failed on device {index}")
        devices.append(
            {"index": index, "name": name, "compute_capability": list(capability)}
        )
    del probe
    torch.cuda.empty_cache()
    subprocess.run(["nvidia-smi"], check=True)

    snapshot_download(
        repo_id=GPT2_REPOSITORY,
        revision=GPT2_REVISION,
        local_dir=MODEL_DIR,
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
    write_json(
        RUN_ROOT / "model-manifest.json",
        {
            "repository": GPT2_REPOSITORY,
            "revision": GPT2_REVISION,
            "files": {
                str(path.relative_to(MODEL_DIR)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(MODEL_DIR.rglob("*"))
                if path.is_file() and ".cache" not in path.parts
            },
        },
    )
    environment = {
        "recorded_at_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": devices,
        "execution_device": "physical cuda:0",
        "second_t4_reserved_unallocated": True,
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "tf32_disabled_by_protocol": True,
    }
    write_json(RUN_ROOT / "environment.json", environment)

    gpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    gpu_env["TOKENIZERS_PARALLELISM"] = "false"
    gpu_env["PYTHONHASHSEED"] = "0"
    run(
        [
            sys.executable,
            "-m",
            "arcgpt2.eggroll_inner_memory",
            "--model-name",
            str(MODEL_DIR),
            "--model-revision",
            GPT2_REVISION,
            "--source-sha",
            SOURCE_SHA,
            "--output-dir",
            str(OUTPUT_DIR),
            "--require-cuda",
        ],
        cwd=SOURCE_DIR,
        env=gpu_env,
        log=LOG_DIR / "eggroll-inner-memory.log",
    )
    result_path = OUTPUT_DIR / "eggroll-result.json"
    if not result_path.is_file():
        raise RuntimeError("required eggroll-result.json is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("protocol") != PROTOCOL or result.get("source_sha") != SOURCE_SHA:
        raise RuntimeError("result provenance does not match the runner")
    if result.get("competition_submission_performed") is not False:
        raise RuntimeError("competition-submission boundary was violated")
    if result.get("model_weights_persisted") is not False:
        raise RuntimeError("model-artifact boundary was violated")
    forbidden = forbidden_model_artifacts()
    if forbidden:
        raise RuntimeError(f"forbidden model artifacts were emitted: {forbidden}")
    execution = result.get("execution_checks", {})
    if not execution or not all(execution.values()):
        raise RuntimeError("one or more execution-contract checks failed")
    receipt = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "completed",
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "result_sha256": sha256_file(result_path),
        "gate_passed": result.get("gate", {}).get("passed") is True,
        "decision": result.get("decision"),
        "dual_t4_verified": len(devices) == 2,
        "execution_device": "physical cuda:0",
        "candidate_evaluations": 8 * 4 * 4 * 1024 + 1024,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "forbidden_model_artifacts": forbidden,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN_ROOT / "receipt.json", receipt)
    write_json(RUN_ROOT / "artifact-manifest.json", artifact_manifest())
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(
            RUN_ROOT / "failure.json",
            {
                "status": "failed",
                "source_sha": SOURCE_SHA,
                "protocol": PROTOCOL,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": utc_now(),
            },
        )
        if RUN_ROOT.exists():
            write_json(RUN_ROOT / "artifact-manifest.json", artifact_manifest())
        raise
