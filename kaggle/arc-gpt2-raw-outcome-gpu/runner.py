"""Private bounded Kaggle T4 runner for the raw-outcome gradient-memory gate.

Runs one pretrained GPT-2 Small seed.  A negative scientific result is a valid
experiment; only infrastructure, provenance, or CUDA failures exit nonzero.
No model is saved and no Kaggle competition API is used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import string
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEED = 314159
GROUP_SEED = 920000
ACTION = "A1"
MAX_OUTER_STEPS = 200
PLATEAU_PATIENCE = 60
RUNNER_SCHEMA_VERSION = 1

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_raw_outcome_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_raw_outcome_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
MODEL_DIR = TEMP_ROOT / "gpt2-small-pinned"
OUTPUT_DIR = RUN_ROOT / "outputs" / "pretrained"
LOG_DIR = RUN_ROOT / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise RuntimeError("runner source SHA was not resolved by the dispatch workflow")


def run(command: list[str], *, cwd: Path | None = None, log: Path | None = None) -> None:
    printable = " ".join(command)
    print(json.dumps({"event": "command_start", "command": printable, "utc": utc_now()}), flush=True)
    handle = log.open("w", encoding="utf-8") if log else None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
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
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        query = "timestamp,name,memory.used,memory.total,utilization.gpu"
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 5:
                        self.samples.append(
                            {
                                "sampled_at_utc": utc_now(),
                                "device_timestamp": parts[0],
                                "name": parts[1],
                                "memory_used_mib": int(parts[2]),
                                "memory_total_mib": int(parts[3]),
                                "utilization_percent": int(parts[4]),
                            }
                        )
            except Exception as exc:
                self.samples.append({"sampled_at_utc": utc_now(), "sampling_error": type(exc).__name__})
            self.stop_event.wait(5)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)


def main() -> None:
    started = time.time()
    validate_source_sha(SOURCE_SHA)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ARC_GPT2_SOURCE_SHA": SOURCE_SHA,
        }
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "transformers==4.57.6",
            "safetensors==0.6.2",
            "pytest==8.4.2",
        ],
        log=LOG_DIR / "dependency-install.log",
    )

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; refusing to spend the run on CPU")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "T4" not in device_name.upper() or tuple(capability) != (7, 5):
        raise RuntimeError(f"expected Tesla T4 capability 7.5, got {device_name!r} {capability!r}")
    probe = torch.ones(256, device="cuda")
    probe_value = float((probe @ probe).item())
    if probe_value != 256.0:
        raise RuntimeError("CUDA compute probe returned an invalid result")

    write_json(
        RUN_ROOT / "environment.json",
        {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "recorded_at_utc": utc_now(),
            "source_sha": SOURCE_SHA,
            "gpt2_repository": GPT2_REPOSITORY,
            "gpt2_revision": GPT2_REVISION,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": device_name,
            "cuda_compute_capability": list(capability),
            "cuda_compute_probe": probe_value,
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
        },
    )
    subprocess.run(["nvidia-smi"], check=True)

    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    run(["git", "init", str(SOURCE_DIR)])
    run(["git", "-C", str(SOURCE_DIR), "remote", "add", "origin", SOURCE_REPOSITORY])
    run(["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", SOURCE_SHA])
    run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"])
    resolved_sha = subprocess.check_output(["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
    if resolved_sha != SOURCE_SHA:
        raise RuntimeError(f"source SHA mismatch: expected {SOURCE_SHA}, got {resolved_sha}")
    if not (SOURCE_DIR / "arcgpt2" / "meta_soft_raw_outcome_overfit.py").is_file():
        raise RuntimeError("pinned source does not contain the raw-outcome module")

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
            str(path.relative_to(MODEL_DIR)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(MODEL_DIR.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        },
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arcgpt2/tests/test_meta_soft_raw_outcome_overfit.py",
            "arcgpt2/tests/test_meta_soft_single_binding_gate.py",
            "arcgpt2/tests/test_meta_soft_second_order.py",
            "arcgpt2/tests/test_completion_scorer.py",
        ],
        cwd=SOURCE_DIR,
        log=LOG_DIR / "preflight-tests.log",
    )

    command = [
        sys.executable,
        "-m",
        "arcgpt2.meta_soft_raw_outcome_overfit",
        "--model-name",
        str(MODEL_DIR),
        "--model-revision",
        GPT2_REVISION,
        "--initialization",
        "pretrained",
        "--output-dir",
        str(OUTPUT_DIR),
        "--seed",
        str(SEED),
        "--group-seed",
        str(GROUP_SEED),
        "--action",
        ACTION,
        "--max-outer-steps",
        str(MAX_OUTER_STEPS),
        "--plateau-patience",
        str(PLATEAU_PATIENCE),
        "--require-cuda",
        "--no-save-model",
    ]
    with GpuSampler() as sampler:
        run(command, cwd=SOURCE_DIR, log=LOG_DIR / "pretrained.log")
    write_json(RUN_ROOT / "gpu-samples.json", sampler.samples)

    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("raw-outcome experiment did not write summary.json")
    summary = load_json(summary_path)
    config = summary.get("config", {})
    gate = summary.get("gate", {})
    support = summary.get("metrics", {}).get("support", {})
    if (OUTPUT_DIR / "model").exists():
        raise RuntimeError("experiment saved a model despite --no-save-model")
    execution_gates = {
        "used_cuda": summary.get("device") == "cuda",
        "tesla_t4_verified": "T4" in device_name.upper() and tuple(capability) == (7, 5),
        "summary_source_sha_matches": summary.get("source_sha") == SOURCE_SHA,
        "raw_outcome_support_objective": summary.get("support_objective") == "raw_outcome_nll",
        "no_counterfactual_training_or_intact_targets": summary.get(
            "counterfactuals_used_in_training_or_intact_update"
        ) is False,
        "corruption_counterfactual_disclosed": summary.get(
            "counterfactuals_used_in_corruption_evaluation"
        ) is True,
        "support_metric_objective": support.get("objective") == "raw_outcome_nll",
        "support_metric_candidate_count": support.get("candidate_count") == 1,
        "support_metric_reduction": support.get("reduction") == "mean",
        "support_metric_unique_target_count": support.get("unique_target_count") == 4,
        "support_metric_no_counterfactual_inner_targets": support.get(
            "counterfactuals_used_in_inner_update"
        ) is False,
        "pretrained_initialization": config.get("initialization") == "pretrained",
        "pinned_model_revision": config.get("model_revision") == GPT2_REVISION,
        "cuda_required_by_experiment": config.get("require_cuda") is True,
        "matched_seed": config.get("seed") == SEED,
        "matched_group_seed": config.get("group_seed") == GROUP_SEED,
        "matched_action": config.get("action") == ACTION,
        "bounded_outer_steps": config.get("max_outer_steps") == MAX_OUTER_STEPS,
        "bounded_plateau_patience": config.get("plateau_patience") == PLATEAU_PATIENCE,
        "model_save_disabled": config.get("save_model") is False,
        "model_not_saved": True,
    }
    failed_execution_gates = [
        name for name, passed in execution_gates.items() if passed is not True
    ]
    if failed_execution_gates:
        raise RuntimeError(
            "raw-outcome execution gate failure: "
            + ", ".join(failed_execution_gates)
        )
    max_memory = max((sample.get("memory_used_mib", 0) for sample in sampler.samples), default=0)
    max_utilization = max((sample.get("utilization_percent", 0) for sample in sampler.samples), default=0)
    result = {
        "status": "pass" if gate.get("passed") is True else "not_yet_passed",
        "scope": "One-seed raw observed-outcome NLL gradient-memory gate; not an ARC-AGI-3 capability claim.",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "source_sha": SOURCE_SHA,
        "gpt2_revision": GPT2_REVISION,
        "seed": SEED,
        "group_seed": GROUP_SEED,
        "action": ACTION,
        "maximum_outer_steps": MAX_OUTER_STEPS,
        "plateau_patience": PLATEAU_PATIENCE,
        "competition_submission_performed": False,
        "gpu": {
            "name": device_name,
            "compute_capability": list(capability),
            "max_observed_memory_used_mib": max_memory,
            "max_observed_utilization_percent": max_utilization,
            "sample_count": len(sampler.samples),
        },
        "execution_gates": execution_gates,
        "scientific_gate": gate,
    }
    write_json(RUN_ROOT / "raw-outcome-result.json", result)
    write_json(
        RUN_ROOT / "artifact-manifest.json",
        {
            str(path.relative_to(RUN_ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "status": "execution_failed",
            "failed_at_utc": utc_now(),
            "source_sha": SOURCE_SHA,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
            "competition_submission_performed": False,
        }
        write_json(RUN_ROOT / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
