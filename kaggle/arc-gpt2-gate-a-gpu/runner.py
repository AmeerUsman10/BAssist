"""Private Kaggle T4 diagnostic for ARC-GPT2 Gate A.

This script deliberately runs one bounded, matched experiment.  A completed
kernel is engineering evidence; only the explicit intact/amnesic/corrupted and
pretrained/random comparisons determine the scientific result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
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
SEED = 271828
WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_gate_a_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_gate_a_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
MODEL_DIR = TEMP_ROOT / "gpt2-small-pinned"
OUTPUT_DIR = RUN_ROOT / "outputs"
LOG_DIR = RUN_ROOT / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    print(json.dumps({"event": "command_complete", "command": printable, "utc": utc_now()}), flush=True)


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
            except Exception as exc:  # Sampling must never kill the experiment.
                self.samples.append({"sampled_at_utc": utc_now(), "sampling_error": type(exc).__name__})
            self.stop_event.wait(5)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    started = time.time()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "ARC_GPT2_META_MAX_PROBES": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    # Pin user-space libraries, but keep Kaggle's CUDA-matched PyTorch build.
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the GPU diagnostic on CPU")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    # Force a real CUDA kernel. This catches the P100/cu128 false-positive case.
    probe = torch.ones(256, device="cuda")
    probe_value = float((probe @ probe).item())
    if probe_value != 256.0:
        raise RuntimeError("CUDA compute probe returned an invalid result")

    environment = {
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
    }
    write_json(RUN_ROOT / "environment.json", environment)
    subprocess.run(["nvidia-smi"], check=True)

    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    run(["git", "init", str(SOURCE_DIR)])
    run(["git", "-C", str(SOURCE_DIR), "remote", "add", "origin", SOURCE_REPOSITORY])
    run(["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", SOURCE_SHA])
    run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", "FETCH_HEAD"])
    resolved_sha = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved_sha != SOURCE_SHA:
        raise RuntimeError(f"source SHA mismatch: expected {SOURCE_SHA}, got {resolved_sha}")

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
    model_manifest = {
        str(path.relative_to(MODEL_DIR)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(MODEL_DIR.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    write_json(RUN_ROOT / "model-manifest.json", model_manifest)

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arcgpt2/tests/test_meta_soft_second_order.py",
            "arcgpt2/tests/test_belief_math.py",
        ],
        cwd=SOURCE_DIR,
        log=LOG_DIR / "preflight-tests.log",
    )

    common = [
        "--model-name",
        str(MODEL_DIR),
        "--seed",
        str(SEED),
        "--train-groups",
        "128",
        "--train-mapping-variants",
        "6",
        "--validation-groups",
        "4",
        "--validation-mapping-variants",
        "4",
        "--max-length",
        "256",
        "--prefix-length",
        "8",
        "--inner-learning-rate",
        "0.2",
        "--outer-learning-rate",
        "0.0001",
        "--episodes-per-outer-step",
        "1",
        "--max-outer-steps",
        "160",
        "--warmup-steps",
        "12",
        "--freeze-first-n-blocks",
        "11",
        "--evaluation-groups",
        "4",
        "--no-save-model",
    ]

    with GpuSampler() as sampler:
        for initialization in ("pretrained", "random"):
            target = OUTPUT_DIR / initialization
            run(
                [
                    sys.executable,
                    "-m",
                    "arcgpt2.meta_soft_second_order",
                    "--initialization",
                    initialization,
                    "--output-dir",
                    str(target),
                    *common,
                ],
                cwd=SOURCE_DIR,
                log=LOG_DIR / f"{initialization}.log",
            )
        run(
            [
                sys.executable,
                "-m",
                "arcgpt2.compare_meta_soft",
                "--pretrained",
                str(OUTPUT_DIR / "pretrained" / "summary.json"),
                "--random",
                str(OUTPUT_DIR / "random" / "summary.json"),
                "--output",
                str(OUTPUT_DIR / "comparison.json"),
            ],
            cwd=SOURCE_DIR,
            log=LOG_DIR / "comparison.log",
        )
    write_json(RUN_ROOT / "gpu-samples.json", sampler.samples)

    pretrained = load_json(OUTPUT_DIR / "pretrained" / "summary.json")
    random_init = load_json(OUTPUT_DIR / "random" / "summary.json")
    comparison = load_json(OUTPUT_DIR / "comparison.json")
    no_evidence = pretrained["final_evaluation"]["summary"]["intact"]["0"]
    execution_gates = {
        "pretrained_used_cuda": pretrained.get("device") == "cuda",
        "random_used_cuda": random_init.get("device") == "cuda",
        "matched_seed": pretrained["config"]["seed"] == random_init["config"]["seed"] == SEED,
        "matched_training_configuration": {
            key: pretrained["config"][key] for key in (
                "train_groups",
                "train_mapping_variants",
                "max_length",
                "prefix_length",
                "inner_learning_rate",
                "outer_learning_rate",
                "max_outer_steps",
                "freeze_first_n_blocks",
                "evaluation_groups",
            )
        }
        == {
            key: random_init["config"][key] for key in (
                "train_groups",
                "train_mapping_variants",
                "max_length",
                "prefix_length",
                "inner_learning_rate",
                "outer_learning_rate",
                "max_outer_steps",
                "freeze_first_n_blocks",
                "evaluation_groups",
            )
        },
    }
    epistemic_gates = {
        "no_evidence_entropy_at_least_4_bits": no_evidence["entropy_bits"] >= 4.0,
        "no_evidence_truth_probability_at_most_0_10": no_evidence["truth_probability"] <= 0.10,
        **comparison["gates"],
    }
    max_memory = max(
        (sample.get("memory_used_mib", 0) for sample in sampler.samples), default=0
    )
    max_utilization = max(
        (sample.get("utilization_percent", 0) for sample in sampler.samples), default=0
    )
    result = {
        "status": "pass" if all(execution_gates.values()) and all(epistemic_gates.values()) else "not_yet_passed",
        "scope": "Synthetic counterfactual Gate A diagnostic; not an ARC-AGI-3 score or capability claim.",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "source_sha": SOURCE_SHA,
        "gpt2_revision": GPT2_REVISION,
        "seed": SEED,
        "gpu": {
            "name": device_name,
            "compute_capability": list(capability),
            "max_observed_memory_used_mib": max_memory,
            "max_observed_utilization_percent": max_utilization,
            "sample_count": len(sampler.samples),
        },
        "execution_gates": execution_gates,
        "epistemic_gates": epistemic_gates,
        "pretrained_no_evidence": no_evidence,
        "comparison": comparison,
    }
    write_json(RUN_ROOT / "gate-a-result.json", result)
    write_json(
        RUN_ROOT / "artifact-manifest.json",
        {
            str(path.relative_to(RUN_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file() and SOURCE_DIR not in path.parents and ".cache" not in path.parts
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
        }
        write_json(RUN_ROOT / "failure.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
