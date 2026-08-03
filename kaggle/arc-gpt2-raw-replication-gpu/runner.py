"""Private dual-T4 replication of the ARC-GPT2 raw-outcome-NLL gate.

Three matched seeds are run sequentially as pairs. Within each pair, the
pretrained GPT-2 occupies physical GPU 0 and the randomly initialized copy
occupies physical GPU 1. Random-init failure is evidence, not a replication
failure; the preregistered scientific gate requires all three pretrained runs
to pass. No competition API is called.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
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
REPLICATIONS = (
    {"seed": 161803, "group_seed": 920000, "action": "A1"},
    {"seed": 141421, "group_seed": 920000, "action": "A1"},
    {"seed": 173205, "group_seed": 920000, "action": "A1"},
)
MAX_OUTER_STEPS = 200
INNER_LEARNING_RATE = 0.2
PREFIX_LEARNING_RATE = 0.001
MODEL_LEARNING_RATE = 0.0001
EVALUATION_INTERVAL = 10
PLATEAU_PATIENCE = 201
NONPASS_CENSORED_T = MAX_OUTER_STEPS + EVALUATION_INTERVAL
REPLICATION_SCHEMA_VERSION = 1
SUPPORT_OBJECTIVE = "raw_outcome_nll"

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_raw_replication_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_raw_replication_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
MODEL_DIR = TEMP_ROOT / "gpt2-small-pinned"
OUTPUT_DIR = RUN_ROOT / "outputs"
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


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(
        json.dumps({"event": "command_start", "command": printable, "utc": utc_now()}),
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
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        code = process.wait()
    finally:
        if handle:
            handle.close()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    print(
        json.dumps({"event": "command_complete", "command": printable, "utc": utc_now()}),
        flush=True,
    )


def run_gpu_pair(
    *,
    seed: int,
    pretrained_command: list[str],
    random_command: list[str],
) -> None:
    """Run one matched pair concurrently with one visible GPU per process."""

    specifications = (
        ("pretrained", pretrained_command, "0"),
        ("random", random_command, "1"),
    )
    processes: list[tuple[str, subprocess.Popen[str], Any, threading.Thread]] = []

    def stream(label: str, process: subprocess.Popen[str], handle: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            handle.write(line)
            handle.flush()

    try:
        for initialization, command, physical_gpu in specifications:
            label = f"seed={seed} {initialization} gpu={physical_gpu}"
            log_path = LOG_DIR / f"seed-{seed}-{initialization}.log"
            handle = log_path.open("w", encoding="utf-8")
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
            process = subprocess.Popen(
                command,
                cwd=str(SOURCE_DIR),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            thread = threading.Thread(
                target=stream,
                args=(label, process, handle),
                daemon=True,
            )
            processes.append((label, process, handle, thread))
            thread.start()

        failures: list[str] = []
        for label, process, _, _ in processes:
            code = process.wait()
            if code != 0:
                failures.append(f"{label} exited {code}")
        for _, _, handle, thread in processes:
            thread.join(timeout=30)
            handle.close()
        if failures:
            raise RuntimeError("; ".join(failures))
    except Exception:
        for _, process, handle, thread in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, handle, thread in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            thread.join(timeout=10)
            if not handle.closed:
                handle.close()
        raise


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        query = "index,timestamp,name,memory.used,memory.total,utilization.gpu"
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
                    if len(parts) == 6:
                        self.samples.append(
                            {
                                "sampled_at_utc": utc_now(),
                                "device_index": int(parts[0]),
                                "device_timestamp": parts[1],
                                "name": parts[2],
                                "memory_used_mib": int(parts[3]),
                                "memory_total_mib": int(parts[4]),
                                "utilization_percent": int(parts[5]),
                            }
                        )
            except Exception as exc:  # Monitoring cannot invalidate execution.
                self.samples.append(
                    {"sampled_at_utc": utc_now(), "sampling_error": type(exc).__name__}
                )
            self.stop_event.wait(5)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)


def experiment_command(
    *,
    initialization: str,
    seed: int,
    group_seed: int,
    action: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "arcgpt2.meta_soft_raw_outcome_overfit",
        "--model-name",
        str(MODEL_DIR),
        "--model-revision",
        GPT2_REVISION,
        "--source-sha",
        SOURCE_SHA,
        "--initialization",
        initialization,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--group-seed",
        str(group_seed),
        "--action",
        action,
        "--max-outer-steps",
        str(MAX_OUTER_STEPS),
        "--inner-learning-rate",
        str(INNER_LEARNING_RATE),
        "--prefix-learning-rate",
        str(PREFIX_LEARNING_RATE),
        "--model-learning-rate",
        str(MODEL_LEARNING_RATE),
        "--evaluation-interval",
        str(EVALUATION_INTERVAL),
        "--plateau-patience",
        str(PLATEAU_PATIENCE),
        "--continue-after-pass",
        "--require-cuda",
        "--no-save-model",
    ]


def summarize_run(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    training = summary.get("metrics", {}).get("training", {})
    final_gate_passed = summary.get("gate", {}).get("passed") is True
    first_passing = training.get("first_passing_step")
    ever_passed = first_passing is not None
    completed = training.get("steps_completed")
    return {
        "seed": config.get("seed"),
        "group_seed": config.get("group_seed"),
        "action": config.get("action"),
        "initialization": config.get("initialization"),
        "device": summary.get("device"),
        "gate_passed": final_gate_passed,
        "ever_passed": ever_passed,
        "steps_to_pass": first_passing,
        "steps_completed": completed,
        "censored_at_steps": None if ever_passed else completed,
        "stopped_reason": training.get("stopped_reason"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "gate_checks": summary.get("gate", {}).get("checks", {}),
    }


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

    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
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
        log=LOG_DIR / "cpu-preflight-tests.log",
        env=cpu_env,
    )

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"replication requires exactly two visible CUDA devices; got {torch.cuda.device_count()}"
        )
    devices: list[dict[str, Any]] = []
    probes: list[float] = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "T4" not in name.upper() or tuple(capability) != (7, 5):
            raise RuntimeError(
                f"device {index} must be Tesla T4 capability 7.5; got {name!r} {capability!r}"
            )
        probe = torch.ones(256, device=f"cuda:{index}")
        value = float((probe @ probe).item())
        if value != 256.0:
            raise RuntimeError(f"CUDA compute probe failed on device {index}")
        probes.append(value)
        devices.append(
            {"index": index, "name": name, "compute_capability": list(capability)}
        )
    del probe
    torch.cuda.empty_cache()

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
        "cuda_devices": devices,
        "cuda_compute_probes": probes,
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
    }
    write_json(RUN_ROOT / "environment.json", environment)
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
    model_manifest = {
        str(path.relative_to(MODEL_DIR)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(MODEL_DIR.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    write_json(RUN_ROOT / "model-manifest.json", model_manifest)

    with GpuSampler() as sampler:
        for replication in REPLICATIONS:
            seed = int(replication["seed"])
            group_seed = int(replication["group_seed"])
            action = str(replication["action"])
            seed_root = OUTPUT_DIR / f"seed-{seed}"
            run_gpu_pair(
                seed=seed,
                pretrained_command=experiment_command(
                    initialization="pretrained",
                    seed=seed,
                    group_seed=group_seed,
                    action=action,
                    output_dir=seed_root / "pretrained",
                ),
                random_command=experiment_command(
                    initialization="random",
                    seed=seed,
                    group_seed=group_seed,
                    action=action,
                    output_dir=seed_root / "random",
                ),
            )
    write_json(RUN_ROOT / "gpu-samples.json", sampler.samples)

    runs: list[dict[str, Any]] = []
    execution_gates: dict[str, bool] = {
        "dual_t4_verified": len(devices) == 2,
        "three_matched_seed_pairs": len(REPLICATIONS) == 3,
        "pilot_seed_excluded": all(item["seed"] != 314159 for item in REPLICATIONS),
    }
    for replication in REPLICATIONS:
        seed = int(replication["seed"])
        group_seed = int(replication["group_seed"])
        action = str(replication["action"])
        pair: dict[str, dict[str, Any]] = {}
        for initialization in ("pretrained", "random"):
            path = OUTPUT_DIR / f"seed-{seed}" / initialization / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"missing replication summary: {path}")
            summary = load_json(path)
            config = summary.get("config", {})
            training = summary.get("metrics", {}).get("training", {})
            support = summary.get("metrics", {}).get("support", {})
            matched = {
                "device_cuda": summary.get("device") == "cuda",
                "source_sha": summary.get("source_sha") == SOURCE_SHA,
                "raw_support_objective": summary.get("support_objective")
                == SUPPORT_OBJECTIVE,
                "no_counterfactual_training_or_intact_targets": summary.get(
                    "counterfactuals_used_in_training_or_intact_update"
                )
                is False,
                "corruption_counterfactual_disclosed": summary.get(
                    "counterfactuals_used_in_corruption_evaluation"
                )
                is True,
                "support_metric_objective": support.get("objective")
                == SUPPORT_OBJECTIVE,
                "support_metric_candidate_count": support.get("candidate_count") == 1,
                "support_metric_reduction": support.get("reduction") == "mean",
                "support_metric_unique_target_count": support.get(
                    "unique_target_count"
                )
                == 4,
                "support_metric_no_counterfactual_inner_targets": support.get(
                    "counterfactuals_used_in_inner_update"
                )
                is False,
                "initialization": config.get("initialization") == initialization,
                "explicit_source_sha": config.get("source_sha") == SOURCE_SHA,
                "pinned_model_path": config.get("model_name") == str(MODEL_DIR),
                "seed": config.get("seed") == seed,
                "group_seed": config.get("group_seed") == group_seed,
                "action": config.get("action") == action,
                "model_revision": config.get("model_revision") == GPT2_REVISION,
                "max_steps": config.get("max_outer_steps") == MAX_OUTER_STEPS,
                "full_curve_enabled": config.get("stop_on_gate_pass") is False,
                "completed_full_curve": training.get("steps_completed")
                == MAX_OUTER_STEPS,
                "stopped_at_max_steps": training.get("stopped_reason") == "max_steps",
                "model_save_disabled": config.get("save_model") is False,
                "cuda_required": config.get("require_cuda") is True,
                "evaluation_interval": config.get("evaluation_interval")
                == EVALUATION_INTERVAL,
                "plateau_disabled_for_budget": config.get("plateau_patience")
                == PLATEAU_PATIENCE,
            }
            execution_gates[f"seed_{seed}_{initialization}_matched"] = all(
                matched.values()
            )
            record = summarize_run(summary)
            record["execution_checks"] = matched
            pair[initialization] = record
            runs.append(record)
            execution_gates[f"seed_{seed}_{initialization}_no_model_artifacts"] = not any(
                candidate.suffix in {".pt", ".bin", ".safetensors"}
                for candidate in path.parent.rglob("*")
                if candidate.is_file()
            )
        execution_gates[f"seed_{seed}_pair_surface_matched"] = (
            pair["pretrained"]["group_seed"] == pair["random"]["group_seed"]
            and pair["pretrained"]["seed"] == pair["random"]["seed"]
            and pair["pretrained"]["action"] == pair["random"]["action"] == action
        )

    pretrained_runs = [item for item in runs if item["initialization"] == "pretrained"]
    random_runs = [item for item in runs if item["initialization"] == "random"]
    pretrained_steps = [
        int(item["steps_to_pass"])
        for item in pretrained_runs
        if item["steps_to_pass"] is not None
    ]
    random_steps = [
        int(item["steps_to_pass"])
        for item in random_runs
        if item["steps_to_pass"] is not None
    ]
    pretrained_t = [
        int(item["steps_to_pass"])
        if item["steps_to_pass"] is not None
        else NONPASS_CENSORED_T
        for item in pretrained_runs
    ]
    random_t = [
        int(item["steps_to_pass"])
        if item["steps_to_pass"] is not None
        else NONPASS_CENSORED_T
        for item in random_runs
    ]
    paired_results: list[dict[str, Any]] = []
    pretrained_faster_count = 0
    for pretrained, random_init in zip(pretrained_runs, random_runs, strict=True):
        pre_t = (
            int(pretrained["steps_to_pass"])
            if pretrained["steps_to_pass"] is not None
            else NONPASS_CENSORED_T
        )
        random_t_value = (
            int(random_init["steps_to_pass"])
            if random_init["steps_to_pass"] is not None
            else NONPASS_CENSORED_T
        )
        faster = pre_t < random_t_value
        pretrained_faster_count += int(faster)
        paired_results.append(
            {
                "seed": int(pretrained["seed"]),
                "pretrained_T": pre_t,
                "random_T": random_t_value,
                "pretrained_faster": faster,
                "random_minus_pretrained_T": random_t_value - pre_t,
                "pretrained_was_censored": pretrained["steps_to_pass"] is None,
                "random_was_censored": random_init["steps_to_pass"] is None,
            }
        )

    all_pretrained_final_passed = all(
        bool(item["gate_passed"]) for item in pretrained_runs
    )
    all_pretrained_passed_by_180 = len(pretrained_steps) == len(REPLICATIONS) and all(
        step <= 180 for step in pretrained_steps
    )
    pretrained_median_t = float(statistics.median(pretrained_t))
    random_median_t = float(statistics.median(random_t))
    median_ratio = pretrained_median_t / random_median_t
    replication_passed = (
        all_pretrained_final_passed
        and all_pretrained_passed_by_180
        and pretrained_median_t <= 150
    )
    scientific_gate = {
        "name": "raw_replication_v1",
        "passed": replication_passed,
        "checks": {
            "all_three_pretrained_final_gates_passed": all_pretrained_final_passed,
            "all_three_pretrained_passed_by_step_180": all_pretrained_passed_by_180,
            "pretrained_median_T_at_most_150": pretrained_median_t <= 150,
        },
        "random_outcome_is_not_a_gate": True,
    }
    pretraining_claim = {
        "name": "pretraining_sample_efficiency_v1",
        "passed": (
            replication_passed
            and pretrained_faster_count >= 2
            and median_ratio <= 0.80
        ),
        "checks": {
            "replication_gate_passed": replication_passed,
            "pretrained_faster_in_at_least_two_pairs": pretrained_faster_count >= 2,
            "median_T_ratio_at_most_0_80": median_ratio <= 0.80,
        },
        "is_replication_gate": False,
    }
    sample_efficiency = {
        "pretrained_steps_to_pass": pretrained_steps,
        "random_steps_to_pass": random_steps,
        "nonpass_censored_T": NONPASS_CENSORED_T,
        "pretrained_T": pretrained_t,
        "random_T": random_t,
        "pretrained_median_T": pretrained_median_t,
        "random_median_T": random_median_t,
        "median_T_ratio_pretrained_over_random": median_ratio,
        "random_pass_count": len(random_steps),
        "random_censored_count": len(REPLICATIONS) - len(random_steps),
        "pretrained_faster_pair_count": pretrained_faster_count,
        "paired_results": paired_results,
        "censoring_note": (
            "Per preregistration, a non-pass by 200 steps receives T=210 for the "
            "paired sample-efficiency comparison. Raw steps_to_pass remains null."
        ),
    }
    per_gpu = {}
    for index in range(2):
        samples = [
            item for item in sampler.samples if item.get("device_index") == index
        ]
        per_gpu[str(index)] = {
            "sample_count": len(samples),
            "max_memory_used_mib": max(
                (item.get("memory_used_mib", 0) for item in samples), default=0
            ),
            "max_utilization_percent": max(
                (item.get("utilization_percent", 0) for item in samples), default=0
            ),
        }
        execution_gates[f"gpu_{index}_sampled"] = len(samples) > 0
        execution_gates[f"gpu_{index}_allocated_memory"] = (
            per_gpu[str(index)]["max_memory_used_mib"] > 0
        )

    if not all(execution_gates.values()):
        failed = [name for name, passed in execution_gates.items() if not passed]
        raise RuntimeError(f"raw replication execution-contract gate failed: {failed}")

    result = {
        "schema_version": REPLICATION_SCHEMA_VERSION,
        "status": (
            "pass"
            if all(execution_gates.values()) and scientific_gate["passed"]
            else "not_yet_passed"
        ),
        "scope": (
            "Three-seed dual-T4 replication of the synthetic raw-outcome-NLL gate; "
            "not an ARC-AGI-3 score or capability claim."
        ),
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "source_sha": SOURCE_SHA,
        "gpt2_revision": GPT2_REVISION,
        "replications": list(REPLICATIONS),
        "maximum_outer_steps": MAX_OUTER_STEPS,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "gpu_assignment": {"pretrained": "physical cuda:0", "random": "physical cuda:1"},
        "gpu_observations": per_gpu,
        "execution_gates": execution_gates,
        "scientific_gate": scientific_gate,
        "pretraining_claim": pretraining_claim,
        "sample_efficiency": sample_efficiency,
        "runs": runs,
    }
    write_json(RUN_ROOT / "replication-result.json", result)
    write_json(
        RUN_ROOT / "artifact-manifest.json",
        {
            str(path.relative_to(RUN_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
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
