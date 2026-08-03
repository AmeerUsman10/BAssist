"""Private dual-T4 runner for the heldout_layout_action_v1 stage gate.

The two initializations run concurrently on separate physical T4s.  The
runner never submits to a competition and never persists model weights.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import string
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROTOCOL = "heldout_layout_action_v1"
EXPERIMENT_SEED = 424_242
TRAIN_SEED_RANGE = (1_100_000, 1_200_000)
VALIDATION_SEED_RANGE = (1_200_000, 1_300_000)
TEST_SEED_RANGE = (1_300_000, 1_400_000)
TRAIN_SIZE = 256
VALIDATION_SIZE = 64
TEST_SIZE = 64
TRAIN_ACTIONS = ("A1",)
TEST_ACTIONS = ("A1", "A2", "A3", "A4")
PRIMARY_TEST_ACTIONS = ("A2", "A3", "A4")
EPOCHS = 2
OPTIMIZER_STEPS = 512
PREFIX_LENGTH = 8
INNER_LEARNING_RATE = 0.2
PREFIX_LEARNING_RATE = 0.001
MODEL_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
PRIOR_LOSS_WEIGHT = 0.25
FREEZE_FIRST_N_BLOCKS = 11
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20_260_803
PAIRED_BOOTSTRAP_SEED = 20_260_804
RUNNER_SCHEMA_VERSION = 2
NO_ADAPTATION_CONTROL = "zero_update_amnesic_prior"
CORRUPTION_CONTROL = (
    "fixed_deranged_quartet_permutation_consistency_not_independent_evidence"
)

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_heldout_binding_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_heldout_binding_gpu")
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


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def regenerate_canonical_manifests() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Regenerate manifests from the pinned core, not from run output."""

    sys.path.insert(0, str(SOURCE_DIR))
    try:
        from arcgpt2.meta_soft_heldout_binding import (  # type: ignore[import-not-found]
            _manifest_dict,
            build_split_manifests,
            canonical_manifest_payload,
            manifest_sha256,
        )

        generated = build_split_manifests()
        manifests = {
            manifest.name: _manifest_dict(manifest)
            for manifest in generated
        }
        payload = canonical_manifest_payload(generated)
        digest = manifest_sha256(generated)
    finally:
        sys.path.remove(str(SOURCE_DIR))
    if digest != canonical_json_sha256(payload):
        raise RuntimeError("core manifest digest disagrees with canonical JSON")
    return manifests, payload, digest


def paired_group_bootstrap(
    pretrained: dict[str, Any], random_init: dict[str, Any]
) -> dict[str, Any]:
    """Bootstrap paired primary A2-A4 group deltas over locked group IDs."""

    def keyed(stratum: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
        return {
            (int(row["game_seed"]), str(row["before_grid_sha256"])): row
            for row in stratum["groups"]
        }

    pretrained_rows = keyed(pretrained)
    random_rows = keyed(random_init)
    if set(pretrained_rows) != set(random_rows) or len(pretrained_rows) != TEST_SIZE:
        raise RuntimeError("pretrained/random primary group IDs are not exactly paired")
    group_ids = sorted(pretrained_rows)
    metrics = ("accuracy", "truth_probability")
    deltas = {
        metric: [
            float(pretrained_rows[group_id][metric])
            - float(random_rows[group_id][metric])
            for group_id in group_ids
        ]
        for metric in metrics
    }
    point = {
        metric: sum(values) / len(values) for metric, values in deltas.items()
    }
    rng = random.Random(PAIRED_BOOTSTRAP_SEED)
    draws = {metric: [] for metric in metrics}
    count = len(group_ids)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(count) for _ in range(count)]
        for metric in metrics:
            draws[metric].append(
                sum(deltas[metric][index] for index in indices) / count
            )
    # Conservative empirical lower 5% order statistic: ceil(.05*N)-1.
    alpha = Decimal(1) - Decimal(str(BOOTSTRAP_CONFIDENCE))
    rank = int(
        (alpha * Decimal(BOOTSTRAP_REPLICATES)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    lower_index = max(0, rank - 1)
    lower_bounds = {
        metric: sorted(values)[lower_index] for metric, values in draws.items()
    }
    accuracy_path = point["accuracy"] >= 0.10 and lower_bounds["accuracy"] > 0.0
    probability_path = (
        point["truth_probability"] >= 0.05
        and lower_bounds["truth_probability"] > 0.0
    )
    return {
        "name": "pretraining_heldout_primary_paired_v1",
        "scope": "Matched locked primary A2-A4 layout-group deltas, pretrained minus random.",
        "passed": accuracy_path or probability_path,
        "checks": {
            "accuracy_delta_at_least_0_10": point["accuracy"] >= 0.10,
            "accuracy_delta_lower_bound_positive": lower_bounds["accuracy"] > 0.0,
            "truth_probability_delta_at_least_0_05": point["truth_probability"] >= 0.05,
            "truth_probability_delta_lower_bound_positive": lower_bounds["truth_probability"] > 0.0,
            "accuracy_or_probability_promotion_path": accuracy_path
            or probability_path,
        },
        "point_deltas_pretrained_minus_random": point,
        "one_sided_lower_bounds": lower_bounds,
        "bootstrap": {
            "unit": "matched_locked_layout_group_id",
            "method": "deterministic_paired_group_percentile",
            "samples": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "seed": PAIRED_BOOTSTRAP_SEED,
            "group_count": count,
        },
    }


def compact_stratum(stratum: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stratum[key]
        for key in (
            "group_count",
            "actions",
            "all_finite",
            "controls",
            "aggregate",
            "bootstrap",
            "per_action",
        )
    }


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


def experiment_command(initialization: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "arcgpt2.meta_soft_heldout_binding",
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
        str(EXPERIMENT_SEED),
        "--prefix-length",
        str(PREFIX_LENGTH),
        "--inner-learning-rate",
        str(INNER_LEARNING_RATE),
        "--prefix-learning-rate",
        str(PREFIX_LEARNING_RATE),
        "--model-learning-rate",
        str(MODEL_LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--no-evidence-weight",
        str(PRIOR_LOSS_WEIGHT),
        "--freeze-first-n-blocks",
        str(FREEZE_FIRST_N_BLOCKS),
        "--bootstrap-samples",
        str(BOOTSTRAP_REPLICATES),
        "--bootstrap-confidence",
        str(BOOTSTRAP_CONFIDENCE),
        "--bootstrap-seed",
        str(BOOTSTRAP_SEED),
        "--require-cuda",
        "--no-save-model",
    ]


def run_gpu_pair() -> None:
    processes: list[tuple[str, subprocess.Popen[str], Any, threading.Thread]] = []

    def stream(label: str, process: subprocess.Popen[str], handle: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            handle.write(line)
            handle.flush()

    try:
        for initialization, physical_gpu in (("pretrained", "0"), ("random", "1")):
            command = experiment_command(initialization, OUTPUT_DIR / initialization)
            handle = (LOG_DIR / f"{initialization}.log").open("w", encoding="utf-8")
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
                args=(f"{initialization} gpu={physical_gpu}", process, handle),
                daemon=True,
            )
            processes.append((initialization, process, handle, thread))
            thread.start()

        failures: list[str] = []
        for initialization, process, _, _ in processes:
            code = process.wait()
            if code:
                failures.append(f"{initialization} exited {code}")
        for _, _, handle, thread in processes:
            thread.join(timeout=30)
            handle.close()
        if failures:
            raise RuntimeError("; ".join(failures))
    except Exception:
        for _, process, _, _ in processes:
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
            except Exception as exc:
                self.samples.append({"sampled_at_utc": utc_now(), "sampling_error": type(exc).__name__})
            self.stop_event.wait(5)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)


def expected_config(initialization: str, output_dir: Path) -> dict[str, Any]:
    return {
        "model_name": str(MODEL_DIR),
        "initialization": initialization,
        "source_sha": SOURCE_SHA,
        "output_dir": str(output_dir),
        "seed": EXPERIMENT_SEED,
        "epochs": EPOCHS,
        "prefix_length": PREFIX_LENGTH,
        "prefix_initialization_std": 0.01,
        "inner_learning_rate": INNER_LEARNING_RATE,
        "prefix_learning_rate": PREFIX_LEARNING_RATE,
        "model_learning_rate": MODEL_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "no_evidence_weight": PRIOR_LOSS_WEIGHT,
        "freeze_first_n_blocks": FREEZE_FIRST_N_BLOCKS,
        "bootstrap_samples": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "require_cuda": True,
        "save_model": False,
        "model_revision": GPT2_REVISION,
    }


def recompute_scientific_gate_checks(summary: dict[str, Any]) -> dict[str, bool]:
    """Recompute the preregistered gate from the recorded locked-test metrics."""

    locked_test = summary["locked_test"]
    geometry = locked_test["geometry_A1"]
    primary = locked_test["primary_A2_A4"]
    checks: dict[str, bool] = {}
    for name, stratum in (("geometry", geometry), ("primary", primary)):
        metrics = stratum["aggregate"]
        lower = stratum["bootstrap"]["lower_bounds"]
        checks.update(
            {
                f"{name}_accuracy": metrics["accuracy"] >= 0.70,
                f"{name}_all_metrics_finite": bool(stratum.get("all_finite", False)),
                f"{name}_truth_probability": metrics["truth_probability"] >= 0.60,
                f"{name}_accuracy_gain": metrics["accuracy_gain"] >= 0.35,
                f"{name}_truth_probability_gain": metrics["truth_probability_gain"] >= 0.25,
                f"{name}_bootstrap_accuracy_above_chance": lower["accuracy"] > 0.25,
                f"{name}_bootstrap_probability_above_chance": lower["truth_probability"] > 0.25,
                f"{name}_bootstrap_accuracy_gain_positive": lower["accuracy_gain"] > 0.0,
                f"{name}_bootstrap_probability_gain_positive": lower["truth_probability_gain"] > 0.0,
                f"{name}_prior_high_entropy": metrics["prior_entropy_bits"] >= 1.95,
                f"{name}_prior_uniform": metrics["prior_max_abs_uniform_deviation"] <= 0.05,
                f"{name}_preupdate_isolation": metrics["max_preupdate_logit_delta"] <= 1e-6,
                f"{name}_corrupted_target_accuracy": metrics["corrupted_target_accuracy"] >= 0.70,
                f"{name}_corrupted_target_probability": metrics["corrupted_mean_target_probability"] >= 0.60,
                f"{name}_corruption_rejects_original_accuracy": metrics["corrupted_original_truth_accuracy"] <= 0.25,
                f"{name}_corruption_rejects_original_probability": metrics["corrupted_mean_original_truth_probability"] <= 0.20,
            }
        )
    for action in PRIMARY_TEST_ACTIONS:
        metrics = primary["per_action"][action]
        checks[f"primary_{action}_accuracy"] = metrics["accuracy"] >= 0.55
        checks[f"primary_{action}_truth_probability"] = metrics["truth_probability"] >= 0.45
        checks[f"primary_{action}_corrupted_target_accuracy"] = metrics["corrupted_target_accuracy"] >= 0.55
        checks[f"primary_{action}_corrupted_target_probability"] = metrics["corrupted_mean_target_probability"] >= 0.45
    invariants = summary["execution_invariants"]
    checks.update(
        {
            "finite_training": bool(invariants["finite_training"]),
            "literal_action_text_audit": bool(
                invariants["literal_action_text_audit"]
            ),
            "inner_update_contract": bool(invariants["inner_update_contract"]),
            "checkpoint_unchanged_during_locked_test": bool(
                invariants["checkpoint_unchanged_during_locked_test"]
            ),
            "selected_validation_matches_history": bool(
                invariants["selected_validation_matches_history"]
            ),
            "epoch_steps_exact": bool(invariants["epoch_steps_exact"]),
        }
    )
    return checks


def validate_summary(
    initialization: str,
    summary: dict[str, Any],
    canonical_manifests: dict[str, Any],
    canonical_payload: dict[str, Any],
    canonical_digest: str,
) -> dict[str, bool]:
    config = summary.get("config", {})
    output_dir = OUTPUT_DIR / initialization
    expected = expected_config(initialization, output_dir)
    manifests = summary.get("manifests", {})
    expected_manifests = {
        "train": (TRAIN_SEED_RANGE, TRAIN_SIZE),
        "validation": (VALIDATION_SEED_RANGE, VALIDATION_SIZE),
        "locked_test": (TEST_SEED_RANGE, TEST_SIZE),
    }
    manifest_exact = manifests == canonical_manifests and all(
        canonical_manifests.get(name, {}).get("seed_range") == list(seed_range)
        and canonical_manifests.get(name, {}).get("requested_groups") == size
        and len(canonical_manifests.get(name, {}).get("accepted", [])) == size
        for name, (seed_range, size) in expected_manifests.items()
    )
    grid_hashes = [
        item.get("before_grid_sha256")
        for name in expected_manifests
        for item in manifests.get(name, {}).get("accepted", [])
    ]
    locked_test = summary.get("locked_test", {})
    selection = summary.get("selection", {})
    validation = summary.get("validation", {})
    geometry = locked_test.get("geometry_A1", {})
    primary = locked_test.get("primary_A2_A4", {})
    expected_controls = {
        "no_adaptation": NO_ADAPTATION_CONTROL,
        "corruption": CORRUPTION_CONTROL,
    }
    corruption_metric_names = {
        "corrupted_target_accuracy",
        "corrupted_mean_target_probability",
        "corrupted_original_truth_accuracy",
        "corrupted_mean_original_truth_probability",
    }
    gate = summary.get("gate", {})
    recomputed_gate_checks = recompute_scientific_gate_checks(summary)
    checkpoint_before = selection.get(
        "frozen_checkpoint_sha256_before_locked_test"
    )
    checkpoint_after = locked_test.get(
        "frozen_checkpoint_sha256_after_locked_test"
    )
    history = summary.get("history", [])
    history_shape = (
        len(history) == 2
        and [item.get("epoch") for item in history] == [1, 2]
        and [item.get("steps_completed") for item in history] == [256, 512]
    )
    if history_shape:
        best_history = min(
            history,
            key=lambda item: (
                float(item["validation_selection_objective"]),
                int(item["epoch"]),
            ),
        )
    else:
        best_history = {}
    inner_contract = summary.get("inner_update_contract", {})
    inner_contract_exact = inner_contract == {
        "objective": "raw_outcome_nll",
        "candidate_count": 1,
        "reduction": "mean",
        "counterfactuals_used_in_inner_update": False,
        "unique_targets_per_quartet": 4,
        "cardinal_semantic_guard_passed": True,
        "attention_implementation": "eager",
        "eager_attention": True,
    }
    recorded_canonical = summary.get("canonical_manifest", {})
    literal_audit = summary.get("literal_action_text_audit", {})
    invariants = summary.get("execution_invariants", {})
    checks = {
        "source_sha": summary.get("source_sha") == SOURCE_SHA,
        "protocol": summary.get("protocol") == PROTOCOL,
        "raw_outcome_nll": summary.get("support_objective", config.get("support_objective")) == "raw_outcome_nll",
        "no_inner_counterfactuals": summary.get("counterfactuals_used_in_inner_update") is False,
        "device_cuda": summary.get("device") == "cuda",
        "config_exact": all(config.get(key) == value for key, value in expected.items()),
        "fixed_split_manifests": manifest_exact,
        "canonical_manifest_payload_regenerated_exactly": recorded_canonical.get(
            "payload"
        )
        == canonical_payload,
        "canonical_manifest_digest_regenerated_exactly": recorded_canonical.get(
            "sha256"
        )
        == canonical_digest
        == canonical_json_sha256(canonical_payload),
        "cross_split_grid_hashes_disjoint": len(grid_hashes) == len(set(grid_hashes)) == 384,
        "full_optimizer_budget": summary.get("training", {}).get("steps_completed") == OPTIMIZER_STEPS,
        "finite_training": summary.get("training", {}).get("all_finite") is True,
        "two_validation_evaluations": selection.get("validation_evaluations") == EPOCHS,
        "epoch_history_exact": history_shape,
        "checkpoint_selected_by_validation": selection.get("best_epoch")
        == best_history.get("epoch"),
        "best_objective_matches_history": selection.get("best_objective")
        == best_history.get("validation_selection_objective"),
        "epoch_objectives_match_history": selection.get("epoch_objectives")
        == [
            {
                "epoch": item.get("epoch"),
                "steps_completed": item.get("steps_completed"),
                "objective": item.get("validation_selection_objective"),
            }
            for item in history
        ],
        "stored_validation_is_selected_history_validation": validation
        == best_history.get("validation")
        and selection.get("selected_validation_equals_best_history_entry") is True,
        "selected_validation_hashes_match": selection.get(
            "selected_validation_sha256"
        )
        == selection.get("best_history_validation_sha256")
        == canonical_json_sha256(validation),
        "frozen_checkpoint_hash_recorded": isinstance(checkpoint_before, str)
        and len(checkpoint_before) == 64
        and checkpoint_before == checkpoint_before.lower()
        and all(character in string.hexdigits for character in checkpoint_before),
        "checkpoint_hash_unchanged": checkpoint_before == checkpoint_after
        and locked_test.get("checkpoint_unchanged") is True,
        "inner_update_contract_exact": inner_contract_exact,
        "literal_action_text_audit_passed": literal_audit.get("passed") is True,
        "execution_invariants_all_true": invariants
        and all(value is True for value in invariants.values()),
        "validation_all_finite": validation.get("all_finite") is True,
        "validation_controls_exact": validation.get("controls") == expected_controls,
        "validation_action_locked": validation.get("actions") == list(TRAIN_ACTIONS),
        "validation_group_count": validation.get("group_count") == VALIDATION_SIZE,
        "locked_test_evaluated_once": locked_test.get("evaluations") == 1,
        "checkpoint_frozen_before_locked_test": locked_test.get("checkpoint_frozen_before_evaluation") is True,
        "geometry_controls_exact": geometry.get("controls") == expected_controls,
        "primary_controls_exact": primary.get("controls") == expected_controls,
        "geometry_corruption_metrics_present": corruption_metric_names <= set(geometry.get("aggregate", {})),
        "primary_corruption_metrics_present": corruption_metric_names <= set(primary.get("aggregate", {})),
        "primary_per_action_corruption_metrics_present": all(
            {"corrupted_target_accuracy", "corrupted_mean_target_probability"}
            <= set(primary.get("per_action", {}).get(action, {}))
            for action in PRIMARY_TEST_ACTIONS
        ),
        "geometry_action_locked": geometry.get("actions") == list(TRAIN_ACTIONS),
        "geometry_group_count": geometry.get("group_count") == TEST_SIZE,
        "geometry_all_finite": geometry.get("all_finite") is True,
        "primary_actions_locked": primary.get("actions") == list(PRIMARY_TEST_ACTIONS),
        "primary_group_count": primary.get("group_count") == TEST_SIZE,
        "primary_all_finite": primary.get("all_finite") is True,
        "gate_identity": gate.get("name") == PROTOCOL,
        "gate_checks_recomputed_exactly": gate.get("checks") == recomputed_gate_checks,
        "gate_pass_recomputed_exactly": gate.get("passed")
        == all(recomputed_gate_checks.values()),
    }
    return checks


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
    resolved_sha = subprocess.check_output(["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
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
            "arcgpt2/tests/test_meta_soft_heldout_binding.py",
            "arcgpt2/tests/test_meta_soft_raw_outcome_overfit.py",
            "arcgpt2/tests/test_meta_soft_single_binding_gate.py",
            "arcgpt2/tests/test_meta_soft_second_order.py",
            "arcgpt2/tests/test_completion_scorer.py",
        ],
        cwd=SOURCE_DIR,
        log=LOG_DIR / "cpu-preflight-tests.log",
        env=cpu_env,
    )

    canonical_manifests, canonical_payload, canonical_digest = (
        regenerate_canonical_manifests()
    )

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(f"held-out gate requires exactly two CUDA devices; got {torch.cuda.device_count()}")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "T4" not in name.upper() or tuple(capability) != (7, 5):
            raise RuntimeError(f"device {index} must be Tesla T4 capability 7.5; got {name!r} {capability!r}")
        probe = torch.ones(256, device=f"cuda:{index}")
        if float((probe @ probe).item()) != 256.0:
            raise RuntimeError(f"CUDA compute probe failed on device {index}")
        devices.append({"index": index, "name": name, "compute_capability": list(capability)})
    del probe
    torch.cuda.empty_cache()

    environment = {
        "recorded_at_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "gpt2_repository": GPT2_REPOSITORY,
        "gpt2_revision": GPT2_REVISION,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": devices,
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
    write_json(
        RUN_ROOT / "model-manifest.json",
        {
            str(path.relative_to(MODEL_DIR)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(MODEL_DIR.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        },
    )

    with GpuSampler() as sampler:
        run_gpu_pair()
    write_json(RUN_ROOT / "gpu-samples.json", sampler.samples)

    execution_gates: dict[str, bool] = {
        "dual_t4_verified": len(devices) == 2,
        "paired_initializations": True,
        "no_trained_model_artifacts": not any(
            path.suffix in {".pt", ".bin", ".safetensors"}
            for path in OUTPUT_DIR.rglob("*")
            if path.is_file()
        ),
    }
    runs: dict[str, Any] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for initialization in ("pretrained", "random"):
        summary_path = OUTPUT_DIR / initialization / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"missing {initialization} summary: {summary_path}")
        summary = load_json(summary_path)
        summaries[initialization] = summary
        checks = validate_summary(
            initialization,
            summary,
            canonical_manifests,
            canonical_payload,
            canonical_digest,
        )
        execution_gates.update({f"{initialization}_{key}": value for key, value in checks.items()})
        runs[initialization] = {
            "summary_path": str(summary_path.relative_to(RUN_ROOT)),
            "execution_checks": checks,
            "capability_gate": summary.get("gate"),
            "locked_test": {
                "geometry_A1": compact_stratum(
                    summary["locked_test"]["geometry_A1"]
                ),
                "primary_A2_A4": compact_stratum(
                    summary["locked_test"]["primary_A2_A4"]
                ),
            },
        }

    per_gpu: dict[str, Any] = {}
    for index in range(2):
        samples = [item for item in sampler.samples if item.get("device_index") == index]
        per_gpu[str(index)] = {
            "sample_count": len(samples),
            "max_memory_used_mib": max((item.get("memory_used_mib", 0) for item in samples), default=0),
            "max_utilization_percent": max((item.get("utilization_percent", 0) for item in samples), default=0),
        }
        execution_gates[f"gpu_{index}_sampled"] = len(samples) > 0
        execution_gates[f"gpu_{index}_allocated_memory"] = per_gpu[str(index)]["max_memory_used_mib"] > 0

    if not all(execution_gates.values()):
        failed = [name for name, passed in execution_gates.items() if not passed]
        raise RuntimeError(f"held-out execution-contract gate failed: {failed}")

    pretrained_gate = runs["pretrained"].get("capability_gate") or {}
    promotion_gate = paired_group_bootstrap(
        summaries["pretrained"]["locked_test"]["primary_A2_A4"],
        summaries["random"]["locked_test"]["primary_A2_A4"],
    )
    scale_promotion_eligible = (
        pretrained_gate.get("passed") is True and promotion_gate["passed"] is True
    )
    result = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "pass" if pretrained_gate.get("passed") is True else "not_yet_passed",
        "scope": "Locked held-out layout/action raw-NLL gradient-memory gate; not an ARC-AGI-3 score.",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "canonical_manifest_sha256": canonical_digest,
        "gpt2_revision": GPT2_REVISION,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "gpu_assignment": {"pretrained": "physical cuda:0", "random": "physical cuda:1"},
        "gpu_observations": per_gpu,
        "execution_gates": execution_gates,
        "capability_gate": pretrained_gate,
        "pretraining_promotion_gate": promotion_gate,
        "scale_promotion": {
            "eligible": scale_promotion_eligible,
            "decision": "promote_scale" if scale_promotion_eligible else "do_not_promote_scale",
            "requires_capability_and_pretraining_promotion": True,
            "capability_gate_passed": pretrained_gate.get("passed") is True,
            "pretraining_promotion_gate_passed": promotion_gate["passed"] is True,
        },
        "scientific_gate": pretrained_gate,
        "random_control_gate": runs["random"].get("capability_gate"),
        "random_control_is_not_required_to_fail": True,
        "runs": runs,
    }
    write_json(RUN_ROOT / "heldout-result.json", result)
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
            "protocol": PROTOCOL,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
            "competition_submission_performed": False,
        }
        write_json(RUN_ROOT / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
