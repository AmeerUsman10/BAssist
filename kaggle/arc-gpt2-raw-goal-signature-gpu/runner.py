"""Private dual-T4 runner for the frozen ``raw_goal_signature_v1`` gate.

Three matched initialization seeds run sequentially.  Within a seed pair the
pretrained lane is isolated on physical T4 0 and the identical randomly
initialized GPT-2 architecture is isolated on physical T4 1.  The runner
performs no competition operation and keeps validation-selected weights only
in memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import string
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
SOURCE_SHA = "__SOURCE_SHA__"
RUN_MODE = "__RUN_MODE__"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROTOCOL = "raw_goal_signature_v1"
MATCHED_SEEDS = (577_215, 618_033, 707_106)
TRAIN_GROUPS = 120
VALIDATION_GROUPS = 24
LOCKED_GROUPS = 72
EPOCHS = 2
OPTIMIZER_STEPS = TRAIN_GROUPS * EPOCHS
PREFIX_LENGTH = 8
PREFIX_INITIALIZATION_STD = 0.01
INNER_LEARNING_RATE = 0.2
PREFIX_LEARNING_RATE = 1e-3
MODEL_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NO_EVIDENCE_WEIGHT = 0.25
FREEZE_FIRST_N_BLOCKS = 11
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20_260_804
HIERARCHICAL_BOOTSTRAP_SEED = 20_260_805
FULL_KERNEL_RUNTIME_CEILING_SECONDS = 16_200
TIMING_TRAIN_GROUPS = 2
TIMING_VALIDATION_GROUPS = 2
TIMING_PROJECTION_MULTIPLIER = 1.25
TIMING_FIXED_ALLOWANCE_SECONDS = 900
RUNNER_SCHEMA_VERSION = 1

WORKING = Path("/kaggle/working")
RUN_ROOT = WORKING / "arc_gpt2_raw_goal_signature_gpu"
TEMP_ROOT = Path("/kaggle/temp/arc_gpt2_raw_goal_signature_gpu")
SOURCE_DIR = TEMP_ROOT / "source"
MODEL_DIR = TEMP_ROOT / "gpt2-small-pinned"
OUTPUT_DIR = RUN_ROOT / "outputs"
LOG_DIR = RUN_ROOT / "logs"
CONSOLE_LOG = RUN_ROOT / "kernel-console.log"

_CONSOLE_HANDLE: Any | None = None
_ORIGINAL_STDOUT: Any | None = None
_ORIGINAL_STDERR: Any | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Normalize arbitrary evidence without ambiguous key stringification.

    Ordinary string-keyed JSON structures retain normal JSON semantics.  A
    mapping with typed/non-string keys becomes a sorted tagged entry list, so
    tuple-keyed audit tables cannot collide (for example ``1`` versus ``"1"``)
    or crash a receipt write.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence JSON may not contain non-finite floats")
        return value
    if isinstance(value, Enum):
        return {
            "__type__": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": json_safe(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: json_safe(item) for key, item in value.items()}
        entries = [
            {"key": json_safe(key), "value": json_safe(item)}
            for key, item in value.items()
        ]
        entries.sort(
            key=lambda row: json.dumps(
                row["key"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
        return {"__type__": "typed_mapping", "entries": entries}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [json_safe(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        )
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


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
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_source_sha(value: str) -> None:
    if len(value) != 40 or any(character not in string.hexdigits for character in value):
        raise RuntimeError("runner source SHA was not resolved by the dispatch workflow")


class _Tee:
    def __init__(self, terminal: Any, file_handle: Any) -> None:
        self.terminal = terminal
        self.file_handle = file_handle

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.file_handle.write(value)
        self.file_handle.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.file_handle.flush()

    def isatty(self) -> bool:
        return False


def install_console_tee() -> None:
    """Persist the complete top-level runner console while still streaming it."""

    global _CONSOLE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _ORIGINAL_STDOUT, _ORIGINAL_STDERR = sys.stdout, sys.stderr
    _CONSOLE_HANDLE = CONSOLE_LOG.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(_ORIGINAL_STDOUT, _CONSOLE_HANDLE)
    sys.stderr = _Tee(_ORIGINAL_STDERR, _CONSOLE_HANDLE)


def close_console_tee() -> None:
    global _CONSOLE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if _CONSOLE_HANDLE is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    _CONSOLE_HANDLE.close()
    _CONSOLE_HANDLE = None


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
                handle.flush()
        code = process.wait()
    finally:
        if handle:
            handle.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


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
            except Exception as exc:  # Monitoring does not define the science.
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


def _worker_command(
    initialization: str, seed: int, output_dir: Path, *, timing_only: bool
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--initialization",
        initialization,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if timing_only:
        command.append("--timing-only")
    return command


def run_gpu_pair(seed: int, *, timing_only: bool = False) -> None:
    """Run one matched seed pair concurrently on separate physical T4s."""

    processes: list[tuple[str, subprocess.Popen[str], Any, threading.Thread]] = []

    def stream(label: str, process: subprocess.Popen[str], handle: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            handle.write(line)
            handle.flush()

    try:
        for initialization, physical_gpu in (("pretrained", "0"), ("random", "1")):
            root = RUN_ROOT / "timing" if timing_only else OUTPUT_DIR
            output_dir = root / f"seed-{seed}" / initialization
            output_dir.mkdir(parents=True, exist_ok=True)
            label = f"seed={seed} {initialization} gpu={physical_gpu}"
            log_prefix = "timing-" if timing_only else ""
            handle = (LOG_DIR / f"{log_prefix}seed-{seed}-{initialization}.log").open(
                "w", encoding="utf-8"
            )
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
            process = subprocess.Popen(
                _worker_command(
                    initialization, seed, output_dir, timing_only=timing_only
                ),
                cwd=str(SOURCE_DIR),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            thread = threading.Thread(
                target=stream, args=(label, process, handle), daemon=True
            )
            processes.append((label, process, handle, thread))
            thread.start()
        failures: list[str] = []
        for label, process, _, _ in processes:
            code = process.wait()
            if code:
                failures.append(f"{label} exited {code}")
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


def _configure_model(model: Any) -> None:
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("second-order soft-memory training requires eager attention")
    model.config.use_cache = False
    for block in model.transformer.h[:FREEZE_FIRST_N_BLOCKS]:
        for parameter in block.parameters():
            parameter.requires_grad_(False)
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad_(False)
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad_(False)
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad_(True)


def _capture_trainable_state(model: Any, prefix: Any) -> tuple[dict[str, Any], Any]:
    return (
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        prefix.detach().cpu().clone(),
    )


def _restore_trainable_state(model: Any, prefix: Any, state: tuple[Mapping[str, Any], Any]) -> None:
    parameters = dict(model.named_parameters())
    import torch

    with torch.no_grad():
        for name, value in state[0].items():
            parameters[name].copy_(value.to(parameters[name].device))
        prefix.copy_(state[1].to(prefix.device))


def _build_config(initialization: str, seed: int) -> Any:
    from arcgpt2.meta_soft_raw_goal_signature import Config

    return Config(
        model_name=GPT2_REPOSITORY,
        model_revision=GPT2_REVISION,
        source_sha=SOURCE_SHA,
        initialization=initialization,
        seed=seed,
        epochs=EPOCHS,
        prefix_length=PREFIX_LENGTH,
        prefix_initialization_std=PREFIX_INITIALIZATION_STD,
        inner_learning_rate=INNER_LEARNING_RATE,
        prefix_learning_rate=PREFIX_LEARNING_RATE,
        model_learning_rate=MODEL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        outer_optimizer="AdamW",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_gradient_norm=1.0,
        no_evidence_weight=NO_EVIDENCE_WEIGHT,
        freeze_first_n_blocks=FREEZE_FIRST_N_BLOCKS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


def _backward_exact_training_group(
    model: Any,
    tokenizer: Any,
    prefix: Any,
    group: Any,
    config: Any,
    device: Any,
) -> tuple[float, bool]:
    """Backpropagate the exact eight world/order mean with bounded memory."""

    import torch
    from arcgpt2.meta_soft_raw_goal_signature import EVIDENCE_ORDERS, world_meta_loss

    group_loss = 0.0
    finite = True
    for order in EVIDENCE_ORDERS:
        for world in group.worlds:
            loss, _ = world_meta_loss(
                model, tokenizer, prefix, world, order, config, device
            )
            scalar = float(loss.detach().item())
            finite &= math.isfinite(scalar)
            (loss / (len(EVIDENCE_ORDERS) * len(group.worlds))).backward()
            group_loss += scalar / (len(EVIDENCE_ORDERS) * len(group.worlds))
    return group_loss, finite


def run_worker(
    initialization: str, seed: int, output_dir: Path, *, timing_only: bool = False
) -> None:
    """Train, select, freeze/hash, and evaluate one lane exactly once."""

    validate_source_sha(SOURCE_SHA)
    if initialization not in {"pretrained", "random"} or seed not in MATCHED_SEEDS:
        raise ValueError("worker received a non-preregistered lane")
    sys.path.insert(0, str(SOURCE_DIR))
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from arcgpt2.meta_soft_binding import initialize_prefix, set_seed
    from arcgpt2.meta_soft_raw_goal_signature import (
        apply_gate,
        evaluate_groups,
        evaluate_validation_groups,
        freeze_for_locked_evaluation,
        from_data_group,
        protocol_provenance,
        validate_protocol_config,
    )
    from arcgpt2.raw_goal_signature_data import (
        build_manifests,
        canonical_manifest_digest,
    )

    config = _build_config(initialization, seed)
    validate_protocol_config(config)
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each worker must see exactly one CUDA device")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if initialization == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, attn_implementation="eager"
        )
    else:
        model_config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        model_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(model_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    _configure_model(model)
    model.to(device)
    model.eval()  # Disable dropout; gradients remain enabled.

    manifests = build_manifests()
    selected_manifests = manifests[:2] if timing_only else manifests
    manifest_digest = canonical_manifest_digest(selected_manifests)
    preflight_audit = load_json(RUN_ROOT / "pretraining-audit.json")
    expected_audit_digest = (
        preflight_audit.get("timing_manifest_sha256")
        if timing_only
        else preflight_audit.get("canonical_manifest_sha256")
    )
    if (
        not preflight_audit.get("passed")
        or expected_audit_digest != manifest_digest
    ):
        raise RuntimeError("worker disagrees with the fail-closed pretraining audit")
    by_name = {
        manifest.name: tuple(from_data_group(group) for group in manifest.groups)
        for manifest in selected_manifests
    }
    expected_sizes = {"train": TRAIN_GROUPS, "validation": VALIDATION_GROUPS}
    if not timing_only:
        expected_sizes["locked_test"] = LOCKED_GROUPS
    if {name: len(groups) for name, groups in by_name.items()} != expected_sizes:
        raise RuntimeError("worker split sizes drifted")

    prefix = initialize_prefix(
        model,
        prefix_length=config.prefix_length,
        std=config.prefix_initialization_std,
        seed=seed,
        device=device,
    )
    model_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": [prefix], "lr": config.prefix_learning_rate, "weight_decay": 0.0},
            {
                "params": model_parameters,
                "lr": config.model_learning_rate,
                "weight_decay": config.weight_decay,
            },
        ],
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    if timing_only:
        # Warm and measure the exact second-order training path without an
        # optimizer mutation, then time intact validation and the locked-shaped
        # evaluator on TRAIN data only.  No locked-test group is converted or
        # queried in this mode.
        optimizer.zero_grad(set_to_none=True)
        _, warmup_finite = _backward_exact_training_group(
            model, tokenizer, prefix, by_name["train"][0], config, device
        )
        warmup_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [prefix, *model_parameters], config.max_gradient_norm
        )
        warmup_finite &= bool(torch.isfinite(warmup_gradient_norm).item())
        torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        timing_started = time.perf_counter()
        timing_losses: list[float] = []
        timing_finite = warmup_finite
        for group in by_name["train"][1 : 1 + TIMING_TRAIN_GROUPS]:
            optimizer.zero_grad(set_to_none=True)
            loss_value, finite = _backward_exact_training_group(
                model, tokenizer, prefix, group, config, device
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [prefix, *model_parameters], config.max_gradient_norm
            )
            finite &= bool(torch.isfinite(gradient_norm).item())
            timing_losses.append(loss_value)
            timing_finite &= finite
        torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - timing_started
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        validation_started = time.perf_counter()
        validation_timing_report = evaluate_validation_groups(
            model,
            tokenizer,
            prefix.detach(),
            by_name["validation"][:TIMING_VALIDATION_GROUPS],
            config,
            device,
        )
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_started

        torch.cuda.synchronize(device)
        locked_shape_started = time.perf_counter()
        timing_evaluation_config = replace(config, bootstrap_samples=1)
        locked_shape_report = evaluate_groups(
            model,
            tokenizer,
            prefix.detach(),
            by_name["train"][:1],
            timing_evaluation_config,
            device,
            locked=False,
        )
        torch.cuda.synchronize(device)
        locked_shape_seconds = time.perf_counter() - locked_shape_started
        train_per_group = training_seconds / TIMING_TRAIN_GROUPS
        validation_per_group = validation_seconds / TIMING_VALIDATION_GROUPS
        projected_lane_raw = (
            train_per_group * OPTIMIZER_STEPS
            + validation_per_group * VALIDATION_GROUPS * EPOCHS
            + locked_shape_seconds * LOCKED_GROUPS
        )
        projected_lane_upper = (
            projected_lane_raw * TIMING_PROJECTION_MULTIPLIER
            + TIMING_FIXED_ALLOWANCE_SECONDS / len(MATCHED_SEEDS)
        )
        timing_summary = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "run_mode": "timing",
            "mode": "timing_only_train_validation_surfaces",
            "source_sha": SOURCE_SHA,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "protocol": PROTOCOL,
            "initialization": initialization,
            "seed": seed,
            "config": asdict(config),
            "canonical_train_validation_manifest_sha256": manifest_digest,
            "locked_test_groups_converted_or_queried": False,
            "optimizer_steps_performed": 0,
            "measurements": {
                "training_groups": TIMING_TRAIN_GROUPS,
                "unmeasured_warmup_training_groups": 1,
                "training_seconds": training_seconds,
                "seconds_per_training_group": train_per_group,
                "validation_groups": TIMING_VALIDATION_GROUPS,
                "validation_seconds": validation_seconds,
                "seconds_per_validation_group": validation_per_group,
                "locked_shape_train_groups": 1,
                "locked_shape_seconds_per_group": locked_shape_seconds,
                "locked_shape_bootstrap_samples": 1,
                "timing_only_config_drift": {"bootstrap_samples": 1},
                "peak_cuda_memory_bytes": peak_memory,
                "projected_full_lane_raw_seconds": projected_lane_raw,
                "projection_multiplier": TIMING_PROJECTION_MULTIPLIER,
                "kernel_fixed_allowance_seconds": TIMING_FIXED_ALLOWANCE_SECONDS,
                "projected_full_lane_upper_seconds": projected_lane_upper,
                "projected_three_sequential_pairs_upper_seconds": (
                    projected_lane_upper * len(MATCHED_SEEDS)
                ),
            },
            "checks": {
                "training_path_finite": timing_finite
                and all(math.isfinite(value) for value in timing_losses),
                "validation_path_finite": validation_timing_report["all_finite"],
                "locked_shape_path_finite": locked_shape_report["execution"]["all_finite"],
                "no_optimizer_step": True,
                "no_locked_test_access": True,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "timing-summary.json", timing_summary)
        print(json.dumps(timing_summary, indent=2, sort_keys=True), flush=True)
        return

    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_objective = math.inf
    best_state: tuple[dict[str, Any], Any] | None = None
    steps = 0
    all_training_finite = True
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_losses: list[float] = []
        for group_index, group in enumerate(by_name["train"], start=1):
            optimizer.zero_grad(set_to_none=True)
            # This micro-batched form is algebraically the exact frozen group
            # objective, while releasing each second-order graph before the
            # next world/order and therefore fitting safely on a T4.
            group_loss, group_finite = _backward_exact_training_group(
                model, tokenizer, prefix, group, config, device
            )
            all_training_finite &= group_finite
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [prefix, *model_parameters], config.max_gradient_norm
            )
            all_training_finite &= bool(torch.isfinite(gradient_norm).item())
            optimizer.step()
            all_training_finite &= all(
                bool(torch.isfinite(parameter.detach()).all().item())
                for parameter in (prefix, *model_parameters)
            )
            steps += 1
            epoch_losses.append(group_loss)
            if group_index % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "initialization": initialization,
                            "seed": seed,
                            "epoch": epoch,
                            "group": group_index,
                            "steps": steps,
                            "mean_epoch_loss": sum(epoch_losses) / len(epoch_losses),
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
        optimizer.zero_grad(set_to_none=True)
        validation = evaluate_validation_groups(
            model,
            tokenizer,
            prefix.detach(),
            by_name["validation"],
            config,
            device,
        )
        objective = float(validation["selection_objective"])
        all_training_finite &= bool(validation["all_finite"]) and math.isfinite(objective)
        history.append(
            {
                "epoch": epoch,
                "steps_completed": steps,
                "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
                "validation_selection_objective": objective,
                "validation": validation,
            }
        )
        # Strict improvement is the preregistered earlier-epoch tie break.
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            best_state = _capture_trainable_state(model, prefix)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "initialization": initialization,
                    "seed": seed,
                    "epoch": epoch,
                    "objective": objective,
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
    if steps != OPTIMIZER_STEPS or best_state is None or best_epoch is None:
        raise RuntimeError("worker did not complete the exact optimizer budget")
    expected_best = min(
        history,
        key=lambda row: (float(row["validation_selection_objective"]), int(row["epoch"])),
    )
    if int(expected_best["epoch"]) != best_epoch:
        raise RuntimeError("checkpoint selection disagrees with validation history")
    _restore_trainable_state(model, prefix, best_state)
    selected_checkpoint_sha256 = freeze_for_locked_evaluation(model, prefix)
    # The only locked evaluation call in this worker occurs after freezing.
    locked_report = evaluate_groups(
        model,
        tokenizer,
        prefix,
        by_name["locked_test"],
        config,
        device,
        locked=True,
    )
    locked_report["bootstrap"]["seed"] = config.bootstrap_seed
    capability_gate = apply_gate(locked_report)
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_mode": "full",
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "initialization": initialization,
        "seed": seed,
        "device": "cuda",
        "config": asdict(config),
        "protocol_provenance": protocol_provenance(
            config, manifest_sha256=manifest_digest
        ),
        "canonical_manifest_sha256": manifest_digest,
        "pretraining_audit_sha256": sha256_file(RUN_ROOT / "pretraining-audit.json"),
        "training": {
            "epochs": EPOCHS,
            "steps_completed": steps,
            "expected_steps": OPTIMIZER_STEPS,
            "all_finite": all_training_finite,
        },
        "history": history,
        "selection": {
            "criterion": "lowest_intact_validation_objective_then_earlier_epoch",
            "best_epoch": best_epoch,
            "best_objective": best_objective,
            "validation_evaluations": EPOCHS,
            "selected_validation": expected_best["validation"],
            "selected_validation_sha256": canonical_json_sha256(
                expected_best["validation"]
            ),
            "checkpoint_sha256_before_locked_test": selected_checkpoint_sha256,
        },
        "locked_test_evaluations": 1,
        "locked_test": locked_report,
        "capability_gate": capability_gate,
        "model_weights_persisted": False,
        "competition_submission_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "worker_complete",
                "initialization": initialization,
                "seed": seed,
                "capability_gate_passed": capability_gate["passed"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        ),
        flush=True,
    )


def run_pretraining_audits(*, timing_only: bool = False) -> tuple[str, dict[str, Any]]:
    """Run the pinned tokenizer and all deterministic audits before GPUs train."""

    sys.path.insert(0, str(SOURCE_DIR))
    from transformers import AutoTokenizer

    from arcgpt2.meta_soft_raw_goal_signature import (
        from_data_group,
        token_length_audit,
    )
    from arcgpt2.raw_goal_signature_data import (
        audit_manifests,
        audit_token_contract,
        build_manifests,
        canonical_manifest_digest,
        canonical_manifest_payload,
    )

    manifests = build_manifests()
    selected_manifests = manifests[:2] if timing_only else manifests
    manifest_report = audit_manifests(selected_manifests)
    manifest_payload = canonical_manifest_payload(selected_manifests)
    manifest_digest = canonical_manifest_digest(selected_manifests)
    timing_manifest_digest = canonical_manifest_digest(manifests[:2])
    if manifest_digest != canonical_json_sha256(manifest_payload):
        raise RuntimeError("canonical manifest digest mismatch")
    write_json(RUN_ROOT / "canonical-manifest.json", manifest_payload)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_reports = [
        audit_token_contract(group, tokenizer)
        for manifest in selected_manifests
        for group in manifest.groups
    ]
    converted = [
        from_data_group(group)
        for manifest in selected_manifests
        for group in manifest.groups
    ]
    model_report = token_length_audit(tokenizer, converted)
    candidate_patterns = Counter(
        tuple(report["candidate_lengths"]) for report in data_reports
    )
    terminal_patterns = Counter(
        tuple(report["terminal_lengths"]) for report in data_reports
    )
    passed = (
        not manifest_report["errors"]
        and all(report["passed"] for report in data_reports)
        and model_report["passed"]
        and len(data_reports)
        == TRAIN_GROUPS + VALIDATION_GROUPS + (0 if timing_only else LOCKED_GROUPS)
        and int(tokenizer.vocab_size) == 50_257
    )
    report = {
        "passed": passed,
        "performed_before_training": True,
        "source_sha": SOURCE_SHA,
        "protocol": PROTOCOL,
        "gpt2_revision": GPT2_REVISION,
        "canonical_manifest_sha256": manifest_digest,
        "timing_manifest_sha256": timing_manifest_digest,
        "scope": "train_validation_only" if timing_only else "all_frozen_splits",
        "manifest_audit": manifest_report,
        "manifest_file_sha256": sha256_file(RUN_ROOT / "canonical-manifest.json"),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "model_max_length": int(tokenizer.model_max_length),
        },
        "data_token_contract": {
            "groups_audited": len(data_reports),
            "groups_passed": sum(bool(report["passed"]) for report in data_reports),
            "candidate_length_patterns": [
                {"lengths": list(pattern), "count": count}
                for pattern, count in sorted(candidate_patterns.items())
            ],
            "terminal_length_patterns": [
                {"lengths": list(pattern), "count": count}
                for pattern, count in sorted(terminal_patterns.items())
            ],
        },
        "model_token_length_audit": model_report,
    }
    write_json(RUN_ROOT / "pretraining-audit.json", report)
    if not passed:
        raise RuntimeError("real pinned-tokenizer or deterministic pretraining audit failed")
    return manifest_digest, report


def _group_metric_rows(summary: Mapping[str, Any], metric: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for group in summary["locked_test"]["groups"]:
        orders = group["orders"]
        if len(orders) != 2:
            raise ValueError("promotion requires both evidence orders")
        key = str(group["group_key"])
        if key in rows:
            raise ValueError("duplicate locked group key")
        values = [float(order[metric]) for order in orders]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("promotion metrics must be finite")
        rows[key] = sum(values) / 2.0
    if len(rows) != LOCKED_GROUPS:
        raise ValueError("promotion requires exactly 72 independent groups")
    return rows


def hierarchical_paired_bootstrap(
    pairs: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    metric: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = HIERARCHICAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Frozen seed-then-group paired median bootstrap for one metric."""

    if len(pairs) != 3 or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("hierarchical promotion requires three seeds and valid draws")
    seed_deltas: list[dict[str, Any]] = []
    for matched_seed, pretrained, random_init in pairs:
        pre = _group_metric_rows(pretrained, metric)
        rnd = _group_metric_rows(random_init, metric)
        if set(pre) != set(rnd):
            raise ValueError("pretrained/random locked group IDs are not paired")
        keys = sorted(pre)
        deltas = [pre[key] - rnd[key] for key in keys]
        if not all(math.isfinite(value) for value in deltas):
            raise ValueError("paired promotion deltas must be finite")
        seed_deltas.append(
            {
                "seed": int(matched_seed),
                "group_keys": keys,
                "deltas": deltas,
                "mean_delta": sum(deltas) / len(deltas),
            }
        )
    if len({item["seed"] for item in seed_deltas}) != 3:
        raise ValueError("matched seeds must be unique")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected_seed_statistics: list[float] = []
        for _ in range(3):
            item = seed_deltas[rng.randrange(3)]
            values = item["deltas"]
            selected_seed_statistics.append(
                sum(values[rng.randrange(LOCKED_GROUPS)] for _ in range(LOCKED_GROUPS))
                / LOCKED_GROUPS
            )
        draws.append(statistics.median(selected_seed_statistics))
    rank = int(
        (
            (Decimal(1) - Decimal(str(confidence))) * Decimal(samples)
        ).to_integral_value(rounding=ROUND_CEILING)
    )
    lower_bound = sorted(draws)[max(0, rank - 1)]
    point_by_seed = {
        str(item["seed"]): float(item["mean_delta"]) for item in seed_deltas
    }
    return {
        "metric": metric,
        "median_seed_delta": statistics.median(point_by_seed.values()),
        "per_seed_mean_deltas": point_by_seed,
        "positive_seed_pairs": sum(value > 0 for value in point_by_seed.values()),
        "one_sided_95_percent_lower_bound": lower_bound,
        "bootstrap": {
            "method": "hierarchical_paired_seed_then_group_percentile_median",
            "seed_resamples_per_draw": 3,
            "group_resamples_per_selected_seed": LOCKED_GROUPS,
            "samples": samples,
            "confidence": confidence,
            "seed": seed,
            "group_unit": "matched_locked_group_id",
        },
    }


def promotion_track(
    pairs: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    *,
    name: str,
    accuracy_metric: str,
    probability_metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    accuracy = hierarchical_paired_bootstrap(
        pairs,
        accuracy_metric,
        seed=HIERARCHICAL_BOOTSTRAP_SEED + seed_offset,
    )
    probability = hierarchical_paired_bootstrap(
        pairs,
        probability_metric,
        seed=HIERARCHICAL_BOOTSTRAP_SEED + seed_offset + 1,
    )
    accuracy_path = (
        accuracy["median_seed_delta"] >= 0.10
        and accuracy["one_sided_95_percent_lower_bound"] > 0
        and accuracy["positive_seed_pairs"] >= 2
    )
    probability_path = (
        probability["median_seed_delta"] >= 0.05
        and probability["one_sided_95_percent_lower_bound"] > 0
        and probability["positive_seed_pairs"] >= 2
    )
    return {
        "name": name,
        "passed": accuracy_path or probability_path,
        "accuracy": accuracy,
        "probability": probability,
        "checks": {
            "accuracy_median_at_least_0_10": accuracy["median_seed_delta"] >= 0.10,
            "accuracy_lower_bound_positive": accuracy["one_sided_95_percent_lower_bound"] > 0,
            "accuracy_positive_in_two_of_three": accuracy["positive_seed_pairs"] >= 2,
            "probability_median_at_least_0_05": probability["median_seed_delta"] >= 0.05,
            "probability_lower_bound_positive": probability["one_sided_95_percent_lower_bound"] > 0,
            "probability_positive_in_two_of_three": probability["positive_seed_pairs"] >= 2,
            "accuracy_or_probability_path": accuracy_path or probability_path,
        },
    }


def _validate_worker_summary(
    summary: Mapping[str, Any], initialization: str, seed: int, manifest_digest: str
) -> dict[str, bool]:
    from arcgpt2.meta_soft_raw_goal_signature import apply_gate

    history = summary.get("history", [])
    expected_best = min(
        history,
        key=lambda row: (float(row["validation_selection_objective"]), int(row["epoch"])),
    ) if history else {}
    locked = summary.get("locked_test", {})
    execution = locked.get("execution", {})
    group_keys = [str(row.get("group_key")) for row in locked.get("groups", [])]
    bootstrap = locked.get("bootstrap", {})
    recorded_gate = summary.get("capability_gate", {})
    recomputed_gate = apply_gate(locked) if locked else {}
    expected_config = asdict(_build_config(initialization, seed))
    checks = {
        "source_sha_exact": summary.get("source_sha") == SOURCE_SHA,
        "protocol_exact": summary.get("protocol") == PROTOCOL,
        "initialization_exact": summary.get("initialization") == initialization,
        "seed_exact": summary.get("seed") == seed,
        "config_exact": summary.get("config") == expected_config,
        "manifest_exact": summary.get("canonical_manifest_sha256") == manifest_digest,
        "pretraining_audit_exact": summary.get("pretraining_audit_sha256")
        == sha256_file(RUN_ROOT / "pretraining-audit.json"),
        "history_two_epochs": len(history) == EPOCHS
        and [row.get("epoch") for row in history] == [1, 2],
        "optimizer_budget_exact": summary.get("training", {}).get("steps_completed")
        == OPTIMIZER_STEPS,
        "epoch_steps_exact": [row.get("steps_completed") for row in history]
        == [TRAIN_GROUPS, OPTIMIZER_STEPS],
        "finite_training": summary.get("training", {}).get("all_finite") is True,
        "intact_only_validation": all(
            row.get("validation", {}).get("mode") == "intact_validation_only"
            and row.get("validation", {}).get("controls_computed") == []
            and row.get("validation", {}).get("group_count") == VALIDATION_GROUPS
            for row in history
        ),
        "earlier_epoch_tie_break": summary.get("selection", {}).get("best_epoch")
        == expected_best.get("epoch"),
        "selected_objective_exact": summary.get("selection", {}).get("best_objective")
        == expected_best.get("validation_selection_objective"),
        "selected_validation_exact": summary.get("selection", {}).get("selected_validation")
        == expected_best.get("validation"),
        "locked_evaluated_once": summary.get("locked_test_evaluations") == 1,
        "locked_group_count": locked.get("group_count") == LOCKED_GROUPS,
        "locked_group_ids_unique": len(group_keys) == len(set(group_keys)) == LOCKED_GROUPS,
        "locked_outputs_finite": execution.get("all_finite") is True,
        "locked_updates_nonzero": float(execution.get("min_prefix_update_l2", 0.0)) > 1e-8,
        "locked_replay_deterministic": float(
            execution.get("deterministic_replay_delta", math.inf)
        ) <= 1e-6,
        "locked_no_trial3_update": execution.get("trial3_updates") == 0,
        "locked_group_bootstrap_unit": execution.get("bootstrap_unit")
        == "independent_group",
        "locked_raw_inner_objective": execution.get("inner_update_objective")
        == "raw_outcome_nll",
        "locked_inner_candidate_count_one": execution.get("inner_candidate_count") == 1,
        "locked_inner_mean_reduction": execution.get("inner_reduction") == "mean",
        "locked_token_audit": execution.get("token_lengths_audited") is True,
        "locked_bootstrap_exact": bootstrap.get("samples") == BOOTSTRAP_SAMPLES
        and bootstrap.get("confidence") == BOOTSTRAP_CONFIDENCE
        and bootstrap.get("seed") == BOOTSTRAP_SEED,
        "locked_evaluation_mode": execution.get("locked_evaluation") is True,
        "frozen_before_locked": execution.get("frozen_before_locked_evaluation") is True,
        "checkpoint_unchanged": execution.get("checkpoint_unchanged") is True,
        "checkpoint_hash_chain": summary.get("selection", {}).get(
            "checkpoint_sha256_before_locked_test"
        )
        == execution.get("checkpoint_sha256_before")
        == execution.get("checkpoint_sha256_after"),
        "gate_recomputed_exact": recorded_gate == recomputed_gate,
        "no_weights_persisted": summary.get("model_weights_persisted") is False,
        "no_competition_submission": summary.get("competition_submission_performed") is False,
    }
    return checks


def _write_artifact_manifest() -> None:
    write_json(
        RUN_ROOT / "artifact-manifest.json",
        {
            str(path.relative_to(RUN_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file()
            and path.name != "artifact-manifest.json"
            and ".cache" not in path.parts
        },
    )


def _collect_timing_result(seed: int) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for initialization in ("pretrained", "random"):
        path = (
            RUN_ROOT
            / "timing"
            / f"seed-{seed}"
            / initialization
            / "timing-summary.json"
        )
        if not path.is_file():
            raise RuntimeError(f"missing timing summary: {path}")
        summary = load_json(path)
        lanes[initialization] = summary
        lane_checks = summary.get("checks", {})
        checks.update(
            {
                f"{initialization}_source": summary.get("source_sha") == SOURCE_SHA,
                f"{initialization}_runner_hash": summary.get(
                    "runner_source_sha256"
                )
                == sha256_file(Path(__file__).resolve()),
                f"{initialization}_seed": summary.get("seed") == seed,
                f"{initialization}_identity": summary.get("initialization")
                == initialization,
                f"{initialization}_no_locked_access": summary.get(
                    "locked_test_groups_converted_or_queried"
                )
                is False,
                f"{initialization}_no_optimizer_step": summary.get(
                    "optimizer_steps_performed"
                )
                == 0,
                f"{initialization}_finite": bool(lane_checks)
                and all(value is True for value in lane_checks.values()),
                f"{initialization}_projection_finite": math.isfinite(
                    float(
                        summary.get("measurements", {}).get(
                            "projected_full_lane_upper_seconds", math.inf
                        )
                    )
                ),
            }
        )
    projected_pair = max(
        float(lanes[name]["measurements"]["projected_full_lane_upper_seconds"])
        for name in ("pretrained", "random")
    )
    projected_kernel = projected_pair * len(MATCHED_SEEDS)
    checks["projected_kernel_within_ceiling"] = (
        projected_kernel <= FULL_KERNEL_RUNTIME_CEILING_SECONDS
    )
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_mode": "timing",
        "status": "within_runtime_ceiling"
        if checks["projected_kernel_within_ceiling"]
        else "above_runtime_ceiling",
        "scope": (
            "Timing-only exact training/intact-validation and locked-shaped compute on "
            "train surfaces; no locked-test group and no optimizer mutation."
        ),
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "seed": seed,
        "runtime_ceiling_seconds": FULL_KERNEL_RUNTIME_CEILING_SECONDS,
        "projected_seconds_per_concurrent_pair": projected_pair,
        "projected_three_pair_kernel_seconds": projected_kernel,
        "checks": checks,
        "lanes": lanes,
        "competition_submission_performed": False,
        "model_weights_persisted": False,
    }


def run_parent(*, timing_only: bool = False) -> None:
    validate_source_sha(SOURCE_SHA)
    started = time.time()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    install_console_tee()
    os.environ.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
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
        raise RuntimeError("cloned source SHA does not match the injected SHA")
    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arcgpt2/tests/test_raw_goal_signature_data.py",
            "arcgpt2/tests/test_meta_soft_raw_goal_signature.py",
            "arcgpt2/tests/test_raw_goal_signature_runner.py",
            "arcgpt2/tests/test_raw_goal_signature_statistics.py",
            "arcgpt2/tests/test_meta_soft_second_order.py",
            "arcgpt2/tests/test_completion_scorer.py",
        ],
        cwd=SOURCE_DIR,
        env=cpu_env,
        log=LOG_DIR / "cpu-preflight-tests.log",
    )
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("raw goal signature gate requires exactly two CUDA devices")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "T4" not in name.upper() or tuple(capability) != (7, 5):
            raise RuntimeError(f"device {index} is not a Tesla T4 capability 7.5")
        probe = torch.ones(256, device=f"cuda:{index}")
        if float((probe @ probe).item()) != 256.0:
            raise RuntimeError(f"CUDA compute probe failed on device {index}")
        devices.append(
            {"index": index, "name": name, "compute_capability": list(capability)}
        )
    del probe
    torch.cuda.empty_cache()
    environment = {
        "recorded_at_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "gpt2_repository": GPT2_REPOSITORY,
        "gpt2_revision": GPT2_REVISION,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
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
            str(path.relative_to(MODEL_DIR)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(MODEL_DIR.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        },
    )
    manifest_digest, pretraining_audit = run_pretraining_audits(
        timing_only=timing_only
    )
    with GpuSampler() as sampler:
        run_gpu_pair(MATCHED_SEEDS[0], timing_only=True)
        timing_result = _collect_timing_result(MATCHED_SEEDS[0])
        write_json(RUN_ROOT / "timing-result.json", timing_result)
        if not timing_only:
            if not all(timing_result["checks"].values()):
                raise RuntimeError(
                    "timing preflight failed or projected runtime exceeds the frozen ceiling"
                )
            for seed in MATCHED_SEEDS:
                run_gpu_pair(seed)
    write_json(RUN_ROOT / "gpu-samples.json", sampler.samples)

    if timing_only:
        print(json.dumps(timing_result, indent=2, sort_keys=True), flush=True)
        close_console_tee()
        _write_artifact_manifest()
        return

    sys.path.insert(0, str(SOURCE_DIR))
    summaries: dict[int, dict[str, dict[str, Any]]] = {}
    execution_gates: dict[str, bool] = {
        "source_sha_resolved": resolved_sha == SOURCE_SHA,
        "dual_t4_verified": len(devices) == 2,
        "three_matched_seed_pairs": len(MATCHED_SEEDS) == 3,
        "pretraining_audit_passed": bool(pretraining_audit["passed"]),
        "canonical_manifest_recorded": (RUN_ROOT / "canonical-manifest.json").is_file(),
        "timing_preflight_passed": all(timing_result["checks"].values()),
    }
    for seed in MATCHED_SEEDS:
        summaries[seed] = {}
        for initialization in ("pretrained", "random"):
            path = OUTPUT_DIR / f"seed-{seed}" / initialization / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"missing worker summary: {path}")
            summary = load_json(path)
            summaries[seed][initialization] = summary
            checks = _validate_worker_summary(
                summary, initialization, seed, manifest_digest
            )
            execution_gates.update(
                {
                    f"seed_{seed}_{initialization}_{name}": passed
                    for name, passed in checks.items()
                }
            )
    weight_artifacts = [
        path
        for path in RUN_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}
    ]
    execution_gates["no_trained_model_artifacts"] = not weight_artifacts
    per_gpu: dict[str, Any] = {}
    for index in range(2):
        rows = [row for row in sampler.samples if row.get("device_index") == index]
        per_gpu[str(index)] = {
            "sample_count": len(rows),
            "max_memory_used_mib": max(
                (int(row.get("memory_used_mib", 0)) for row in rows), default=0
            ),
            "max_utilization_percent": max(
                (int(row.get("utilization_percent", 0)) for row in rows), default=0
            ),
        }
        execution_gates[f"gpu_{index}_sampled"] = bool(rows)
        execution_gates[f"gpu_{index}_allocated_memory"] = (
            per_gpu[str(index)]["max_memory_used_mib"] > 0
        )
    if not all(execution_gates.values()):
        failed = [name for name, passed in execution_gates.items() if not passed]
        raise RuntimeError(f"execution-contract gate failed: {failed}")

    pairs = [
        (seed, summaries[seed]["pretrained"], summaries[seed]["random"])
        for seed in MATCHED_SEEDS
    ]
    identification = promotion_track(
        pairs,
        name="identification_pretraining_promotion",
        accuracy_metric="goal_accuracy",
        probability_metric="goal_probability",
        seed_offset=0,
    )
    semantic = promotion_track(
        pairs,
        name="trial3_semantic_pretraining_promotion",
        accuracy_metric="trial3_accuracy",
        probability_metric="trial3_probability",
        seed_offset=10,
    )
    pretrained_absolute = {
        str(seed): bool(summaries[seed]["pretrained"]["capability_gate"]["passed"])
        for seed in MATCHED_SEEDS
    }
    all_pretrained_capability = all(pretrained_absolute.values())
    pretraining_promotion = identification["passed"] and semantic["passed"]
    scale_eligible = all_pretrained_capability and pretraining_promotion
    result = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_mode": "full",
        "status": "pass" if scale_eligible else "not_yet_passed",
        "scope": (
            "Locked balanced synthetic four-atom raw terminal-report gradient-memory gate; "
            "not an ARC-AGI-3 score or general Goal-DSL result."
        ),
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "source_sha": SOURCE_SHA,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": PROTOCOL,
        "gpt2_revision": GPT2_REVISION,
        "matched_seeds": list(MATCHED_SEEDS),
        "canonical_manifest_sha256": manifest_digest,
        "pretraining_audit_sha256": sha256_file(RUN_ROOT / "pretraining-audit.json"),
        "competition_submission_performed": False,
        "model_weights_persisted": False,
        "gpu_assignment": {"pretrained": "physical cuda:0", "random": "physical cuda:1"},
        "gpu_observations": per_gpu,
        "execution_gates": execution_gates,
        "absolute_pretrained_capability": {
            "passed": all_pretrained_capability,
            "every_seed_required": True,
            "per_seed": pretrained_absolute,
        },
        "random_control_not_required_to_fail": True,
        "pretraining_promotion": {
            "passed": pretraining_promotion,
            "identification": identification,
            "semantic_trial3": semantic,
        },
        "scale_promotion": {
            "eligible": scale_eligible,
            "decision": "promote_scale" if scale_eligible else "do_not_promote_scale",
            "requires_every_pretrained_seed_absolute_gate": True,
            "requires_identification_and_semantic_pretraining_promotion": True,
        },
        "runs": {
            str(seed): {
                initialization: {
                    "summary_path": str(
                        (OUTPUT_DIR / f"seed-{seed}" / initialization / "summary.json").relative_to(RUN_ROOT)
                    ),
                    "capability_gate": summaries[seed][initialization]["capability_gate"],
                }
                for initialization in ("pretrained", "random")
            }
            for seed in MATCHED_SEEDS
        },
    }
    write_json(RUN_ROOT / "raw-goal-signature-result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    close_console_tee()
    _write_artifact_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--initialization", choices=("pretrained", "random"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timing-only", action="store_true")
    arguments = parser.parse_args()
    if RUN_MODE not in {"__RUN_MODE__", "full", "timing"}:
        parser.error(f"invalid injected RUN_MODE: {RUN_MODE!r}")
    if not arguments.worker and RUN_MODE == "timing":
        arguments.timing_only = True
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    try:
        if arguments.worker:
            if arguments.initialization is None or arguments.seed is None or arguments.output_dir is None:
                raise ValueError("worker arguments are incomplete")
            run_worker(
                arguments.initialization,
                arguments.seed,
                arguments.output_dir,
                timing_only=arguments.timing_only,
            )
        else:
            run_parent(timing_only=arguments.timing_only)
    except Exception as exc:
        failure = {
            "status": "execution_failed",
            "failed_at_utc": utc_now(),
            "source_sha": SOURCE_SHA,
            "protocol": PROTOCOL,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
            "competition_submission_performed": False,
            "model_weights_persisted": False,
        }
        if arguments.worker and arguments.output_dir is not None:
            write_json(arguments.output_dir / "failure.json", failure)
        else:
            write_json(RUN_ROOT / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        close_console_tee()
        if not arguments.worker:
            _write_artifact_manifest()
        raise
