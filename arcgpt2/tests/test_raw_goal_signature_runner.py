"""Execution-contract tests for the raw-goal Kaggle runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from arcgpt2.raw_goal_signature_data import audit_manifests, build_manifests


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "kaggle"
    / "arc-gpt2-raw-goal-signature-gpu"
    / "runner.py"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("raw_goal_signature_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_runner_identity_and_budget(runner):
    assert runner.PROTOCOL == "raw_goal_signature_v1"
    assert runner.MATCHED_SEEDS == (577_215, 618_033, 707_106)
    assert (runner.TRAIN_GROUPS, runner.VALIDATION_GROUPS, runner.LOCKED_GROUPS) == (
        120,
        24,
        72,
    )
    assert runner.EPOCHS == 2
    assert runner.OPTIMIZER_STEPS == 240
    assert runner.BOOTSTRAP_SAMPLES == 10_000
    assert runner.BOOTSTRAP_CONFIDENCE == 0.95
    assert runner.BOOTSTRAP_SEED == 20_260_804
    assert runner.HIERARCHICAL_BOOTSTRAP_SEED == 20_260_805
    assert runner.RUN_MODE == "__RUN_MODE__"
    with pytest.raises(RuntimeError):
        runner.validate_source_sha(runner.SOURCE_SHA)


def test_json_safe_typed_mapping_is_collision_free(runner):
    normalized = runner.json_safe({1: "integer", "1": "string", (1, "1"): "tuple"})
    assert normalized["__type__"] == "typed_mapping"
    assert len(normalized["entries"]) == 3
    encoded_keys = {
        json.dumps(row["key"], sort_keys=True) for row in normalized["entries"]
    }
    assert len(encoded_keys) == 3


def test_real_manifest_audit_receipt_roundtrip(runner, tmp_path):
    """Exercise the tuple/Enum-keyed tables that previously crashed JSON."""

    audit = audit_manifests(build_manifests())
    target = tmp_path / "audit.json"
    runner.write_json(target, {"manifest_audit": audit})
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["manifest_audit"]["errors"] == []
    train = loaded["manifest_audit"]["splits"]["train"]
    assert train["signature_permutation_counts"]["__type__"] == "typed_mapping"
    assert len(train["signature_permutation_counts"]["entries"]) == 24


def test_timing_mode_is_bounded_to_nonlocked_surfaces(runner):
    assert runner.TIMING_TRAIN_GROUPS >= 1
    assert runner.TIMING_VALIDATION_GROUPS >= 1
    assert runner.TIMING_PROJECTION_MULTIPLIER == 1.25
    assert runner.TIMING_FIXED_ALLOWANCE_SECONDS == 900
    assert runner.FULL_KERNEL_RUNTIME_CEILING_SECONDS <= 11 * 60 * 60
