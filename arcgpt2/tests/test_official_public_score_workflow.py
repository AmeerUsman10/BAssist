"""Static and fake-integration checks for the one-shot official score runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPOSITORY / "kaggle/arc-gpt2-official-public-score-gpu/runner.py"
)
METADATA_PATH = (
    REPOSITORY / "kaggle/arc-gpt2-official-public-score-gpu/kernel-metadata.json"
)
WORKFLOW_PATH = (
    REPOSITORY / ".github/workflows/arc-gpt2-official-public-score-gpu.yml"
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("official_public_score_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_identity_hashes_and_13_12_manifest_shards(runner):
    assert runner.PROTOCOL == "official_public_score_v1"
    assert runner.ARC_AGI_VERSION == "0.9.9"
    assert runner.ARCENGINE_VERSION == "0.9.3"
    assert runner.PYTHON_VERSION == (3, 12)
    assert runner.GPT2_REVISION == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert runner.LOCK_SHA256 == (
        "0b6ba06bbb5104c49608dcfc25e54cdb54202844057b519c65395765ca0c3bec"
    )
    assert runner.PUBLIC_MANIFEST_SHA256 == (
        "77b441ebbba044653c6abe64d17ab3023e54fd1ed6fec732bf0c525614e2e7dc"
    )
    with pytest.raises(RuntimeError):
        runner.validate_source_sha(runner.SOURCE_SHA)

    manifest = runner.load_public_manifest(
        REPOSITORY / "arcgpt2/official_public_manifest.json"
    )
    first, second = runner.score_shards(manifest["game_ids"])
    assert first["lane"] == "cuda:0"
    assert second["lane"] == "cuda:1"
    assert len(first["game_ids"]) == 13
    assert len(second["game_ids"]) == 12
    assert first["game_ids"] + second["game_ids"] == manifest["game_ids"]


def test_text_model_manifest_allowed_but_payloads_blocked(runner, tmp_path):
    (tmp_path / "model-manifest.json").write_text(
        json.dumps({"files": {"model.safetensors": {"sha256": "a" * 64}}}),
        encoding="utf-8",
    )
    clean = runner.output_boundary_violations(tmp_path)
    assert clean == {
        "unexpected_files": [],
        "model_payloads": [],
        "secret_leaks": [],
    }

    (tmp_path / "model.safetensors").write_bytes(b"not-a-real-checkpoint")
    blocked = runner.output_boundary_violations(tmp_path)
    assert blocked["model_payloads"] == ["model.safetensors"]
    assert "model.safetensors" in blocked["unexpected_files"]


def test_evidence_sanitizer_removes_sdk_identity_and_credentials(runner):
    sanitized = runner.sanitize_evidence(
        {
            "card_id": "card-secret",
            "api_key": "api-secret",
            "nested": {"guid": "guid-secret", "score": 1.25},
            "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            "safe": "kept",
        }
    )
    assert sanitized == {"nested": {"score": 1.25}, "safe": "kept"}


def test_lane_factory_binds_driver_seam_in_manifest_order_without_parallel_claim(runner):
    manifest = runner.load_public_manifest(
        REPOSITORY / "arcgpt2/official_public_manifest.json"
    )
    game_ids = manifest["game_ids"]
    lane_calls = []
    lanes = (
        runner.ScoreLane("cuda:0", "cuda:0", object(), object()),
        runner.ScoreLane("cuda:1", "cuda:1", object(), object()),
    )

    def memory_factory(lane):
        lane_calls.append(lane.name)
        return {"lane": lane.name}

    factory = runner.LaneProviderFactory(lanes, memory_factory)
    providers = [factory() for _ in game_ids]
    assert [provider["lane"] for provider in providers] == (
        ["cuda:0"] * 13 + ["cuda:1"] * 12
    )
    assert lane_calls == ["cuda:0"] * 13 + ["cuda:1"] * 12
    driver = factory.receipt(game_ids)
    assert driver["lane_game_counts"] == {"cuda:0": 13, "cuda:1": 12}
    assert driver["concurrent_scorecard_writers"] == 0
    assert driver["parallel_score_execution"] is False
    assert driver["parallel_score_execution_reason"] == (
        "arc_agi_0_9_9_shared_scorecard_thread_safety_not_proven"
    )
    assert driver["score_speedup_claimed"] is False
    assert driver["one_scorecard"] is True
    with pytest.raises(RuntimeError, match="exceeded"):
        factory()


@pytest.mark.parametrize(
    "change",
    [
        {"enabled": False},
        {"failures": [{"phase": "observe"}]},
        {"fallback_used": True},
        {"reset_calls": 0},
        {"reset_calls": 2},
    ],
)
def test_provider_gate_rejects_disabled_failure_fallback_and_reset_drift(runner, change):
    receipt = {
        "enabled": True,
        "failures": [],
        "fallback_used": False,
        "reset_calls": 1,
    }
    runner.validate_provider_receipt(receipt)
    receipt.update(change)
    with pytest.raises(RuntimeError, match="provider receipt"):
        runner.validate_provider_receipt(receipt)


def test_failed_provider_contract_cannot_retain_a_model_score_claim(runner):
    receipt = {
        "aggregate": {
            "valid": True,
            "status": "valid",
            "official_score": 12.5,
            "reasons": [],
        }
    }
    output = runner.driver_receipt_for_output(
        receipt,
        model_score_eligible=False,
        failure_reason="runner_provider_contract_failure",
    )
    assert output["aggregate"] == {
        "valid": False,
        "status": "partial",
        "official_score": None,
        "reasons": ["runner_provider_contract_failure"],
    }
    assert output["runner_model_score_contract"]["eligible"] is False


def test_private_metadata_has_no_external_sources():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["code_file"] == "runner.py"
    assert metadata["competition_sources"] == []
    assert metadata["dataset_sources"] == []
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []


def test_workflow_has_one_shot_guard_secret_split_and_failure_publication():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "branches:\n      - arc-gpt2-official-score-v1" in workflow
    assert "workflow_dispatch:" not in workflow
    authorized = re.search(r'^  AUTHORIZED_PARENT_SHA: "([^"]+)"$', workflow, re.MULTILINE)
    assert authorized is not None
    authorized_value = authorized.group(1)
    placeholder_present = authorized_value == "__AUTHORIZED_PARENT_SHA__"
    frozen_sha_present = re.fullmatch(r"[0-9a-f]{40}", authorized_value) is not None
    assert placeholder_present != frozen_sha_present
    assert workflow.count(authorized_value) == 1
    assert "github.event.before" in workflow
    assert "launch commit must change only the official workflow" in workflow
    assert "[run-kaggle-official-public-score-v1]" in workflow
    assert workflow.count("persist-credentials: false") >= 2
    assert "preflight:\n" in workflow and "dispatch:\n" in workflow and "publish:\n" in workflow
    dispatch = workflow.split("  dispatch:\n", 1)[1].split("  publish:\n", 1)[0]
    publish = workflow.split("  publish:\n", 1)[1]
    assert "permissions:\n      contents: read" in dispatch
    assert "permissions:\n      actions: read\n      contents: write" in publish
    assert "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}" in dispatch
    assert "KAGGLE_API_TOKEN" not in publish
    assert "actions/upload-artifact@65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08" in dispatch
    assert "if: always() && needs.dispatch.result != 'skipped'" in publish
    assert "|| true" not in workflow
    assert "kaggle competitions" not in workflow.lower()
    assert "OperationMode.ONLINE" not in runner_source
    assert "OperationMode.COMPETITION" not in runner_source
    assert '"competition_sources": []' in METADATA_PATH.read_text(encoding="utf-8")
    assert '["git", "-C", str(SOURCE_DIR), "diff", "--exit-code", "--", "."]' in runner_source
    assert runner_source.index("source_integrity = verify_runtime_source_integrity") < runner_source.index(
        "    import torch\n"
    )
    assert 'target = Path("evidence-bundle")' not in dispatch
    assert 'Path(os.environ["RUNNER_TEMP"])' in dispatch
    assert 'unsafe_files_uploaded": False' in dispatch
    assert "os.replace(ready, upload)" in dispatch
    assert dispatch.index("if leaks or model_payloads:") < dispatch.index(
        "os.replace(ready, upload)"
    )
