"""Official Instella-MoE checkpoint catalog and memory arithmetic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    repository_id: str
    revision: str
    stage: str
    chat_tuned: bool
    preferred_use: str


# Revisions were resolved by the no-weight audit on 2026-08-03. Pinning them
# prevents a silent upstream change from invalidating a frozen comparison.
CHECKPOINTS: dict[str, CheckpointSpec] = {
    "base": CheckpointSpec(
        key="base",
        repository_id="amd/Instella-MoE-16B-A3B-Base",
        revision="5ca845e88b237ca66c9c8e1f2551933a47b0daf9",
        stage="long-context base",
        chat_tuned=False,
        preferred_use="clean adaptation and causal-scoring control",
    ),
    "dpo": CheckpointSpec(
        key="dpo",
        repository_id="amd/Instella-MoE-16B-A3B-DPO",
        revision="ef5a850b1e5638a98b2e28cf321a6c1b63ccde39",
        stage="direct preference optimization",
        chat_tuned=True,
        preferred_use="first ARC-specific parameter-efficient tuning substrate",
    ),
    "think": CheckpointSpec(
        key="think",
        repository_id="amd/Instella-MoE-16B-A3B-Think",
        revision="e67a4a54d81b19692ec85ea1d1c777aa5c0bfd83",
        stage="reinforcement learning / multi-teacher on-policy distillation",
        chat_tuned=True,
        preferred_use="strongest frozen reasoning baseline",
    ),
}


@dataclass(frozen=True)
class MemoryEstimate:
    weight_gib: float
    runtime_floor_gib: float
    note: str


def gib(byte_count: int | float) -> float:
    return float(byte_count) / (1024.0**3)


def estimate_weight_memory(total_weight_bytes: int) -> dict[str, MemoryEstimate]:
    """Return conservative load-time weight estimates.

    Quantized estimates are planning values, not claims that a given backend can
    quantize every custom MoE module. The runtime audit records module support
    separately before a GPU run is attempted.
    """

    bf16 = gib(total_weight_bytes)
    return {
        "bf16_or_fp16": MemoryEstimate(
            weight_gib=bf16,
            runtime_floor_gib=bf16 * 1.10,
            note="Raw two-byte checkpoint plus a 10% minimum allocator/module overhead.",
        ),
        "int8": MemoryEstimate(
            weight_gib=bf16 / 2.0,
            runtime_floor_gib=(bf16 / 2.0) * 1.18,
            note="One-byte planning estimate plus scales and runtime overhead.",
        ),
        "int4": MemoryEstimate(
            weight_gib=bf16 / 4.0,
            runtime_floor_gib=(bf16 / 4.0) * 1.30,
            note="Half-byte planning estimate plus double-quantization metadata and runtime overhead.",
        ),
    }


def catalog_payload() -> dict[str, Any]:
    return {
        "checkpoints": {key: asdict(value) for key, value in CHECKPOINTS.items()},
        "selection_policy": {
            "frozen_order": ["think", "dpo", "base"],
            "tuning_order": ["dpo", "base", "think"],
            "one_checkpoint_per_run": True,
            "pinned_upstream_revisions": True,
        },
    }


def stable_sha256(value: Mapping[str, Any] | list[Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
