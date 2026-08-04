from __future__ import annotations

import math

from instella_arc.catalog import CHECKPOINTS, catalog_payload, estimate_weight_memory


def test_catalog_contains_the_three_decision_checkpoints() -> None:
    assert set(CHECKPOINTS) == {"base", "dpo", "think"}
    assert CHECKPOINTS["base"].chat_tuned is False
    assert CHECKPOINTS["dpo"].chat_tuned is True
    assert CHECKPOINTS["think"].repository_id.endswith("-Think")
    assert catalog_payload()["selection_policy"]["one_checkpoint_per_run"] is True


def test_memory_estimates_are_monotonic_and_conservative() -> None:
    estimates = estimate_weight_memory(32 * 1024**3)
    assert estimates["bf16_or_fp16"].weight_gib == 32.0
    assert estimates["int8"].weight_gib == 16.0
    assert estimates["int4"].weight_gib == 8.0
    for estimate in estimates.values():
        assert estimate.runtime_floor_gib > estimate.weight_gib
        assert math.isfinite(estimate.runtime_floor_gib)
