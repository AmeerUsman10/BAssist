from __future__ import annotations

import json

from arcgpt2.build_epistemic_counterfactual_dataset import (
    build_counterfactual_group,
    build_dataset,
    mapping_variants,
)


def _variant_rows(rows, variant: int, probe: int = 0):
    return [
        row
        for row in rows
        if int(row["mapping_variant_index"]) == variant
        and int(row["probe_count"]) == probe
    ]


def test_mapping_variants_are_distinct_and_reproducible() -> None:
    first = mapping_variants(9001, 24)
    second = mapping_variants(9001, 24)
    assert first == second
    assert len(set(first)) == 24


def test_counterfactual_twins_share_surface_but_have_different_truths() -> None:
    rows = build_counterfactual_group(9002, 6)
    assert len(rows) == 6 * 20
    surface_hashes = {row["surface_grid_sha256"] for row in rows}
    initial_grids = {json.dumps(row["initial_grid"]) for row in rows}
    truth_mappings = {
        json.dumps(row["truth_mapping"], sort_keys=True)
        for row in rows
    }
    assert len(surface_hashes) == 1
    assert len(initial_grids) == 1
    assert len(truth_mappings) == 6


def test_amnesic_context_is_identical_across_mapping_variants() -> None:
    rows = build_counterfactual_group(9003, 8)
    for action in ("A1", "A2", "A3", "A4"):
        contexts = {
            row["amnesic_context"]
            for row in rows
            if row["probe_count"] == 0 and row["query_action"] == action
        }
        targets = {
            row["truth_word"]
            for row in rows
            if row["probe_count"] == 0 and row["query_action"] == action
        }
        assert len(contexts) == 1
        assert len(targets) > 1


def test_intervention_history_separates_counterfactual_variants() -> None:
    rows = build_counterfactual_group(9004, 6)
    full_contexts = set()
    truth_mappings = set()
    for variant in range(6):
        variant_rows = _variant_rows(rows, variant, probe=4)
        assert len(variant_rows) == 4
        full_contexts.add(variant_rows[0]["context"])
        truth_mappings.add(json.dumps(variant_rows[0]["truth_mapping"], sort_keys=True))
        assert all(len(row["allowed_words"]) == 1 for row in variant_rows)
    assert len(full_contexts) == 6
    assert len(truth_mappings) == 6


def test_counterfactual_groups_never_cross_splits(tmp_path) -> None:
    manifest = build_dataset(
        tmp_path,
        train_groups=3,
        validation_groups=2,
        test_groups=2,
        mapping_variants_per_group=4,
        seed_base=9100,
    )
    groups = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        groups[split] = {int(row["counterfactual_group_seed"]) for row in rows}
        expected = manifest["splits"][split]["counterfactual_groups"] * 4 * 20
        assert len(rows) == expected
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
