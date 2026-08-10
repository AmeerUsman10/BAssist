"""Build counterfactual-twin action-binding episodes for GPT-2.

Every geometry/palette/goal seed is expanded into several worlds that differ
*only* in the hidden permutation from A1..A4 to cardinal directions. This
removes a major shortcut: the initial grid cannot predict the mapping because
identical grids are paired with contradictory mappings. Only the literal
intervention history distinguishes the variants.

All variants of one base seed stay in the same dataset split.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Sequence

from .build_epistemic_dataset import (
    _DIRECTION_WORD,
    _factorial,
    _sha256,
    _write_jsonl,
    allowed_directions,
    probe_order,
)
from .natural_protocol import direction_words, mapping_prompt, rotate_action_labels
from .phase0_hidden_action import Action, Direction, HiddenActionGame, generate_game


def mapping_variants(seed: int, count: int) -> tuple[tuple[Direction, ...], ...]:
    """Return deterministic distinct permutations for one counterfactual group."""

    if not 1 <= count <= 24:
        raise ValueError("mapping variant count must lie between 1 and 24")
    variants = list(itertools.permutations(tuple(Direction)))
    random.Random(seed ^ 0xC0A7_EFEC).shuffle(variants)
    return tuple(variants[:count])


def _grid_sha256(grid) -> str:
    payload = json.dumps([list(row) for row in grid], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_variant_examples(
    base_seed: int,
    variant_index: int,
    permutation: Sequence[Direction],
) -> list[dict[str, object]]:
    base = generate_game(base_seed)
    mapping = {
        action: direction
        for action, direction in zip(tuple(Action), permutation, strict=True)
    }
    spec = replace(base, action_to_direction=mapping)
    game = HiddenActionGame(spec)
    initial = game.frame
    order = probe_order(base_seed)
    records = []
    examples: list[dict[str, object]] = []
    words = direction_words()
    surface_hash = _grid_sha256(initial)

    for prefix_length in range(len(order) + 1):
        if prefix_length > 0:
            record = game.step(order[prefix_length - 1])
            if record.status != "ACTIVE":
                raise RuntimeError("the safe probe arena terminated unexpectedly")
            records.append(record)

        observed = {
            record.action: spec.action_to_direction[record.action] for record in records
        }
        shuffled_labels = rotate_action_labels(records)
        observed_json = {
            action.value: direction.value
            for action, direction in sorted(observed.items(), key=lambda item: item[0].value)
        }

        for query_action in Action:
            allowed = allowed_directions(spec, records, query_action)
            allowed_words = tuple(_DIRECTION_WORD[direction] for direction in allowed)
            probability = 1.0 / len(allowed_words)
            target_distribution = {
                word: probability if word in allowed_words else 0.0 for word in words
            }
            truth = spec.action_to_direction[query_action]
            examples.append(
                {
                    "schema": "arcgpt2.epistemic_binding.counterfactual.v1",
                    "counterfactual_group_seed": base_seed,
                    "surface_grid_sha256": surface_hash,
                    "mapping_variant_index": variant_index,
                    "mapping_variant_count": None,
                    "game_seed": base_seed,
                    "probe_count": prefix_length,
                    "probe_order": [action.value for action in order],
                    "observed_actions": [record.action.value for record in records],
                    "observed_mapping": observed_json,
                    "query_action": query_action.value,
                    "query_was_observed": query_action in observed,
                    "context": mapping_prompt(initial, tuple(records), query_action),
                    "amnesic_context": mapping_prompt(
                        initial,
                        (),
                        query_action,
                        include_evidence=False,
                    ),
                    "shuffled_context": mapping_prompt(
                        initial,
                        tuple(records),
                        query_action,
                        displayed_actions=shuffled_labels,
                    ),
                    "candidate_words": list(words),
                    "allowed_words": list(allowed_words),
                    "target_distribution": target_distribution,
                    "truth_word": _DIRECTION_WORD[truth],
                    "truth_mapping": {
                        action.value: _DIRECTION_WORD[spec.action_to_direction[action]]
                        for action in Action
                    },
                    "consistent_mapping_count": _factorial(len(Action) - prefix_length),
                    "initial_grid": [list(row) for row in initial],
                }
            )
    return examples


def build_counterfactual_group(seed: int, variant_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_index, permutation in enumerate(mapping_variants(seed, variant_count)):
        variant_rows = build_variant_examples(seed, variant_index, permutation)
        for row in variant_rows:
            row["mapping_variant_count"] = variant_count
        rows.extend(variant_rows)
    return rows


def _build_split(
    output_dir: Path,
    name: str,
    seeds: range,
    *,
    variant_count: int,
) -> dict[str, object]:
    data_path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        data_path,
        (
            row
            for seed in seeds
            for row in build_counterfactual_group(seed, variant_count)
        ),
    )
    return {
        "counterfactual_groups": len(seeds),
        "mapping_variants_per_group": variant_count,
        "examples": count,
        "examples_per_group": variant_count * (len(Action) + 1) * len(Action),
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "file": data_path.name,
        "sha256": _sha256(data_path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_groups: int,
    validation_groups: int,
    test_groups: int,
    mapping_variants_per_group: int,
    seed_base: int,
) -> dict[str, object]:
    if min(train_groups, validation_groups, test_groups) < 1:
        raise ValueError("each split requires at least one counterfactual group")
    if not 1 <= mapping_variants_per_group <= 24:
        raise ValueError("mapping_variants_per_group must lie between 1 and 24")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = range(seed_base, seed_base + train_groups)
    validation = range(train.stop, train.stop + validation_groups)
    test = range(validation.stop, validation.stop + test_groups)
    manifest = {
        "schema": "arcgpt2.epistemic_binding.counterfactual.v1",
        "scope": (
            "Set-valued hidden-action binding with counterfactual twins that "
            "hold geometry, palette, and goal fixed while changing only the "
            "action permutation. Controlled gate, not ARC-AGI-3 evaluation."
        ),
        "shortcut_control": (
            "Initial grids are identical within each group and therefore carry "
            "zero information about which mapping variant is active."
        ),
        "mapping_variants_per_group": mapping_variants_per_group,
        "seed_base": seed_base,
        "splits": {
            "train": _build_split(
                output_dir,
                "train",
                train,
                variant_count=mapping_variants_per_group,
            ),
            "validation": _build_split(
                output_dir,
                "validation",
                validation,
                variant_count=mapping_variants_per_group,
            ),
            "test": _build_split(
                output_dir,
                "test",
                test,
                variant_count=mapping_variants_per_group,
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/epistemic_counterfactual/data"),
    )
    parser.add_argument("--train-groups", type=int, default=64)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=16)
    parser.add_argument("--mapping-variants-per-group", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=314_159)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_groups=args.train_groups,
        validation_groups=args.validation_groups,
        test_groups=args.test_groups,
        mapping_variants_per_group=args.mapping_variants_per_group,
        seed_base=args.seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
