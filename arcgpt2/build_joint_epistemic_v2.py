"""Build one-checkpoint action, goal, and contact-mechanics curriculum.

This is the first joint dataset that asks the same GPT-2 weights to maintain
uncertainty over three different latent-variable classes:

- hidden action semantics;
- hidden terminal predicates;
- hidden object-contact mechanics.

Every task uses ordinary completion candidates and set-valued targets derived
from exact replay. Counterfactual groups never cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .build_contact_version_dataset import build_group_examples
from .build_joint_epistemic_dataset import (
    JointDatasetError,
    action_rows,
    goal_rows,
    validate_joint_row,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact_rows(seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in build_group_examples(seed):
        output.append(
            {
                "schema": "arcgpt2.joint_epistemic.v2",
                "task": "contact_mechanics",
                "source_id": (
                    f"contact:{seed}:mode:{row['mode_variant']}:"
                    f"prefix:{row['prefix_length']}"
                ),
                "context": str(row["context"]),
                "null_context": str(row["amnesic_context"]),
                "control_contexts": {
                    "amnesic": str(row["amnesic_context"]),
                    "precontact": str(row["precontact_context"]),
                    "shuffled_contact": str(row["shuffled_contact_context"]),
                },
                "candidate_texts": [str(value) for value in row["candidate_texts"]],
                "candidate_labels": [str(value) for value in row["candidate_labels"]],
                "target_distribution": [
                    float(value) for value in row["target_distribution"]
                ],
                "truth_index": int(row["truth_index"]),
                "consistent_indices": [
                    int(value) for value in row["consistent_indices"]
                ],
                "information_level": int(row["prefix_length"]),
                "metadata": {
                    "counterfactual_group_seed": seed,
                    "mode_variant": row["mode_variant"],
                    "direct_contact_observed": row["direct_contact_observed"],
                    "consistent_mode_count": row["consistent_mode_count"],
                },
            }
        )
    return output


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_joint_row(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _rows(
    action_seeds: range,
    goal_seeds: range,
    contact_seeds: range,
    *,
    mapping_variants: int,
):
    for seed in action_seeds:
        yield from action_rows(seed, mapping_variants)
    for seed in goal_seeds:
        yield from goal_rows(seed)
    for seed in contact_seeds:
        yield from contact_rows(seed)


def _build_split(
    output_dir: Path,
    name: str,
    action_seeds: range,
    goal_seeds: range,
    contact_seeds: range,
    *,
    mapping_variants: int,
) -> dict[str, Any]:
    path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        path,
        _rows(
            action_seeds,
            goal_seeds,
            contact_seeds,
            mapping_variants=mapping_variants,
        ),
    )
    return {
        "examples": count,
        "action_groups": len(action_seeds),
        "goal_games": len(goal_seeds),
        "contact_groups": len(contact_seeds),
        "mapping_variants_per_action_group": mapping_variants,
        "action_seed_start": action_seeds.start,
        "action_seed_stop_exclusive": action_seeds.stop,
        "goal_seed_start": goal_seeds.start,
        "goal_seed_stop_exclusive": goal_seeds.stop,
        "contact_seed_start": contact_seeds.start,
        "contact_seed_stop_exclusive": contact_seeds.stop,
        "file": path.name,
        "sha256": _sha256(path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_action_groups: int,
    validation_action_groups: int,
    test_action_groups: int,
    train_goal_games: int,
    validation_goal_games: int,
    test_goal_games: int,
    train_contact_groups: int,
    validation_contact_groups: int,
    test_contact_groups: int,
    mapping_variants: int,
    action_seed_base: int,
    goal_seed_base: int,
    contact_seed_base: int,
) -> dict[str, Any]:
    counts = (
        train_action_groups,
        validation_action_groups,
        test_action_groups,
        train_goal_games,
        validation_goal_games,
        test_goal_games,
        train_contact_groups,
        validation_contact_groups,
        test_contact_groups,
    )
    if min(counts) < 1:
        raise JointDatasetError("every task and split requires at least one source group")
    if not 1 <= mapping_variants <= 24:
        raise JointDatasetError("mapping_variants must lie between 1 and 24")
    output_dir.mkdir(parents=True, exist_ok=True)

    action_train = range(action_seed_base, action_seed_base + train_action_groups)
    action_validation = range(
        action_train.stop, action_train.stop + validation_action_groups
    )
    action_test = range(
        action_validation.stop, action_validation.stop + test_action_groups
    )
    goal_train = range(goal_seed_base, goal_seed_base + train_goal_games)
    goal_validation = range(goal_train.stop, goal_train.stop + validation_goal_games)
    goal_test = range(goal_validation.stop, goal_validation.stop + test_goal_games)
    contact_train = range(contact_seed_base, contact_seed_base + train_contact_groups)
    contact_validation = range(
        contact_train.stop, contact_train.stop + validation_contact_groups
    )
    contact_test = range(
        contact_validation.stop, contact_validation.stop + test_contact_groups
    )

    manifest = {
        "schema": "arcgpt2.joint_epistemic.v2",
        "scope": (
            "One-GPT-2 set-valued action, goal, and contact-mechanics "
            "curriculum; controlled gates, not ARC-AGI-3 evaluation."
        ),
        "tasks": ["action_binding", "goal_inference", "contact_mechanics"],
        "splits": {
            "train": _build_split(
                output_dir,
                "train",
                action_train,
                goal_train,
                contact_train,
                mapping_variants=mapping_variants,
            ),
            "validation": _build_split(
                output_dir,
                "validation",
                action_validation,
                goal_validation,
                contact_validation,
                mapping_variants=mapping_variants,
            ),
            "test": _build_split(
                output_dir,
                "test",
                action_test,
                goal_test,
                contact_test,
                mapping_variants=mapping_variants,
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/joint_v2/data"))
    parser.add_argument("--train-action-groups", type=int, default=24)
    parser.add_argument("--validation-action-groups", type=int, default=3)
    parser.add_argument("--test-action-groups", type=int, default=6)
    parser.add_argument("--train-goal-games", type=int, default=24)
    parser.add_argument("--validation-goal-games", type=int, default=3)
    parser.add_argument("--test-goal-games", type=int, default=6)
    parser.add_argument("--train-contact-groups", type=int, default=24)
    parser.add_argument("--validation-contact-groups", type=int, default=3)
    parser.add_argument("--test-contact-groups", type=int, default=6)
    parser.add_argument("--mapping-variants", type=int, default=4)
    parser.add_argument("--action-seed-base", type=int, default=818_034)
    parser.add_argument("--goal-seed-base", type=int, default=341_421)
    parser.add_argument("--contact-seed-base", type=int, default=271_828_2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_action_groups=args.train_action_groups,
        validation_action_groups=args.validation_action_groups,
        test_action_groups=args.test_action_groups,
        train_goal_games=args.train_goal_games,
        validation_goal_games=args.validation_goal_games,
        test_goal_games=args.test_goal_games,
        train_contact_groups=args.train_contact_groups,
        validation_contact_groups=args.validation_contact_groups,
        test_contact_groups=args.test_contact_groups,
        mapping_variants=args.mapping_variants,
        action_seed_base=args.action_seed_base,
        goal_seed_base=args.goal_seed_base,
        contact_seed_base=args.contact_seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
