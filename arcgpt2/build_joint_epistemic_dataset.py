"""Build a single-checkpoint epistemic curriculum for GPT-2.

The final system cannot use one model for action semantics and another for goal
inference. This builder converts both controlled tasks into one candidate-set
schema so the same GPT-2 weights learn to preserve uncertainty and contract it
from intervention evidence across multiple latent-variable types.

Action examples use counterfactual twins: identical grids are paired with
contradictory hidden action mappings. Goal examples use exact mechanics and a
bounded atomic Goal DSL. Complete source groups remain inside one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .build_epistemic_counterfactual_dataset import build_counterfactual_group
from . import build_goal_version_dataset as goal_builder
from .build_goal_version_compact import compact_candidate_goals
from .natural_protocol import answer_text


class JointDatasetError(ValueError):
    """Raised when a joint candidate-set row is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_joint_row(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def validate_joint_row(row: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task",
        "source_id",
        "context",
        "null_context",
        "control_contexts",
        "candidate_texts",
        "target_distribution",
        "truth_index",
        "consistent_indices",
        "information_level",
        "metadata",
    }
    missing = required.difference(row)
    if missing:
        raise JointDatasetError(f"joint row is missing fields: {sorted(missing)}")
    candidates = list(row["candidate_texts"])
    targets = [float(value) for value in row["target_distribution"]]
    consistent = [int(value) for value in row["consistent_indices"]]
    if not candidates or any(not str(candidate) for candidate in candidates):
        raise JointDatasetError("candidate_texts must be non-empty strings")
    if len(candidates) != len(targets):
        raise JointDatasetError("candidate and target lengths differ")
    if any(value < 0.0 for value in targets):
        raise JointDatasetError("target probabilities may not be negative")
    if abs(sum(targets) - 1.0) > 1e-9:
        raise JointDatasetError("target distribution must sum to one")
    expected_consistent = {index for index, value in enumerate(targets) if value > 0.0}
    if set(consistent) != expected_consistent:
        raise JointDatasetError("consistent_indices do not match non-zero targets")
    truth_index = int(row["truth_index"])
    if not 0 <= truth_index < len(candidates):
        raise JointDatasetError("truth_index is out of range")
    if truth_index not in expected_consistent:
        raise JointDatasetError("truth must remain in the exact version space")
    controls = row["control_contexts"]
    if not isinstance(controls, dict) or not controls:
        raise JointDatasetError("at least one control context is required")


def action_rows(seed: int, mapping_variants: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in build_counterfactual_group(seed, mapping_variants):
        words = [str(value) for value in row["candidate_words"]]
        targets = [float(row["target_distribution"][word]) for word in words]
        consistent = [index for index, value in enumerate(targets) if value > 0.0]
        variant = int(row["mapping_variant_index"])
        output.append(
            {
                "schema": "arcgpt2.joint_epistemic.v1",
                "task": "action_binding",
                "source_id": (
                    f"action:{seed}:variant:{variant}:probe:{row['probe_count']}:"
                    f"query:{row['query_action']}"
                ),
                "context": str(row["context"]),
                "null_context": str(row["amnesic_context"]),
                "control_contexts": {
                    "amnesic": str(row["amnesic_context"]),
                    "shuffled_evidence": str(row["shuffled_context"]),
                },
                "candidate_texts": [answer_text(word) for word in words],
                "candidate_labels": words,
                "target_distribution": targets,
                "truth_index": words.index(str(row["truth_word"])),
                "consistent_indices": consistent,
                "information_level": int(row["probe_count"]),
                "metadata": {
                    "counterfactual_group_seed": seed,
                    "mapping_variant_index": variant,
                    "mapping_variant_count": mapping_variants,
                    "surface_grid_sha256": row["surface_grid_sha256"],
                    "query_action": row["query_action"],
                    "query_was_observed": row["query_was_observed"],
                    "consistent_mapping_count": row["consistent_mapping_count"],
                },
            }
        )
    return output


def goal_rows(seed: int) -> list[dict[str, Any]]:
    original = goal_builder.candidate_goals
    goal_builder.candidate_goals = compact_candidate_goals
    try:
        source_rows = goal_builder.build_examples(seed)
    finally:
        goal_builder.candidate_goals = original

    output: list[dict[str, Any]] = []
    for row in source_rows:
        targets = [float(value) for value in row["target_distribution"]]
        output.append(
            {
                "schema": "arcgpt2.joint_epistemic.v1",
                "task": "goal_inference",
                "source_id": f"goal:{seed}:prefix:{row['prefix_length']}",
                "context": str(row["context"]),
                "null_context": str(row["amnesic_context"]),
                "control_contexts": {
                    "amnesic": str(row["amnesic_context"]),
                    "statusless": str(row["statusless_context"]),
                    "shuffled_status": str(row["shuffled_status_context"]),
                },
                "candidate_texts": [str(value) for value in row["candidate_texts"]],
                "candidate_labels": [str(value) for value in row["candidate_programs"]],
                "target_distribution": targets,
                "truth_index": int(row["truth_index"]),
                "consistent_indices": [int(value) for value in row["consistent_indices"]],
                "information_level": int(row["observed_terminal_count"]),
                "metadata": {
                    "game_seed": seed,
                    "prefix_length": row["prefix_length"],
                    "total_records": row["total_records"],
                    "observed_terminal_count": row["observed_terminal_count"],
                    "consistent_goal_count": row["consistent_goal_count"],
                    "truth_sha256": row["truth_sha256"],
                    "mechanics_sha256": row["mechanics_sha256"],
                },
            }
        )
    return output


def _split_rows(
    action_seeds: range,
    goal_seeds: range,
    *,
    mapping_variants: int,
) -> Iterable[dict[str, Any]]:
    for seed in action_seeds:
        yield from action_rows(seed, mapping_variants)
    for seed in goal_seeds:
        yield from goal_rows(seed)


def _build_split(
    output_dir: Path,
    name: str,
    action_seeds: range,
    goal_seeds: range,
    *,
    mapping_variants: int,
) -> dict[str, Any]:
    path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        path,
        _split_rows(action_seeds, goal_seeds, mapping_variants=mapping_variants),
    )
    return {
        "examples": count,
        "action_groups": len(action_seeds),
        "goal_games": len(goal_seeds),
        "mapping_variants_per_action_group": mapping_variants,
        "action_seed_start": action_seeds.start,
        "action_seed_stop_exclusive": action_seeds.stop,
        "goal_seed_start": goal_seeds.start,
        "goal_seed_stop_exclusive": goal_seeds.stop,
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
    mapping_variants: int,
    action_seed_base: int,
    goal_seed_base: int,
) -> dict[str, Any]:
    counts = (
        train_action_groups,
        validation_action_groups,
        test_action_groups,
        train_goal_games,
        validation_goal_games,
        test_goal_games,
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

    manifest = {
        "schema": "arcgpt2.joint_epistemic.v1",
        "scope": (
            "One-checkpoint set-valued action-semantics and latent-goal "
            "curriculum; controlled gate, not ARC-AGI-3 evaluation."
        ),
        "candidate_scoring": (
            "Ordinary GPT-2 completion likelihood over variable candidate sets; "
            "no auxiliary learned heads."
        ),
        "splits": {
            "train": _build_split(
                output_dir,
                "train",
                action_train,
                goal_train,
                mapping_variants=mapping_variants,
            ),
            "validation": _build_split(
                output_dir,
                "validation",
                action_validation,
                goal_validation,
                mapping_variants=mapping_variants,
            ),
            "test": _build_split(
                output_dir,
                "test",
                action_test,
                goal_test,
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/joint/data"))
    parser.add_argument("--train-action-groups", type=int, default=32)
    parser.add_argument("--validation-action-groups", type=int, default=4)
    parser.add_argument("--test-action-groups", type=int, default=8)
    parser.add_argument("--train-goal-games", type=int, default=32)
    parser.add_argument("--validation-goal-games", type=int, default=4)
    parser.add_argument("--test-goal-games", type=int, default=8)
    parser.add_argument("--mapping-variants", type=int, default=4)
    parser.add_argument("--action-seed-base", type=int, default=618_034)
    parser.add_argument("--goal-seed-base", type=int, default=141_421)
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
        mapping_variants=args.mapping_variants,
        action_seed_base=args.action_seed_base,
        goal_seed_base=args.goal_seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
