"""Build reproducible Stage-0.1 set-valued action datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .codec import token_inventory
from .phase0_hidden_action import Action, game_summary, phase0_special_tokens
from .stage01_hidden_action import (
    SetValuedDecisionExample,
    build_all_variant_examples,
    canonical_probe_counts,
    probe_support_counts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(example: SetValuedDecisionExample) -> dict[str, object]:
    return {
        "game_seed": example.game_seed,
        "variant_index": example.variant_index,
        "probe_order": [action.value for action in example.probe_order],
        "level_index": example.level_index,
        "step_index": example.step_index,
        "context": example.context,
        "target": example.canonical_target,
        "valid_targets": list(example.valid_targets),
        "decision_phase": example.decision_phase,
        "mapping_known_count": example.mapping_known_count,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    all_examples: list[SetValuedDecisionExample] = []
    summaries: list[dict[str, object]] = []

    for seed in seeds:
        examples = build_all_variant_examples(seed)
        all_examples.extend(examples)
        base_summary = game_summary(
            __import__(
                "arcgpt2.phase0_hidden_action",
                fromlist=["generate_game"],
            ).generate_game(seed)
        )
        base_summary["probe_order_variants"] = 24
        base_summary["example_count_across_variants"] = len(examples)
        summaries.append(base_summary)

    data_path = output_dir / f"{name}.jsonl"
    example_count = _write_jsonl(data_path, (_row(example) for example in all_examples))
    summary_path = output_dir / f"{name}_games.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    phase_counts = Counter(example.decision_phase for example in all_examples)
    canonical_counts = Counter(example.canonical_target for example in all_examples)
    valid_support = Counter()
    navigation_targets = Counter()
    mapping_stage_counts = Counter()
    valid_cardinality_counts = Counter()
    for example in all_examples:
        mapping_stage_counts[str(example.mapping_known_count)] += 1
        valid_cardinality_counts[str(len(example.valid_targets))] += 1
        for target in example.valid_targets:
            valid_support[target] += 1
        if example.decision_phase == "navigate":
            navigation_targets[example.canonical_target] += 1

    probe_support = probe_support_counts(all_examples)
    probe_canonical = canonical_probe_counts(all_examples)
    probe_support_values = list(probe_support.values())
    probe_canonical_values = list(probe_canonical.values())

    return {
        "base_games": len(summaries),
        "probe_order_variants_per_game": 24,
        "trajectory_variants": len(summaries) * 24,
        "examples": example_count,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "data_file": data_path.name,
        "data_sha256": _sha256(data_path),
        "game_manifest_file": summary_path.name,
        "game_manifest_sha256": _sha256(summary_path),
        "decision_phase_counts": dict(sorted(phase_counts.items())),
        "mapping_known_count_distribution": dict(sorted(mapping_stage_counts.items())),
        "valid_set_cardinality_distribution": dict(
            sorted(valid_cardinality_counts.items())
        ),
        "canonical_target_counts": dict(sorted(canonical_counts.items())),
        "valid_target_support_counts": dict(sorted(valid_support.items())),
        "navigation_target_counts": dict(sorted(navigation_targets.items())),
        "probe_valid_target_support_counts": probe_support,
        "probe_canonical_target_counts": probe_canonical,
        "probe_valid_support_balanced": len(set(probe_support_values)) == 1,
        "probe_canonical_targets_balanced": len(set(probe_canonical_values)) == 1,
    }


def build_dataset(
    output_dir: Path,
    *,
    train_games: int,
    validation_games: int,
    test_games: int,
    seed_base: int,
) -> dict[str, object]:
    if min(train_games, validation_games, test_games) < 1:
        raise ValueError("all splits require at least one base game")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = range(seed_base, seed_base + train_games)
    validation_start = train_seeds.stop
    validation_seeds = range(validation_start, validation_start + validation_games)
    test_start = validation_seeds.stop
    test_seeds = range(test_start, test_start + test_games)

    special_tokens = sorted(set(token_inventory()) | set(phase0_special_tokens()))
    token_path = output_dir / "special_tokens.json"
    token_path.write_text(json.dumps(special_tokens, indent=2), encoding="utf-8")

    manifest: dict[str, object] = {
        "schema": "arcgpt2.stage01.set-valued.v1",
        "seed_base": seed_base,
        "objective": {
            "probe_loss": (
                "negative log probability mass assigned to all still-untried "
                "actions"
            ),
            "navigation_loss": "single shortest-path action",
            "probe_orders": "all 24 permutations per base game",
            "first_probe_supervision": (
                "all four actions valid; therefore zero action-selection gradient"
            ),
        },
        "special_tokens_file": token_path.name,
        "special_tokens_sha256": _sha256(token_path),
        "splits": {
            "train": _build_split(output_dir, "train", train_seeds),
            "validation": _build_split(
                output_dir,
                "validation",
                validation_seeds,
            ),
            "test": _build_split(output_dir, "test", test_seeds),
        },
    }

    for split_name, split in manifest["splits"].items():  # type: ignore[union-attr]
        if not split["probe_valid_support_balanced"]:  # type: ignore[index]
            raise RuntimeError(f"probe valid support is imbalanced in {split_name}")
        if not split["probe_canonical_targets_balanced"]:  # type: ignore[index]
            raise RuntimeError(f"canonical probe targets are imbalanced in {split_name}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage01/data"),
    )
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=2)
    parser.add_argument("--test-games", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_games=args.train_games,
        validation_games=args.validation_games,
        test_games=args.test_games,
        seed_base=args.seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
