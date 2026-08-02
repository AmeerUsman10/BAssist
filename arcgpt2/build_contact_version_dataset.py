"""Build counterfactual hidden-contact version-space data for GPT-2.

For every base geometry and action mapping, five worlds are created that differ
only in the contact primitive applied to one special color. Their observations
are identical until direct contact. This makes grid appearance, palette,
action names, and geometry useless shortcuts for predicting the rule.

Targets are set-valued: probability is uniform over all contact modes retained
by exact replay of the partial history.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contact_protocol import contact_completion, contact_prompt
from .mechanics_v2 import ContactMode, filter_consistent_v2
from .phase0_hidden_action import Direction
from .primitive_contact_game import (
    ContactGameSpec,
    ContactStepRecord,
    PrimitiveContactGame,
    enumerate_contact_programs,
    generate_contact_game,
)


_MODE_ORDER = tuple(ContactMode)


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
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def controlled_contact_history(spec: ContactGameSpec) -> tuple[ContactStepRecord, ...]:
    """Probe the action mapping, approach the special cell, then contact it."""

    game = PrimitiveContactGame(spec)
    records = [game.step(action) for action in spec.probe_order]
    right_action = next(
        action
        for action, direction in spec.action_to_direction.items()
        if direction is Direction.RIGHT
    )
    records.append(game.step(right_action))
    records.append(game.step(right_action))
    if records[-1].contacted_color != spec.palette.interaction:
        raise RuntimeError("controlled history failed to contact the special color")
    if any(record.status != "ACTIVE" for record in records):
        raise RuntimeError("controlled contact history terminated unexpectedly")
    return tuple(records)


def contact_variants(seed: int) -> dict[ContactMode, tuple[ContactGameSpec, tuple[ContactStepRecord, ...]]]:
    base = generate_contact_game(seed)
    variants: dict[
        ContactMode, tuple[ContactGameSpec, tuple[ContactStepRecord, ...]]
    ] = {}
    for mode in _MODE_ORDER:
        spec = replace(base, contact_mode=mode)
        variants[mode] = (spec, controlled_contact_history(spec))
    return variants


def allowed_modes(
    spec: ContactGameSpec,
    records: Sequence[ContactStepRecord],
) -> tuple[ContactMode, ...]:
    observations = tuple(record.as_observation() for record in records)
    surviving = filter_consistent_v2(enumerate_contact_programs(spec), observations)
    modes = tuple(
        mode
        for mode in _MODE_ORDER
        if any(program.contact_mode is mode for program in surviving)
    )
    if not modes:
        raise RuntimeError("contact mechanics version space became empty")
    return modes


def build_group_examples(seed: int) -> list[dict[str, Any]]:
    variants = contact_variants(seed)
    examples: list[dict[str, Any]] = []
    prefixes = (0, 4, 5, 6)

    for mode_index, mode in enumerate(_MODE_ORDER):
        spec, records = variants[mode]
        shuffled_mode = _MODE_ORDER[(mode_index + 1) % len(_MODE_ORDER)]
        _, shuffled_records = variants[shuffled_mode]
        initial = records[0].before
        candidate_texts = [
            contact_completion(candidate, spec.palette.interaction)
            for candidate in _MODE_ORDER
        ]

        for prefix_length in prefixes:
            prefix = records[:prefix_length]
            support = allowed_modes(spec, prefix)
            probability = 1.0 / len(support)
            target = [
                probability if candidate in support else 0.0
                for candidate in _MODE_ORDER
            ]
            corrupted_prefix = list(prefix)
            if prefix_length == len(records):
                corrupted_prefix[-1] = shuffled_records[-1]
            examples.append(
                {
                    "schema": "arcgpt2.contact_version_space.v1",
                    "counterfactual_group_seed": seed,
                    "mode_variant": mode.value,
                    "prefix_length": prefix_length,
                    "direct_contact_observed": prefix_length == len(records),
                    "context": contact_prompt(
                        initial,
                        prefix,
                        spec.palette.interaction,
                    ),
                    "amnesic_context": contact_prompt(
                        initial,
                        (),
                        spec.palette.interaction,
                    ),
                    "precontact_context": contact_prompt(
                        initial,
                        records[:5],
                        spec.palette.interaction,
                    ),
                    "shuffled_contact_context": contact_prompt(
                        initial,
                        tuple(corrupted_prefix),
                        spec.palette.interaction,
                    ),
                    "candidate_texts": candidate_texts,
                    "candidate_labels": [candidate.value for candidate in _MODE_ORDER],
                    "target_distribution": target,
                    "consistent_indices": [
                        index
                        for index, value in enumerate(target)
                        if value > 0.0
                    ],
                    "truth_index": _MODE_ORDER.index(mode),
                    "truth_mode": mode.value,
                    "consistent_mode_count": len(support),
                    "action_mapping": {
                        action.value: direction.value
                        for action, direction in spec.action_to_direction.items()
                    },
                    "palette": {
                        "background": spec.palette.background,
                        "wall": spec.palette.wall,
                        "agent": spec.palette.agent,
                        "goal": spec.palette.goal,
                        "interaction": spec.palette.interaction,
                    },
                    "history": [
                        {
                            "action": record.action.value,
                            "status": record.status,
                            "before": [list(row) for row in record.before],
                            "after": [list(row) for row in record.after],
                        }
                        for record in prefix
                    ],
                }
            )
    return examples


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, Any]:
    path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        path,
        (row for seed in seeds for row in build_group_examples(seed)),
    )
    return {
        "counterfactual_groups": len(seeds),
        "mode_variants_per_group": len(_MODE_ORDER),
        "examples": count,
        "examples_per_group": len(_MODE_ORDER) * 4,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "file": path.name,
        "sha256": _sha256(path),
    }


def build_dataset(
    output_dir: Path,
    *,
    train_groups: int,
    validation_groups: int,
    test_groups: int,
    seed_base: int,
) -> dict[str, Any]:
    if min(train_groups, validation_groups, test_groups) < 1:
        raise ValueError("every split requires at least one counterfactual group")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = range(seed_base, seed_base + train_groups)
    validation = range(train.stop, train.stop + validation_groups)
    test = range(validation.stop, validation.stop + test_groups)
    manifest = {
        "schema": "arcgpt2.contact_version_space.v1",
        "scope": (
            "Set-valued hidden contact-mechanics inference with identical-world "
            "counterfactual variants; controlled Gate C, not ARC-AGI-3 evaluation."
        ),
        "target": (
            "Uniform probability over every contact mode whose complete mechanics "
            "program exactly replays the partial history."
        ),
        "splits": {
            "train": _build_split(output_dir, "train", train),
            "validation": _build_split(output_dir, "validation", validation),
            "test": _build_split(output_dir, "test", test),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contact/data"))
    parser.add_argument("--train-groups", type=int, default=64)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=271_828_1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_groups=args.train_groups,
        validation_groups=args.validation_groups,
        test_groups=args.test_groups,
        seed_base=args.seed_base,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
