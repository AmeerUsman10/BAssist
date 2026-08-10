"""Build the natural-language factorized action-semantics dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .natural_protocol import answer_text, mapping_prompt, rotate_action_labels
from .phase0_hidden_action import Action, HiddenActionGame, generate_game


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def probe_records(seed: int):
    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    initial = game.frame
    records = tuple(game.step(action) for action in Action)
    if game.level_index != 0 or game.step_index != 4:
        raise RuntimeError("Phase-0 probe arena terminated unexpectedly")
    return spec, initial, records


def build_examples(seed: int) -> list[dict[str, object]]:
    spec, initial, records = probe_records(seed)
    rows: list[dict[str, object]] = []
    for action in Action:
        direction = spec.action_to_direction[action]
        rows.append(
            {
                "game_seed": seed,
                "query_action": action.value,
                "context": mapping_prompt(initial, records, action),
                "target": answer_text(direction.value.lower()),
                "direction": direction.value.lower(),
                "mapping": {
                    candidate.value: spec.action_to_direction[candidate].value.lower()
                    for candidate in Action
                },
                "palette": {
                    "background": spec.palette.background,
                    "wall": spec.palette.wall,
                    "agent": spec.palette.agent,
                    "goal": spec.palette.goal,
                },
            }
        )
    return rows


def build_control_prompt(seed: int, action: Action, mode: str) -> str:
    _, initial, records = probe_records(seed)
    if mode == "full":
        return mapping_prompt(initial, records, action)
    if mode == "amnesic":
        return mapping_prompt(initial, records, action, include_evidence=False)
    if mode == "shuffled":
        return mapping_prompt(
            initial,
            records,
            action,
            displayed_actions=rotate_action_labels(records),
        )
    raise ValueError("mode must be full, amnesic, or shuffled")


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    data_path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        data_path,
        (example for seed in seeds for example in build_examples(seed)),
    )
    return {
        "games": len(seeds),
        "examples": count,
        "examples_per_game": len(Action),
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "file": data_path.name,
        "sha256": _sha256(data_path),
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
        raise ValueError("each split requires at least one game")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = range(seed_base, seed_base + train_games)
    validation = range(train.stop, train.stop + validation_games)
    test = range(validation.stop, validation.stop + test_games)

    manifest = {
        "schema": "arcgpt2.factorized_action_semantics.v1",
        "scope": (
            "Natural-language four-way action semantics with an exact permutation "
            "constraint; controlled Phase-0 gate, not ARC-AGI-3 evaluation."
        ),
        "representation": (
            "Exact natural-language grids and cell deltas using only the original "
            "GPT-2 vocabulary; no learned visual encoder or semantic parser."
        ),
        "seed_base": seed_base,
        "splits": {
            "train": _build_split(output_dir, "train", train),
            "validation": _build_split(output_dir, "validation", validation),
            "test": _build_split(output_dir, "test", test),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/factorized/data"))
    parser.add_argument("--train-games", type=int, default=512)
    parser.add_argument("--validation-games", type=int, default=64)
    parser.add_argument("--test-games", type=int, default=128)
    parser.add_argument("--seed-base", type=int, default=112_358)
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
