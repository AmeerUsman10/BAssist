"""Build reproducible Phase-0 algorithm-distillation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .codec import token_inventory
from .phase0_hidden_action import build_decision_examples, game_summary, generate_game, phase0_special_tokens


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    summaries: list[dict[str, object]] = []

    def rows():
        for seed in seeds:
            spec = generate_game(seed)
            summaries.append(game_summary(spec))
            for example in build_decision_examples(spec):
                yield {
                    "game_seed": example.game_seed,
                    "level_index": example.level_index,
                    "step_index": example.step_index,
                    "context": example.context,
                    "target": example.target,
                }

    data_path = output_dir / f"{name}.jsonl"
    example_count = _write_jsonl(data_path, rows())
    summary_path = output_dir / f"{name}_games.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "games": len(summaries),
        "examples": example_count,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "data_file": data_path.name,
        "data_sha256": _sha256(data_path),
        "game_manifest_file": summary_path.name,
        "game_manifest_sha256": _sha256(summary_path),
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
        raise ValueError("all splits require at least one game")
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
        "schema": "arcgpt2.phase0.v1",
        "seed_base": seed_base,
        "special_tokens_file": token_path.name,
        "special_tokens_sha256": _sha256(token_path),
        "splits": {
            "train": _build_split(output_dir, "train", train_seeds),
            "validation": _build_split(output_dir, "validation", validation_seeds),
            "test": _build_split(output_dir, "test", test_seeds),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0/data"))
    parser.add_argument("--train-games", type=int, default=512)
    parser.add_argument("--validation-games", type=int, default=64)
    parser.add_argument("--test-games", type=int, default=128)
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
