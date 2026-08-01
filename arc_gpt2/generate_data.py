from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

try:
    from .world import generate_episode
except ImportError:  # direct script execution
    from world import generate_episode


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def _records_for_range(start_seed: int, episodes: int, max_steps: int):
    for seed in range(start_seed, start_seed + episodes):
        yield from generate_episode(seed, max_steps=max_steps)


def build_dataset(
    output_dir: Path,
    *,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    seed: int,
    max_steps: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "train": (seed, train_episodes),
        "validation": (seed + 100_000, val_episodes),
        "test": (seed + 200_000, test_episodes),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": "pure-gpt2-hidden-rule-grid-v0",
        "base_seed": seed,
        "max_steps": max_steps,
        "splits": {},
    }

    for split, (start_seed, episodes) in split_specs.items():
        path = output_dir / f"{split}.jsonl"
        count, sha256 = _write_jsonl(
            path,
            _records_for_range(start_seed, episodes, max_steps),
        )
        manifest["splits"][split] = {  # type: ignore[index]
            "path": str(path),
            "start_seed": start_seed,
            "episodes": episodes,
            "turn_records": count,
            "sha256": sha256,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-0 pure GPT-2 ARC traces")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--val-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.output_dir,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
