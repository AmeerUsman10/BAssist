"""Build a symmetry-aware hidden-action binding curriculum for GPT-2.

The earlier Phase-0 datasets assigned one arbitrary target to an information
state even when several action meanings remained observationally possible. This
builder preserves the full version space. Each query carries a probability
distribution that is uniform over every direction still consistent with the
literal action/outcome history.

That distinction matters for ARC-AGI-3: an agent must keep uncertainty alive
until evidence removes it rather than learning a confident tie-breaking habit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from .natural_protocol import (
    direction_words,
    mapping_prompt,
    rotate_action_labels,
)
from .phase0_hidden_action import Action, Direction, HiddenActionGame, generate_game


_DIRECTION_WORD = {
    Direction.UP: "north",
    Direction.DOWN: "south",
    Direction.LEFT: "west",
    Direction.RIGHT: "east",
}
_WORD_DIRECTION = {word: direction for direction, word in _DIRECTION_WORD.items()}


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


def probe_order(seed: int) -> tuple[Action, ...]:
    """Return a deterministic but seed-varying order of the four probe actions."""

    actions = list(Action)
    random.Random(seed ^ 0xE915_7EED).shuffle(actions)
    return tuple(actions)


def _observed_mapping(spec, records) -> dict[Action, Direction]:
    return {record.action: spec.action_to_direction[record.action] for record in records}


def allowed_directions(spec, records, query_action: Action) -> tuple[Direction, ...]:
    """Return every direction compatible with the exact partial history."""

    observed = _observed_mapping(spec, records)
    if query_action in observed:
        return (observed[query_action],)
    used = set(observed.values())
    return tuple(direction for direction in Direction if direction not in used)


def build_examples(seed: int) -> list[dict[str, object]]:
    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    initial = game.frame
    order = probe_order(seed)
    records = []
    examples: list[dict[str, object]] = []
    words = direction_words()

    for prefix_length in range(len(order) + 1):
        if prefix_length > 0:
            record = game.step(order[prefix_length - 1])
            if record.status != "ACTIVE":
                raise RuntimeError("the safe Phase-0 probe arena terminated unexpectedly")
            records.append(record)

        observed = _observed_mapping(spec, records)
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
                    "schema": "arcgpt2.epistemic_binding.v1",
                    "game_seed": seed,
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
                    "consistent_mapping_count": 1
                    if prefix_length == len(Action)
                    else _factorial(len(Action) - prefix_length),
                }
            )
    return examples


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def build_control_context(seed: int, prefix_length: int, query_action: Action, mode: str) -> str:
    """Reconstruct a control prompt independently of a stored JSONL record."""

    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    initial = game.frame
    records = tuple(game.step(action) for action in probe_order(seed)[:prefix_length])
    if mode == "full":
        return mapping_prompt(initial, records, query_action)
    if mode == "amnesic":
        return mapping_prompt(initial, (), query_action, include_evidence=False)
    if mode == "shuffled":
        return mapping_prompt(
            initial,
            records,
            query_action,
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
        "examples_per_game": (len(Action) + 1) * len(Action),
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
        "schema": "arcgpt2.epistemic_binding.v1",
        "scope": (
            "Set-valued natural-language action binding with exact permutation "
            "version spaces; controlled meta-learning gate, not ARC-AGI-3 evaluation."
        ),
        "representation": (
            "Ordinary GPT-2 vocabulary, exact grids, exact changed cells, and no "
            "learned encoder or semantic action label supplied by the wrapper."
        ),
        "target": (
            "Uniform probability over every cardinal direction consistent with "
            "the observed partial action mapping."
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/epistemic/data"))
    parser.add_argument("--train-games", type=int, default=256)
    parser.add_argument("--validation-games", type=int, default=32)
    parser.add_argument("--test-games", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=271_828)
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
