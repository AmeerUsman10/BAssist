"""Build exact counterexample-guided GPT-2 program-repair data.

For every hidden-action game, generate near-correct mappings by swapping two
action effects. Exact replay finds the first transition that falsifies each
candidate. GPT-2 receives the candidate, the concrete predicted/observed
outcomes, and the evidence history; its target is the corrected compact mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

from .build_program_dataset import PROGRAM_SPECIAL_TOKENS
from .codec import encode_transition, token_inventory, tokens_to_text
from .dsl import (
    Program,
    execute,
    program_from_phase0_spec,
    replay,
)
from .mapping_target import (
    MAPPING_SPECIAL_TOKENS,
    compact_mapping,
    expand_mapping,
    program_mapping,
)
from .phase0_hidden_action import (
    ACTION_TOKEN,
    Action,
    append_record_tokens,
    generate_game,
    initial_transcript,
    phase0_special_tokens,
    simulate_source_history,
)


REPAIR_SPECIAL_TOKENS = [
    "<CANDIDATE_MAPPING>",
    "</CANDIDATE_MAPPING>",
    "<COUNTEREXAMPLE>",
    "</COUNTEREXAMPLE>",
    "<PREDICTED>",
    "</PREDICTED>",
    "<OBSERVED>",
    "</OBSERVED>",
    "<REPAIR_MAPPING>",
    "<COUNTEREXAMPLE_FULL>",
    "<COUNTEREXAMPLE_MISSING>",
    "<COUNTEREXAMPLE_IRRELEVANT>",
]


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


def one_swap_programs(truth: Program) -> tuple[Program, ...]:
    mapping = program_mapping(truth)
    candidates: list[Program] = []
    actions = tuple(Action)
    for left, right in itertools.combinations(actions, 2):
        swapped = dict(mapping)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        candidates.append(expand_mapping(swapped, truth))
    return tuple(candidates)


def _history_tokens(spec, records, *, through_index: int) -> list[str]:
    transcript = initial_transcript(spec)
    for record in records[: through_index + 1]:
        append_record_tokens(transcript, record, next_frame=None)
    return transcript


def _counterexample_tokens(wrong: Program, record) -> list[str]:
    predicted = execute(wrong, record.before, record.action)
    predicted_transition = encode_transition(record.before, predicted.after)
    return [
        "<COUNTEREXAMPLE>",
        "<ACTION>",
        ACTION_TOKEN[record.action],
        "</ACTION>",
        "<PREDICTED>",
        "<TRANSITION_DELTA>" if predicted_transition.kind == "delta" else "<TRANSITION_FULL>",
        *predicted_transition.tokens,
        "</PREDICTED>",
        "<OBSERVED>",
        "<TRANSITION_DELTA>" if record.transition.kind == "delta" else "<TRANSITION_FULL>",
        *record.transition.tokens,
        "</OBSERVED>",
        "</COUNTEREXAMPLE>",
    ]


def build_examples(seed: int) -> list[dict[str, object]]:
    spec = generate_game(seed)
    truth = program_from_phase0_spec(spec)
    records = simulate_source_history(spec)
    output: list[dict[str, object]] = []

    for wrong_index, wrong in enumerate(one_swap_programs(truth)):
        result = replay(wrong, records)
        if result.consistent or result.mismatch is None:
            raise RuntimeError("one-swap candidate unexpectedly matched all evidence")
        mismatch_index = result.mismatch.index
        record = records[mismatch_index]
        context_tokens = [
            *_history_tokens(spec, records, through_index=mismatch_index),
            "<CANDIDATE_MAPPING>",
            *compact_mapping(wrong).split(),
            "</CANDIDATE_MAPPING>",
            "<COUNTEREXAMPLE_FULL>",
            *_counterexample_tokens(wrong, record),
            "<REPAIR_MAPPING>",
        ]
        output.append(
            {
                "game_seed": seed,
                "wrong_index": wrong_index,
                "mismatch_index": mismatch_index,
                "mismatch_action": record.action.value,
                "differing_cells": result.mismatch.differing_cells,
                "context": tokens_to_text(context_tokens),
                "target": compact_mapping(truth),
                "wrong_mapping": compact_mapping(wrong),
                "truth_mapping": compact_mapping(truth),
                "wrong_program_sha256": wrong.sha256,
                "truth_program_sha256": truth.sha256,
            }
        )
    return output


def build_control_context(seed: int, wrong_index: int, mode: str) -> str:
    examples = build_examples(seed)
    if wrong_index < 0 or wrong_index >= len(examples):
        raise ValueError("wrong_index is outside the one-swap candidate set")
    selected = examples[wrong_index]
    full = str(selected["context"])
    if mode == "full":
        return full

    start = full.index("<COUNTEREXAMPLE_FULL>")
    repair = full.rindex("<REPAIR_MAPPING>")
    prefix = full[:start]
    if mode == "missing":
        return prefix + "<COUNTEREXAMPLE_MISSING> <REPAIR_MAPPING>"
    if mode == "irrelevant":
        other = examples[(wrong_index + 1) % len(examples)]
        other_full = str(other["context"])
        other_start = other_full.index("<COUNTEREXAMPLE_FULL>")
        other_repair = other_full.rindex("<REPAIR_MAPPING>")
        other_counterexample = other_full[other_start:other_repair]
        return prefix + "<COUNTEREXAMPLE_IRRELEVANT> " + other_counterexample + "<REPAIR_MAPPING>"
    raise ValueError("mode must be full, missing, or irrelevant")


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    data_path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(
        data_path,
        (
            example
            for seed in seeds
            for example in build_examples(seed)
        ),
    )
    return {
        "games": len(seeds),
        "examples": count,
        "examples_per_game": len(tuple(itertools.combinations(tuple(Action), 2))),
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

    special_tokens = sorted(
        set(token_inventory())
        | set(phase0_special_tokens())
        | set(PROGRAM_SPECIAL_TOKENS)
        | set(MAPPING_SPECIAL_TOKENS)
        | set(REPAIR_SPECIAL_TOKENS)
    )
    token_path = output_dir / "special_tokens.json"
    token_path.write_text(json.dumps(special_tokens, indent=2), encoding="utf-8")

    manifest = {
        "schema": "arcgpt2.program_repair.phase0.v1",
        "scope": (
            "One-swap hidden-action program repair from exact replay "
            "counterexamples; controlled synthetic gate, not ARC-AGI-3."
        ),
        "seed_base": seed_base,
        "special_tokens_file": token_path.name,
        "special_tokens_sha256": _sha256(token_path),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/program_repair/data"))
    parser.add_argument("--train-games", type=int, default=256)
    parser.add_argument("--validation-games", type=int, default=32)
    parser.add_argument("--test-games", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=71_003)
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
