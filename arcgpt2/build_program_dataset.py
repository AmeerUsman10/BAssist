"""Build the Phase-0 GPT-2 executable-program induction dataset.

Each example contains four exact action/outcome probes from a hidden-action game.
The supervised target is a compact atomic action-to-direction mapping. That
mapping deterministically expands into the complete ARC-DSL transition program
stored alongside the example.

The first smoke run used the full DSL program as the language-model target. Most
of those tokens were shared by all 24 candidates, diluting the actual hidden
mapping signal. Version 2 isolates the latent variable while preserving exact
downstream program execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .codec import token_inventory, tokens_to_text
from .dsl import canonical_program, program_from_phase0_spec
from .mapping_target import MAPPING_SPECIAL_TOKENS, compact_mapping
from .phase0_hidden_action import (
    Action,
    HiddenActionGame,
    append_record_tokens,
    generate_game,
    initial_transcript,
    phase0_special_tokens,
)


PROGRAM_SPECIAL_TOKENS = [
    "<INDUCE_MAPPING>",
    "<EVIDENCE_FULL>",
    "<EVIDENCE_AMNESIC>",
    "<EVIDENCE_SHUFFLED>",
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


def _probe_game(seed: int):
    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    transcript = initial_transcript(spec)
    records = []

    # Canonical probes remove arbitrary teacher tie-breaking while keeping the
    # hidden action semantics random. The safe first level guarantees all four
    # probes move and return the agent to its starting cell.
    for action in Action:
        record = game.step(action)
        records.append(record)
        append_record_tokens(transcript, record, next_frame=None)

    return spec, game, transcript, records


def _rotate_historical_actions(transcript: list[str]) -> list[str]:
    rotation = {
        "<A1>": "<A2>",
        "<A2>": "<A3>",
        "<A3>": "<A4>",
        "<A4>": "<A1>",
    }
    output: list[str] = []
    inside_action = False
    for token in transcript:
        if token == "<ACTION>":
            inside_action = True
            output.append(token)
        elif token == "</ACTION>":
            inside_action = False
            output.append(token)
        elif inside_action:
            output.append(rotation.get(token, token))
        else:
            output.append(token)
    return output


def build_example(seed: int) -> dict[str, object]:
    spec, game, transcript, records = _probe_game(seed)
    program = program_from_phase0_spec(spec)
    context_tokens = [*transcript, "<EVIDENCE_FULL>", "<INDUCE_MAPPING>"]
    return {
        "game_seed": seed,
        "context": tokens_to_text(context_tokens),
        "target": compact_mapping(program),
        "target_program_sha256": program.sha256,
        "full_program": canonical_program(program),
        "mapping": {
            action.value: direction.value
            for action, direction in sorted(
                spec.action_to_direction.items(), key=lambda item: item[0].value
            )
        },
        "palette": {
            "background": spec.palette.background,
            "wall": spec.palette.wall,
            "agent": spec.palette.agent,
            "goal": spec.palette.goal,
        },
        "probe_records": len(records),
        "post_probe_level": game.level_index,
        "post_probe_step": game.step_index,
    }


def build_control_context(seed: int, mode: str) -> str:
    spec, _, transcript, _ = _probe_game(seed)
    if mode == "full":
        evidence = transcript
        marker = "<EVIDENCE_FULL>"
    elif mode == "amnesic":
        evidence = initial_transcript(spec)
        marker = "<EVIDENCE_AMNESIC>"
    elif mode == "shuffled":
        evidence = _rotate_historical_actions(transcript)
        marker = "<EVIDENCE_SHUFFLED>"
    else:
        raise ValueError("mode must be full, amnesic, or shuffled")
    return tokens_to_text([*evidence, marker, "<INDUCE_MAPPING>"])


def _build_split(output_dir: Path, name: str, seeds: range) -> dict[str, object]:
    data_path = output_dir / f"{name}.jsonl"
    count = _write_jsonl(data_path, (build_example(seed) for seed in seeds))
    return {
        "games": count,
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
    )
    special_tokens_path = output_dir / "special_tokens.json"
    special_tokens_path.write_text(json.dumps(special_tokens, indent=2), encoding="utf-8")

    manifest = {
        "schema": "arcgpt2.program_induction.phase0.v2",
        "scope": (
            "Four-probe induction over 24 hidden movement mappings with compact "
            "atomic targets and exact DSL expansion; controlled gate, not an "
            "ARC-AGI-3 score."
        ),
        "seed_base": seed_base,
        "target": "<MAP> <A1> <DIRECTION> ... <A4> <DIRECTION> </MAP>",
        "special_tokens_file": special_tokens_path.name,
        "special_tokens_sha256": _sha256(special_tokens_path),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/program_induction/data"))
    parser.add_argument("--train-games", type=int, default=512)
    parser.add_argument("--validation-games", type=int, default=64)
    parser.add_argument("--test-games", type=int, default=128)
    parser.add_argument("--seed-base", type=int, default=31_337)
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
