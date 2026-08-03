"""Mode-correct training examples for Instella hidden-action adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from .action_replication import DIRECTIONS, mode_allowed_words
from .prompts import Prompt, SYSTEM_TEXT, TaskKind


@dataclass(frozen=True)
class ActionTrainingExample:
    example_id: str
    game_seed: int
    probe_count: int
    query_action: str
    mode: str
    prompt: Prompt
    candidate_words: tuple[str, ...]
    target_distribution: tuple[float, ...]
    allowed_words: tuple[str, ...]
    prompt_sha256: str


def prompt_for_context(context: str) -> Prompt:
    return Prompt(
        task=TaskKind.INFER_ACTION,
        messages=(
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": context},
        ),
        legal_actions=(),
    )


def target_for_mode(
    row: Mapping[str, Any], mode: str
) -> tuple[float, ...]:
    candidates = tuple(str(word) for word in row["candidate_words"])
    allowed = mode_allowed_words(row, mode)
    probability = 1.0 / len(allowed)
    return tuple(probability if word in allowed else 0.0 for word in candidates)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mode_context(row: Mapping[str, Any], mode: str) -> str:
    fields = {
        "intact": "context",
        "amnesic": "amnesic_context",
        "shuffled": "shuffled_context",
    }
    try:
        return str(row[fields[mode]])
    except KeyError as exc:
        raise ValueError(f"unknown training mode: {mode}") from exc


def build_action_training_examples(
    seeds: Iterable[int],
    *,
    probe_counts: Sequence[int] = (0, 1, 2, 4),
    include_modes: Sequence[str] = ("intact", "amnesic", "shuffled"),
) -> list[ActionTrainingExample]:
    """Build deduplicated examples without crossing seed boundaries.

    Amnesic prompts repeat at every probe depth, so prompt+target deduplication
    prevents the no-evidence control from dominating the curriculum. Shuffled
    depth-zero examples are also duplicates of a no-evidence world and are
    omitted by the same rule.
    """

    from arcgpt2.build_epistemic_dataset import build_examples

    requested_depths = set(int(value) for value in probe_counts)
    modes = tuple(str(mode) for mode in include_modes)
    unknown = set(modes) - {"intact", "amnesic", "shuffled"}
    if unknown:
        raise ValueError(f"unknown training modes: {sorted(unknown)}")

    examples: list[ActionTrainingExample] = []
    seen: set[tuple[int, str, tuple[float, ...]]] = set()
    for seed in seeds:
        for row in build_examples(int(seed)):
            depth = int(row["probe_count"])
            if depth not in requested_depths:
                continue
            candidates = tuple(str(word) for word in row["candidate_words"])
            if candidates != DIRECTIONS:
                raise ValueError("candidate direction order changed unexpectedly")
            for mode in modes:
                context = _mode_context(row, mode)
                prompt = prompt_for_context(context)
                target = target_for_mode(row, mode)
                rendered = prompt.plain_text()
                digest = _sha256(rendered)
                dedupe_key = (int(seed), digest, target)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                allowed = mode_allowed_words(row, mode)
                examples.append(
                    ActionTrainingExample(
                        example_id=(
                            f"action-train:{seed}:{depth}:"
                            f"{row['query_action']}:{mode}"
                        ),
                        game_seed=int(seed),
                        probe_count=depth,
                        query_action=str(row["query_action"]),
                        mode=mode,
                        prompt=prompt,
                        candidate_words=candidates,
                        target_distribution=target,
                        allowed_words=allowed,
                        prompt_sha256=digest,
                    )
                )
    return examples


def manifest(examples: Sequence[ActionTrainingExample]) -> dict[str, Any]:
    by_mode: dict[str, int] = {}
    by_depth: dict[int, int] = {}
    seeds: set[int] = set()
    for example in examples:
        by_mode[example.mode] = by_mode.get(example.mode, 0) + 1
        by_depth[example.probe_count] = by_depth.get(example.probe_count, 0) + 1
        seeds.add(example.game_seed)
    return {
        "schema": "instella_arc.action_training_data.v1",
        "examples": len(examples),
        "games": len(seeds),
        "by_mode": dict(sorted(by_mode.items())),
        "by_probe_count": {
            str(key): value for key, value in sorted(by_depth.items())
        },
        "candidate_words": list(DIRECTIONS),
        "examples_sha256": hashlib.sha256(
            "\n".join(
                f"{example.example_id}:{example.prompt_sha256}:"
                f"{example.target_distribution}"
                for example in examples
            ).encode("utf-8")
        ).hexdigest(),
    }


def serializable_example(example: ActionTrainingExample) -> dict[str, Any]:
    payload = asdict(example)
    payload["prompt"] = list(example.prompt.messages)
    return payload
