"""Mode-correct, lexical-prior-calibrated hidden-action replication.

The first Instella smoke run established that the full checkpoint loads and can
score candidates, but its control metric reused the original-world truth for the
amnesic and shuffled prompts. That is not a valid causal test:

* an amnesic prompt admits all four directions;
* a shuffled-label prompt defines a different observed world and therefore a
  different mode-consistent answer;
* raw direction-token likelihoods contain stable lexical priors unrelated to
  the episode evidence.

This module fixes all three issues. For each seed and query action it scores an
intact, amnesic, and shuffled prompt. It reports both raw probabilities and a
paired evidence-only posterior obtained by subtracting the matched amnesic
log-score from each evidence-conditioned candidate score before softmax.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .backend import LoadPlan, TransformersBackend
from .catalog import CHECKPOINTS
from .prompts import Prompt, SYSTEM_TEXT, TaskKind


DIRECTIONS = ("north", "south", "west", "east")
ACTION_ROTATION = {"A1": "A2", "A2": "A3", "A3": "A4", "A4": "A1"}


class ScoringBackend(Protocol):
    metadata: Mapping[str, Any]

    def score_completions(
        self, prompt: Prompt, completions: Sequence[str]
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ScoredMode:
    mode: str
    prompt_sha256: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    allowed_words: tuple[str, ...]
    mode_truth_word: str | None
    original_truth_word: str
    allowed_mass: float
    mode_truth_probability: float | None
    original_truth_probability: float
    top1_word: str
    top1_mode_consistent: bool
    latency_seconds: float


@dataclass(frozen=True)
class ReplicationCase:
    case_id: str
    game_seed: int
    query_action: str
    original_truth_word: str
    shuffled_truth_word: str
    candidate_words: tuple[str, ...]
    modes: tuple[ScoredMode, ...]
    calibrated_intact_probabilities: tuple[float, ...]
    calibrated_shuffled_probabilities: tuple[float, ...]
    calibrated_intact_truth_probability: float
    calibrated_shuffled_truth_probability: float
    calibrated_shuffled_original_probability: float
    calibrated_intact_top1_word: str
    calibrated_shuffled_top1_word: str
    calibrated_intact_correct: bool
    calibrated_shuffled_follows_corruption: bool
    calibrated_shuffled_prefers_corruption_margin: float


def _prompt(context: str) -> Prompt:
    return Prompt(
        task=TaskKind.INFER_ACTION,
        messages=(
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": context},
        ),
        legal_actions=(),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _softmax(scores: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(score) for score in scores)
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _uniform(words: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(word) for word in words)


def shuffled_observed_mapping(row: Mapping[str, Any]) -> dict[str, str]:
    """Return displayed-action -> direction after deterministic label rotation."""

    observed_mapping = dict(row["observed_mapping"])
    return {
        ACTION_ROTATION[action]: _direction_word(direction)
        for action, direction in observed_mapping.items()
    }


def _direction_word(value: str) -> str:
    mapping = {
        "UP": "north",
        "DOWN": "south",
        "LEFT": "west",
        "RIGHT": "east",
        "north": "north",
        "south": "south",
        "west": "west",
        "east": "east",
    }
    try:
        return mapping[str(value)]
    except KeyError as exc:
        raise ValueError(f"unsupported direction value: {value!r}") from exc


def mode_allowed_words(row: Mapping[str, Any], mode: str) -> tuple[str, ...]:
    candidates = tuple(str(word) for word in row["candidate_words"])
    if mode == "intact":
        return tuple(str(word) for word in row["allowed_words"])
    if mode == "amnesic":
        return _uniform(candidates)
    if mode == "shuffled":
        displayed = shuffled_observed_mapping(row)
        query = str(row["query_action"])
        if query in displayed:
            return (displayed[query],)
        used = set(displayed.values())
        return tuple(word for word in candidates if word not in used)
    raise ValueError(f"unknown evidence mode: {mode}")


def mode_truth_word(row: Mapping[str, Any], mode: str) -> str | None:
    allowed = mode_allowed_words(row, mode)
    return allowed[0] if len(allowed) == 1 else None


def score_mode(
    backend: ScoringBackend,
    *,
    mode: str,
    context: str,
    row: Mapping[str, Any],
) -> ScoredMode:
    from arcgpt2.natural_protocol import answer_text

    candidates = tuple(str(word) for word in row["candidate_words"])
    completions = tuple(answer_text(word) for word in candidates)
    prompt = _prompt(context)
    started = time.perf_counter()
    scores = tuple(float(value) for value in backend.score_completions(prompt, completions))
    latency = time.perf_counter() - started
    probabilities = _softmax(scores)
    allowed = mode_allowed_words(row, mode)
    allowed_indices = [candidates.index(word) for word in allowed]
    truth = mode_truth_word(row, mode)
    original = str(row["truth_word"])
    top1_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return ScoredMode(
        mode=mode,
        prompt_sha256=_sha256(prompt.plain_text()),
        scores=scores,
        probabilities=probabilities,
        allowed_words=allowed,
        mode_truth_word=truth,
        original_truth_word=original,
        allowed_mass=sum(probabilities[index] for index in allowed_indices),
        mode_truth_probability=(
            probabilities[candidates.index(truth)] if truth is not None else None
        ),
        original_truth_probability=probabilities[candidates.index(original)],
        top1_word=candidates[top1_index],
        top1_mode_consistent=candidates[top1_index] in allowed,
        latency_seconds=latency,
    )


def calibrated_probabilities(
    evidence_scores: Sequence[float],
    amnesic_scores: Sequence[float],
) -> tuple[float, ...]:
    if len(evidence_scores) != len(amnesic_scores):
        raise ValueError("paired score vectors must have equal length")
    return _softmax(
        tuple(
            float(evidence) - float(baseline)
            for evidence, baseline in zip(
                evidence_scores, amnesic_scores, strict=True
            )
        )
    )


def build_replication_rows(seeds: Iterable[int]) -> list[dict[str, Any]]:
    from arcgpt2.build_epistemic_dataset import build_examples

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for row in build_examples(int(seed)):
            if int(row["probe_count"]) != 4:
                continue
            rows.append(dict(row))
    return rows


def run_replication_case(
    backend: ScoringBackend,
    row: Mapping[str, Any],
) -> ReplicationCase:
    candidates = tuple(str(word) for word in row["candidate_words"])
    contexts = {
        "intact": str(row["context"]),
        "amnesic": str(row["amnesic_context"]),
        "shuffled": str(row["shuffled_context"]),
    }
    scored = {
        mode: score_mode(backend, mode=mode, context=context, row=row)
        for mode, context in contexts.items()
    }
    intact_calibrated = calibrated_probabilities(
        scored["intact"].scores,
        scored["amnesic"].scores,
    )
    shuffled_calibrated = calibrated_probabilities(
        scored["shuffled"].scores,
        scored["amnesic"].scores,
    )
    original_truth = str(row["truth_word"])
    shuffled_truth = mode_truth_word(row, "shuffled")
    if shuffled_truth is None:
        raise RuntimeError("full-evidence shuffled case must have one induced truth")
    original_index = candidates.index(original_truth)
    shuffled_index = candidates.index(shuffled_truth)
    intact_top1 = candidates[
        max(range(len(intact_calibrated)), key=intact_calibrated.__getitem__)
    ]
    shuffled_top1 = candidates[
        max(range(len(shuffled_calibrated)), key=shuffled_calibrated.__getitem__)
    ]
    return ReplicationCase(
        case_id=(
            f"action-replication:{row['game_seed']}:{row['query_action']}"
        ),
        game_seed=int(row["game_seed"]),
        query_action=str(row["query_action"]),
        original_truth_word=original_truth,
        shuffled_truth_word=shuffled_truth,
        candidate_words=candidates,
        modes=tuple(scored[mode] for mode in ("intact", "amnesic", "shuffled")),
        calibrated_intact_probabilities=intact_calibrated,
        calibrated_shuffled_probabilities=shuffled_calibrated,
        calibrated_intact_truth_probability=intact_calibrated[original_index],
        calibrated_shuffled_truth_probability=shuffled_calibrated[shuffled_index],
        calibrated_shuffled_original_probability=shuffled_calibrated[original_index],
        calibrated_intact_top1_word=intact_top1,
        calibrated_shuffled_top1_word=shuffled_top1,
        calibrated_intact_correct=intact_top1 == original_truth,
        calibrated_shuffled_follows_corruption=shuffled_top1 == shuffled_truth,
        calibrated_shuffled_prefers_corruption_margin=(
            shuffled_calibrated[shuffled_index]
            - shuffled_calibrated[original_index]
        ),
    )


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    generator = random.Random(seed)
    count = len(values)
    estimates = []
    for _ in range(samples):
        estimates.append(
            statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        )
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return float(lower), float(upper)


def _mean_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    lower, upper = _bootstrap_interval(values, seed=seed)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95": [lower, upper],
        "count": len(values),
    }


def summarize(cases: Sequence[ReplicationCase]) -> dict[str, Any]:
    if not cases:
        raise ValueError("replication requires at least one case")
    intact_mass = [case.calibrated_intact_truth_probability for case in cases]
    shuffled_mode_mass = [
        case.calibrated_shuffled_truth_probability for case in cases
    ]
    shuffled_original_mass = [
        case.calibrated_shuffled_original_probability for case in cases
    ]
    corruption_margin = [
        case.calibrated_shuffled_prefers_corruption_margin for case in cases
    ]
    intact_correct = [1.0 if case.calibrated_intact_correct else 0.0 for case in cases]
    shuffled_correct = [
        1.0 if case.calibrated_shuffled_follows_corruption else 0.0
        for case in cases
    ]

    per_direction: dict[str, Any] = {}
    for direction in DIRECTIONS:
        selected = [case for case in cases if case.original_truth_word == direction]
        per_direction[direction] = {
            "cases": len(selected),
            "intact_top1_accuracy": (
                statistics.fmean(
                    1.0 if case.calibrated_intact_correct else 0.0
                    for case in selected
                )
                if selected
                else None
            ),
            "intact_truth_probability": (
                statistics.fmean(
                    case.calibrated_intact_truth_probability for case in selected
                )
                if selected
                else None
            ),
        }

    metrics = {
        "calibrated_intact_truth_probability": _mean_summary(
            intact_mass, seed=0x1A7AC7
        ),
        "calibrated_shuffled_mode_truth_probability": _mean_summary(
            shuffled_mode_mass, seed=0x5AFF1E
        ),
        "calibrated_shuffled_original_truth_probability": _mean_summary(
            shuffled_original_mass, seed=0x0A161A1
        ),
        "calibrated_corruption_preference_margin": _mean_summary(
            corruption_margin, seed=0xC0AAE57
        ),
        "calibrated_intact_top1_accuracy": _mean_summary(
            intact_correct, seed=0x1A7ACC
        ),
        "calibrated_shuffled_follows_corruption_top1_accuracy": _mean_summary(
            shuffled_correct, seed=0x5AFFACC
        ),
    }

    # Promotion requires evidence use in both the real and deliberately corrupted
    # worlds. Thresholds are deliberately above chance but are not an ARC score.
    lower_intact_mass = metrics[
        "calibrated_intact_truth_probability"
    ]["bootstrap_95"][0]
    lower_intact_accuracy = metrics[
        "calibrated_intact_top1_accuracy"
    ]["bootstrap_95"][0]
    lower_corruption_margin = metrics[
        "calibrated_corruption_preference_margin"
    ]["bootstrap_95"][0]
    lower_shuffled_accuracy = metrics[
        "calibrated_shuffled_follows_corruption_top1_accuracy"
    ]["bootstrap_95"][0]
    gates = {
        "intact_probability_above_chance": lower_intact_mass > 0.25,
        "intact_top1_above_chance": lower_intact_accuracy > 0.25,
        "corrupted_evidence_preferred_over_original": lower_corruption_margin > 0.0,
        "shuffled_world_top1_above_chance": lower_shuffled_accuracy > 0.25,
        "no_direction_collapse": all(
            value["cases"] > 0
            and value["intact_truth_probability"] is not None
            and value["intact_truth_probability"] > 0.10
            for value in per_direction.values()
        ),
    }
    gates["promote_to_frozen_goal_and_mechanics"] = all(gates.values())
    gates["promote_to_qlora"] = False
    return {
        "metrics": metrics,
        "per_direction": per_direction,
        "gates": gates,
        "interpretation": (
            "Promotion here means only that the frozen checkpoint follows exact "
            "action-binding evidence across unseen synthetic worlds. It is not an "
            "ARC-AGI-3 score and does not itself authorize QLoRA."
        ),
    }


def run_replication(
    backend: ScoringBackend,
    *,
    seed_base: int,
    games: int,
) -> dict[str, Any]:
    rows = build_replication_rows(range(seed_base, seed_base + games))
    cases = [run_replication_case(backend, row) for row in rows]
    return {
        "schema": "instella_arc.action_replication.v2",
        "backend": dict(getattr(backend, "metadata", {})),
        "seed_base": seed_base,
        "games": games,
        "queries_per_game": 4,
        "case_count": len(cases),
        "candidate_words": list(DIRECTIONS),
        "calibration": (
            "For each candidate, subtract the matched amnesic prompt log-score "
            "from the intact or shuffled prompt log-score, then softmax."
        ),
        "summary": summarize(cases),
        "cases": [asdict(case) for case in cases],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", choices=tuple(CHECKPOINTS), default="think"
    )
    parser.add_argument(
        "--quantization", choices=("int4", "int8", "none"), default="int4"
    )
    parser.add_argument("--seed-base", type=int, default=940_000)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=6144)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = CHECKPOINTS[args.checkpoint]
    backend = TransformersBackend.from_plan(
        LoadPlan(
            checkpoint_key=args.checkpoint,
            quantization=args.quantization,
            dtype="float16",
            revision=spec.revision,
            max_context_tokens=args.max_context_tokens,
        )
    )
    report = run_replication(
        backend,
        seed_base=args.seed_base,
        games=args.games,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
