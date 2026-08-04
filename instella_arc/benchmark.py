"""Frozen-checkpoint causal evidence benchmark for Instella-MoE.

The benchmark reuses the exact counterfactual/version-space generators from the
GPT-2 program. It never treats one arbitrary action or hidden rule as the only
correct target when the observation history still admits alternatives.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .backend import LoadPlan, TransformersBackend
from .prompts import Prompt, SYSTEM_TEXT, TaskKind


class ScoringBackend(Protocol):
    metadata: Mapping[str, Any]

    def score_completions(
        self, prompt: Prompt, completions: Sequence[str]
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class Case:
    case_id: str
    task: str
    mode: str
    evidence_depth: int
    prompt: Prompt
    completions: tuple[str, ...]
    target: tuple[float, ...]
    truth_index: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    task: str
    mode: str
    evidence_depth: int
    prompt_sha256: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    target: tuple[float, ...]
    set_cross_entropy: float
    brier: float
    consistent_mass: float
    top1_consistent: bool
    truth_probability: float | None
    truth_rank: int | None
    latency_seconds: float
    metadata: dict[str, Any]


def _prompt(task: TaskKind, context: str) -> Prompt:
    return Prompt(
        task=task,
        messages=(
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": context},
        ),
        legal_actions=(),
    )


def _normalized_target(values: Iterable[float]) -> tuple[float, ...]:
    target = tuple(float(value) for value in values)
    if not target or any(value < 0.0 for value in target):
        raise ValueError("target probabilities must be non-negative and non-empty")
    total = sum(target)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("target probabilities must have positive finite mass")
    return tuple(value / total for value in target)


def _softmax(scores: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(score) for score in scores)
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = sum(weights)
    return tuple(value / total for value in weights)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_action_cases(seeds: Iterable[int]) -> list[Case]:
    from arcgpt2.build_epistemic_dataset import build_examples
    from arcgpt2.natural_protocol import answer_text

    cases: list[Case] = []
    for seed in seeds:
        for row in build_examples(int(seed)):
            depth = int(row["probe_count"])
            if depth not in {0, 1, 2, 4}:
                continue
            completions = tuple(answer_text(word) for word in row["candidate_words"])
            target_map = row["target_distribution"]
            target = _normalized_target(target_map[word] for word in row["candidate_words"])
            truth_index = list(row["candidate_words"]).index(row["truth_word"])
            for mode, field in (
                ("intact", "context"),
                ("amnesic", "amnesic_context"),
                ("shuffled", "shuffled_context"),
            ):
                case_id = f"action:{seed}:{depth}:{row['query_action']}:{mode}"
                cases.append(
                    Case(
                        case_id=case_id,
                        task="action_binding",
                        mode=mode,
                        evidence_depth=depth,
                        prompt=_prompt(TaskKind.INFER_ACTION, str(row[field])),
                        completions=completions,
                        target=target,
                        truth_index=truth_index,
                        metadata={
                            "game_seed": int(seed),
                            "query_action": row["query_action"],
                            "consistent_mapping_count": row["consistent_mapping_count"],
                        },
                    )
                )
    return cases


def build_goal_cases(seeds: Iterable[int]) -> list[Case]:
    from arcgpt2.build_goal_version_dataset import build_examples

    cases: list[Case] = []
    for seed in seeds:
        for row in build_examples(int(seed)):
            depth = int(row["observed_terminal_count"])
            completions = tuple(str(value) for value in row["candidate_texts"])
            target = _normalized_target(row["target_distribution"])
            for mode, field in (
                ("intact", "context"),
                ("amnesic", "amnesic_context"),
                ("statusless", "statusless_context"),
                ("shuffled", "shuffled_status_context"),
            ):
                case_id = f"goal:{seed}:{row['prefix_length']}:{mode}"
                cases.append(
                    Case(
                        case_id=case_id,
                        task="goal_inference",
                        mode=mode,
                        evidence_depth=depth,
                        prompt=_prompt(TaskKind.INFER_GOAL, str(row[field])),
                        completions=completions,
                        target=target,
                        truth_index=int(row["truth_index"]),
                        metadata={
                            "game_seed": int(seed),
                            "prefix_length": int(row["prefix_length"]),
                            "consistent_goal_count": int(row["consistent_goal_count"]),
                        },
                    )
                )
    return cases


def build_contact_cases(seeds: Iterable[int]) -> list[Case]:
    from arcgpt2.build_contact_version_dataset import build_group_examples

    cases: list[Case] = []
    for seed in seeds:
        for row in build_group_examples(int(seed)):
            depth = int(bool(row["direct_contact_observed"]))
            completions = tuple(str(value) for value in row["candidate_texts"])
            target = _normalized_target(row["target_distribution"])
            for mode, field in (
                ("intact", "context"),
                ("amnesic", "amnesic_context"),
                ("precontact", "precontact_context"),
                ("shuffled", "shuffled_contact_context"),
            ):
                case_id = (
                    f"contact:{seed}:{row['mode_variant']}:"
                    f"{row['prefix_length']}:{mode}"
                )
                cases.append(
                    Case(
                        case_id=case_id,
                        task="contact_mechanics",
                        mode=mode,
                        evidence_depth=depth,
                        prompt=_prompt(TaskKind.INFER_MECHANICS, str(row[field])),
                        completions=completions,
                        target=target,
                        truth_index=int(row["truth_index"]),
                        metadata={
                            "counterfactual_group_seed": int(seed),
                            "mode_variant": row["mode_variant"],
                            "prefix_length": int(row["prefix_length"]),
                            "consistent_mode_count": int(row["consistent_mode_count"]),
                        },
                    )
                )
    return cases


def run_case(backend: ScoringBackend, case: Case) -> CaseResult:
    started = time.perf_counter()
    scores = tuple(float(value) for value in backend.score_completions(case.prompt, case.completions))
    latency = time.perf_counter() - started
    if len(scores) != len(case.target):
        raise ValueError("backend returned the wrong number of candidate scores")
    probabilities = _softmax(scores)
    cross_entropy = -sum(
        target * math.log(max(probability, 1e-30))
        for target, probability in zip(case.target, probabilities, strict=True)
    )
    brier = sum(
        (probability - target) ** 2
        for probability, target in zip(probabilities, case.target, strict=True)
    )
    consistent = [index for index, value in enumerate(case.target) if value > 0.0]
    top1 = max(range(len(probabilities)), key=probabilities.__getitem__)
    truth_probability = (
        probabilities[case.truth_index] if case.truth_index is not None else None
    )
    truth_rank = None
    if case.truth_index is not None:
        order = sorted(
            range(len(probabilities)), key=probabilities.__getitem__, reverse=True
        )
        truth_rank = order.index(case.truth_index) + 1
    rendered = case.prompt.plain_text()
    return CaseResult(
        case_id=case.case_id,
        task=case.task,
        mode=case.mode,
        evidence_depth=case.evidence_depth,
        prompt_sha256=_sha256(rendered),
        scores=scores,
        probabilities=probabilities,
        target=case.target,
        set_cross_entropy=cross_entropy,
        brier=brier,
        consistent_mass=sum(probabilities[index] for index in consistent),
        top1_consistent=top1 in consistent,
        truth_probability=truth_probability,
        truth_rank=truth_rank,
        latency_seconds=latency,
        metadata=case.metadata,
    )


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[CaseResult]] = {}
    for result in results:
        groups.setdefault(
            (result.task, result.mode, result.evidence_depth), []
        ).append(result)
    grouped: dict[str, Any] = {}
    for (task, mode, depth), values in sorted(groups.items()):
        key = f"{task}/{mode}/depth_{depth}"
        count = len(values)
        truth_probabilities = [
            value.truth_probability
            for value in values
            if value.truth_probability is not None
        ]
        truth_ranks = [
            value.truth_rank for value in values if value.truth_rank is not None
        ]
        grouped[key] = {
            "cases": count,
            "set_cross_entropy": sum(value.set_cross_entropy for value in values) / count,
            "brier": sum(value.brier for value in values) / count,
            "consistent_mass": sum(value.consistent_mass for value in values) / count,
            "top1_consistent": sum(value.top1_consistent for value in values) / count,
            "truth_probability": (
                sum(truth_probabilities) / len(truth_probabilities)
                if truth_probabilities
                else None
            ),
            "truth_rank": (
                sum(truth_ranks) / len(truth_ranks) if truth_ranks else None
            ),
            "latency_seconds": sum(value.latency_seconds for value in values) / count,
        }

    evidence_deltas: dict[str, Any] = {}
    for task in sorted({result.task for result in results}):
        task_results = [result for result in results if result.task == task]
        max_depth = max(result.evidence_depth for result in task_results)
        at_depth = [result for result in task_results if result.evidence_depth == max_depth]
        means: dict[str, float] = {}
        for mode in sorted({result.mode for result in at_depth}):
            selected = [result for result in at_depth if result.mode == mode]
            means[mode] = sum(value.consistent_mass for value in selected) / len(selected)
        intact = means.get("intact")
        evidence_deltas[task] = {
            "max_evidence_depth": max_depth,
            "consistent_mass_by_mode": means,
            "intact_minus_amnesic": (
                intact - means["amnesic"]
                if intact is not None and "amnesic" in means
                else None
            ),
            "intact_minus_shuffled": (
                intact - means["shuffled"]
                if intact is not None and "shuffled" in means
                else None
            ),
        }
    return {"groups": grouped, "evidence_deltas": evidence_deltas}


def run_benchmark(
    backend: ScoringBackend,
    *,
    tasks: Sequence[str],
    seed_base: int,
    games_per_task: int,
) -> dict[str, Any]:
    seeds = range(seed_base, seed_base + games_per_task)
    builders = {
        "action": build_action_cases,
        "goal": build_goal_cases,
        "contact": build_contact_cases,
    }
    cases: list[Case] = []
    for task in tasks:
        try:
            builder = builders[task]
        except KeyError as exc:
            raise ValueError(f"unknown task: {task}") from exc
        cases.extend(builder(seeds))
    results = [run_case(backend, case) for case in cases]
    return {
        "schema": "instella_arc.frozen_benchmark.v1",
        "backend": dict(getattr(backend, "metadata", {})),
        "tasks": list(tasks),
        "seed_base": seed_base,
        "games_per_task": games_per_task,
        "case_count": len(results),
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=("base", "dpo", "think"), default="think")
    parser.add_argument("--quantization", choices=("none", "int8", "int4"), default="int4")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--tasks", nargs="+", choices=("action", "goal", "contact"), default=["action", "goal", "contact"])
    parser.add_argument("--seed-base", type=int, default=880_000)
    parser.add_argument("--games-per-task", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = TransformersBackend.from_plan(
        LoadPlan(
            checkpoint_key=args.checkpoint,
            quantization=args.quantization,
            dtype=args.dtype,
            max_context_tokens=args.max_context_tokens,
        )
    )
    report = run_benchmark(
        backend,
        tasks=args.tasks,
        seed_base=args.seed_base,
        games_per_task=args.games_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
