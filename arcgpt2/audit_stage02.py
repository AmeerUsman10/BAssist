"""Audit Stage-0.2 tokenization and evidence retention before more training.

This script performs no learning. It measures whether GPT-2 actually receives
complete evidence under each context budget and whether proposed answer strings
are represented by compact original GPT-2 tokens.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer

from . import stage02_decomposed as dense
from .train_stage02 import truncate_ids


def percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return int(ordered[index])


def candidate_audit(tokenizer: Any) -> dict[str, Any]:
    candidates = [
        "N", "E", "S", "W", "?", "1", "2", "3", "4",
        " north", " east", " south", " west", " unknown",
        " one", " two", " three", " four",
        "north", "east", "south", "west", "unknown",
    ]
    return {
        value: {
            "token_ids": tokenizer.encode(value, add_special_tokens=False),
            "tokens": tokenizer.tokenize(value),
            "round_trip": tokenizer.decode(
                tokenizer.encode(value, add_special_tokens=False)
            ),
        }
        for value in candidates
    }


def rows_for_seeds(seeds: range) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(dense.build_rows(seed))
    return rows


def length_audit(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
) -> dict[str, Any]:
    lengths: list[int] = []
    lengths_by_task: dict[str, list[int]] = defaultdict(list)
    lengths_by_history: dict[int, list[int]] = defaultdict(list)
    truncation_counts = Counter()
    known_mapping_rows = 0
    evidence_retained = Counter()
    per_budget_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    encoded_rows: list[tuple[Mapping[str, Any], list[int]]] = []
    for row in rows:
        ids = tokenizer.encode(str(row["prompt"]), add_special_tokens=False)
        encoded_rows.append((row, ids))
        length = len(ids)
        lengths.append(length)
        task = str(row["task"])
        history_length = int(row.get("metadata", {}).get("history_length", -1))
        lengths_by_task[task].append(length)
        lengths_by_history[history_length].append(length)

    for row, ids in encoded_rows:
        task = str(row["task"])
        target = str(row["target"])
        detail = str(row.get("task_detail", ""))
        action_digit = detail.removeprefix("map_") if detail.startswith("map_") else None
        requires_mapping_evidence = task == "mapping" and target != "?" and action_digit
        if requires_mapping_evidence:
            known_mapping_rows += 1
        for budget in budgets:
            truncated = truncate_ids(ids, budget=budget, prefix_keep=budget // 2)
            was_truncated = len(ids) > budget
            truncation_counts[(budget, "rows")] += 1
            truncation_counts[(budget, "truncated")] += int(was_truncated)
            if requires_mapping_evidence:
                decoded = tokenizer.decode(truncated)
                marker = f"action {action_digit};"
                retained = marker in decoded
                evidence_retained[(budget, "required")] += 1
                evidence_retained[(budget, "retained")] += int(retained)
                if was_truncated and not retained and len(per_budget_examples[str(budget)]) < 5:
                    per_budget_examples[str(budget)].append(
                        {
                            "task_detail": detail,
                            "target": target,
                            "history_length": row.get("metadata", {}).get("history_length"),
                            "original_tokens": len(ids),
                            "decoded_prefix": decoded[:500],
                            "decoded_suffix": decoded[-500:],
                        }
                    )

    def describe(values: Sequence[int]) -> dict[str, Any]:
        return {
            "count": len(values),
            "min": min(values, default=0),
            "median": median(values) if values else 0,
            "mean": mean(values) if values else 0,
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0),
        }

    return {
        "overall_lengths": describe(lengths),
        "lengths_by_task": {
            task: describe(values) for task, values in sorted(lengths_by_task.items())
        },
        "lengths_by_history": {
            str(history): describe(values)
            for history, values in sorted(lengths_by_history.items())
        },
        "budgets": {
            str(budget): {
                "rows": truncation_counts[(budget, "rows")],
                "truncated_rows": truncation_counts[(budget, "truncated")],
                "truncated_fraction": truncation_counts[(budget, "truncated")]
                / max(truncation_counts[(budget, "rows")], 1),
                "known_mapping_rows": evidence_retained[(budget, "required")],
                "known_mapping_evidence_retained": evidence_retained[(budget, "retained")],
                "known_mapping_evidence_retention_rate": evidence_retained[(budget, "retained")]
                / max(evidence_retained[(budget, "required")], 1),
                "missing_evidence_examples": per_budget_examples[str(budget)],
            }
            for budget in budgets
        },
        "known_mapping_rows": known_mapping_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[128, 192, 256, 320, 384, 512, 768, 1024]
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rows = rows_for_seeds(range(args.games))
    report = {
        "scope": "Stage-0.2 tokenizer and context audit; no capability claim",
        "model_name": args.model_name,
        "games": args.games,
        "rows": len(rows),
        "candidate_tokenization": candidate_audit(tokenizer),
        "dense_protocol": length_audit(tokenizer, rows, args.budgets),
    }

    # Import only after dense rows are materialized because the sparse module
    # deliberately installs its exact codec into the shared protocol module.
    from . import stage02_sparse as _sparse  # noqa: F401

    sparse_rows = rows_for_seeds(range(args.games))
    report["sparse_protocol"] = length_audit(
        tokenizer, sparse_rows, args.budgets
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
