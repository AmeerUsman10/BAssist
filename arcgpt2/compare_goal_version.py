"""Apply predeclared information-use gates to latent-goal GPT-2 runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_counts(summary: dict[str, Any], mode: str) -> list[int]:
    return sorted(
        int(key)
        for key in summary["controls"][mode]["by_observed_terminal_count"]
    )


def _metrics(
    summary: dict[str, Any], mode: str, terminal_count: int
) -> dict[str, float]:
    return summary["controls"][mode]["by_observed_terminal_count"][
        str(terminal_count)
    ]


def _choose_informative_count(summary: dict[str, Any]) -> int:
    """Choose the earliest prefix containing terminal evidence and future data."""

    counts = _available_counts(summary, "full")
    positive = [count for count in counts if count > 0]
    if not positive:
        raise ValueError("goal run contains no terminal-evidence prefixes")
    for count in positive:
        metrics = _metrics(summary, "full", count)
        if float(metrics.get("held_out_rows", 0.0)) > 0.0:
            return count
    return positive[0]


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    count = _choose_informative_count(pretrained)
    pre_full = _metrics(pretrained, "full", count)
    pre_amnesic = _metrics(pretrained, "amnesic", count)
    pre_statusless = _metrics(pretrained, "statusless", count)
    pre_shuffled = _metrics(pretrained, "shuffled_status", count)
    random_full = _metrics(random_init, "full", count)

    history_gain = pre_full["consistent_mass"] - pre_amnesic["consistent_mass"]
    terminal_gain = pre_full["consistent_mass"] - pre_statusless["consistent_mass"]
    corruption_drop = pre_full["consistent_mass"] - pre_shuffled["consistent_mass"]
    pretraining_gain = pre_full["consistent_mass"] - random_full["consistent_mass"]
    held_out_accuracy = float(pre_full.get("held_out_terminal_accuracy", float("nan")))

    gates = {
        "full_consistent_mass_at_least_0_50": pre_full["consistent_mass"] >= 0.50,
        "full_map_is_consistent_at_least_0_60": pre_full["map_consistent"] >= 0.60,
        "history_gain_at_least_0_20": history_gain >= 0.20,
        "terminal_report_gain_at_least_0_20": terminal_gain >= 0.20,
        "shuffled_status_drop_at_least_0_15": corruption_drop >= 0.15,
        "pretrained_beats_random_consistent_mass": pretraining_gain > 0.0,
        "held_out_terminal_accuracy_at_least_0_80": (
            math.isfinite(held_out_accuracy) and held_out_accuracy >= 0.80
        ),
        "validation_set_cross_entropy_improves": (
            pretrained["final_validation"]["set_cross_entropy"]
            < pretrained["initial_validation"]["set_cross_entropy"]
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Atomic latent-goal information-use gate under known mechanics. "
            "A pass is not ARC-AGI-3 completion."
        ),
        "informative_terminal_count": count,
        "gates": gates,
        "deltas": {
            "full_minus_amnesic_consistent_mass": history_gain,
            "full_minus_statusless_consistent_mass": terminal_gain,
            "full_minus_shuffled_status_consistent_mass": corruption_drop,
            "pretrained_minus_random_consistent_mass": pretraining_gain,
        },
        "pretrained": {
            "initial_validation": pretrained["initial_validation"],
            "final_validation": pretrained["final_validation"],
            "full": pre_full,
            "amnesic": pre_amnesic,
            "statusless": pre_statusless,
            "shuffled_status": pre_shuffled,
        },
        "random_initialization": {
            "initial_validation": random_init["initial_validation"],
            "final_validation": random_init["final_validation"],
            "full": random_full,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(_load(args.pretrained), _load(args.random))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
