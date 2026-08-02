"""Apply matched gates to the minimal one-shot GPT-2 memory experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    pre = pretrained["final_evaluation"]["summary"]
    rnd = random_init["final_evaluation"]["summary"]
    intact = pre["intact"]
    frozen = pre["no_adaptation"]
    shuffled = pre["shuffled_outcome"]
    random_intact = rnd["intact"]

    adaptation_accuracy_gain = intact["accuracy"] - frozen["accuracy"]
    adaptation_probability_gain = (
        intact["truth_probability"] - frozen["truth_probability"]
    )
    shuffled_accuracy_drop = intact["accuracy"] - shuffled["accuracy"]
    shuffled_probability_drop = (
        intact["truth_probability"] - shuffled["truth_probability"]
    )
    pretraining_accuracy_gain = intact["accuracy"] - random_intact["accuracy"]
    pretraining_probability_gain = (
        intact["truth_probability"] - random_intact["truth_probability"]
    )
    gates = {
        "intact_accuracy_at_least_0_70": intact["accuracy"] >= 0.70,
        "intact_truth_probability_at_least_0_60": (
            intact["truth_probability"] >= 0.60
        ),
        "online_accuracy_gain_at_least_0_35": adaptation_accuracy_gain >= 0.35,
        "online_probability_gain_at_least_0_25": (
            adaptation_probability_gain >= 0.25
        ),
        "shuffled_accuracy_drop_at_least_0_35": shuffled_accuracy_drop >= 0.35,
        "shuffled_probability_drop_at_least_0_25": (
            shuffled_probability_drop >= 0.25
        ),
        "pretrained_beats_random_accuracy": pretraining_accuracy_gain > 0.0,
        "pretrained_beats_random_probability": pretraining_probability_gain > 0.0,
        "support_gradient_is_nonzero": float(intact["gradient_norm"]) > 0.0,
        "query_loss_is_finite": float(pretrained["mean_query_loss"]) < float("inf"),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Minimal one-observation soft-prefix binding gate. A pass would show "
            "that one GPT-2 can write and read one causal fact through gradient "
            "memory; it is not ARC-AGI-3 completion."
        ),
        "gates": gates,
        "deltas": {
            "intact_minus_no_adaptation_accuracy": adaptation_accuracy_gain,
            "intact_minus_no_adaptation_truth_probability": adaptation_probability_gain,
            "intact_minus_shuffled_accuracy": shuffled_accuracy_drop,
            "intact_minus_shuffled_truth_probability": shuffled_probability_drop,
            "pretrained_minus_random_accuracy": pretraining_accuracy_gain,
            "pretrained_minus_random_truth_probability": pretraining_probability_gain,
        },
        "pretrained": {
            "initial": pretrained["initial_evaluation"]["summary"],
            "final": pre,
            "mean_query_loss": pretrained["mean_query_loss"],
            "mean_support_loss": pretrained["mean_support_loss"],
            "mean_gradient_norm": pretrained["mean_gradient_norm"],
        },
        "random_initialization": {
            "initial": random_init["initial_evaluation"]["summary"],
            "final": rnd,
            "mean_query_loss": random_init["mean_query_loss"],
            "mean_support_loss": random_init["mean_support_loss"],
            "mean_gradient_norm": random_init["mean_gradient_norm"],
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
