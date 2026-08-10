"""Compare pretrained and random GPT-2 on factorized action induction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    pre_full = pretrained["controls"]["full"]
    pre_amnesic = pretrained["controls"]["amnesic"]
    pre_shuffled = pretrained["controls"]["shuffled"]
    rnd_full = random_init["controls"]["full"]

    exact_history_gain = (
        pre_full["exact_mapping_accuracy"] - pre_amnesic["exact_mapping_accuracy"]
    )
    exact_shuffle_drop = (
        pre_full["exact_mapping_accuracy"] - pre_shuffled["exact_mapping_accuracy"]
    )
    action_history_gain = (
        pre_full["per_action_accuracy"] - pre_amnesic["per_action_accuracy"]
    )
    pretraining_gain = (
        pre_full["exact_mapping_accuracy"] - rnd_full["exact_mapping_accuracy"]
    )

    gates = {
        "full_exact_mapping_accuracy_at_least_0_70": (
            pre_full["exact_mapping_accuracy"] >= 0.70
        ),
        "full_per_action_accuracy_at_least_0_80": (
            pre_full["per_action_accuracy"] >= 0.80
        ),
        "exact_history_gain_at_least_0_30": exact_history_gain >= 0.30,
        "exact_shuffled_history_drop_at_least_0_25": exact_shuffle_drop >= 0.25,
        "per_action_history_gain_at_least_0_20": action_history_gain >= 0.20,
        "pretrained_beats_random_exact_mapping": pretraining_gain > 0.0,
        "full_downstream_level_completion_at_least_0_70": (
            pre_full["downstream_level_completion_rate"] >= 0.70
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Natural-language factorized hidden-action gate with an exact "
            "permutation posterior. It isolates action semantics and is not an "
            "ARC-AGI-3 score."
        ),
        "gates": gates,
        "deltas": {
            "full_minus_amnesic_exact_mapping_accuracy": exact_history_gain,
            "full_minus_shuffled_exact_mapping_accuracy": exact_shuffle_drop,
            "full_minus_amnesic_per_action_accuracy": action_history_gain,
            "pretrained_minus_random_full_exact_mapping_accuracy": pretraining_gain,
        },
        "pretrained": {
            "initial_full": pretrained["initial_full"],
            "full": pre_full,
            "amnesic": pre_amnesic,
            "shuffled": pre_shuffled,
            "validation_loss": {
                "initial": pretrained["initial_validation_loss"],
                "final": pretrained["final_validation_loss"],
            },
        },
        "random_initialization": {
            "initial_full": random_init["initial_full"],
            "full": rnd_full,
            "validation_loss": {
                "initial": random_init["initial_validation_loss"],
                "final": random_init["final_validation_loss"],
            },
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
