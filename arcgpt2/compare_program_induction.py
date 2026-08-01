"""Compare pretrained and random-initialized GPT-2 program induction runs."""

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

    history_gain = (
        pre_full["exact_program_accuracy"] - pre_amnesic["exact_program_accuracy"]
    )
    shuffled_gain = (
        pre_full["exact_program_accuracy"] - pre_shuffled["exact_program_accuracy"]
    )
    pretraining_gain = (
        pre_full["exact_program_accuracy"] - rnd_full["exact_program_accuracy"]
    )
    downstream_history_gain = (
        pre_full["downstream_level_completion_rate"]
        - pre_amnesic["downstream_level_completion_rate"]
    )

    gates = {
        "full_exact_program_accuracy_at_least_0_70": (
            pre_full["exact_program_accuracy"] >= 0.70
        ),
        "history_exact_accuracy_gain_at_least_0_40": history_gain >= 0.40,
        "shuffled_history_drop_at_least_0_30": shuffled_gain >= 0.30,
        "pretrained_beats_random_exact_accuracy": pretraining_gain > 0.0,
        "full_downstream_level_completion_at_least_0_70": (
            pre_full["downstream_level_completion_rate"] >= 0.70
        ),
        "history_improves_downstream_completion": downstream_history_gain > 0.0,
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Finite 24-program Phase-0 gate. Passing is evidence for executable "
            "program induction, not an ARC-AGI-3 score."
        ),
        "gates": gates,
        "deltas": {
            "full_minus_amnesic_exact_program_accuracy": history_gain,
            "full_minus_shuffled_exact_program_accuracy": shuffled_gain,
            "pretrained_minus_random_full_exact_program_accuracy": pretraining_gain,
            "full_minus_amnesic_downstream_level_completion": downstream_history_gain,
        },
        "pretrained": {
            "initial_full": pretrained["initial_full_selection"],
            "full": pre_full,
            "amnesic": pre_amnesic,
            "shuffled": pre_shuffled,
            "validation_loss": {
                "initial": pretrained["initial_validation_loss"],
                "final": pretrained["final_validation_loss"],
            },
        },
        "random_initialization": {
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
