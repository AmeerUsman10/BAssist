"""Compare pretrained and random-initialized Phase-0 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary["closed_loop"][name]


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    pre_intact = _mode(pretrained, "intact")
    pre_amnesic = _mode(pretrained, "amnesic")
    pre_shuffled = _mode(pretrained, "shuffled")
    rnd_intact = _mode(random_init, "intact")

    history_gain = pre_intact["level_completion_rate"] - pre_amnesic["level_completion_rate"]
    shuffle_gain = pre_intact["level_completion_rate"] - pre_shuffled["level_completion_rate"]
    pretraining_gain = pre_intact["level_completion_rate"] - rnd_intact["level_completion_rate"]

    gates = {
        "intact_level_completion_at_least_0_50": pre_intact["level_completion_rate"] >= 0.50,
        "history_gain_at_least_0_20": history_gain >= 0.20,
        "shuffled_history_drop_at_least_0_15": shuffle_gain >= 0.15,
        "pretraining_gain_positive": pretraining_gain > 0.0,
        "test_action_accuracy_above_random": pretrained["test_classification"]["accuracy"] > 0.25,
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "gates": gates,
        "deltas": {
            "intact_minus_amnesic_level_completion": history_gain,
            "intact_minus_shuffled_level_completion": shuffle_gain,
            "pretrained_minus_random_intact_level_completion": pretraining_gain,
        },
        "pretrained": {
            "test_classification": pretrained["test_classification"],
            "intact": pre_intact,
            "amnesic": pre_amnesic,
            "shuffled": pre_shuffled,
        },
        "random_initialization": {
            "test_classification": random_init["test_classification"],
            "intact": rnd_intact,
        },
        "interpretation": (
            "A pass would demonstrate in-context use of action/outcome history on held-out games. "
            "It would not constitute an ARC-AGI-3 score or evidence of hidden-goal generalization."
        ),
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
