"""Apply predeclared information-use gates to contact-mechanics GPT-2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(summary: dict[str, Any], mode: str, prefix: int) -> dict[str, float]:
    return summary["controls"][mode]["by_prefix_length"][str(prefix)]


def _mean_cross_entropy(evaluation: dict[str, Any]) -> float:
    levels = evaluation["by_prefix_length"]
    total = sum(float(values["rows"]) for values in levels.values())
    return sum(
        float(values["set_cross_entropy"]) * float(values["rows"])
        for values in levels.values()
    ) / total


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    full = _metrics(pretrained, "full", 6)
    amnesic = _metrics(pretrained, "amnesic", 6)
    precontact = _metrics(pretrained, "precontact", 6)
    shuffled = _metrics(pretrained, "shuffled_contact", 6)
    random_full = _metrics(random_init, "full", 6)
    before_contact = _metrics(pretrained, "full", 5)

    history_gain = full["truth_probability"] - amnesic["truth_probability"]
    contact_gain = full["truth_probability"] - precontact["truth_probability"]
    corruption_drop = full["truth_probability"] - shuffled["truth_probability"]
    pretraining_gain = full["truth_probability"] - random_full["truth_probability"]
    gates = {
        "direct_contact_truth_probability_at_least_0_60": (
            full["truth_probability"] >= 0.60
        ),
        "direct_contact_map_accuracy_at_least_0_70": full["map_consistent"] >= 0.70,
        "amnesia_truth_probability_drop_at_least_0_25": history_gain >= 0.25,
        "precontact_truth_probability_drop_at_least_0_25": contact_gain >= 0.25,
        "shuffled_contact_truth_probability_drop_at_least_0_20": (
            corruption_drop >= 0.20
        ),
        "pretrained_beats_random_truth_probability": pretraining_gain > 0.0,
        "precontact_entropy_at_least_2_bits": before_contact["entropy_bits"] >= 2.0,
        "validation_set_cross_entropy_improves": (
            _mean_cross_entropy(pretrained["final_validation"])
            < _mean_cross_entropy(pretrained["initial_validation"])
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Counterfactual hidden-contact Gate C. A pass shows evidence-dependent "
            "primitive inference, not ARC-AGI-3 completion."
        ),
        "gates": gates,
        "deltas": {
            "full_minus_amnesic_truth_probability": history_gain,
            "full_minus_precontact_truth_probability": contact_gain,
            "full_minus_shuffled_truth_probability": corruption_drop,
            "pretrained_minus_random_truth_probability": pretraining_gain,
        },
        "pretrained": {
            "before_contact": before_contact,
            "after_contact_full": full,
            "after_contact_amnesic": amnesic,
            "after_contact_precontact": precontact,
            "after_contact_shuffled": shuffled,
            "initial_validation": pretrained["initial_validation"],
            "final_validation": pretrained["final_validation"],
        },
        "random_initialization": {
            "after_contact_full": random_full,
            "initial_validation": random_init["initial_validation"],
            "final_validation": random_init["final_validation"],
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
