"""Compare matched GPT-2 soft-memory meta-learning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _final(summary: dict[str, Any], mode: str, probe_count: int) -> dict[str, float]:
    return summary["final_evaluation"]["summary"][mode][str(probe_count)]


def _last_probe(summary: dict[str, Any]) -> int:
    return max(
        int(value)
        for value in summary["final_evaluation"]["summary"]["intact"]
    )


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    probe_count = _last_probe(pretrained)
    if probe_count != _last_probe(random_init):
        raise ValueError("matched runs have different support lengths")

    pre_intact = _final(pretrained, "intact", probe_count)
    pre_frozen = _final(pretrained, "no_adaptation", probe_count)
    pre_shuffled = _final(pretrained, "shuffled_outcome", probe_count)
    random_intact = _final(random_init, "intact", probe_count)

    adaptation_gain = (
        pre_intact["consistent_mass"] - pre_frozen["consistent_mass"]
    )
    corruption_drop = (
        pre_intact["consistent_mass"] - pre_shuffled["consistent_mass"]
    )
    pretraining_gain = (
        pre_intact["consistent_mass"] - random_intact["consistent_mass"]
    )
    gates = {
        "intact_consistent_mass_at_least_0_40": (
            pre_intact["consistent_mass"] >= 0.40
        ),
        "online_adaptation_gain_at_least_0_10": adaptation_gain >= 0.10,
        "shuffled_outcome_drop_at_least_0_10": corruption_drop >= 0.10,
        "per_action_accuracy_at_least_0_50": (
            pre_intact["per_action_accuracy"] >= 0.50
        ),
        "pretrained_beats_random_consistent_mass": pretraining_gain > 0.0,
        "meta_query_loss_is_finite": (
            float(pretrained["mean_meta_query_loss"]) < float("inf")
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Soft-prefix hidden-action meta-learning control. A pass would show "
            "evidence-dependent temporary memory, not ARC-AGI-3 completion."
        ),
        "support_probe_count": probe_count,
        "gates": gates,
        "deltas": {
            "intact_minus_no_adaptation_consistent_mass": adaptation_gain,
            "intact_minus_shuffled_consistent_mass": corruption_drop,
            "pretrained_minus_random_consistent_mass": pretraining_gain,
        },
        "pretrained": {
            "intact": pre_intact,
            "no_adaptation": pre_frozen,
            "shuffled_outcome": pre_shuffled,
            "mean_meta_query_loss": pretrained["mean_meta_query_loss"],
            "mean_support_nll": pretrained["mean_support_nll"],
        },
        "random_initialization": {
            "intact": random_intact,
            "mean_meta_query_loss": random_init["mean_meta_query_loss"],
            "mean_support_nll": random_init["mean_support_nll"],
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
