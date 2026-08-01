"""Apply predeclared gates to matched epistemic-binding runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe(summary: dict[str, Any], mode: str, count: int) -> dict[str, float]:
    return summary["controls"][mode]["by_probe_count"][str(count)]


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    pre_0 = _probe(pretrained, "full", 0)
    pre_4 = _probe(pretrained, "full", 4)
    pre_amnesic_4 = _probe(pretrained, "amnesic", 4)
    pre_shuffled_4 = _probe(pretrained, "shuffled", 4)
    random_4 = _probe(random_init, "full", 4)

    history_gain = pre_4["consistent_mass"] - pre_amnesic_4["consistent_mass"]
    corruption_drop = pre_4["consistent_mass"] - pre_shuffled_4["consistent_mass"]
    pretraining_gain = pre_4["map_exact"] - random_4["map_exact"]
    entropy_drop = pre_0["joint_entropy_bits"] - pre_4["joint_entropy_bits"]

    gates = {
        "full_four_probe_exact_mapping_at_least_0_60": pre_4["map_exact"] >= 0.60,
        "full_four_probe_consistent_mass_at_least_0_70": pre_4["consistent_mass"] >= 0.70,
        "history_consistent_mass_gain_at_least_0_40": history_gain >= 0.40,
        "shuffled_evidence_drop_at_least_0_25": corruption_drop >= 0.25,
        "pretrained_beats_random_exact_mapping": pretraining_gain > 0.0,
        "posterior_entropy_contracts_by_at_least_2_bits": entropy_drop >= 2.0,
        "validation_set_cross_entropy_improves": (
            pretrained["final_validation"]["set_cross_entropy"]
            < pretrained["initial_validation"]["set_cross_entropy"]
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "Partial-evidence hidden-action version-space gate. A pass would "
            "show calibrated in-context binding, not ARC-AGI-3 completion."
        ),
        "gates": gates,
        "deltas": {
            "full_minus_amnesic_consistent_mass_at_four_probes": history_gain,
            "full_minus_shuffled_consistent_mass_at_four_probes": corruption_drop,
            "pretrained_minus_random_exact_mapping_at_four_probes": pretraining_gain,
            "full_entropy_drop_zero_to_four_probes_bits": entropy_drop,
        },
        "pretrained": {
            "initial_validation": pretrained["initial_validation"],
            "final_validation": pretrained["final_validation"],
            "full": pretrained["controls"]["full"]["by_probe_count"],
            "amnesic": pretrained["controls"]["amnesic"]["by_probe_count"],
            "shuffled": pretrained["controls"]["shuffled"]["by_probe_count"],
        },
        "random_initialization": {
            "initial_validation": random_init["initial_validation"],
            "final_validation": random_init["final_validation"],
            "full": random_init["controls"]["full"]["by_probe_count"],
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
