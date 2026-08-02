"""Apply matched gates to one-checkpoint action-and-goal GPT-2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_CONTROLS = {
    "action_binding": ("amnesic", "shuffled_evidence"),
    "goal_inference": ("amnesic", "statusless", "shuffled_status"),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _levels(summary: dict[str, Any], mode: str, task: str) -> dict[str, Any]:
    return summary["final_controls"][mode]["by_task"][task]["by_information_level"]


def _maximum_shared_level(
    summary: dict[str, Any], task: str, modes: tuple[str, ...]
) -> int:
    sets = [set(int(key) for key in _levels(summary, mode, task)) for mode in modes]
    shared = set.intersection(*sets)
    if not shared:
        raise ValueError(f"no shared information level for {task}: {modes}")
    return max(shared)


def _metric(
    summary: dict[str, Any], mode: str, task: str, level: int
) -> dict[str, float]:
    return _levels(summary, mode, task)[str(level)]


def compare(pretrained: dict[str, Any], random_init: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for task, controls in TASK_CONTROLS.items():
        modes = ("full", *controls)
        level = _maximum_shared_level(pretrained, task, modes)
        full = _metric(pretrained, "full", task, level)
        random_full = _metric(random_init, "full", task, level)
        control_metrics = {
            mode: _metric(pretrained, mode, task, level) for mode in controls
        }
        deltas = {
            f"full_minus_{mode}_consistent_mass": (
                full["consistent_mass"] - values["consistent_mass"]
            )
            for mode, values in control_metrics.items()
        }
        pretraining_gain = (
            full["consistent_mass"] - random_full["consistent_mass"]
        )
        task_gates = {
            "full_consistent_mass_at_least_0_50": full["consistent_mass"] >= 0.50,
            "full_map_consistent_at_least_0_60": full["map_consistent"] >= 0.60,
            "pretrained_beats_random_consistent_mass": pretraining_gain > 0.0,
            **{
                f"full_beats_{mode}_by_at_least_0_10": delta >= 0.10
                for mode, delta in deltas.items()
            },
        }
        for name, value in task_gates.items():
            gates[f"{task}:{name}"] = value
        tasks[task] = {
            "information_level": level,
            "pretrained_full": full,
            "pretrained_controls": control_metrics,
            "random_full": random_full,
            "deltas": {
                **deltas,
                "pretrained_minus_random_consistent_mass": pretraining_gain,
            },
            "gates": task_gates,
        }

    validation_improved = all(
        _mean_task_cross_entropy(pretrained["final_validation"], task)
        < _mean_task_cross_entropy(pretrained["initial_validation"], task)
        for task in TASK_CONTROLS
    )
    gates["both_tasks:validation_cross_entropy_improves"] = validation_improved
    return {
        "status": "pass" if all(gates.values()) else "not_yet_passed",
        "scope": (
            "One-GPT-2 multi-query epistemic gate. A pass would show joint "
            "information use without catastrophic task loss, not ARC-AGI-3 completion."
        ),
        "gates": gates,
        "tasks": tasks,
        "pretrained_training_loss_by_task": pretrained["mean_training_loss_by_task"],
        "random_training_loss_by_task": random_init["mean_training_loss_by_task"],
    }


def _mean_task_cross_entropy(evaluation: dict[str, Any], task: str) -> float:
    levels = evaluation["by_task"][task]["by_information_level"]
    total_rows = sum(float(values["rows"]) for values in levels.values())
    return sum(
        float(values["set_cross_entropy"]) * float(values["rows"])
        for values in levels.values()
    ) / total_rows


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
