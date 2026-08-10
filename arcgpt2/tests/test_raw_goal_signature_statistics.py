"""Independent contract tests for raw-goal promotion statistics."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import importlib.util
import math
from pathlib import Path
import random
import statistics

import pytest


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "kaggle"
    / "arc-gpt2-raw-goal-signature-gpu"
    / "runner.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "raw_goal_signature_statistics_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(metric: str, values: dict[str, float]) -> dict:
    return {
        "locked_test": {
            "groups": [
                {
                    "group_key": key,
                    "orders": [{metric: value}, {metric: value}],
                }
                for key, value in values.items()
            ]
        }
    }


def _constant_delta_pair(metric: str, seed: int, delta: float):
    # Deliberately reverse one side: pairing must use group IDs, not row order.
    baseline = {f"g{index:02d}": (index % 7) / 100 for index in range(72)}
    pretrained = {key: value + delta for key, value in baseline.items()}
    return seed, _summary(metric, pretrained), _summary(
        metric, dict(reversed(tuple(baseline.items())))
    )


def test_hierarchical_bootstrap_matches_frozen_seed_then_group_algorithm():
    runner = _load_runner()
    metric = "goal_probability"
    pairs = [
        _constant_delta_pair(metric, 11, 0.20),
        _constant_delta_pair(metric, 22, 0.10),
        _constant_delta_pair(metric, 33, -0.05),
    ]
    samples = 257
    confidence = 0.95
    seed = 991
    observed = runner.hierarchical_paired_bootstrap(
        pairs, metric, samples=samples, confidence=confidence, seed=seed
    )

    # Reproduce the frozen algorithm independently, including a fresh group
    # resample for each occurrence of a resampled seed.
    deltas_by_seed = []
    for _, pretrained, random_init in pairs:
        pre = runner._group_metric_rows(pretrained, metric)
        rnd = runner._group_metric_rows(random_init, metric)
        deltas_by_seed.append([pre[key] - rnd[key] for key in sorted(pre)])
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = []
        for _ in range(3):
            values = deltas_by_seed[rng.randrange(3)]
            selected.append(
                sum(values[rng.randrange(72)] for _ in range(72)) / 72
            )
        draws.append(statistics.median(selected))
    rank = int(
        ((Decimal(1) - Decimal(str(confidence))) * Decimal(samples))
        .to_integral_value(rounding=ROUND_CEILING)
    )
    expected_lower = sorted(draws)[max(0, rank - 1)]

    assert observed["one_sided_95_percent_lower_bound"] == expected_lower
    assert observed["median_seed_delta"] == pytest.approx(0.10)
    assert observed["positive_seed_pairs"] == 2
    assert runner.hierarchical_paired_bootstrap(
        pairs, metric, samples=samples, confidence=confidence, seed=seed
    ) == observed


def test_hierarchical_bootstrap_rejects_unpaired_or_nonfinite_group_rows():
    runner = _load_runner()
    metric = "goal_accuracy"
    pairs = [
        _constant_delta_pair(metric, 11, 0.2),
        _constant_delta_pair(metric, 22, 0.2),
        _constant_delta_pair(metric, 33, 0.2),
    ]
    missing = pairs[0][2]["locked_test"]["groups"].pop()
    with pytest.raises(ValueError):
        runner.hierarchical_paired_bootstrap(pairs, metric, samples=10)
    pairs[0][2]["locked_test"]["groups"].append(missing)
    pairs[1][1]["locked_test"]["groups"][0]["orders"][0][metric] = math.nan
    with pytest.raises(ValueError):
        runner.hierarchical_paired_bootstrap(pairs, metric, samples=10)


def test_promotion_or_paths_cannot_mix_threshold_and_bound_across_metrics(monkeypatch):
    runner = _load_runner()
    results = {
        "accuracy": {
            "median_seed_delta": 0.11,
            "one_sided_95_percent_lower_bound": -0.001,
            "positive_seed_pairs": 3,
        },
        "probability": {
            "median_seed_delta": 0.049,
            "one_sided_95_percent_lower_bound": 0.01,
            "positive_seed_pairs": 3,
        },
    }

    def fake_bootstrap(_pairs, metric, **_kwargs):
        return results[metric]

    monkeypatch.setattr(runner, "hierarchical_paired_bootstrap", fake_bootstrap)
    failed = runner.promotion_track(
        [],
        name="identification",
        accuracy_metric="accuracy",
        probability_metric="probability",
        seed_offset=0,
    )
    assert failed["passed"] is False

    results["probability"]["median_seed_delta"] = 0.05
    passed = runner.promotion_track(
        [],
        name="identification",
        accuracy_metric="accuracy",
        probability_metric="probability",
        seed_offset=0,
    )
    assert passed["passed"] is True

