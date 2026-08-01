from __future__ import annotations

from arcgpt2.sample_stage01_eval import stratified_sample, stratum_key


def make_row(phase: str, target: str, valid_size: int, level: int, marker: int):
    return {
        "decision_phase": phase,
        "target": target,
        "valid_targets": [f"<A{index}>" for index in range(1, valid_size + 1)],
        "level_index": level,
        "marker": marker,
    }


def test_stratified_sample_is_deterministic_and_preserves_strata() -> None:
    rows = []
    marker = 0
    for phase in ("probe", "navigate"):
        for target in ("<A1>", "<A2>"):
            for valid_size in (1, 2):
                for level in (0, 1):
                    for _ in range(4):
                        rows.append(make_row(phase, target, valid_size, level, marker))
                        marker += 1

    first = stratified_sample(rows, limit=32, seed=42)
    second = stratified_sample(rows, limit=32, seed=42)
    assert first == second
    assert len(first) == 32
    assert {stratum_key(row) for row in first} == {stratum_key(row) for row in rows}


def test_stratified_sample_returns_all_rows_when_limit_is_large() -> None:
    rows = [make_row("probe", "<A1>", 4, 0, marker) for marker in range(3)]
    assert stratified_sample(rows, limit=10, seed=1) == rows
