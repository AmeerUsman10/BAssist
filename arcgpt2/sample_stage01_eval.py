"""Create a bounded deterministic Stage-0.1 evaluation sample.

The training split contains all 24 probe-order variants for every base game.
Evaluating every row twice on CPU is expensive and unnecessary for the overfit
sanity gate. This utility samples in round-robin order across decision phase,
canonical target, valid-set size, and level so all important strata remain
represented.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "decision_phase",
                "target",
                "valid_targets",
                "level_index",
            }
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"invalid row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"input dataset is empty: {path}")
    return rows


def stratum_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["decision_phase"]),
        str(row["target"]),
        len(row["valid_targets"]),
        int(row["level_index"]),
    )


def stratified_sample(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not rows:
        raise ValueError("rows must not be empty")
    if limit >= len(rows):
        return list(rows)

    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)

    rng = random.Random(seed)
    queues: list[deque[dict[str, Any]]] = []
    for key in sorted(groups, key=str):
        values = list(groups[key])
        rng.shuffle(values)
        queues.append(deque(values))

    selected: list[dict[str, Any]] = []
    while queues and len(selected) < limit:
        remaining: list[deque[dict[str, Any]]] = []
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    selected = stratified_sample(rows, limit=args.limit, seed=args.seed)
    write_jsonl(args.output, selected)
    input_strata = {stratum_key(row) for row in rows}
    output_strata = {stratum_key(row) for row in selected}
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "selected_rows": len(selected),
                "input_strata": len(input_strata),
                "selected_strata": len(output_strata),
                "all_strata_preserved": input_strata == output_strata,
                "seed": args.seed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
