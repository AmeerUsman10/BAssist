"""Semantic-free sparse serialization variant of Stage 0.2.

The dense row string in the first Stage-0.2 run is exact, but GPT-2's byte-pair
tokenizer can merge long digit runs in ways that obscure coordinates.  This
variant mechanically chooses the most frequent cell value as a baseline and
lists every exception with its exact row, column, and value.  It identifies no
objects and discards no information.

Importing this module installs the sparse codec into ``stage02_decomposed`` for
the lifetime of the current Python process.  The environment, labels, prompts,
policy calls, and purity contract are otherwise unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from . import stage02_decomposed as base

Grid = tuple[tuple[int, ...], ...]


def sparse_grid_text(grid: Sequence[Sequence[int]]) -> str:
    rows = base.normalize_grid(grid)
    counts = Counter(value for row in rows for value in row)
    baseline = min(
        (value for value, count in counts.items() if count == max(counts.values())),
        default=0,
    )
    exceptions = [
        f"r{row}c{column}={value:x}"
        for row, values in enumerate(rows)
        for column, value in enumerate(values)
        if value != baseline
    ]
    body = ",".join(exceptions) if exceptions else "none"
    return f"{len(rows)}x{len(rows[0])};b{baseline:x};{body}"


def parse_sparse_grid_text(value: str) -> Grid:
    try:
        dimensions, raw_baseline, body = value.split(";", 2)
        raw_height, raw_width = dimensions.split("x", 1)
        height = int(raw_height)
        width = int(raw_width)
        if not raw_baseline.startswith("b"):
            raise ValueError("missing baseline marker")
        baseline = int(raw_baseline[1:], 16)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid sparse grid serialization: {value!r}") from exc
    canvas = [[baseline for _ in range(width)] for _ in range(height)]
    if body != "none":
        touched: set[tuple[int, int]] = set()
        for item in body.split(","):
            try:
                coordinate, raw_cell = item.split("=", 1)
                if not coordinate.startswith("r") or "c" not in coordinate:
                    raise ValueError("invalid coordinate")
                raw_row, raw_column = coordinate[1:].split("c", 1)
                row = int(raw_row)
                column = int(raw_column)
                cell = int(raw_cell, 16)
            except ValueError as exc:
                raise ValueError(f"invalid sparse cell: {item!r}") from exc
            if not (0 <= row < height and 0 <= column < width):
                raise ValueError("sparse coordinate outside grid")
            if (row, column) in touched:
                raise ValueError("duplicate sparse coordinate")
            touched.add((row, column))
            canvas[row][column] = cell
    return base.normalize_grid(canvas)


SPARSE_WORLD_HEADER = (
    "This is an exact grid-control record. A grid is written as HxW;bV;cells. "
    "V is the baseline value for every unlisted cell; listed cells use exact "
    "rROWcCOLUMN=VALUE coordinates. Values are 0 empty, 1 wall, 2 mover, "
    "3 goal. Row 0 is the top and column 0 is the left. N decreases row, E "
    "increases column, S increases row, and W decreases column. Actions 1, 2, "
    "3, and 4 are a different permutation of N, E, S, and W in every game. "
    "Infer the permutation only from observed cell changes."
)

# The base protocol functions resolve these names in their defining module at
# call time, so this changes only the exact serialization and explanatory
# grammar.  It adds no learned or decision-making component.
base.grid_text = sparse_grid_text
base.parse_grid_text = parse_sparse_grid_text
base.WORLD_HEADER = SPARSE_WORLD_HEADER

# Re-export the public experiment API after installing the codec.
build_dataset = base.build_dataset
build_rows = base.build_rows
format_history = base.format_history
make_stage02_spec = base.make_stage02_spec
simulate_context = base.simulate_context


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
