"""Lossless text codecs for ARC-style integer grids.

The codec deliberately performs no semantic interpretation. It only converts a
rectangular integer grid to compact reversible text and encodes literal changed
cells between two equally-shaped grids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

Grid: TypeAlias = list[list[int]]


def normalize_grid(grid: Sequence[Sequence[int]]) -> Grid:
    """Return a validated mutable rectangular grid of non-negative integers."""
    rows = [[int(value) for value in row] for row in grid]
    if not rows:
        raise ValueError("grid must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError("grid rows must not be empty")
    if any(len(row) != width for row in rows):
        raise ValueError("grid must be rectangular")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("grid values must be non-negative")
    return rows


def _encode_row(row: Sequence[int]) -> str:
    if not row:
        raise ValueError("row must not be empty")
    parts: list[str] = []
    current = int(row[0])
    count = 1
    for raw_value in row[1:]:
        value = int(raw_value)
        if value == current:
            count += 1
            continue
        parts.append(f"{current}*{count}" if count > 1 else str(current))
        current = value
        count = 1
    parts.append(f"{current}*{count}" if count > 1 else str(current))
    return ",".join(parts)


def _decode_row(text: str) -> list[int]:
    if text == "":
        raise ValueError("encoded row must not be empty")
    row: list[int] = []
    for part in text.split(","):
        if "*" in part:
            value_text, count_text = part.split("*", maxsplit=1)
            value = int(value_text)
            count = int(count_text)
            if count <= 0:
                raise ValueError("run length must be positive")
            row.extend([value] * count)
        else:
            row.append(int(part))
    return row


def encode_grid(grid: Sequence[Sequence[int]]) -> str:
    """Encode a rectangular grid with per-row run-length encoding.

    Example: ``G[2,4]:0*2,1*2/2,0*3``.
    """
    rows = normalize_grid(grid)
    height = len(rows)
    width = len(rows[0])
    body = "/".join(_encode_row(row) for row in rows)
    return f"G[{height},{width}]:{body}"


def decode_grid(text: str) -> Grid:
    """Decode :func:`encode_grid` output and validate its declared shape."""
    if not text.startswith("G[") or "]:" not in text:
        raise ValueError("invalid grid encoding")
    header, body = text[2:].split("]:", maxsplit=1)
    height_text, width_text = header.split(",", maxsplit=1)
    height = int(height_text)
    width = int(width_text)
    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive")
    rows = [_decode_row(row_text) for row_text in body.split("/")]
    if len(rows) != height:
        raise ValueError("row count does not match encoded height")
    if any(len(row) != width for row in rows):
        raise ValueError("row width does not match encoded width")
    return normalize_grid(rows)


def changed_cells(
    before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]
) -> list[tuple[int, int, int]]:
    """Return ``(row, column, new_value)`` triples for every changed cell."""
    left = normalize_grid(before)
    right = normalize_grid(after)
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError("delta grids must have the same shape")
    changes: list[tuple[int, int, int]] = []
    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        for column_index, (left_value, right_value) in enumerate(
            zip(left_row, right_row, strict=True)
        ):
            if left_value != right_value:
                changes.append((row_index, column_index, right_value))
    return changes


def encode_delta(
    before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]
) -> str:
    """Encode only the literal changed cells between equally-shaped grids."""
    left = normalize_grid(before)
    right = normalize_grid(after)
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError("delta grids must have the same shape")
    height = len(left)
    width = len(left[0])
    changes = changed_cells(left, right)
    if not changes:
        return f"D[{height},{width}]:SAME"
    body = ";".join(f"{row},{column}={value}" for row, column, value in changes)
    return f"D[{height},{width}]:{body}"


def apply_delta(grid: Sequence[Sequence[int]], text: str) -> Grid:
    """Apply an encoded delta to a grid and return the exact resulting grid."""
    result = normalize_grid(grid)
    if not text.startswith("D[") or "]:" not in text:
        raise ValueError("invalid delta encoding")
    header, body = text[2:].split("]:", maxsplit=1)
    height_text, width_text = header.split(",", maxsplit=1)
    height = int(height_text)
    width = int(width_text)
    if len(result) != height or len(result[0]) != width:
        raise ValueError("delta shape does not match grid")
    if body == "SAME":
        return result
    seen: set[tuple[int, int]] = set()
    for part in body.split(";"):
        coordinate_text, value_text = part.split("=", maxsplit=1)
        row_text, column_text = coordinate_text.split(",", maxsplit=1)
        row = int(row_text)
        column = int(column_text)
        value = int(value_text)
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("delta coordinate is outside the grid")
        if (row, column) in seen:
            raise ValueError("delta contains a duplicate coordinate")
        if value < 0:
            raise ValueError("grid values must be non-negative")
        seen.add((row, column))
        result[row][column] = value
    return result


def extract_last_grid(frame: object) -> Grid:
    """Extract the last 2-D numeric grid from common ARC frame containers.

    Official frame payloads can expose either one 2-D grid or a sequence of
    animation grids. Choosing the final literal grid is a deterministic transport
    convention, not object recognition.
    """

    def is_scalar(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if hasattr(frame, "tolist"):
        frame = frame.tolist()  # type: ignore[assignment, union-attr]
    if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
        raise ValueError("frame does not contain a sequence")
    values = list(frame)
    if not values:
        raise ValueError("frame is empty")

    first = values[0]
    if hasattr(first, "tolist"):
        values = [value.tolist() if hasattr(value, "tolist") else value for value in values]
        first = values[0]

    # Direct 2-D grid.
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        first_row = list(first)
        if first_row and all(is_scalar(value) for value in first_row):
            return normalize_grid(values)  # type: ignore[arg-type]

    # Sequence of 2-D grids or nested animation payload; recurse into the last.
    return extract_last_grid(values[-1])


def encode_many(grids: Iterable[Sequence[Sequence[int]]]) -> list[str]:
    """Convenience helper used by dataset and test code."""
    return [encode_grid(grid) for grid in grids]
