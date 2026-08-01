"""Deterministic, lossless frame serialization for ARC-GPT2.

The codec is deliberately semantic-free: it never identifies objects, goals, or
salient regions. It only chooses the shortest exact representation among fixed
mechanical encodings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

Grid = tuple[tuple[int, ...], ...]


class CodecError(ValueError):
    """Raised when an encoded token stream is malformed."""


def normalize_grid(grid: Sequence[Sequence[int]]) -> Grid:
    rows = tuple(tuple(int(value) for value in row) for row in grid)
    if not rows:
        raise CodecError("grid must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise CodecError("grid rows must contain at least one cell")
    if any(len(row) != width for row in rows):
        raise CodecError("grid must be rectangular")
    if len(rows) > 64 or width > 64:
        raise CodecError("ARC grids are limited to 64x64")
    if any(value < 0 or value > 15 for row in rows for value in row):
        raise CodecError("cell values must be integers in the range 0..15")
    return rows


def _dim_token(kind: str, value: int) -> str:
    return f"<{kind}_{value}>"


def _parse_indexed(token: str, kind: str, *, minimum: int, maximum: int) -> int:
    prefix = f"<{kind}_"
    if not token.startswith(prefix) or not token.endswith(">"):
        raise CodecError(f"expected {kind} token, received {token!r}")
    raw = token[len(prefix) : -1]
    try:
        value = int(raw)
    except ValueError as exc:
        raise CodecError(f"invalid {kind} token: {token!r}") from exc
    if value < minimum or value > maximum:
        raise CodecError(f"{kind} value out of range: {value}")
    return value


def _color_token(value: int) -> str:
    return _dim_token("C", value)


def encode_rle(grid: Sequence[Sequence[int]]) -> list[str]:
    normalized = normalize_grid(grid)
    height = len(normalized)
    width = len(normalized[0])
    tokens = ["<RLE>", _dim_token("H", height), _dim_token("W", width)]

    for row in normalized:
        tokens.append("<ROW>")
        start = 0
        while start < width:
            color = row[start]
            end = start + 1
            while end < width and row[end] == color:
                end += 1
            tokens.extend((_color_token(color), _dim_token("N", end - start)))
            start = end
        tokens.append("<ENDROW>")

    tokens.append("<ENDGRID>")
    return tokens


def decode_rle(tokens: Sequence[str]) -> Grid:
    stream = iter(tokens)
    if next(stream, None) != "<RLE>":
        raise CodecError("RLE stream must start with <RLE>")
    height = _parse_indexed(next(stream, ""), "H", minimum=1, maximum=64)
    width = _parse_indexed(next(stream, ""), "W", minimum=1, maximum=64)

    rows: list[tuple[int, ...]] = []
    for _ in range(height):
        if next(stream, None) != "<ROW>":
            raise CodecError("expected <ROW>")
        row: list[int] = []
        while True:
            token = next(stream, None)
            if token is None:
                raise CodecError("truncated RLE row")
            if token == "<ENDROW>":
                break
            color = _parse_indexed(token, "C", minimum=0, maximum=15)
            count = _parse_indexed(next(stream, ""), "N", minimum=1, maximum=64)
            row.extend([color] * count)
            if len(row) > width:
                raise CodecError("RLE row exceeds declared width")
        if len(row) != width:
            raise CodecError("RLE row does not match declared width")
        rows.append(tuple(row))

    if next(stream, None) != "<ENDGRID>":
        raise CodecError("expected <ENDGRID>")
    if next(stream, None) is not None:
        raise CodecError("unexpected trailing RLE tokens")
    return tuple(rows)


def _uniform_color(grid: Grid, y0: int, x0: int, height: int, width: int) -> int | None:
    first = grid[y0][x0]
    for y in range(y0, y0 + height):
        for x in range(x0, x0 + width):
            if grid[y][x] != first:
                return None
    return first


def encode_quadtree(grid: Sequence[Sequence[int]]) -> list[str]:
    normalized = normalize_grid(grid)
    height = len(normalized)
    width = len(normalized[0])
    tokens = ["<QT>", _dim_token("H", height), _dim_token("W", width)]

    def visit(y0: int, x0: int, h: int, w: int) -> None:
        color = _uniform_color(normalized, y0, x0, h, w)
        if color is not None:
            tokens.append(_color_token(color))
            return

        if w >= h and w > 1:
            left_width = w // 2
            tokens.append("<SX>")
            visit(y0, x0, h, left_width)
            visit(y0, x0 + left_width, h, w - left_width)
            return

        if h > 1:
            top_height = h // 2
            tokens.append("<SY>")
            visit(y0, x0, top_height, w)
            visit(y0 + top_height, x0, h - top_height, w)
            return

        raise CodecError("non-uniform 1x1 region is impossible")

    visit(0, 0, height, width)
    tokens.append("<ENDGRID>")
    return tokens


def decode_quadtree(tokens: Sequence[str]) -> Grid:
    stream = iter(tokens)
    if next(stream, None) != "<QT>":
        raise CodecError("quadtree stream must start with <QT>")
    height = _parse_indexed(next(stream, ""), "H", minimum=1, maximum=64)
    width = _parse_indexed(next(stream, ""), "W", minimum=1, maximum=64)
    canvas = [[-1 for _ in range(width)] for _ in range(height)]

    def fill(y0: int, x0: int, h: int, w: int) -> None:
        token = next(stream, None)
        if token is None:
            raise CodecError("truncated quadtree stream")
        if token.startswith("<C_"):
            color = _parse_indexed(token, "C", minimum=0, maximum=15)
            for y in range(y0, y0 + h):
                for x in range(x0, x0 + w):
                    canvas[y][x] = color
            return
        if token == "<SX>":
            if w <= 1 or w < h:
                raise CodecError("invalid horizontal split")
            left_width = w // 2
            fill(y0, x0, h, left_width)
            fill(y0, x0 + left_width, h, w - left_width)
            return
        if token == "<SY>":
            if h <= 1 or (w >= h and w > 1):
                raise CodecError("invalid vertical split")
            top_height = h // 2
            fill(y0, x0, top_height, w)
            fill(y0 + top_height, x0, h - top_height, w)
            return
        raise CodecError(f"unexpected quadtree token: {token!r}")

    fill(0, 0, height, width)
    if next(stream, None) != "<ENDGRID>":
        raise CodecError("expected <ENDGRID>")
    if next(stream, None) is not None:
        raise CodecError("unexpected trailing quadtree tokens")
    if any(value < 0 for row in canvas for value in row):
        raise CodecError("quadtree did not fill the complete canvas")
    return tuple(tuple(row) for row in canvas)


def encode_frame(grid: Sequence[Sequence[int]]) -> list[str]:
    """Return the shorter exact full-frame representation.

    Ties are resolved in favor of row RLE so output is deterministic.
    """

    rle = encode_rle(grid)
    quadtree = encode_quadtree(grid)
    return rle if len(rle) <= len(quadtree) else quadtree


def decode_frame(tokens: Sequence[str]) -> Grid:
    if not tokens:
        raise CodecError("empty frame token stream")
    if tokens[0] == "<RLE>":
        return decode_rle(tokens)
    if tokens[0] == "<QT>":
        return decode_quadtree(tokens)
    raise CodecError(f"unknown full-frame encoding: {tokens[0]!r}")


def encode_delta(previous: Sequence[Sequence[int]], current: Sequence[Sequence[int]]) -> list[str]:
    before = normalize_grid(previous)
    after = normalize_grid(current)
    if len(before) != len(after) or len(before[0]) != len(after[0]):
        raise CodecError("delta frames must have identical dimensions")

    height = len(before)
    width = len(before[0])
    tokens = ["<DELTA>", _dim_token("H", height), _dim_token("W", width)]
    changed = False

    for y in range(height):
        x = 0
        while x < width:
            old = before[y][x]
            new = after[y][x]
            if old == new:
                x += 1
                continue
            changed = True
            end = x + 1
            while (
                end < width
                and before[y][end] == old
                and after[y][end] == new
                and before[y][end] != after[y][end]
            ):
                end += 1
            tokens.extend(
                (
                    "<RUN>",
                    _dim_token("Y", y),
                    _dim_token("X", x),
                    _dim_token("N", end - x),
                    _dim_token("OLD", old),
                    _dim_token("NEW", new),
                )
            )
            x = end

    tokens.append("<ENDDELTA>")
    if not changed:
        return ["<NOCHANGE>", _dim_token("H", height), _dim_token("W", width)]
    return tokens


def apply_delta(previous: Sequence[Sequence[int]], tokens: Sequence[str]) -> Grid:
    before = normalize_grid(previous)
    if not tokens:
        raise CodecError("empty delta token stream")

    stream = iter(tokens)
    first = next(stream, None)
    if first == "<NOCHANGE>":
        height = _parse_indexed(next(stream, ""), "H", minimum=1, maximum=64)
        width = _parse_indexed(next(stream, ""), "W", minimum=1, maximum=64)
        if (height, width) != (len(before), len(before[0])):
            raise CodecError("NOCHANGE dimensions do not match previous frame")
        if next(stream, None) is not None:
            raise CodecError("unexpected trailing NOCHANGE tokens")
        return before

    if first != "<DELTA>":
        raise CodecError("delta stream must start with <DELTA> or <NOCHANGE>")
    height = _parse_indexed(next(stream, ""), "H", minimum=1, maximum=64)
    width = _parse_indexed(next(stream, ""), "W", minimum=1, maximum=64)
    if (height, width) != (len(before), len(before[0])):
        raise CodecError("delta dimensions do not match previous frame")

    canvas = [list(row) for row in before]
    touched: set[tuple[int, int]] = set()
    while True:
        token = next(stream, None)
        if token is None:
            raise CodecError("truncated delta stream")
        if token == "<ENDDELTA>":
            break
        if token != "<RUN>":
            raise CodecError(f"expected <RUN>, received {token!r}")
        y = _parse_indexed(next(stream, ""), "Y", minimum=0, maximum=63)
        x = _parse_indexed(next(stream, ""), "X", minimum=0, maximum=63)
        count = _parse_indexed(next(stream, ""), "N", minimum=1, maximum=64)
        old = _parse_indexed(next(stream, ""), "OLD", minimum=0, maximum=15)
        new = _parse_indexed(next(stream, ""), "NEW", minimum=0, maximum=15)
        if y >= height or x + count > width:
            raise CodecError("delta run lies outside the grid")
        for xi in range(x, x + count):
            key = (y, xi)
            if key in touched:
                raise CodecError("overlapping delta runs")
            touched.add(key)
            if canvas[y][xi] != old:
                raise CodecError("delta OLD value does not match previous frame")
            canvas[y][xi] = new

    if next(stream, None) is not None:
        raise CodecError("unexpected trailing delta tokens")
    return tuple(tuple(row) for row in canvas)


@dataclass(frozen=True)
class TransitionEncoding:
    kind: str
    tokens: tuple[str, ...]


def encode_transition(
    previous: Sequence[Sequence[int]], current: Sequence[Sequence[int]]
) -> TransitionEncoding:
    """Choose the shorter representation of a new frame.

    `kind` is either ``delta`` or ``full``. Ties favor delta because it exposes
    action consequences explicitly while remaining exact.
    """

    delta = encode_delta(previous, current)
    full = encode_frame(current)
    if len(delta) <= len(full):
        return TransitionEncoding("delta", tuple(delta))
    return TransitionEncoding("full", tuple(full))


def decode_transition(
    previous: Sequence[Sequence[int]], encoding: TransitionEncoding
) -> Grid:
    if encoding.kind == "delta":
        return apply_delta(previous, encoding.tokens)
    if encoding.kind == "full":
        return decode_frame(encoding.tokens)
    raise CodecError(f"unknown transition kind: {encoding.kind!r}")


def token_inventory(max_dimension: int = 64) -> list[str]:
    """Return every atomic token required by the deterministic codec."""

    if max_dimension < 1 or max_dimension > 64:
        raise CodecError("max_dimension must be in 1..64")
    fixed = [
        "<RLE>",
        "<QT>",
        "<ROW>",
        "<ENDROW>",
        "<ENDGRID>",
        "<SX>",
        "<SY>",
        "<DELTA>",
        "<NOCHANGE>",
        "<RUN>",
        "<ENDDELTA>",
    ]
    indexed: list[str] = []
    for value in range(16):
        indexed.extend((_dim_token("C", value), _dim_token("OLD", value), _dim_token("NEW", value)))
    for value in range(max_dimension + 1):
        indexed.extend(
            (
                _dim_token("H", value),
                _dim_token("W", value),
                _dim_token("X", value),
                _dim_token("Y", value),
                _dim_token("N", value),
            )
        )
    return fixed + indexed


def tokens_to_text(tokens: Iterable[str]) -> str:
    return " ".join(tokens)
