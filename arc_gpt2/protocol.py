from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

Grid = list[list[int]]

PROTOCOL_TOKENS = [
    "<PAD>",
    "<TASK>",
    "</TASK>",
    "<PLAY>",
    "<FAST>",
    "<MEM>",
    "</MEM>",
    "<MAP>",
    "<OBS>",
    "</OBS>",
    "<PREV>",
    "</PREV>",
    "<LEGAL>",
    "</LEGAL>",
    "<OUT>",
    "<CF>",
    "</CF>",
    "<ACT>",
    "</ACT>",
    "<END>",
    "<RLE>",
    "<SPARSE>",
    "<BG>",
    "<EGRID>",
    "<DELTA>",
    "<EDELTA>",
    "<NOCHANGE>",
    "<NONE>",
    "<STATE>",
    "</STATE>",
    "<RUN>",
    "<WIN>",
    "<LOSE>",
    "<UNK>",
    "<N>",
    "<S>",
    "<W>",
    "<E>",
    "<MOVE>",
    "<NOOP>",
    "<RESET>",
    "<CHUNK>",
    "</CHUNK>",
]

ACTION_TOKENS = [f"<A{i}>" for i in range(1, 8)]
COLOR_TOKENS = [f"<C{i}>" for i in range(16)]
RUN_TOKENS = [f"<R{i}>" for i in range(1, 65)]
X_TOKENS = [f"<X{i}>" for i in range(64)]
Y_TOKENS = [f"<Y{i}>" for i in range(64)]
H_TOKENS = [f"<H{i}>" for i in range(1, 65)]
W_TOKENS = [f"<W{i}>" for i in range(1, 65)]

SPECIAL_TOKENS = list(
    dict.fromkeys(
        PROTOCOL_TOKENS
        + ACTION_TOKENS
        + COLOR_TOKENS
        + RUN_TOKENS
        + X_TOKENS
        + Y_TOKENS
        + H_TOKENS
        + W_TOKENS
    )
)

_TOKEN_RE = re.compile(r"<[^>]+>")
_ACTION_RE = re.compile(r"<A([1-7])>")


def _validate_grid(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")
    height = len(grid)
    width = len(grid[0])
    if height > 64 or width > 64:
        raise ValueError("Grid dimensions cannot exceed 64x64")
    if any(len(row) != width for row in grid):
        raise ValueError("Grid rows must have equal width")
    for row in grid:
        for value in row:
            if not isinstance(value, int) or not 0 <= value <= 15:
                raise ValueError("Grid values must be integers in [0, 15]")
    return height, width


def atom_count(text: str) -> int:
    """Count protocol atoms without requiring a tokenizer."""
    stripped = _TOKEN_RE.sub(" ", text)
    plain = [part for part in stripped.split() if part]
    return len(_TOKEN_RE.findall(text)) + len(plain)


def _runs(row: Sequence[int]) -> Iterable[tuple[int, int]]:
    current = row[0]
    length = 1
    for value in row[1:]:
        if value == current and length < 64:
            length += 1
        else:
            yield current, length
            current = value
            length = 1
    yield current, length


def encode_grid_rle(grid: Sequence[Sequence[int]]) -> str:
    height, width = _validate_grid(grid)
    parts = ["<RLE>", f"<H{height}>", f"<W{width}>"]
    for y, row in enumerate(grid):
        parts.append(f"<Y{y}>")
        for color, run_length in _runs(row):
            parts.extend((f"<C{color}>", f"<R{run_length}>"))
    parts.append("<EGRID>")
    return "".join(parts)


def encode_grid_sparse(grid: Sequence[Sequence[int]]) -> str:
    height, width = _validate_grid(grid)
    counts = Counter(value for row in grid for value in row)
    background = counts.most_common(1)[0][0]
    parts = [
        "<SPARSE>",
        f"<H{height}>",
        f"<W{width}>",
        "<BG>",
        f"<C{background}>",
    ]
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != background:
                parts.extend((f"<X{x}>", f"<Y{y}>", f"<C{value}>"))
    parts.append("<EGRID>")
    return "".join(parts)


def encode_grid(grid: Sequence[Sequence[int]]) -> str:
    """Return the shorter of two lossless, semantics-free encodings."""
    rle = encode_grid_rle(grid)
    sparse = encode_grid_sparse(grid)
    return sparse if atom_count(sparse) < atom_count(rle) else rle


def _number(token: str, prefix: str) -> int:
    expected_start = f"<{prefix}"
    if not token.startswith(expected_start) or not token.endswith(">"):
        raise ValueError(f"Expected {prefix} token, received {token}")
    return int(token[len(expected_start) : -1])


def decode_grid(encoded: str) -> Grid:
    tokens = _TOKEN_RE.findall(encoded)
    if len(tokens) < 4:
        raise ValueError("Encoded grid is incomplete")

    mode = tokens[0]
    height = _number(tokens[1], "H")
    width = _number(tokens[2], "W")

    if mode == "<RLE>":
        grid = [[0 for _ in range(width)] for _ in range(height)]
        index = 3
        seen_rows: set[int] = set()
        while index < len(tokens):
            token = tokens[index]
            if token == "<EGRID>":
                break
            y = _number(token, "Y")
            if y in seen_rows or not 0 <= y < height:
                raise ValueError("Invalid or duplicate RLE row")
            seen_rows.add(y)
            index += 1
            row: list[int] = []
            while index < len(tokens) and not tokens[index].startswith("<Y"):
                if tokens[index] == "<EGRID>":
                    break
                color = _number(tokens[index], "C")
                if index + 1 >= len(tokens):
                    raise ValueError("Missing run length")
                run_length = _number(tokens[index + 1], "R")
                row.extend([color] * run_length)
                index += 2
            if len(row) != width:
                raise ValueError("Decoded RLE row has incorrect width")
            grid[y] = row
        if len(seen_rows) != height:
            raise ValueError("RLE encoding omitted rows")
        _validate_grid(grid)
        return grid

    if mode == "<SPARSE>":
        if len(tokens) < 6 or tokens[3] != "<BG>":
            raise ValueError("Sparse encoding lacks background marker")
        background = _number(tokens[4], "C")
        grid = [[background for _ in range(width)] for _ in range(height)]
        index = 5
        while index < len(tokens):
            if tokens[index] == "<EGRID>":
                break
            if index + 2 >= len(tokens):
                raise ValueError("Incomplete sparse coordinate")
            x = _number(tokens[index], "X")
            y = _number(tokens[index + 1], "Y")
            color = _number(tokens[index + 2], "C")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("Sparse coordinate outside grid")
            grid[y][x] = color
            index += 3
        _validate_grid(grid)
        return grid

    raise ValueError(f"Unknown grid encoding mode: {mode}")


def encode_delta(previous: Sequence[Sequence[int]], current: Sequence[Sequence[int]]) -> str:
    previous_height, previous_width = _validate_grid(previous)
    current_height, current_width = _validate_grid(current)
    if (previous_height, previous_width) != (current_height, current_width):
        raise ValueError("Delta grids must have identical dimensions")

    changes: list[tuple[int, int, int]] = []
    for y in range(current_height):
        for x in range(current_width):
            if previous[y][x] != current[y][x]:
                changes.append((x, y, current[y][x]))

    if not changes:
        return "<DELTA><NOCHANGE><EDELTA>"

    parts = ["<DELTA>"]
    for x, y, color in changes:
        parts.extend((f"<X{x}>", f"<Y{y}>", f"<C{color}>"))
    parts.append("<EDELTA>")
    return "".join(parts)


def apply_delta(previous: Sequence[Sequence[int]], encoded_delta: str) -> Grid:
    _validate_grid(previous)
    output = [list(row) for row in previous]
    tokens = _TOKEN_RE.findall(encoded_delta)
    if not tokens or tokens[0] != "<DELTA>" or tokens[-1] != "<EDELTA>":
        raise ValueError("Malformed delta encoding")
    if tokens[1:-1] == ["<NOCHANGE>"]:
        return output
    index = 1
    while index < len(tokens) - 1:
        if index + 2 >= len(tokens) - 0:
            raise ValueError("Incomplete delta coordinate")
        x = _number(tokens[index], "X")
        y = _number(tokens[index + 1], "Y")
        color = _number(tokens[index + 2], "C")
        if not (0 <= y < len(output) and 0 <= x < len(output[0])):
            raise ValueError("Delta coordinate outside grid")
        output[y][x] = color
        index += 3
    return output


def chunk_protocol(text: str, max_atoms: int = 256) -> list[str]:
    if max_atoms < 8:
        raise ValueError("max_atoms must be at least 8")
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return [text]
    chunks = [tokens[index : index + max_atoms] for index in range(0, len(tokens), max_atoms)]
    total = len(chunks)
    return [
        f"<CHUNK>{index + 1}/{total}" + "".join(chunk) + "</CHUNK>"
        for index, chunk in enumerate(chunks)
    ]


def initial_memory(actions: Sequence[int] = (1, 2, 3, 4)) -> str:
    mapping = "".join(f"<A{action}><UNK>" for action in actions)
    return f"<MEM><MAP>{mapping}</MEM>"


def build_prompt(
    *,
    memory: str,
    grid: Sequence[Sequence[int]],
    legal_actions: Sequence[int],
    state: str = "RUN",
    previous_action: int | None = None,
    previous_delta: str | None = None,
) -> str:
    if state not in {"RUN", "WIN", "LOSE"}:
        raise ValueError("state must be RUN, WIN, or LOSE")
    legal = "".join(f"<A{action}>" for action in legal_actions)
    if previous_action is None:
        previous = "<PREV><NONE></PREV>"
    else:
        delta = previous_delta or "<DELTA><NOCHANGE><EDELTA>"
        previous = f"<PREV><A{previous_action}>{delta}</PREV>"
    return (
        "<TASK><PLAY><FAST></TASK>"
        + memory
        + "<OBS>"
        + encode_grid(grid)
        + f"<STATE><{state}></STATE>"
        + previous
        + "</OBS>"
        + f"<LEGAL>{legal}</LEGAL>"
        + "<OUT>"
    )


def extract_action(text: str, legal_actions: Sequence[int]) -> int | None:
    legal = set(legal_actions)
    act_section = text.split("<ACT>", 1)[1] if "<ACT>" in text else text
    for match in _ACTION_RE.finditer(act_section):
        action = int(match.group(1))
        if action in legal:
            return action
    return None


def extract_memory(text: str) -> str | None:
    start = text.find("<MEM>")
    end = text.find("</MEM>", start + 5)
    if start < 0 or end < 0:
        return None
    memory = text[start : end + len("</MEM>")]
    if atom_count(memory) > 192:
        return None
    return memory


def parse_coordinates(text: str) -> tuple[int, int] | None:
    x_match = re.search(r"<X([0-9]|[1-5][0-9]|6[0-3])>", text)
    y_match = re.search(r"<Y([0-9]|[1-5][0-9]|6[0-3])>", text)
    if not x_match or not y_match:
        return None
    return int(x_match.group(1)), int(y_match.group(1))
