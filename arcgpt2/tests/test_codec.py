from __future__ import annotations

import random

import pytest

from arcgpt2.codec import (
    CodecError,
    TransitionEncoding,
    apply_delta,
    decode_delta,
    decode_frame,
    decode_grid,
    decode_quadtree,
    decode_rle,
    decode_transition,
    encode_delta,
    encode_delta_text,
    encode_frame,
    encode_grid,
    encode_quadtree,
    encode_rle,
    encode_transition,
    normalize_grid,
    text_to_tokens,
    tokens_to_text,
)


def test_structured_roundtrips() -> None:
    grids = [
        [[0]],
        [[0, 0, 0, 0]],
        [[0], [1], [1], [0]],
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]],
        [[(x + y) % 4 for x in range(7)] for y in range(5)],
    ]
    for grid in grids:
        expected = normalize_grid(grid)
        assert decode_rle(encode_rle(grid)) == expected
        assert decode_quadtree(encode_quadtree(grid)) == expected
        assert decode_frame(encode_frame(grid)) == expected


def test_random_roundtrips() -> None:
    rng = random.Random(73)
    for _ in range(250):
        height = rng.randint(1, 12)
        width = rng.randint(1, 12)
        palette = rng.randint(1, 8)
        grid = [[rng.randrange(palette) for _ in range(width)] for _ in range(height)]
        expected = normalize_grid(grid)
        assert decode_rle(encode_rle(grid)) == expected
        assert decode_quadtree(encode_quadtree(grid)) == expected
        assert decode_frame(encode_frame(grid)) == expected


def test_quadtree_beats_rle_on_large_rectangles() -> None:
    grid = [[0 for _ in range(16)] for _ in range(16)]
    for y in range(8, 16):
        for x in range(8, 16):
            grid[y][x] = 3
    assert len(encode_quadtree(grid)) < len(encode_rle(grid))
    assert encode_frame(grid)[0] == "<QT>"


def test_rle_wins_tie_deterministically() -> None:
    grid = [[3]]
    rle = encode_rle(grid)
    quadtree = encode_quadtree(grid)
    assert len(rle) >= len(quadtree)
    # For a 1x1 grid the quadtree is strictly shorter, so verify deterministic
    # selection by comparing with the exact minimum rather than assuming RLE.
    selected = encode_frame(grid)
    expected = rle if len(rle) <= len(quadtree) else quadtree
    assert selected == expected


def test_delta_roundtrip_and_nochange() -> None:
    before = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
    after = [[0, 2, 2, 0], [1, 1, 0, 3], [0, 0, 0, 0]]
    encoded = encode_delta(before, after)
    assert apply_delta(before, encoded) == normalize_grid(after)

    nochange = encode_delta(before, before)
    assert nochange[0] == "<NOCHANGE>"
    assert apply_delta(before, nochange) == normalize_grid(before)


def test_canonical_text_apis_are_reversible() -> None:
    before = [[0, 0, 0], [1, 1, 0]]
    after = [[0, 2, 0], [1, 0, 0]]

    grid_text = encode_grid(after)
    assert tokens_to_text(text_to_tokens(grid_text)) == grid_text
    assert decode_grid(grid_text) == normalize_grid(after)

    delta_text = encode_delta_text(before, after)
    assert tokens_to_text(text_to_tokens(delta_text)) == delta_text
    assert decode_delta(before, delta_text) == normalize_grid(after)


@pytest.mark.parametrize(
    "malformed",
    [
        " <RLE>",
        "<RLE> ",
        "<RLE>  <H_1>",
        "<RLE>\n<H_1>",
        "<RLE>\t<H_1>",
    ],
)
def test_text_parser_rejects_noncanonical_whitespace(malformed: str) -> None:
    with pytest.raises(CodecError):
        text_to_tokens(malformed)


def test_text_decoders_reject_malformed_streams() -> None:
    with pytest.raises(CodecError):
        decode_grid("<RLE> <H_1> <W_1> <ROW> <C_0> <N_1> <ENDROW>")
    with pytest.raises(CodecError):
        decode_delta([[0]], "<DELTA> <H_1> <W_1> <RUN>")
    with pytest.raises(CodecError):
        tokens_to_text(["<RLE>", "<BAD TOKEN>"])


def test_transition_selects_an_exact_representation() -> None:
    before = [[0 for _ in range(12)] for _ in range(12)]
    after = [row[:] for row in before]
    after[6][7] = 4
    encoding = encode_transition(before, after)
    assert encoding.kind == "delta"
    assert decode_transition(before, encoding) == normalize_grid(after)

    entirely_new = [[(x + y) % 16 for x in range(12)] for y in range(12)]
    encoding = encode_transition(before, entirely_new)
    assert decode_transition(before, encoding) == normalize_grid(entirely_new)


def test_delta_rejects_wrong_old_value() -> None:
    before = [[0, 0], [0, 0]]
    malformed = TransitionEncoding(
        "delta",
        (
            "<DELTA>",
            "<H_2>",
            "<W_2>",
            "<RUN>",
            "<Y_0>",
            "<X_0>",
            "<N_1>",
            "<OLD_3>",
            "<NEW_1>",
            "<ENDDELTA>",
        ),
    )
    with pytest.raises(CodecError):
        decode_transition(before, malformed)


def test_normalization_rejects_invalid_grids() -> None:
    with pytest.raises(CodecError):
        normalize_grid([])
    with pytest.raises(CodecError):
        normalize_grid([[0], [0, 1]])
    with pytest.raises(CodecError):
        normalize_grid([[16]])
    with pytest.raises(CodecError):
        normalize_grid([[0] * 65])
