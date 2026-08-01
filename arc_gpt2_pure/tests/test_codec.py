import pytest

from arc_gpt2.codec import (
    apply_delta,
    decode_grid,
    encode_delta,
    encode_grid,
    extract_last_grid,
)


def test_grid_round_trip() -> None:
    grid = [
        [0, 0, 0, 1, 1, 1],
        [0, 2, 0, 0, 3, 0],
        [4, 4, 4, 4, 4, 4],
    ]
    encoded = encode_grid(grid)
    assert encoded == "G[3,6]:0*3,1*3/0,2,0*2,3,0/4*6"
    assert decode_grid(encoded) == grid


def test_delta_round_trip() -> None:
    before = [[0, 2, 0], [0, 0, 3]]
    after = [[0, 0, 2], [0, 0, 3]]
    encoded = encode_delta(before, after)
    assert encoded == "D[2,3]:0,1=0;0,2=2"
    assert apply_delta(before, encoded) == after


def test_same_delta() -> None:
    grid = [[0, 1], [2, 3]]
    assert encode_delta(grid, grid) == "D[2,2]:SAME"
    assert apply_delta(grid, "D[2,2]:SAME") == grid


def test_extract_last_animation_grid() -> None:
    first = [[0, 2], [0, 3]]
    second = [[2, 0], [0, 3]]
    assert extract_last_grid([first, second]) == second
    assert extract_last_grid(second) == second


def test_invalid_nonrectangular_grid() -> None:
    with pytest.raises(ValueError):
        encode_grid([[0, 1], [2]])


def test_delta_rejects_out_of_bounds_coordinate() -> None:
    with pytest.raises(ValueError):
        apply_delta([[0]], "D[1,1]:1,0=2")
