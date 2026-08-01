from __future__ import annotations

import random

from arc_gpt2.protocol import (
    apply_delta,
    atom_count,
    chunk_protocol,
    decode_grid,
    encode_delta,
    encode_grid,
    extract_action,
    extract_memory,
)
from arc_gpt2.world import HiddenRuleGrid, generate_episode


def test_grid_codec_round_trips_random_grids() -> None:
    rng = random.Random(7)
    for _ in range(100):
        height = rng.randint(1, 64)
        width = rng.randint(1, 64)
        palette = rng.sample(range(16), rng.randint(1, 6))
        grid = [[rng.choice(palette) for _ in range(width)] for _ in range(height)]
        encoded = encode_grid(grid)
        assert decode_grid(encoded) == grid


def test_grid_codec_handles_sparse_and_dense_extremes() -> None:
    sparse = [[0 for _ in range(64)] for _ in range(64)]
    sparse[1][2] = 7
    sparse[63][63] = 15
    dense = [[(x + y) % 16 for x in range(64)] for y in range(64)]
    assert decode_grid(encode_grid(sparse)) == sparse
    assert decode_grid(encode_grid(dense)) == dense


def test_delta_round_trip() -> None:
    previous = [[0, 0, 1], [2, 2, 2], [0, 0, 0]]
    current = [[0, 4, 1], [2, 0, 2], [0, 0, 9]]
    delta = encode_delta(previous, current)
    assert apply_delta(previous, delta) == current
    assert apply_delta(current, encode_delta(current, current)) == current


def test_chunk_protocol_is_bounded() -> None:
    grid = [[(x + y) % 16 for x in range(64)] for y in range(64)]
    chunks = chunk_protocol(encode_grid(grid), max_atoms=128)
    assert len(chunks) > 1
    assert all(atom_count(chunk) <= 134 for chunk in chunks)


def test_output_parsers_are_strict() -> None:
    text = "noise<MEM><MAP><A1><N><A2><UNK></MEM><ACT><A2></ACT>tail"
    assert extract_memory(text) == "<MEM><MAP><A1><N><A2><UNK></MEM>"
    assert extract_action(text, (1, 2, 3, 4)) == 2
    assert extract_action("<ACT><A7></ACT>", (1, 2, 3, 4)) is None


def test_generated_worlds_are_reachable_and_action_mappings_are_permutations() -> None:
    for seed in range(50):
        world = HiddenRuleGrid.generate(seed)
        assert world.shortest_path_directions()
        assert set(world.action_to_direction) == {1, 2, 3, 4}
        assert set(world.action_to_direction.values()) == {"N", "S", "W", "E"}


def test_oracle_learning_histories_complete_held_out_worlds() -> None:
    for seed in range(1000, 1040):
        records = generate_episode(seed, max_steps=40)
        assert records
        assert records[-1]["metadata"]["completed"] is True
        assert all("<MEM>" in str(record["completion"]) for record in records)
        assert all("<CF>" in str(record["completion"]) for record in records)
        assert all("<ACT>" in str(record["completion"]) for record in records)
