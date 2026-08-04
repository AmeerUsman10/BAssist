"""String wrappers around the token-level ARC-GPT2 codec.

The core codec deliberately returns token lists. Evidence records and official
FrameData adapters need stable single-line strings, so this module provides the
lossless conversion without changing the token-level API used by training code.
"""

from __future__ import annotations

from collections.abc import Sequence

from .codec import Grid, apply_delta, decode_frame, encode_delta, encode_frame, tokens_to_text


def _tokens(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return value.split()
    return [str(token) for token in value]


def encode_grid(grid) -> str:
    """Return one reversible full-grid string."""
    return tokens_to_text(encode_frame(grid))


def decode_grid(value: str | Sequence[str]) -> Grid:
    """Decode a grid string or an existing token sequence."""
    return decode_frame(_tokens(value))


def encode_delta_text(previous, current) -> str:
    """Return one reversible delta string."""
    return tokens_to_text(encode_delta(previous, current))


def decode_delta(previous, value: str | Sequence[str]) -> Grid:
    """Apply a delta string or token sequence to a previous frame."""
    return apply_delta(previous, _tokens(value))
