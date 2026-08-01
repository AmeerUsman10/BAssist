"""Pure GPT-2 experiments for adaptive ARC-AGI-3 play."""

from .codec import apply_delta, decode_grid, encode_delta, encode_grid
from .curriculum import MotionEpisode, generate_episode

__all__ = [
    "MotionEpisode",
    "apply_delta",
    "decode_grid",
    "encode_delta",
    "encode_grid",
    "generate_episode",
]

__version__ = "0.1.0"
