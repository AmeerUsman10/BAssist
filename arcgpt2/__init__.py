"""GPT-2-only research code for ARC-AGI-3."""

from .codec import CodecError, TransitionEncoding, encode_frame, encode_transition

__all__ = ["CodecError", "TransitionEncoding", "encode_frame", "encode_transition"]
