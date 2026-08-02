"""Run Stage-0.2 with compact prompts and pretrained natural answer tokens."""

from __future__ import annotations

# Install compact exact serialization and natural-language prompts before the
# shared trainer imports protocol functions.
from . import stage02_natural as natural
from . import train_stage02 as trainer

_original_builder = trainer.build_model_and_tokenizer


def build_model_and_natural_tokenizer(config: trainer.TrainConfig):
    model, tokenizer, _ = _original_builder(config)
    label_token_ids: dict[str, int] = {}
    for canonical_label, surface in natural.LABEL_SURFACE.items():
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"natural Stage-0.2 surface must be one GPT-2 token: "
                f"{canonical_label!r} -> {surface!r} -> {token_ids}"
            )
        label_token_ids[canonical_label] = token_ids[0]
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise ValueError("natural Stage-0.2 surfaces are not distinct tokens")
    return model, tokenizer, label_token_ids


trainer.build_model_and_tokenizer = build_model_and_natural_tokenizer


def main() -> None:
    trainer.main()


if __name__ == "__main__":
    main()
