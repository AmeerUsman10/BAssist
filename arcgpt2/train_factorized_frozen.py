"""Preserve GPT-2's lexical/spatial embeddings during factorized tuning.

The factorized task uses only GPT-2's original vocabulary, so there is no need
to relearn token embeddings from a small synthetic dataset. This wrapper freezes
word and positional embeddings for both pretrained and random-initialized
controls while retaining the same trainable upper transformer blocks.
"""

from __future__ import annotations

from . import train_factorized as base


_original_builder = base.build_model_and_tokenizer


def _frozen_builder(config):
    model, tokenizer = _original_builder(config)
    for parameter in model.transformer.wte.parameters():
        parameter.requires_grad = False
    for parameter in model.transformer.wpe.parameters():
        parameter.requires_grad = False
    return model, tokenizer


base.build_model_and_tokenizer = _frozen_builder

Config = base.Config
evaluate = base.evaluate
score_completions = base.score_completions
select_mapping = base.select_mapping
train = base.train


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
