"""High-signal Phase-0 mapping-induction runner.

The original executable-program smoke runner trained and scored full ARC-DSL
program strings. This wrapper retains the same GPT-2 training loop, controls,
candidate programs, exact replay, and downstream planner, but scores each
candidate through its compact atomic action mapping. The selected mapping still
indexes a complete executable Program.

Keeping this as an explicit runner preserves the first full-program experiment
for comparison instead of silently rewriting its evidence path.
"""

from __future__ import annotations

from . import train_program_induction as base
from .mapping_target import compact_mapping


# `score_candidates` resolves this module-level function dynamically from the
# imported base module. Replacing it before `base.main()` makes both initial and
# final candidate ranking use the same compact target as the v2 dataset.
base._candidate_target = compact_mapping

Config = base.Config
ProgramDataset = base.ProgramDataset
build_model_and_tokenizer = base.build_model_and_tokenizer
evaluate_program_selection = base.evaluate_program_selection
score_candidates = base.score_candidates
train = base.train


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
