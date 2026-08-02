"""Run the Stage-0.2 trainer with the semantic-free sparse grid codec."""

from __future__ import annotations

# Import for its deliberate protocol installation side effect before importing
# the shared trainer.  The learned component remains exactly one GPT-2.
from . import stage02_sparse as _stage02_sparse  # noqa: F401
from .train_stage02 import main


if __name__ == "__main__":
    main()
