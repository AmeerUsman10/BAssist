"""Train one Stage-0.3 GPT-2 component without running a policy evaluator.

This is a diagnostic entry point, not a separate learned system.  It uses the
same standard GPT-2 and natural one-token answer surfaces as the eventual agent.
Closed-loop evaluation is deliberately skipped because a single isolated
component cannot act by itself; only held-out classification is claimed.
"""

from __future__ import annotations

from typing import Any

# Installs natural candidate token IDs into the shared Stage-0.2 trainer.
from . import train_stage02_natural as _natural_trainer  # noqa: F401
from . import train_stage02 as trainer


def skipped_closed_loop(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        "history_mode": str(kwargs.get("history_mode", "skipped")),
        "games": 0,
        "games_won": 0,
        "game_win_rate": 0.0,
        "levels_completed": 0,
        "level_completion_rate": 0.0,
        "total_actions": 0,
        "mean_actions_per_game": 0.0,
        "composed_valid_action_accuracy": 0.0,
        "direct_oracle_action_accuracy": 0.0,
        "mapping_slot_accuracy": 0.0,
        "need_accuracy": 0.0,
        "repeated_action_transition_rate": 0.0,
        "predicted_action_counts": {},
        "dominant_action_share": 0.0,
        "prediction_distribution_entropy": 0.0,
        "traces": [],
        "note": "closed loop intentionally skipped for isolated component diagnostic",
    }


trainer.evaluate_closed_loop = skipped_closed_loop


def main() -> None:
    trainer.main()


if __name__ == "__main__":
    main()
