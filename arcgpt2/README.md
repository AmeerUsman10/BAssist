# ARC-GPT2

**Constraint:** the only learned model used to act is the original GPT-2 family.

No vision encoder, external LLM, value network, random forest, symbolic planner, BFS/MCTS controller, or game-specific policy is permitted at inference. Deterministic code may only serialize an exact ARC frame, maintain the transcript, call the same GPT-2 checkpoint, validate its output grammar, and execute the selected legal action.

The project asks a precise question:

> Can GPT-2 be converted into an in-context learning algorithm for unseen interactive grid worlds, then carry that learned algorithm into ARC-AGI-3?

The chosen method is **Counterfactual Fracture Distillation**. Offline, exact algorithms and human replays produce learning histories containing exploration, failed hypotheses, resets, and eventual solutions. GPT-2 is trained autoregressively on those histories. At inference, the algorithms are gone: GPT-2 must reproduce the learning process from its own interaction history.

## Core design

One GPT-2 checkpoint is reused in several prompted modes:

- `NARRATE` — convert exact frame changes into a compact event description;
- `BELIEF` — update a bounded recurrent memory of action meanings, dynamics, hazards, goals, and failed prefixes;
- `IMAGINE` — predict the consequence of a candidate action;
- `ACT` — emit one legal ARC action, including coordinates for ACTION6;
- `REFLECT` — compress an ended attempt into memory before RESET.

All modes use the same weights and language-model head. Multi-pass deliberation is allowed because it is repeated inference from the same GPT-2, not an ensemble.

## First falsifiable gate

`phase0_hidden_action` tests whether fine-tuned GPT-2 can learn an action-remapping algorithm in context.

Each synthetic game contains several levels. The mapping from `A1..A4` to cardinal movement is randomly permuted once per game and remains fixed across levels. Frames, colors, obstacle layouts, goals, and permutations are held out during evaluation. The model receives only prior frames, actions, outcomes, and terminal signals.

The first gate passes only if:

1. intact-history GPT-2 solves held-out games materially better than an amnesic version given only the current frame;
2. later-level efficiency improves after the model has observed the action mapping;
3. the gain survives unseen color, layout, and action permutations;
4. a shuffled-history control destroys most of the gain;
5. pretrained GPT-2 is compared with an identically sized randomly initialized GPT-2 architecture.

This is deliberately narrower than ARC-AGI-3. It tests the missing primitive before spending GPU quota on the full benchmark.

## Repository layout

- `DESIGN.md` — complete research and engineering plan
- `codec.py` — deterministic lossless frame/delta serialization
- `phase0_hidden_action.py` — generated multi-level hidden-action environment and source learning histories
- `build_phase0_dataset.py` — reproducible train/validation/held-out data builder
- `train_phase0.py` — GPT-2 action-only objective and evaluation
- `tests/` — codec and environment invariants

## Status

The prior ARC work established useful exploration mechanisms but not broad transfer: the strongest repaired public policy solved 2/25 development-involved games and none of the other 23. ARC-GPT2 therefore starts from a clean locked split and treats any synthetic result only as a gate, never as an ARC score.
