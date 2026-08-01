# Pure GPT-2 for ARC-AGI-3

## Research question

Can one original GPT-2 causal language model be transformed into an agent that learns an unseen interactive grid world from its own observation history and prediction errors?

This project deliberately excludes every other learned model. There is no vision encoder, planner network, value network, retrieval model, teacher LLM, model ensemble, or game-specific learned component.

The runtime is restricted to:

1. one `GPT2LMHeadModel` checkpoint;
2. one tokenizer;
3. a reversible grid codec;
4. environment plumbing that carries GPT-2's own memory text forward, constrains the final token to a legal action, and executes it.

The codec is lossless compression, not perception: it does not identify objects, goals, affordances, or useful regions.

## Stage 0: learn the act of learning

The first falsifiable milestone is a procedurally generated hidden-rule grid world:

- a player must reach a target;
- walls and layouts vary;
- colors vary;
- `ACTION1` to `ACTION4` are randomly remapped to north, south, west, and east in every episode;
- GPT-2 is not given the mapping;
- it must infer mappings from observed frame deltas, preserve them in its own recurrent memory packet, and act efficiently on held-out layouts and mappings.

Each supervised turn contains:

- the exact compressed frame;
- GPT-2's carried memory from the previous turn;
- the previous action and exact observed delta;
- legal actions;
- a target completion containing revised memory, counterfactual consequences for each action, and the next action.

The same GPT-2 generates all three outputs. At closed-loop evaluation, the generated memory—not oracle memory—is carried to the next turn.

## Why this milestone matters

A model that cannot infer four remapped actions in small held-out worlds has no credible path to ARC-AGI-3. A model that can do so provides evidence for the core mechanism: in-context policy improvement through a self-authored recurrent state.

The stage is not considered successful because training loss falls. It must beat random action selection in closed-loop held-out episodes and show better action selection after informative transitions.

## Files

- `protocol.py` — special-token vocabulary, reversible frame/delta codecs, recurrent prompt protocol, output parsing.
- `world.py` — procedural hidden-rule worlds and an oracle trace generator used only to produce training data.
- `generate_data.py` — deterministic train/validation/test creation with disjoint seed ranges.
- `train.py` — completion-masked fine-tuning of one GPT-2 checkpoint.
- `evaluate.py` — teacher-forced action accuracy and genuine closed-loop held-out play.
- `official_agent.py` — adapter skeleton for the official ARC-AGI-3 Agents repository; it still invokes only GPT-2 for learned decisions.
- `configs/stage0.json` — reproducible first-run configuration.
- `tests/test_core.py` — codec, delta, environment, and trace invariants.

## Reproducibility

```bash
python arc_gpt2/generate_data.py \
  --output-dir arc_gpt2/data/stage0 \
  --train-episodes 64 --val-episodes 12 --test-episodes 12

python arc_gpt2/train.py \
  --config arc_gpt2/configs/stage0.json

python arc_gpt2/evaluate.py \
  --model-dir outputs/arc-gpt2-stage0 \
  --data-dir arc_gpt2/data/stage0 \
  --output reports/arc_gpt2/stage0-evaluation.json
```

The GitHub Actions workflow performs the same sequence unattended, uploads the full checkpoint, and commits compact execution receipts and metrics back to `reports/arc_gpt2/`.

## Progression after Stage 0

Scaling is gated by evidence rather than model size:

1. remapped movement and recurrent memory;
2. sparse/dense 64×64 streaming reads;
3. randomized controllable-object identity;
4. keys, doors, switches, pushing, delayed effects, and irreversible hazards;
5. hidden and compositional goals;
6. complete failed-to-successful learning histories;
7. leave-one-environment-out evaluation on public ARC-AGI-3 games;
8. only then GPT-2 Medium, Large, and XL.

The final competition system remains one GPT-2 checkpoint. Small and Medium are development instruments, not an ensemble.

## Current claim boundary

The project begins with a synthetic capability gate. It does not yet claim an ARC-AGI-3 score or generalization to the private environments.
