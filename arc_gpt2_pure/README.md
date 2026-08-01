# Pure GPT-2 for ARC-AGI-3

This directory is an executable research program for one deliberately strict question:

> Can the original GPT-2 causal language model be trained to learn and act in an unseen interactive grid world from the trajectory in its own context?

## Purity contract

At inference time the only learned component is one standard GPT-2 causal language model and its original tokenizer. There is no second neural network, vision model, retrieval system, symbolic solver, search policy, reward model, LLM teacher, model ensemble, or game-specific rule base.

Deterministic runtime code is limited to lossless grid serialization, maintaining the literal interaction transcript, constraining a final choice to actions exposed by the environment, invoking the same GPT-2 more than once when needed, and converting its selected action into the ARC API type.

## First experiment: remapped-control in-context learning

Version 0 does not claim to solve ARC-AGI-3. It establishes the first necessary capability under controlled conditions:

1. Each episode randomly permutes the meanings of `A1` through `A4`.
2. GPT-2 observes zero to four state/action/next-state transitions.
3. It writes a compact memory of the action mapping.
4. It selects either the shortest-path action toward a goal or an untested action when the required direction remains unknown.
5. It predicts the state delta caused by the selected action.
6. Evaluation uses unseen layouts, action permutations, and random seeds.

The crucial metric is end-to-end action accuracy using GPT-2's own generated memory. Training loss alone is not accepted as evidence.

## Repository layout

- `arc_gpt2/codec.py` — reversible frame and delta codec.
- `arc_gpt2/curriculum.py` — procedural hidden-action environments and datasets.
- `arc_gpt2/protocol.py` — prompt protocol and strict output parsing.
- `arc_gpt2/modeling.py` — GPT-2 loading, candidate scoring, memory generation, and evaluation.
- `arc_gpt2/train.py` — masked causal-LM training and before/after evaluation.
- `arc_gpt2/official_agent.py` — adapter for the official ARC-AGI-3 agent interface.
- `tests/` — codec, curriculum, protocol, and tiny-model tests.
- `kaggle/` — GPU runner material; no credentials are stored in this repository.

## Current scientific gates

- **G0 — infrastructure:** tests pass and a pretrained GPT-2 checkpoint can train and save reproducibly.
- **G1 — codec:** every generated frame and delta round-trips exactly.
- **G2 — remapped controls:** tuned GPT-2 beats both random choice and the untouched GPT-2 checkpoint on unseen episodes.
- **G3 — self-memory:** action accuracy using generated memory remains materially above action accuracy with memory removed.
- **G4 — partial information:** GPT-2 probes an unobserved action when the shortest-path direction has not yet been identified.
- **G5 — composition:** add collections, keys, toggles, coordinate actions, delayed effects, and hidden terminal conditions only after G2-G4 are met.
- **G6 — ARC transfer:** integrate public ARC-AGI-3 recordings and use whole-game leave-one-out evaluation.

A successful workflow writes exact configuration, hashes, losses, baseline/tuned action accuracy, generated examples, and the downloadable model artifact. A failed workflow writes its log tail rather than silently disappearing.

## Run locally

```bash
cd arc_gpt2_pure
python -m pip install -e .[dev]
python -m arc_gpt2.curriculum --train-out data/train.jsonl --eval-out data/eval.jsonl
python -m arc_gpt2.train \
  --train-file data/train.jsonl \
  --eval-file data/eval.jsonl \
  --output-dir outputs/gpt2-arc-v0 \
  --max-steps 40 \
  --freeze-first-n-blocks 10
```

The project intentionally begins with GPT-2 Small. Model size is increased only after a measured held-out gain.
