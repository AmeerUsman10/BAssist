# ARC-GPT2 Experiment Matrix

The repository now contains two intentionally independent GPT-2-only lines:

- `arc_gpt2/` — **direct recurrent agent**. GPT-2 writes its own bounded memory,
  counterfactuals, and action; deterministic code only encodes observations,
  parses output, and executes the action.
- `arcgpt2/` — **history policy and executable-program agent**. GPT-2 either
  predicts the action from a learning history or induces ARC-DSL programs that
  are checked and planned through by deterministic code.

They should not be merged conceptually before matched evidence exists. Their
independent failure modes are useful.

## What “just GPT-2” means in every line

- exactly one original GPT-2-family checkpoint is the learned model at inference;
- no external LLM, vision encoder, value model, reward model, embedding model,
  classifier, reranker, or model ensemble;
- no game-specific solution branches;
- the random-initialized GPT-2 architecture is a mandatory control;
- deterministic code may losslessly encode data, constrain grammar, check exact
  consequences, and enforce legal actions;
- any stronger deterministic operation must be reported explicitly rather than
  hidden inside the phrase “GPT-2 solved it.”

## Matched systems

| ID | Learned decision mechanism | Deterministic assistance | Scientific question |
|---|---|---|---|
| D0 | Raw GPT-2 action likelihood | legal-action mask | Does unmodified GPT-2 contain any useful prior? |
| D1 | Fine-tuned direct action GPT-2 | codec + legal mask | Can a static policy transfer? |
| D2 | Recurrent-memory GPT-2 | codec + memory parser | Can GPT-2 improve in context from its own history? |
| D3 | Multi-pass recurrent-memory GPT-2 | repeated tied inference | Does more GPT-2 compute improve decisions? |
| P1 | GPT-2 ranks executable programs | finite candidate enumeration + replay | Can GPT-2 infer an exact latent transition program? |
| P2 | GPT-2 generates executable programs | grammar + compiler + replay | Can it synthesize beyond a finite family? |
| P3 | GPT-2 program posterior | compiler + generic search + disagreement | Does verified planning improve completion/efficiency? |
| P4 | GPT-2 program repair | exact counterexample extraction | Can it revise rather than restart after contradiction? |
| Z1 | GPT-2 with per-game soft-prefix particles | gradient update on real transitions | Can one checkpoint maintain useful latent hypotheses? |

## Required controls

Every claimed gain must include the relevant controls below.

### Information controls

- intact interaction history;
- current observation only;
- action labels shuffled while observations remain intact;
- transitions shuffled across games;
- terminal labels removed;
- recurrent memory replaced by an equally sized random or stale memory;
- exact frame versus compressed frame.

### Model controls

- pretrained GPT-2;
- same GPT-2 architecture from random initialization;
- frozen GPT-2 with only new token embeddings trained;
- full or selective fine-tuning under matched optimizer steps;
- GPT-2 Small versus Medium only after Small passes a transfer gate.

### Tool controls

- direct action without programs;
- generated program without replay checking;
- replay checking without generic planning;
- planning with the true program as an upper bound;
- planning with a random consistent program;
- myopic disagreement versus reachable fracture search.

## Evaluation ladder

### L0 — representation validity

- every frame and delta round-trips exactly;
- every generated DSL program is grammar-valid or rejected;
- replay mismatches identify an exact action and differing cells;
- no semantic label is introduced by the codec.

### L1 — hidden action semantics

Random action remapping across multi-level games. Passing requires:

- intact history beats amnesia and shuffled history;
- action efficiency improves after informative transitions;
- held-out action permutations, palettes, and layouts;
- pretrained GPT-2 compared with random initialization.

### L2 — exact executable program induction

After a bounded evidence sequence, GPT-2 selects or generates the correct
transition program. Passing requires:

- exact program or behavioral-equivalence accuracy;
- exact replay on held-out transitions;
- downstream level completion using the selected program;
- full evidence materially beats amnesic/shuffled controls.

### L3 — contradiction and repair

Programs are near-correct but fail one or more transitions. Passing requires:

- the repaired program preserves all previously correct behavior;
- the repair resolves the counterexample;
- minimal repair length is reported;
- repair beats independent resampling under matched GPT-2 calls.

### L4 — compositional mechanics and latent goals

Hold out mechanic compositions and goal predicates, not just seeds. Measure:

- behavioral program equivalence;
- goal posterior calibration;
- recovery after terminal failure;
- cross-level transfer;
- action efficiency relative to a known-program oracle.

### L5 — locked public ARC-AGI-3 games

A result counts only when the target game has contributed no trajectory, human
replay, environment-derived label, hand-written branch, or mutation to training
or development. All prompts, checkpoints, seeds, action budgets, and hashes are
frozen before the run.

## Promotion rules

1. Do not promote a lower validation loss.
2. Do not promote a synthetic win without the matched information controls.
3. Do not promote a development-involved public game as transfer.
4. Do not move from Small to Medium until at least one learning/induction gate
   passes on locked generated families.
5. Do not submit to official evaluation until a frozen configuration improves
   at least one locked public game or materially improves efficiency across
   several locked games.
6. Preserve negative results. A direct-agent failure can still justify the
   program route; a program-route failure can still show that generic search,
   not GPT-2 pretraining, supplied the value.

## Current status

- exact semantic-free codec: implemented and tested;
- hidden-action multi-level generator: implemented and tested;
- direct action-history training/evaluation: implemented; smoke execution open;
- recurrent-memory direct agent: independently implemented under `arc_gpt2/`;
- initial ARC-DSL kernel, exact replay, hypothesis elimination, disagreement,
  and planning: implemented and tested;
- finite-family program-induction data/training/evaluation: implemented; smoke
  execution open;
- official ARC-AGI-3 transfer: not yet demonstrated.
