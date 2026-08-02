# Stage 0.2 — One GPT-2, Decomposed In-Context Control

## Why Stage 0.1 was rejected

Stage 0.1 executed successfully but did not demonstrate a learned controller. The model collapsed to one action on every classification example and every closed-loop step. The one-game micro diagnostic also collapsed. Lower set-valued loss therefore was not treated as capability evidence.

The likely confounds were introduced simultaneously:

- hundreds of newly appended opaque protocol and codec tokens;
- randomized palette, action semantics, walls, memory, and navigation;
- one next-token decision forced to parse transitions, recover action meanings, locate the mover and goal, plan, and choose an action;
- probe supervision whose first decision intentionally supplied no action-selection gradient;
- a direct-policy objective that could be reduced by global action bias.

## Stage-0.2 intervention

The acting system remains one ordinary GPT-2 causal language model with the original GPT-2 tokenizer and embedding table. It is queried multiple times, but every answer comes from the same weights:

1. infer the direction of action 1;
2. infer the direction of action 2;
3. infer the direction of action 3;
4. infer the direction of action 4;
5. infer the currently needed direction;
6. compose those GPT-2 answers into a proposed action using another GPT-2 call;
7. independently produce a direct action as an ablation.

The harness performs no planning. It losslessly serializes the grid and observed changes, masks the answer to a fixed legal one-character label set, copies GPT-2's first-call answers into the composition prompt, and executes GPT-2's selected action.

## Controlled world

Stage 0.2 deliberately uses:

- cells `0=empty`, `1=wall`, `2=mover`, `3=goal`;
- no walls;
- three levels per game;
- a different permutation of four cardinal actions in each game;
- exact raw row serialization and exact changed-cell coordinates;
- all 65 ordered distinct action-history prefixes;
- 64 repeat-recovery histories;
- held-out action permutations and layouts.

This is not intended as an ARC score. It is a causal gate: the static initial frame cannot reveal action meanings, so success requires extracting them from the literal in-context transitions.

## Training objective

No new special tokens are added. Targets are existing single-character GPT-2 tokens:

- mapping and direction: `N E S W ?`;
- action: `1 2 3 4`.

Training uses candidate-only cross-entropy, equal total sampling mass for mapping, needed-direction, composition, and direct-action tasks, and equal target mass within each task. GPT-2's original embedding table is frozen; only upper transformer blocks and the final layer norm are updated.

## Evidence requirements

Stage 0.2 is not passed by loss reduction. Required evidence is:

- high held-out mapping, direction, and composition accuracy;
- positive closed-loop level completion on unseen layouts/action permutations;
- intact history outperforming the same GPT-2 with history removed;
- no single-action collapse;
- literal action traces and probability distributions preserved in the report.

The first full and micro experiments are defined in:

- `.github/workflows/arc-gpt2-stage02.yml`
- `.github/workflows/arc-gpt2-stage02-micro.yml`

Durable receipts are written to:

- `reports/arcgpt2-stage02/`
- `reports/arcgpt2-stage02-micro/`

## Next expansion, only after the gate

1. semantic-free sparse/dense codec selection;
2. random palette and role inference;
3. walls and blocked-action inference;
4. collection, toggles, coordinate actions, and delayed effects;
5. hidden goals and across-attempt learning histories;
6. GPT-2-written recurrent memory for the 1,024-token context limit;
7. whole-game leave-one-out evaluation on public ARC-AGI-3 environments.
