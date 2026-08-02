# Pure GPT-2 → ARC-AGI-3 Research Ledger

This is the durable decision record for the strict **one GPT-2** research program.

## Non-negotiable inference contract

The only learned component at inference is one standard GPT-2 causal language model and its original tokenizer. The model may be queried repeatedly, and its own answers may be copied into later prompts. There is no second neural network, LLM, retrieval system, learned value/reward model, ensemble, game-specific rule base, symbolic solver, search policy, or test-time optimizer.

Deterministic code may only:

- serialize an exact grid or literal frame delta without semantic interpretation;
- retain the literal transcript or copy GPT-2 outputs into fixed memory fields;
- restrict the final answer to environment-provided legal actions or coordinates;
- invoke the same GPT-2 again;
- execute the selected action and record the observed result.

Offline dataset generation and scoring may use environment truth, but that truth may never be consulted by the acting policy.

## Claim levels

1. **Execution:** code ran without error.
2. **Memorization:** a checkpoint can overfit controlled examples.
3. **Held-out component generalization:** one necessary operation transfers to unseen layouts or mappings.
4. **Closed-loop synthetic learning:** one GPT-2 improves from its own interaction history and completes unseen generated games.
5. **Public whole-game transfer:** a public ARC-AGI-3 game excluded entirely from training is completed under first-run conditions.
6. **Official private score:** only a valid competition evaluation may support this claim.

A lower level must never be reported as a higher one.

## Experiment ledger

### R-000 — Ordinary GPT-2 fine-tuning infrastructure

- Result: completed.
- Scope: 16 prompt/response examples, 12 optimizer steps, final two blocks trained.
- Evidence: real weight update and exported model artifact.
- Interpretation: infrastructure proof only; unrelated to ARC capability.

### R-010 — Original Phase-0 monolithic hidden-action policy

- Result: execution succeeded, capability gate failed.
- Intervention: opaque special-token codec, randomized palette, hidden action mapping, walls, memory, and action prediction in one next-token task.
- Failure: prediction collapse and no convincing held-out closed-loop learning.
- Decision: preserved as historical baseline; workflow is manual-only.

### R-020 — Stage-0.1 balanced/set-valued policy

- Result: execution succeeded, capability gate failed.
- Failure: lower loss without behavioral diversity; one-action collapse in classification and closed loop.
- Decision: loss reduction is not accepted as capability evidence.

### R-030 — Stage-0.2 decomposed dense protocol

- Hypothesis: one GPT-2 can separately infer four action meanings, infer a useful direction, compose those answers into an action, and outperform an amnesic ablation.
- Dense full run: GitHub Actions run `30725085305`.
- Status: pending at last ledger update.

### R-031 — Stage-0.2 micro diagnostic

- Run: `30725166280`.
- Execution: success.
- Capability gate: failed.
- Final behavior: mapping and needed direction collapsed to `N`; composition collapsed to action `4`; closed loop repeated action `4` for all 20 steps and completed zero levels.
- Important confound: only 80 optimizer steps, less than one effective epoch; 320-token context omitted some required evidence.

### R-032 — Stage-0.2 mapping-only diagnostic

- Run: `30725686400`.
- Execution: success.
- Capability gate: failed.
- Final behavior: predicted `?` on all 516 examples; accuracy remained 0.2403.
- Important confounds: only 160 sampled examples for a 516-example dataset and a 256-token context that retained required mapping evidence on only 47.96% of known-mapping rows.
- Decision: this result rejects the particular run, not GPT-2's ability to learn the component.

### R-033 — Token/context audit

- Run: `30725981952`.
- Execution: success.
- Findings:
  - natural answer surfaces such as ` north`, ` east`, ` one`, and ` two` are distinct single tokens in the original GPT-2 vocabulary;
  - dense prompts require 384 tokens for zero truncation in the current controlled curriculum;
  - 256 tokens retain required mapping evidence on only 47.96% of known rows;
  - 320 tokens retain 93.88%, still not a clean causal test;
  - the first sparse codec was longer than dense because its explanatory header was too verbose.
- Decision: future causal mapping tests use compact prompts, natural pretrained tokens, and an audited no-truncation budget.

### R-034 — Natural-token mapping overfit diagnostic

- Run: `30726132979`.
- Intervention: compact exact sparse codec, natural one-token outputs, 384-token budget, batch 16, 128 optimizer steps, final two blocks trainable.
- Status: pending at last ledger update.
- Gate: high, non-collapsed train/evaluation accuracy on the same controlled one-game mapping set. This is a memorization sanity check, not generalization.

### R-035 — Exact sparse Stage-0.2 full run

- Run: `30725476066`.
- Status: pending at last ledger update.
- Known confound: the workflow used a 256-token budget before the audit showed the sparse prompt can reach 411 tokens. Any failure is not a clean model-capability result.

### R-040 — Stage-0.3 isolated components

- Run: `30726260737`.
- Intervention: isolate four necessary operations using natural GPT-2 tokens and exact compact grids:
  1. raw before/action/after transition → movement direction;
  2. current mover/goal grid → useful direction;
  3. explicit action map + useful direction → action;
  4. explicit memory + raw current grid → action.
- Status: pending at last ledger update.
- Gates: held-out accuracy of 0.90, 0.90, 0.98, and 0.90 respectively, with no dominant-label collapse.
- Interpretation: separate checkpoints are diagnostic instruments only. The final agent must use one joint checkpoint.

## Current causal decision tree

1. **If natural mapping cannot overfit:** remove the unknown class, train only known single-transition direction examples, unfreeze more GPT-2 blocks, and verify the optimization path on a tiny fixed batch.
2. **If transition perception fails but text composition passes:** the bottleneck is grid/coordinate representation; redesign the exact codec and spatial curriculum before policy work.
3. **If all isolated components pass:** train one joint checkpoint, then run a closed loop where the same GPT-2 writes action meanings from each transition, reads the current direction, and composes the next action.
4. **If the joint checkpoint passes only with canonical memory:** introduce scheduled sampling so training progressively replaces canonical memory with GPT-2's own predicted memory.
5. **Only after unseen synthetic games are solved:** add walls, blocked moves, collection, toggles, coordinate actions, delayed effects, hidden goals, recurrent GPT-2-written memory, and whole-public-game leave-one-out evaluation.

## Scaling policy

- GPT-2 Small remains the experimental model until a held-out behavioral gate passes.
- Larger GPT-2 checkpoints, longer runs, or GPU use are not justified by lower loss alone.
- GPU execution requires explicit approval because earlier GPU use was stopped by the owner.

## Latest validated repository state

- 44 ARC-GPT2 tests passed in GitHub Actions run `30726231127`.
- Canonical branch: `main`.
- Exploratory branch `arc-gpt2-programs` is not automatically merged; its executable-posterior/planner experiments violate the strict runtime contract, and its pure soft-memory/program-induction gates have not shown causal gains.
