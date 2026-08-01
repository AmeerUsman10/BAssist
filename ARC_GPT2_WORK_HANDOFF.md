# Work-chat handoff: Pure GPT-2 ARC-AGI-3

Prepared: 2026-08-02  
Repository: `AmeerUsman10/BAssist`  
Branch: `main`

## Objective and non-negotiable constraint

Build a solver for ARC-AGI-3 in which the **only learned model used for inference is one checkpoint from the original GPT-2 family**.

Allowed: one `GPT2LMHeadModel`, one tokenizer, repeated calls to the same checkpoint, a reversible semantic-free grid codec, transcript/context management, legal-action grammar validation, and environment plumbing.

Forbidden: any second learned model, external LLM, vision encoder, value/reward model, retrieval model, learned reranker, ensemble, game-specific policy, or symbolic/BFS/MCTS planner that selects actions at inference.

The current target is not to imitate a chat assistant. It is to distil an in-context learning algorithm into GPT-2 so the same weights observe action consequences, preserve beliefs in their own token stream, revise those beliefs, and choose the next action.

## What is complete

### Shared infrastructure

- A real pretrained GPT-2 124M download/fine-tuning/export path has run successfully.
- GitHub Actions can train unattended on CPU, upload checkpoints, and commit compact evidence receipts.
- The repository contains two independently implemented Stage/Phase-0 lines. They should now be consolidated rather than expanded in parallel.

### `arcgpt2/` — clean experimental/control line

Implemented:

- strict purity contract and full design document;
- exact reversible RLE/quadtree/delta grid codec;
- multi-level hidden-action environment with per-game randomized action mapping;
- reproducible train/validation/test generation;
- action-only GPT-2 objective;
- closed-loop intact-history, amnesic, and shuffled-history controls;
- pretrained-versus-random-initialization comparison;
- self-reporting Phase-0 workflow;
- 13 passing unit tests.

### `arc_gpt2/` — richer recurrent-protocol line

Implemented:

- reversible frame/delta protocol;
- GPT-2-authored `<MEM>` mapping state;
- generated counterfactual span and `<ACT>` span from the same LM head;
- procedural hidden-rule worlds and complete learning traces;
- completion-masked training;
- teacher-forced and genuine closed-loop evaluation;
- an `official_agent.py` adapter skeleton;
- self-reporting Stage-0 workflow and checkpoint artifact.

## Verified evidence

### `arcgpt2/` Phase 0

The workflow completed successfully as infrastructure, but the capability gate failed.

- held-out action classification accuracy: `0.34597` versus `0.25` random chance;
- intact-history closed-loop: `0/4` games, `0` levels;
- amnesic closed-loop: `0/4` games, `0` levels;
- shuffled-history closed-loop: `0/4` games, `0` levels;
- pretrained and random-initialized models had the same measured closed-loop result;
- the policy collapsed to repeatedly emitting `A3`.

Gate status: `not_yet_passed`.

### `arc_gpt2/` Stage 0

Training completed and loss fell substantially, but behavior did not improve.

- initial validation loss: `1.53509`;
- final validation loss: `0.30000`;
- teacher-forced action accuracy: `9/48 = 0.1875`, below the `0.25` random-action baseline;
- model closed-loop: `0/4` wins;
- random closed-loop: `2/4` wins;
- oracle trace: `4/4` wins with `5.25` mean actions;
- generated memory remained effectively all `<UNK>` and the policy repeatedly emitted `A1`.

## Claim boundary

There is **no official ARC-AGI-3 score** from this GPT-2 line. The successful workflow status means the code executed and artifacts were produced; it does not mean the research gate passed.

The strongest honest conclusion is:

> The pure GPT-2 infrastructure is operational, but neither current Stage-0 formulation has yet learned the required in-context action-remapping algorithm.

## Diagnosis

1. **Irreducible teacher noise:** the `arcgpt2` source learner uses a per-game randomized probe order. Early probe targets can therefore be arbitrary rather than inferable from the transcript.
2. **Action collapse:** both systems converge to one frequent action (`A3` or `A1`). Current reports lack strong class-balance and output-entropy gates.
3. **Loss/behavior mismatch:** in `arc_gpt2`, the model can reduce completion loss by learning easy protocol text while still failing the action decision and memory update.
4. **Premature protocol complexity:** memory plus counterfactual generation was attempted before basic action-mapping recovery was demonstrated.
5. **Smoke-scale evaluation:** four closed-loop episodes are enough to expose failure, not enough to establish a positive result.

## Canonicalization decision

Continue from **`arcgpt2/` as the canonical scientific line** because it has the cleanest falsifiable controls and isolates the action decision. Keep `arc_gpt2/` intact as an experimental archive and later port only proven components such as the official adapter and recurrent-memory protocol.

Do not delete either directory until the useful pieces have been reconciled.

## Immediate Stage-0.1 experiment

The next work chat should do this before adding mechanics, hidden goals, public ARC games, online LoRA, or larger GPT-2 checkpoints:

1. Remove arbitrary probe-order targets. Use a deterministic probe curriculum or train against the set of equally valid informative actions rather than one random tie-break.
2. Report target counts by action and decision stage; enforce a balanced action distribution in training/evaluation.
3. Add confusion matrix, action entropy, repeated-action rate, mapping-recovery accuracy, and per-level efficiency.
4. Separate the probe/identification phase from the navigation phase in metrics.
5. Create an **overfit sanity gate**: eight fixed games must reach near-perfect teacher-forced action accuracy and at least 90% closed-loop completion. Failure here means a software/objective bug, not insufficient scale.
6. Then run locked held-out games across at least five training seeds and compare intact, amnesic, shuffled, pretrained, and random initialization.
7. Promote only if intact history materially beats both amnesia and shuffled history, later levels become more efficient, and pretrained GPT-2 beats the randomly initialized architecture.

## Stop conditions

Do not move to GPT-2 Medium, real ARC-AGI-3 evaluation, or Kaggle-scale training merely because loss decreases. Stage 0.1 must first demonstrate genuine history-dependent closed-loop improvement.

Do not report synthetic completion as an ARC score.

## Durable evidence paths

- `arcgpt2/DESIGN.md`
- `arcgpt2/README.md`
- `reports/arcgpt2/latest-unit-tests.json`
- `reports/arcgpt2/latest-phase0-status.json`
- `reports/arcgpt2/latest-phase0-comparison.json`
- `arc_gpt2/README.md`
- `reports/arc_gpt2/latest-status.json`
- `reports/arc_gpt2/latest-training-summary.json`
- `reports/arc_gpt2/stage0-evaluation.json`

## Instruction to the receiving work chat

Take over from this file and the durable reports. Do not restart the project from a generic GPT-2 notebook. First repair Stage 0.1, make the overfit sanity gate pass, and commit every result with exact configuration, seed, checksum, raw traces, and a literal pass/fail interpretation. Preserve the strict one-GPT-2 inference contract throughout.
