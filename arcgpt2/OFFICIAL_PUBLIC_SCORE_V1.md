# Official ARC-AGI-3 Public Score V1

## Decision

`official_public_score_v1` is one bounded measurement of the current
one-original-GPT-2 system on the official public ARC-AGI-3 environments.  It
is the first repository experiment allowed to report the official scorecard
formula.  It is not a competition submission, an online leaderboard run, a
locked transfer result, or evidence about private games.

The prior rank-one EGGROLL experiment is not promoted into this run.  It
retained only 0.189857 of the exact-gradient probability gain against a frozen
minimum of 0.50.  This protocol uses the independently validated exact-gradient
temporary memory instead.

## Frozen source and dependencies

- stacked base commit: `11c2f62786744cb772a7fae359448a58c6d4e3f3`;
- model: `openai-community/gpt2` revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- tokenizer: the tokenizer shipped at the same model revision;
- official toolkit: `arc-agi==0.9.9` on Python 3.12;
- engine floor selected by that toolkit: `arcengine==0.9.3`;
- public environment manifest: `official_public_manifest.json`.

The runner must resolve and record every installed version and hash its source,
manifest, result, receipt, and public trace.  It may not persist trained model
weights or tokenizer files.

## Model preparation and canary

The runner reproduces the known-positive A1-only training/validation phase:

- 256 synthetic training layouts and 64 disjoint validation layouts;
- two epochs and exactly 512 AdamW steps;
- original GPT-2 Small initialization only;
- first 11 transformer blocks frozen;
- prefix length 8;
- model learning rate `1e-4`, prefix learning rate `1e-3`;
- exact online prefix step size `0.2`;
- no old locked synthetic split evaluation.

Before opening the public scorecard, all training values must be finite and the
validation checkpoint must have accuracy at least 0.70 and truth probability
at least 0.60.  Failure aborts before any public-game action.

The selected model weights are then frozen.  Game-specific learning may modify
only an in-memory soft prefix using exact observations made during that game.

## Public score run

- `OperationMode.NORMAL` only; `ONLINE` and `COMPETITION` are forbidden.
- Verify the available versioned environment IDs equal the frozen 25-game
  manifest before taking a scored action.
- One local official scorecard, one deterministic run per game, seed 0.
- Fixed manifest order, at most 160 environment actions and three resets per
  game.
- No game-ID branches, game-source inspection, manual intervention, retry after
  seeing a score, threshold change, or post-score tuning.

The primary score is the toolkit's official local scorecard: per completed
level, `min(115, 100 * (human_actions / agent_actions)^2)`; levels are weighted
by their one-indexed level number; the total is the mean game score.

## Frozen policy

The deterministic shell treats actions as opaque and observations as exact
state.  It may:

1. preserve every animation frame and the persistent final frame;
2. maintain an exact state/action transition graph;
3. prefer an untried legal action at the current state;
4. prefer actions with fewer trials and more previously novel transitions;
5. replay a known reversible edge to reach an unresolved frontier;
6. validate legality, count actions, and enforce reset/budget limits;
7. ask the one trained GPT-2 memory to score the direction of simple actions
   after eligible exact transition observations.

For complex coordinate actions, the fixed semantic-free schedule is:

1. representatives of non-modal-color connected components in row-major
   order;
2. deterministic low-discrepancy coverage of the complete frame.

These rules do not assign objects or goals.  They expose the first direct score
of a generic observation-driven frontier policy with the validated GPT-2
action-memory channel.

## Required preflight

All conditions are conjunctive:

- codec and official-observation round trips pass, including animation and
  integer official-action normalization;
- the complete ARC-GPT2 test suite collects and passes;
- fake-environment policy tests cover determinism, simple and complex actions,
  level transitions, resets, budgets, and strict receipts;
- the pinned official SDK imports under Python 3.12;
- runner-source injection and dependency/model/manifest pins are exact;
- no `ONLINE`, `COMPETITION`, or competition-submission path is reachable;
- the artifact scan blocks checkpoint and model payloads while allowing text
  manifests;
- a timed projection remains below 10,800 seconds.

## Evidence and interpretation

The run must preserve a strict-JSON summary, sanitized official scorecard,
environment manifest, training canary, transition JSONL, environment receipt,
model manifest, runner receipt, failure receipt when applicable, and a complete
artifact hash manifest.

A zero score is a valid result.  Any nonzero score establishes only public
development-game level completion under this exact fixed policy.  It does not
establish locked generalization or competition performance.
