# The GPT-2-Only ARC Core

## Decision

The primary route is no longer “fine-tune GPT-2 to emit the next action.” The
first closed-loop direct-policy smoke result was worse than random, and its
output collapsed to a single action. The result is useful: it falsifies the
simplest route at the tested scale.

The primary route is now:

> **One GPT-2 checkpoint acts as a calibrated scorer and generator over a live
> version space of executable world hypotheses.**

No other learned model is permitted. Deterministic code may expose exact grid
facts, enforce types and grammar, execute GPT-2-authored hypotheses, calculate
probabilities, and carry out generic search through consequences predicted by
those hypotheses.

This is still “just GPT-2” in the consequential sense: GPT-2 is the only source
of semantic claims, latent variables, programs, memories, predicted values, and
action preferences. The rest is a calculator and compiler.

## Why direct imitation was structurally wrong

### 1. Early exploration is set-valued

Before an action has been tested, several actions can be equally valid probes.
Training one arbitrary expert choice as the only correct label creates false
supervision. A language model then learns a tie-breaking habit rather than an
uncertainty-aware learning algorithm.

### 2. A monolithic target dilutes the useful signal

A full program contains much syntax and many fields shared by every candidate.
Ordinary next-token loss can fall while the few tokens representing the actual
latent rule remain wrong.

### 3. A generated memory can become a self-confirming error

If the same model emits one confident textual theory, then conditions its next
action on that theory, an early mistake is repeatedly amplified. The runtime
must preserve alternative explanations until evidence eliminates them.

### 4. Static policy training does not teach learning

ARC-AGI-3 rewards improvement during an unseen game. Training isolated
state-action examples teaches an average policy, not a procedure that updates
from intervention outcomes.

## The epistemic state

At time `t`, the agent carries a version space over typed latent variables:

```text
action semantics
controlled entity / selector
collision policy
contact effects
persistent inventory or toggles
temporal phase / counters
goal predicate
level-local geometry versus game-global mechanics
```

GPT-2 supplies a log score for each candidate value using ordinary-language or
compact DSL completions. Deterministic constraints combine those local scores
into a joint posterior. Examples:

- the four movement actions form a one-to-one permutation;
- one color cannot simultaneously be both a wall and the unique controlled
  entity unless a candidate program explicitly models that overlap;
- a candidate transition program must replay every recorded transition;
- a candidate goal predicate must agree with every observed terminal and
  non-terminal state;
- persistent facts must remain valid across level transitions.

The posterior is not forced to select one story. Observationally equivalent
programs remain separate or are grouped by behavioral equivalence.

## Symmetry-aware training objective

For prompt `H` and candidate values `v`, GPT-2 yields completion scores
`s_theta(v | H)`. Let `C(H)` be every value still consistent with the exact
history. The current calibrated target is uniform over that set:

```text
q(v | H) = 1 / |C(H)|  if v is consistent
           0            otherwise
```

Training minimizes:

```text
L = - sum_v q(v | H) log softmax(s_theta(v | H))
```

This has three useful properties:

1. no evidence produces an intentionally broad posterior;
2. evidence contracts the posterior without arbitrary labels;
3. the loss can supervise partial histories, not only solved episodes.

The first implementation applies this to hidden action semantics. The same
construction then extends to mechanics clauses and goal predicates.

## Executable hypotheses

A top joint assignment is compiled into ARC-DSL. Every candidate must survive:

1. grammar/type checking;
2. exact replay against all observed state/action/next-state triples;
3. exact agreement with terminal signals;
4. cross-level persistence checks for facts declared game-global.

A contradiction is returned to GPT-2 as the smallest exact counterexample. The
same checkpoint then repairs the hypothesis. A repair must preserve all prior
passing traces, so self-correction is measured rather than asserted.

## Action selection

For every legal action, execute the surviving GPT-2-authored programs. This
partitions the posterior by predicted consequence.

The controller first asks whether the current posterior already supports a
high-confidence plan. If not, it searches for a short **fracture sequence**: a
reachable intervention after which leading hypotheses predict different
observations.

The deterministic score is:

```text
expected goal progress from GPT-2-authored goal programs
+ reachable posterior reduction
- predicted irreversible failure mass
- action cost
```

No hand-written game objective enters this calculation. If GPT-2 has not yet
proposed a goal hypothesis, the agent is explicitly in an exploration state.

## One model, repeated computation

The same weights are called in several modes:

- `SCORE VARIABLE` — rank typed candidate values;
- `PROPOSE PROGRAM` — generate a mechanics or goal program;
- `REPAIR PROGRAM` — revise from an exact counterexample;
- `PREDICT RESIDUAL` — predict a transition not explained by the current DSL;
- `VALUE STATE` — score progress under its own goal hypotheses;
- `CHOOSE ACTION` — make the final decision from verified rollouts.

Multiple passes are allowed because they are recurrent use of one checkpoint,
not an ensemble. Every pass is logged and can be ablated.

## Continuous per-game memory

Text/program state is the first implementation. The next adaptive layer will be
small soft-prefix particles attached to the same GPT-2:

```text
z_i <- z_i - eta * gradient_z_i[-log P(actual next event | history, action, z_i)]
```

Each `z_i` is temporary per-game state, not another pretrained model. Particles
are reweighted by their predictive likelihood. Their disagreement supplies a
continuous residual uncertainty signal when the discrete DSL is incomplete.

This layer will not be promoted until the discrete version-space route passes
its information controls.

## Evidence ladder

### Gate 0 — exact infrastructure

- codecs round-trip every grid and delta;
- typed programs compile or fail loudly;
- replay identifies a concrete contradiction;
- all artifacts and splits are checksum-versioned.

### Gate 1 — epistemic action binding

- posterior starts broad with no evidence;
- intact evidence contracts it;
- shuffled evidence destroys the gain;
- amnesia does not imitate intact history;
- pretrained GPT-2 beats the same architecture from random initialization.

### Gate 2 — primitive executable induction

- GPT-2 identifies movement, collision, collect, toggle, push, teleport, paint,
  and click primitives from held-out trajectories;
- selected programs exactly replay unseen transitions;
- program-guided completion beats direct action GPT-2 under the same action
  budget.

### Gate 3 — composition and repair

- train on primitives and selected combinations;
- hold out complete compositions;
- require a valid novel program and successful repair after contradiction.

### Gate 4 — latent goals and cross-level transfer

- mechanics and goals are inferred separately;
- goal posterior is calibrated against terminal evidence;
- game-global rules transfer to new levels while local geometry is replaced.

### Gate 5 — locked ARC-AGI-3 games

A result counts only when the game contributed no trajectory, replay,
environment-derived label, mutation, or game-specific code to training or
development. A frozen checkpoint/configuration must solve an additional locked
game or materially improve efficiency across several locked games.

## Current execution

The repository now contains:

- exact frame/delta codecs;
- direct-policy and recurrent-memory negative controls;
- a typed mechanics DSL and typed goal DSL;
- exact replay, contradiction extraction, repair checks, hypothesis posteriors,
  disagreement, and generic planning;
- a set-valued partial-evidence dataset;
- a matched pretrained-versus-random GPT-2 training/evaluation workflow for
  calibrated version-space contraction.

The next promotion decision depends on executed evidence, not preference. If the
set-valued binding gate fails after adequate optimization, the representation or
use of GPT-2 pretraining is wrong and the project will change before adding more
mechanics.
