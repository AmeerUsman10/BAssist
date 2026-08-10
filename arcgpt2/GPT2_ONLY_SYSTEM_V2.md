# GPT-2-Only ARC-AGI-3 System V2

## Non-negotiable boundary

There is one learned model: an original GPT-2-family causal language model.

Not permitted:

- another LLM;
- a vision encoder;
- a learned tokenizer, object detector, reward model, value model, embedding
  model, router, verifier, or ensemble;
- labels derived from a stronger teacher model;
- game-specific handwritten policies.

Permitted deterministic machinery:

- lossless encoding of the exact grid and exact changed cells;
- legal-action and coordinate validation;
- a typed grammar and compiler;
- execution of GPT-2-authored programs;
- arithmetic, probability normalization, replay, and generic search;
- temporary state belonging to the same GPT-2 checkpoint.

The deterministic shell may reject an impossible claim. It may not invent a
semantic claim.

## The central correction

A direct action policy is the wrong primary object. Early in an unseen game,
there is usually no uniquely correct action. Several probes can be equally
rational and several world rules can be observationally equivalent. Training a
single action label turns uncertainty into false supervision and encourages
mode collapse.

The primary learned object is therefore:

> `P(next observation, terminal event, latent rule | exact history, candidate action)`

GPT-2 is trained as a conditional density estimator and hypothesis scorer. An
action is selected only after the model has represented what it does and does
not know.

## One GPT-2, three time scales

### 1. Global weights

The checkpoint acquires reusable priors over intervention, movement, collision,
collection, toggling, ordering, transformation, spatial relations, terminal
rules, and scientific hypothesis revision.

### 2. Discrete per-game version space

GPT-2 scores candidate values and executable programs. Exact constraints and
replay preserve every candidate still compatible with the observations. This
is the inspectable belief state.

### 3. Continuous per-game soft belief

Several small soft prefixes are attached to the same frozen GPT-2 weights. Each
prefix is an alternative temporary latent state. After a real transition, the
prefixes are weighted by predictive likelihood and updated by gradient descent
on that transition alone. They model residual regularities not yet captured by
the discrete DSL.

The particles are not independently trained models. They are alternate states
of one model, analogous to particles in a Bayesian filter.

## Training data: scientist episodes, not demonstrations

A training episode contains:

```text
latent mechanics program
latent goal program
initial exact grid
intervention history
exact next-state deltas
terminal reports
complete version space after each observation
information value of each legal intervention
short successful plan once the posterior is sufficient
```

The source generator can see the latent program. GPT-2 never receives it unless
it is the target of an explicitly measured query.

### Counterfactual twins

For every geometry, create worlds with identical grids, colors, objects, and
goals but different hidden action mappings or mechanics. The only difference in
GPT-2's input is the intervention evidence. This makes surface shortcuts
mathematically useless.

### Set-valued targets

When several answers are still compatible with the evidence, probability is
spread across all of them. The model is not punished for retaining a true
alternative and is punished for assigning mass to a contradicted one.

### Contrastive causal pairs

Pair histories that differ in exactly one observation. Require the posterior to
change only where that observation is causally relevant. Pair histories with
identical observations but different hidden source programs. Require identical
posterior predictions over all behaviorally equivalent futures.

### Held-out families

Evaluation holds out complete mechanics compositions and goal families, not
just random seeds. A model that saw `push + toggle` during training has not
proved compositional transfer by solving a new layout of `push + toggle`.

## Representation

GPT-2 receives two synchronized views generated without learning:

1. a compact lossless grid/delta codec;
2. an ordinary-language relational account of the same facts.

Examples:

```text
Before: one color-2 cell at row 4 column 7.
Action A3.
After: color 2 left row 4 column 7 and entered row 4 column 8.
No other cell changed. The environment did not report success.
```

The compact view prevents linguistic omission. The language view lets GPT-2's
pretraining contribute. Training randomly renames colors, actions, coordinates,
and description order so fixed words cannot become game semantics.

## Queries made to the same checkpoint

Every mode is represented as text and candidate completions. No learned head is
required.

### `SCORE ACTION MEANING`

Score direction/effect candidates for one unknown action.

### `SCORE MECHANICS CLAUSE`

Score typed clauses such as movement, collision, collect, push, toggle,
teleport, paint, spawn, delete, and phase change.

### `SCORE GOAL`

Score terminal predicates against terminal and non-terminal evidence.

### `PREDICT DELTA`

Score exact candidate next-state deltas for a proposed action.

### `PROPOSE PROGRAM`

Generate a complete typed mechanics or goal program under grammar-constrained
decoding.

### `REPAIR PROGRAM`

Receive the smallest exact replay contradiction and generate a minimally
changed replacement.

### `PROPOSE INTERVENTION`

Choose an action or short reachable action sequence predicted to split the
current posterior.

### `PROPOSE PLAN`

Once uncertainty is sufficiently low, generate a plan. The same model predicts
its consequences under every surviving belief particle.

## Posterior update

For candidate hypothesis `h`, history `H`, and observed transition `o`:

```text
log w_t(h) = log w_(t-1)(h)
             + log P_GPT2(o | H, action, h)
             + exact_replay_constraint(h, o)
```

The exact constraint is `0` for a replay match and negative infinity for a
contradiction. GPT-2 therefore guides search among valid hypotheses but cannot
argue an invalid program into existence.

Soft-prefix particle weights are updated with the same predictive likelihood.
Particles are resampled only when effective sample size collapses.

## Active experiment selection

The agent does not ask GPT-2 for a free-form guess and obey it. It evaluates
legal interventions using the same checkpoint's posterior predictions.

For an action sequence `a`, calculate:

```text
expected reduction in posterior entropy
+ expected reduction in predictive disagreement
+ probability of terminal progress under surviving goal hypotheses
- irreversible-failure probability
- action cost
```

Search is restricted to reachable sequences. A theoretically informative state
that cannot be reached is worthless.

A **fracture sequence** is the shortest safe sequence for which leading
hypotheses predict different observations. It is preferred during exploration.
Once leading hypotheses agree on the consequences and a high-probability goal
plan exists, the controller switches to exploitation.

## Online learning

After every real transition:

1. score the transition under each soft prefix before adaptation;
2. update particle weights by predictive likelihood;
3. update each prefix for a few gradient steps on exact next-delta prediction;
4. extract the smallest discrete contradiction, if any;
5. ask the same GPT-2 to repair or extend the program posterior;
6. recompute action information values.

No reward label or hidden source rule is used online. The only supervision is
what the environment actually did.

## Curriculum

### Gate A — action semantics

Infer a hidden permutation from partial interventions. Counterfactual twins have
identical initial grids and contradictory mappings.

### Gate B — terminal goals

Infer a set-valued posterior over bounded Goal-DSL predicates. Statusless and
shuffled-status controls must remove the gain.

### Gate C — one-clause mechanics

Infer and exactly replay movement, collision, collect, toggle, push, teleport,
paint, spawn, delete, and click effects.

### Gate D — composition

Train on atoms and selected pairs; hold out complete three- and four-clause
compositions. Require executable novel programs, not textual similarity.

### Gate E — repair

Supply one contradictory transition. Require the repaired program to explain it
without breaking any earlier passing trace.

### Gate F — active learning

Under a fixed action budget, the model must select interventions that identify
rules faster than random, fixed-order probing, and direct-action GPT-2.

### Gate G — cross-level persistence

Separate game-global mechanics and goals from level-local geometry. Transfer the
former without carrying the latter.

### Gate H — locked ARC-AGI-3 games

Freeze checkpoint, codec, grammar, prompts, update hyperparameters, and action
budget. A game counts only if no trajectory, source mutation, or game-specific
code influenced development.

## Baselines that remain mandatory

- random legal actions;
- fixed-order probing;
- direct GPT-2 policy;
- GPT-2 with history removed;
- GPT-2 with evidence labels shuffled;
- identical GPT-2 architecture from random initialization;
- executable posterior with uniform rather than GPT-2 prior;
- frozen versus online soft belief.

The project advances only when the intact pretrained system wins against these
controls on locked families.

## Scaling policy

GPT-2 Small remains the research model until it passes Gates A and B with clear
history dependence and a positive pretraining advantage. Medium is introduced
only if Small is bottlenecked by capacity after the algorithm works. Large and
XL are final scaling tests, not substitutes for a failed learning procedure.

## Current concrete implementation

The repository now contains:

- exact codecs, typed mechanics/goal programs, replay, contradiction extraction,
  repair, posterior constraints, and generic planning;
- direct-policy negative controls;
- set-valued action-binding and latent-goal datasets;
- counterfactual-twin action mappings;
- matched pretrained/random GPT-2 scoring experiments;
- a numerically stable particle posterior;
- online soft-prefix adaptation for one frozen GPT-2 checkpoint;
- safe probes for any pre-existing Kaggle credentials.

The next promotion is evidence-driven: finish action-binding controls, then run
the goal posterior, then join both into a single multi-query checkpoint. No
public ARC score is claimed before the locked-game gate.
