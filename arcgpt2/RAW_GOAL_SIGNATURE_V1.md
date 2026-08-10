# Raw Goal Signature v1

Status: design freeze candidate. This protocol must pass CPU invariants and an
independent leakage review before it is preregistered or dispatched to Kaggle.

## Claim boundary

A pass may show that, on locked balanced synthetic atomic-goal worlds, raw
terminal-report next-token NLL updates to GPT-2's soft prefix identify one of
four candidate predicates and support a terminal prediction for that predicate
on a new transition. It does not establish general Goal-DSL induction,
unknown-goal inference outside the four atoms, planning, ARC-AGI-3 capability,
or a reason to scale without the separate pretraining-promotion gate.

## Group construction

Each independent group contains four counterfactual worlds and these candidate
families, with distinct semantic color roles:

1. `CONTACT Cx`
2. `ABSENT Cy`
3. `COUNT Cz EQ 1`
4. `TOUCH Ca Cb FOUR`

Trials 1 and 2 are resettable, one-step transitions under known mechanics. The
four predicates' two-bit terminal signatures are a bijection onto `00`, `01`,
`10`, and `11`. The four worlds share byte-identical mechanics, grids, actions,
changed cells, candidate order, queries, and initial prefix. Only the two
observed terminal-report tokens differ.

Trial 3 is a new resettable transition with its terminal report hidden. It is
byte-identical in all four worlds, receives no inner update, and makes exactly
two of the four predicates true. GPT-2 must score ordinary ` yes.` and ` no.`
completions from the exact new transition and the prefix adapted on Trials 1
and 2. Deterministic Goal-DSL execution supplies and checks the target only; it
must not supply the prediction. Trial-3 truth is balanced within family and is
independent of signature, candidate position, semantic color, and trial order.

A verified base construction uses background `0`, controlled color `1`, and
semantic roles `2` through `6`. Its identification truth vectors, ordered as
CONTACT/ABSENT/COUNT/TOUCH, are `1010` for Trial 1 and `0110` for Trial 2.
Colors and geometry are transformed under the balancing schedule; these literal
values are not exposed as group identifiers.

## Exact splits and balance

- train: 120 groups; five occurrences of each predicate-to-signature permutation
- validation: 24 groups; one occurrence of each permutation
- locked test: 72 groups; three occurrences of each permutation
- all 24 candidate-order permutations occur 5/1/3 times
- each family occupies each candidate position 30/6/18 times
- each of six semantic-role colors occurs 20/4/12 times in every named role
- twelve full five-role color tuples occur 10/2/6 times, and 24 inert nuisance
  layouts occur 5/1/3 times
- signature-assignment/candidate-order, signature-assignment/full-color-tuple,
  and signature-assignment/nuisance-layout edges are each unique within a split
  and have zero overlap between every pair of splits
- all six two-of-four Trial-3 masks are balanced in every split
- both evidence orders, Trial 1 then 2 and Trial 2 then 1, are evaluated

Candidate order is assigned once per group and shared across its four worlds.
The canonical physical-surface hash excludes candidate order, terminal values,
group IDs, split-bearing level identity, and all other identifiers. It includes
candidate programs and colors plus exact before/action/after states,
coordinates, and dimensions for all three trials. Exact hashes must be
disjoint across train, validation, and locked test. The accepted balanced
nuisance-layout schedule is frozen directly in the data contract; the runtime
builder performs no rejection or resampling. Manifests retain the level identity
as non-model metadata, an explicit empty `rejected_surfaces` field, and a
canonical manifest digest. Any future builder that rejects a candidate must
retain its reason and canonical payload and constitutes a new protocol version.
The frozen canonical manifest digest is
`02e59b60dab038e16f45fbfe03dc9dadece1c0d1fc8082210834965e7634ab04`.

No seed, group/world ID, signature assignment, truth bit, candidate index, or
rejection index may enter model text. Token audits must prove that the two
identification supports differ between counterfactual worlds only at their
terminal-report tokens. Candidate completions use a fixed equal token length.
The binary terminal completions and the neutral statusless replacement also use
equal token lengths.

## Learning contract

One GPT-2 Small checkpoint, its original tokenizer and LM head, and one soft
prefix are used. There is no auxiliary learned model or head. Each observed
trial performs one sequential soft-prefix update using candidate-count-one,
mean next-token NLL of the exact raw changed-cell and terminal report. No
counterfactual candidates enter an inner update.

Candidate evidence logits are the post-update mean completion log probability
minus the same candidate's no-update mean completion log probability. Raw scores
remain diagnostics. Temperature is fixed at one.

The checkpoint/tokenizer are pinned to `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The learned initial soft prefix has
length 8, Gaussian initialization standard deviation `.01`, and inner update
learning rate `.2`.

For every training group, reset to the same prefix and evaluate both orders:

`Louter = 0.25*Lprior + (Lsingle1 + Lsingle2 + Lfinal12 + Lfinal21 + Lsemantic12 + Lsemantic21)/6`

The single-bit targets are uniform on the exactly two consistent predicates;
the final goal targets are one-hot; semantic targets are binary. Statusless and
deranged modes are locked-test causal/consistency controls and are never special
training modes. Statusless replaces each terminal field with a token-count-
matched neutral field. Derangement uses a fixed other valid signature.

Validation selects the epoch with the lowest preregistered total objective,
breaking ties toward the earlier epoch. The complete selected checkpoint is
hashed and frozen before one locked-test evaluation.

The outer loop runs exactly two ordered passes over the 120 training groups
(240 optimizer steps, no shuffle). It uses AdamW with prefix/model learning
rates `1e-3`/`1e-4`, weight decay `0` on the prefix and `.01` on trainable GPT-2
parameters, betas `(.9, .999)`, epsilon `1e-8`, and global gradient-norm clipping
at `1.0`. GPT-2 blocks 0 through 10 plus token and position embeddings are
frozen; block 11 and final layer normalization are trainable. Dropout, TF32,
early stopping, and learned checkpoint persistence are disabled.

## Absolute pretrained capability gate

Every check is required, separately by evidence order where stated:

- one-bit consistent-pair mass at least `.75`, exact top-two set accuracy at
  least `.70`, and conditional pair imbalance at most `.15`; group-bootstrap
  lower bounds exceed `.50` mass and `1/6` set accuracy
- final identification accuracy at least `.70`, truth probability at least
  `.60`, statusless-subtracted gains at least `.35` and `.25`; bootstrap lower
  bounds exceed `.25`, `.25`, `0`, and `0`; every family reaches `.55` accuracy
  and `.45` truth probability
- cross-order argmax agreement at least `.90` and mean total-variation distance
  at most `.10` after canonical candidate alignment
- deranged-target accuracy/probability at least `.70`/`.60`, original-goal
  accuracy/probability at most `.25`/`.20`, and every corrupted target family
  reaches `.55`/`.45`
- Trial-3 binary accuracy at least `.75`, truth probability at least `.65`, and
  Brier score at most `.20`; bootstrap lower bounds for accuracy/probability
  exceed `.50`; every family reaches `.65`/`.60`
- Trial-3 intact-minus-statusless accuracy/probability gains at least
  `.25`/`.15`, with both bootstrap lower bounds above zero
- statusless goal entropy at least `1.95` bits, maximum uniform deviation at
  most `.05`, and cross-world goal-logit delta at most `1e-6`
- statusless Trial-3 cross-world logit delta at most `1e-6`; quartet-aggregated
  statusless goal accuracy/probability equal `.25 +/- 1e-6` and Trial-3
  accuracy/probability equal `.50 +/- 1e-6`
- all losses, logits, probabilities, gradients, and updates are finite; every
  prefix update L2 exceeds `1e-8`; deterministic replay delta is at most `1e-6`

All uncertainty intervals use a deterministic 10,000-draw one-sided 95% group
bootstrap over the 72 independent locked groups, never the 288 correlated
counterfactual worlds.

## Separate pretraining and scale promotion

Run three preregistered matched initialization seeds. Each pair uses pretrained
GPT-2 on one T4 and the identical randomly initialized architecture on the
other, with identical data, order, and hyperparameters. Every pretrained seed
must pass the absolute gate.

The exact matched seeds are `(577215, 618033, 707106)`. Within each seed the
pretrained and random lanes run concurrently on physical T4 0 and T4 1; the
three matched pairs run sequentially. The independent-group bootstrap uses base
seed `20260804`. The hierarchical promotion bootstrap uses seeds `20260805` and
`20260806` for identification accuracy/probability, and `20260815` and
`20260816` for Trial-3 accuracy/probability. Every bootstrap uses exactly 10,000
draws and a one-sided 95% lower bound.

Before a full allocation may perform any optimizer step, a timing-only dual-T4
guard uses seed `577215` and train/validation surfaces only. Per lane it runs one
unmeasured exact-group warmup, times two exact training-group backward/clip
paths with gradients cleared between groups, times two intact validation
groups, and times one locked-shaped evaluator on a training group with the
timing-only bootstrap reduced to one draw. It performs zero optimizer steps and
does not convert or query a locked-test group. The raw three-sequential-pair
projection is conservatively transformed to `1.25 * projection + 900 seconds`;
the full kernel may continue only when this upper projection is at most 16,200
seconds. Timing results cannot satisfy any scientific or promotion gate.

For a metric, each hierarchical bootstrap draw resamples the three seeds and
then the 72 matched groups within each selected seed, computes one mean paired
delta per selected seed, and takes their median. The lower bound is the fifth
percentile of 10,000 deterministic draws.

Identification promotion requires either median accuracy delta at least `.10`
or truth-probability delta at least `.05`, with the qualifying lower bound above
zero and the delta positive in at least two of three seed pairs. Semantic
promotion applies the same rule to Trial-3 accuracy/probability. Pretraining
promotion requires both. Random initialization is not required to fail.

Scale promotion requires both the absolute capability gate and the separate
pretraining-promotion gate. Thresholds are not relaxed after results are seen.

## Launch ledger

- Timing allocation preregistered in issue #12 as
  `arc-gpt2-raw-goal-timing-prereg-20260803-001`; the launch commit changes
  documentation only and leaves the frozen implementation and manifest unchanged.
