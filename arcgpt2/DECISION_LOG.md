# ARC-GPT2 Decision Log

This log separates implemented evidence from architectural preference. Synthetic
results are never promoted as ARC-AGI-3 scores.

## 2026-08-01 — Direct recurrent Stage 0 completed

**System:** one pretrained GPT-2 Small checkpoint generated a bounded memory,
counterfactual section, and action. No other learned model was used.

**Run:** GitHub Actions `30720890533`, source commit
`adc807973d6f87d7162174e35be914dad93296ab`.

**Observed:**

- validation loss improved from `1.5351` to `0.3000` over 96 optimizer steps;
- teacher-forced held-out action accuracy was `9/48 = 18.75%`, below the 25%
  four-action random reference;
- closed-loop GPT-2 won `0/4` episodes;
- matched random action won `2/4` episodes;
- oracle traces won `4/4` with mean 5.25 actions;
- GPT-2 repeatedly emitted A1 and kept all action meanings `<UNK>` despite
  producing syntactically valid memory and action sections.

**Interpretation:** sequence loss and protocol validity did not translate into
history-dependent action learning. The model learned the output shell and a
low-entropy action habit. This falsifies the claim that the current direct
recurrent formulation is already a useful meta-learner.

**Decision:** keep the direct system as a strict-purity control. Do not scale it
until a redesigned objective demonstrates intact-history advantage over
amnesia, shuffled history, and random action.

## 2026-08-01 — Full ARC-DSL completion smoke completed

**System:** GPT-2 ranked all 24 Phase-0 action-mapping programs by conditional
likelihood of their full canonical ARC-DSL text. The selected program was
executed and planned through exactly.

**Observed:** on the one-game CPU smoke, the pretrained checkpoint ranked the
truth second before training and mapped two of four actions correctly after
training, whereas random initialization ranked the truth much lower before
training. Exact recovery and downstream completion remained zero.

**Interpretation:** the code path works and the pretrained model appears to have
a weak prior over code-like descriptions, but shared program syntax dominated
the target and one example cannot support a capability claim.

**Decision:** retain full-program scoring as a control, not the main induction
interface.

## 2026-08-01 — Compact special-token mapping smoke completed

**System:** the target was compressed to atomic tokens such as
`<MAP> <A1> <UP> ...` while retaining exact expansion to ARC-DSL.

**Observed:** after eight optimizer steps on 16 games, exact program recovery was
zero and the candidate posterior remained effectively uniform. Full,
amnesic, and shuffled evidence produced no material separation. Pretrained GPT-2
did not beat random initialization on the one-game smoke.

**Interpretation:** merely shortening the target did not preserve GPT-2's useful
pretraining. Replacing ordinary language/code with many new atomic tokens turned
the task into nearly-from-scratch representation learning.

**Decision:** do not use an all-special-token interface as the primary route.

## 2026-08-01 — Natural factorized induction selected

**Reasoning:** GPT-2 should be used through the representational medium it was
pretrained on. The hidden action mapping is therefore decomposed into four
ordinary-language completion questions. Exact grid-cell changes are rendered in
lossless English and numbers; GPT-2 scores `north`, `south`, `west`, and `east`.
A deterministic 24-permutation assignment enforces global consistency and the
selected mapping expands into an executable ARC-DSL program.

**Controls:**

- full evidence;
- no action-outcome evidence;
- action labels shuffled while outcomes are unchanged;
- pretrained GPT-2;
- the same GPT-2 architecture from random initialization.

**Status:** implementation and matched CPU experiment launched. No result is
claimed until the durable comparison receipt exists.

## Standing architecture decision

The main route is now:

1. GPT-2 answers typed, natural-language hypothesis questions or proposes code-
   like clauses;
2. deterministic code assembles complete candidate programs;
3. exact replay eliminates contradicted programs;
4. the surviving program posterior is used for planning and fracture-directed
   exploration;
5. counterexamples become GPT-2 repair examples;
6. one GPT-2 checkpoint remains the only learned source of hypotheses, memory,
   semantic ranking, and action choice.

This preserves the user's `just GPT-2` constraint while refusing to make GPT-2
perform exact simulation or graph search unreliably inside one text completion.
