# EGGROLL inner-memory compatibility gate

Protocol: `eggroll_inner_memory_v1`

Preregistration: GitHub issue #14, comment `5232184075`.

## Purpose

This is the smallest direct test of whether the low-rank perturbation idea in
EGGROLL is useful inside the existing ARC-GPT2 system. The known-positive
held-out binding protocol first trains the original pinned GPT-2 Small model to
write one observed outcome into an 8x768 temporary soft prefix with an exact
raw-NLL gradient. The selected checkpoint is then frozen. On fresh layouts, one
rank-one antithetic evolution-strategy step replaces that exact online write.

This is a technical compatibility test. It is not an ARC-AGI-3 score, a claim
about EGGROLL's hyperscale throughput, or permission to scale the raw-goal run.

## Estimator

For prefix `Z`, rank-one perturbation `E_i = a_i b_i^T`, support loss `L`, and
fixed `sigma`:

```text
d_i     = (L(Z + sigma E_i) - L(Z - sigma E_i)) / (2 sigma)
g_raw   = mean_i d_i E_i
g_norm  = g_raw / ||g_raw|| * sqrt(mean_i d_i^2)
Z'      = Z - 0.2 g_norm
```

The RMS norm correction uses only black-box directional derivatives. Exact
gradients are computed separately for audit metrics and never enter the update.

Frozen settings are rank 1, 512 antithetic pairs (population 1,024), sigma
0.001, one update, and 32-candidate forward batches. The 128/256/512-pair nested
ladder is diagnostic only and uses prefixes from the same frozen draw sequence.

## Evidence boundary

- Canonical train and validation data and AdamW settings are unchanged.
- The old locked test is not evaluated.
- The audit takes the first eight valid unique layouts from seeds
  `[1_400_000, 1_500_000)` and must be seed- and grid-hash-disjoint from every
  canonical split.
- A1-A4 and all four balanced counterfactual directions are evaluated.
- Model weights and the selected initial prefix are hashed before and after the
  single audit.
- No model weights are persisted and no competition submission is made.

The full conjunctive thresholds and stop rules live in the preregistration and
are enforced by `apply_gate` in `eggroll_inner_memory.py`.
