# Instella-MoE × ARC-AGI-3 — execution status

Recorded: 2026-08-03

## Decision

The engineering and no-weight evidence phase is complete. The project stops here until one real pinned checkpoint is loaded on a compatible GPU. Starting ARC-specific QLoRA before validating the frozen runtime and evidence controls would spend compute without resolving the current uncertainty.

## Repository state

- Branch: `instella-arc`
- Draft PR: #3, targeting `arc-gpt2-programs`
- GPU execution issue: #4
- Upstream model revisions are pinned in `instella_arc/catalog.py`.

## Verified

- 141 deterministic tests pass.
- Base, DPO, and Think configuration, tokenizer, and official remote model code load.
- The complete 15,862,787,584-parameter architecture instantiates on the meta device.
- A shape-compatible tiny form of Base, DPO, and Think completes forward inference and autoregressive generation.
- Official `FrameData` animation frames are separated from the final persistent state and persistent action delta.
- Hidden action, goal, and contact-mechanics benchmarks use set-valued targets and intact/amnesic/corrupted evidence controls.
- The bounded smoke profile selects one no-evidence case and one matched maximum-evidence case for every control mode.
- Checkpoint revisions, prompts, metrics, latencies, GPU inventory, and load failures are written to machine-readable receipts.

## Exact audited model footprint

All three selected checkpoints consist of six Safetensors shards totaling approximately 31.726 GB decimal, or 29.547 GiB of raw two-byte weights. The planning floors are:

- BF16/FP16: 32.50 GiB including minimum runtime overhead;
- INT8: 17.43 GiB planning floor;
- INT4: 9.60 GiB planning floor.

These INT8/INT4 values are feasibility estimates. The custom MoE's actual bitsandbytes compatibility remains an execution gate.

## Adapter decision

The first rational adaptation, if the frozen gate passes, is DPO QLoRA with the router and all experts frozen:

- target: `q_proj` and `o_proj` only;
- rank: 8;
- modules: 54;
- trainable parameters: 1,769,472;
- FP16 adapter storage: 3.375 MiB;
- estimated adapter optimizer/gradient state: 20.25 MiB, excluding activations and quantized-base workspaces.

An all-projection rank-8 adapter would create 145,464,576 trainable parameters across 5,208 modules, including routed experts. It is rejected for the first free-GPU experiment.

## Prepared real-weight gate

The first external run is fully specified:

- checkpoint: `amd/Instella-MoE-16B-A3B-Think`;
- revision: `e67a4a54d81b19692ec85ea1d1c777aa5c0bfd83`;
- accelerator: NVIDIA T4;
- load order: NF4 INT4, then INT8;
- benchmark: one hidden-action counterfactual world;
- evidence cases: no evidence, intact maximum evidence, amnesic maximum evidence, and shuffled maximum evidence;
- outputs: structured generation, candidate log probabilities, set-valued cross entropy, Brier score, consistent probability mass, truth rank, latency, GPU inventory, and every load error.

Runnable assets:

- `instella_arc/kaggle/instella_arc_kaggle.ipynb`
- `instella_arc/kaggle/instella_arc_kaggle.py`
- `.github/workflows/instella-arc-kaggle-dispatch.yml`

## Current blocker

The safe GitHub dispatch probe found no Kaggle API credential and no kernel identifier in repository secrets. It recorded `blocked`, made no Kaggle write, and produced no model output.

Autonomous execution needs:

- `KAGGLE_API_TOKEN`;
- `KAGGLE_KERNEL_ID`, shaped as `username/instella-arc-frozen-probe`.

Legacy `KAGGLE_USERNAME` plus `KAGGLE_KEY` is also supported. Credential values must not be committed, posted in an issue, or pasted into chat.

## Promotion rule

1. Frozen Think must load and finish the bounded evidence gate.
2. Intact evidence must improve the posterior relative to amnesia and change appropriately under corrupted evidence.
3. Only then expand frozen tests to goals and contact mechanics.
4. Compare Think, DPO, and Base under identical locked cases.
5. Only then run the rank-8 attention QLoRA plan.
6. Only after held-out synthetic gains proceed to whole-game ARC-AGI-3 leave-one-game-out evaluation.

## Claims not supported yet

- No full Instella checkpoint has been executed in this project.
- No quantized compatibility result exists yet.
- No ARC-AGI-3 level or game has been completed by Instella.
- No public or private ARC score is claimed.
- No evidence yet shows Instella uses interaction history correctly in this harness.
