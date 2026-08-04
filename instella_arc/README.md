# Instella-MoE × ARC-AGI-3

This branch is the score-oriented open-model continuation of the pure GPT-2 research program.

## Decision

Instella-MoE is not assumed to solve ARC-AGI-3 because it performs well on ordinary language benchmarks. It must pass the same causal evidence controls that GPT-2 has struggled with:

1. intact interaction history must outperform amnesia;
2. deliberately corrupted history must change the inferred rule in the corresponding wrong direction;
3. uncertainty must remain broad when several hidden worlds are observationally equivalent;
4. executable hypotheses must exactly replay observed transitions;
5. selected experiments must identify rules faster than random or fixed-order exploration;
6. all positive claims must survive whole-family and whole-game holdouts.

The primary checkpoints are:

- `amd/Instella-MoE-16B-A3B-Base` — clean adaptation substrate and long-context control;
- `amd/Instella-MoE-16B-A3B-DPO` — likely practical fine-tuning substrate;
- `amd/Instella-MoE-16B-A3B-Think` — strongest frozen reasoning baseline.

Only one checkpoint is loaded in an experiment. The ARC shell remains deterministic: exact grid serialization, exact replay, probability arithmetic, legal-action validation, and generic search. Instella is the only learned source of semantic hypotheses, predictions, plans, and repairs.

## Execution ladder

### I0 — artifact and architecture audit

Without downloading 32 GB of weights:

- read every official checkpoint configuration and repository revision;
- calculate exact shard sizes;
- load the official tokenizer and remote model code;
- instantiate the full architecture on the meta device;
- instantiate and execute a tiny shape-compatible Instella-MoE;
- detect quantization and LoRA target modules;
- estimate memory under BF16, INT8, and INT4;
- audit prompt token budgets.

### I1 — frozen checkpoint shootout

Run Base, DPO, and Think on the locked synthetic gates already used by the GPT-2 track:

- hidden action mapping;
- latent terminal goals;
- hidden contact mechanics;
- next-transition prediction;
- experiment selection;
- executable program proposal and repair.

Each task includes intact, amnesic, and corrupted-evidence controls. The benchmark records candidate log-probabilities, calibration, latency, memory, exact prompts, model revision, and configuration hash.

### I2 — closed-loop synthetic worlds

Use Instella priors inside the existing exact version-space agent. Compare:

- uniform hypothesis prior;
- Instella prior;
- Instella with history removed;
- direct-action Instella;
- random legal actions.

A gain counts only if fewer actions are required on completely held-out world families.

### I3 — ARC-specific adaptation

Fine-tune DPO and Base separately using environment-generated learning histories rather than teacher answers. Objectives include exact delta prediction, set-valued rule scoring, goal inference, program repair, and information-seeking action selection. The router remains frozen in the first experiments.

### I4 — test-time game memory

Attach a temporary soft prefix or LoRA state to the same checkpoint. Update it only from the environment's observed `state + action -> next state` transition. Compare frozen and adaptive runs under an identical action budget.

### I5 — public ARC-AGI-3 leave-one-game-out

Freeze the model and harness, exclude the entire target game and its replays from training and development, then run first-attempt evaluation through the official toolkit. No public score is promoted as evidence of private generalization without the holdout controls.

## Hardware reality

The official BF16 checkpoint is split into six shards and is approximately 31.7 GiB before runtime overhead. Full-precision loading is therefore not a sensible first use of a free 16 GB GPU. The first practical GPU path is quantized frozen inference; LoRA training follows only after load and quality are validated. `audit.py` produces exact current numbers from Hugging Face metadata rather than relying on this document.

## Files

- `catalog.py` — official checkpoint catalog and memory calculations.
- `prompts.py` — strict role prompts and output parsers.
- `backend.py` — Transformers and OpenAI-compatible inference backends.
- `audit.py` — no-weight architecture/tokenizer/feasibility audit.
- `benchmark.py` — frozen model evidence-control benchmark.
- `kaggle_runner.py` — one-checkpoint quantized Kaggle execution entry point.
- `tests/` — deterministic tests that do not require the 16B weights.
- `kaggle/` — kernel metadata and launch instructions.

## Promotion rule

Instella becomes the main ARC model only after a frozen or tuned checkpoint shows a reproducible intact-history advantage over amnesia and corrupted history on locked families. Normal benchmark strength, lower training loss, or an impressive free-form explanation does not satisfy that gate.
