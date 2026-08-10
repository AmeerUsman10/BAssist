# Muse Glimmer ARC/CFS Gate V1 — Executable Handoff

This branch is the isolated execution lane for issue #18. It preserves the frozen GPT-2 public-development evidence in PR #17 and does not modify that branch.

## Execution owner

The private GitHub Actions workflow dispatches a source-derived private Kaggle kernel using the existing encrypted `KAGGLE_API_TOKEN` secret. The Kaggle runner is the execution worker. It must return one forced disposition: `ADOPT`, `SPECIALIZE`, `CONTINUE_EXPERIMENTALLY`, or `KILL`.

## Frozen conditions

- Model repository/revision: `meta-models/Muse-Glimmer-30B-GGUF@93769bc7ab5ad1e9cd22d857e3138cf5d977ae81`
- Quant: `muse-glimmer-30B-kquant-17gb.gguf`
- Projector: `mmproj-kquant.gguf`
- Backend: `ggml-org/llama.cpp@dd1ea524333b1e697489067d7a4c39c60d32beee`
- DFlash: disabled
- Official surface: exact 25-game local NORMAL public-development manifest
- Per-game matched budgets: 8 model calls, 32 environment actions, 2 resets
- Representation: exact text grid; multimodal projector is independently smoke-tested

## Gate order

1. Exact source/hardware/disk checks.
2. Exact backend build and model/projector hashing.
3. Text JSON, determinism, image, and legal-action deployment checks.
4. Twenty-task matched synthetic qualification for both arms.
5. Eight-task local evidence/tool transfer benchmark.
6. Runtime projection.
7. Exact public-development native arm.
8. Server restart and model rehash.
9. Exact public-development CFS-Lite arm.
10. Evidence boundary validation and forced decision.

A failed gate stops later stages. Failure is a result, not permission to retune V1.

## Safety boundaries

No competition submission, leaderboard action, private/locked evaluation, paid endpoint, model-generated code execution, credential persistence, model-weight persistence, hidden-reasoning persistence, or modification of prior score evidence is permitted.

## Durable outputs

The private Kaggle output contains manifests, sanitized JSON score receipts, provider call hashes/latencies, synthetic/transfer summaries, official scorecards when reached, a boundary audit, and `final-decision.json`. Model files remain in temporary Kaggle storage and are excluded from outputs.
