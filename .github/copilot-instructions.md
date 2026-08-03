# BAssist repository instructions for coding agents

Use the root `AGENTS.md` as the primary operating contract.

Before modifying code:

1. Identify the correct active branch and assigned issue or pull request.
2. Read the relevant status and evidence files.
3. Continue the existing implementation rather than recreating it.

Keep changes narrow, testable, and evidence-driven. Preserve negative results. Run the smallest relevant tests before broader suites. Do not claim ARC capability from model loading, training loss, synthetic-oracle results, or unit tests alone.

Never expose credentials. Do not create or commit `kaggle.json`; do not echo token-bearing environment variables; use GitHub Actions secrets and variables. If an account-level permission or secret is missing, record the exact blocker and request one precise manual action.

For the Instella track, do not begin QLoRA until a pinned frozen checkpoint completes the intact/amnesic/corrupted-history evidence gate. For the pure GPT-2 track, do not introduce another learned model.

Every pull request should state:

- what changed;
- what was actually executed;
- exact tests and artifacts;
- what remains uncertain;
- the next promotion or kill gate.
