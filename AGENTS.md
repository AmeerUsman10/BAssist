# Agent Operating Contract — BAssist

Read this file before changing the repository. It is the durable coordination contract for ChatGPT/Codex, GitHub Copilot, and other coding agents.

## 1. Locate the real source of truth

Do not assume the default branch contains the active experiment.

- Pure GPT-2 ARC work: `arc-gpt2-programs`
- Instella-MoE ARC work: `instella-arc`
- Generic GPT-2 training proof: `main`

Before editing, inspect the assigned issue or pull request and then read the nearest current status documents, especially:

- `reports/instella_arc/STATUS.md`
- `arcgpt2/EPISTEMIC_CORE.md`
- `arcgpt2/GPT2_ONLY_SYSTEM_V2.md`
- any `latest-*-status.json`, comparison report, or log tail referenced by the task

Continue existing work. Do not restart the project from a generic notebook or rewrite working infrastructure merely because another design is possible.

## 2. Evidence rules

Separate these explicitly in reports and pull requests:

- verified execution result;
- strong inference from evidence;
- unresolved uncertainty;
- proposed next experiment.

Never promote falling loss, passing unit tests, synthetic-oracle performance, a successful model load, or a persuasive generated explanation into an ARC capability claim.

A result counts only when the exact configuration, checkpoint revision, data split, seed, prompt or protocol version, raw output, metrics, and durable artifact paths are recorded.

When an experiment fails, preserve the failure and identify what it falsified. Do not hide or overwrite negative evidence.

## 3. Security and authority

Never print, commit, upload, quote, or place credentials in issues, pull requests, logs, artifacts, notebooks, or generated files.

In particular:

- never create or commit `kaggle.json`;
- never echo `KAGGLE_API_TOKEN`, `KAGGLE_KEY`, GitHub tokens, or Hugging Face tokens;
- use repository Actions secrets or variables only;
- redact library exceptions if they could contain an environment value;
- do not submit to a competition, create paid resources, place orders, or change external accounts unless the assigned task explicitly authorizes it.

A missing credential is a legitimate blocker. Record it precisely rather than bypassing it insecurely.

## 4. Working method

1. Inspect current code, reports, open issues, and related pull requests.
2. State the smallest testable objective.
3. Make the minimum coherent change.
4. Run the narrow tests first, then the relevant broader suite.
5. Record exact failures and iterate only when the new attempt addresses a concrete cause.
6. Commit with a descriptive message.
7. Open or update a pull request rather than pushing unrelated work directly into another active branch.

Do not perform broad refactors during a capability experiment unless a verified defect requires them.

## 5. ARC research boundaries

### Pure GPT-2 track

The only learned model is one original GPT-2-family causal language model. Deterministic code may serialize exact grids, validate legal actions, execute GPT-2-authored programs, replay transitions, normalize probabilities, and perform generic search. It may not invent semantic rules or introduce another learned model.

### Instella track

Use one pinned Instella checkpoint per run. The deterministic shell may serialize, verify, replay, calculate, and search. Instella remains the only learned source of hypotheses, predictions, plans, repairs, and action preferences.

Do not begin QLoRA merely because the checkpoint loads. The frozen checkpoint must first show a correct intact-history advantage over amnesia and corrupted-history controls.

## 6. Required controls

For any claimed evidence-sensitive capability, include as applicable:

- intact history;
- history removed or amnesic;
- evidence deliberately shuffled or corrupted;
- random legal action baseline;
- fixed-order exploration baseline;
- direct-policy model baseline;
- randomly initialized architecture control;
- whole-family or whole-game holdout.

Surface-identical counterfactual worlds with contradictory hidden rules are preferred because they make shortcut learning detectable.

## 7. Definition of done

A task is complete only when:

- code is committed on the correct branch;
- relevant tests pass or the exact failure is recorded;
- durable machine-readable evidence is written when the task executes an experiment;
- the pull request or issue states what changed, what was verified, what remains unknown, and the next gated action;
- no unsupported ARC score or capability claim is made.

When blocked by an account-level action, provide one precise manual action, explain why it is unavoidable, and stop without exposing secrets.
