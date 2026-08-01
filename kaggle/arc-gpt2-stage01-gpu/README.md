# ARC-GPT2 Stage 0.1 — Kaggle GPU package

This directory is ready for unattended GPU execution once a Kaggle credential is available to the GitHub workflow or local CLI.

The kernel runs only one learned model: the original GPT-2 Small causal LM. It generates the deterministic synthetic curriculum, applies set-valued probe supervision and balanced offline sampling, evaluates intact/amnesic/shuffled histories, and exports the checkpoint plus exact reports.

## Launch contract

1. Copy `kernel-metadata.template.json` to `kernel-metadata.json`.
2. Replace `YOUR_KAGGLE_USERNAME` with the authenticated account name.
3. Run:

```bash
kaggle kernels push \
  --accelerator NvidiaTeslaP100 \
  -p kaggle/arc-gpt2-stage01-gpu
```

The latest official Kaggle CLI supports explicit accelerator selection during `kernels push`; the default GPU metadata remains enabled as a fallback.

## Outputs

Kaggle writes these under `/kaggle/working/arc-gpt2-stage01-gpu/`:

- `data/manifest.json`
- `pretrained/summary.json`
- `pretrained/model/`
- `KAGGLE_RUN_RESULT.json`

No credential is stored in this directory or committed to the repository.
