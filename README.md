# GPT-2 Autonomous Fine-Tuning

This repository contains a reproducible, unattended GPT-2 fine-tuning pipeline.

A GitHub Actions run downloads the public `openai-community/gpt2` checkpoint, fine-tunes the final two transformer blocks on the versioned JSONL dataset, evaluates on a held-out split, generates a sample, and uploads the complete tuned model as an Actions artifact.

## Inputs

- `data/train.jsonl` — prompt/response training examples
- `train_gpt2.py` — deterministic training and evaluation script
- `.github/workflows/train-gpt2.yml` — unattended runner

## Outputs

The workflow uploads an artifact named `gpt2-tuned-model` containing:

- full model weights in safetensors format
- tokenizer files
- training configuration and metrics
- generated evaluation sample

The initial run is intentionally small and evidence-oriented. It verifies the complete training path before larger or private datasets are introduced.
