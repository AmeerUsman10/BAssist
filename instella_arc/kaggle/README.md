# Kaggle GPU execution

The first run is a private Python script kernel. It loads one checkpoint, tries
4-bit then 8-bit quantization, evaluates one hidden-action world under intact,
amnesic, and shuffled-evidence conditions, and writes machine-readable evidence
to `/kaggle/working/instella_arc_results`.

## Manual route

1. Create a Kaggle Code notebook or script.
2. Enable Internet.
3. Select an NVIDIA T4 accelerator. The current default Kaggle image may report
   a P100 as CUDA-visible but lack compatible `sm_60` PyTorch kernels, so the T4
   is the safer supported choice.
4. Copy or upload `instella_arc_kaggle.py` as the script and run it.
5. Download the output directory after the kernel completes.

No Hugging Face token is required for the public model, although unauthenticated
download rate limits can make the first load slower.

## Official Kaggle CLI route

The directory can be pushed as a kernel using the official CLI:

```bash
cp kernel-metadata.template.json kernel-metadata.json
# Replace __KAGGLE_USERNAME__ in kernel-metadata.json.
pip install --upgrade kaggle
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 21600
kaggle kernels status YOUR_USERNAME/instella-arc-frozen-probe
kaggle kernels output YOUR_USERNAME/instella-arc-frozen-probe \
  -p downloaded-output --force
```

The GitHub workflow `instella-arc-kaggle-dispatch.yml` performs the same process
when repository secrets are available. It never prints or commits a credential.

## Credentials expected by the GitHub workflow

Preferred:

- `KAGGLE_API_TOKEN` — current Kaggle API token.
- `KAGGLE_KERNEL_ID` — complete `username/instella-arc-frozen-probe` identifier.

Legacy authentication is also accepted through:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

The repository currently contains no Kaggle credentials. A missing credential
produces a durable `blocked` receipt rather than an apparent successful run.

## Why the first run is small

The BF16 checkpoint is roughly 32 GB before activations and cache. The T4 path
is therefore an engineering gate for custom-code quantization support. A single
action-binding game is enough to determine whether:

- the custom MoE modules load under bitsandbytes;
- inference completes within memory;
- candidate log-probability scoring is numerically valid;
- the model uses intact evidence differently from amnesia and corrupted history.

Goal and contact-mechanics tasks are enabled only after that gate completes.
