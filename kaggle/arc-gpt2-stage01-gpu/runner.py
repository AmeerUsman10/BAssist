"""Kaggle GPU runner for the strict one-GPT-2 Stage-0.1 gate.

The script clones the public source repository, generates deterministic data,
trains only one original GPT-2 model, evaluates intact/amnesic/shuffled history,
and copies all evidence plus the checkpoint into /kaggle/working.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/AmeerUsman10/BAssist.git"
CHECKOUT = Path("/kaggle/working/BAssist")
OUTPUT_ROOT = CHECKOUT / "outputs" / "stage01-gpu"
EXPORT_ROOT = Path("/kaggle/working/arc-gpt2-stage01-gpu")


def run(*args: str, cwd: Path | None = None) -> None:
    command = [str(value) for value in args]
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(cwd) if cwd else None)


def main() -> None:
    if CHECKOUT.exists():
        shutil.rmtree(CHECKOUT)
    run("git", "clone", "--depth", "1", REPOSITORY, str(CHECKOUT))
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "transformers>=4.48,<6",
        "safetensors>=0.4",
        "pytest>=8.2,<9",
    )
    run(sys.executable, "-m", "pytest", "-q", "arcgpt2/tests", cwd=CHECKOUT)
    run(
        sys.executable,
        "-m",
        "arcgpt2.build_stage01_dataset",
        "--output-dir",
        str(OUTPUT_ROOT / "data"),
        "--train-games",
        "8",
        "--validation-games",
        "2",
        "--test-games",
        "2",
        "--seed-base",
        "1729",
        cwd=CHECKOUT,
    )
    run(
        sys.executable,
        "-m",
        "arcgpt2.sample_stage01_eval",
        "--input",
        str(OUTPUT_ROOT / "data" / "train.jsonl"),
        "--output",
        str(OUTPUT_ROOT / "data" / "overfit_eval_sample.jsonl"),
        "--limit",
        "1536",
        "--seed",
        "42",
        cwd=CHECKOUT,
    )
    run(
        sys.executable,
        "-m",
        "arcgpt2.train_stage01_balanced",
        "--initialization",
        "pretrained",
        "--data-dir",
        str(OUTPUT_ROOT / "data"),
        "--train-file",
        str(OUTPUT_ROOT / "data" / "train.jsonl"),
        "--evaluation-file",
        str(OUTPUT_ROOT / "data" / "overfit_eval_sample.jsonl"),
        "--output-dir",
        str(OUTPUT_ROOT / "pretrained"),
        "--max-length",
        "512",
        "--prefix-keep",
        "192",
        "--train-batch-size",
        "8",
        "--eval-batch-size",
        "16",
        "--gradient-accumulation-steps",
        "1",
        "--learning-rate",
        "1e-4",
        "--max-optimizer-steps",
        "600",
        "--warmup-steps",
        "30",
        "--freeze-first-n-blocks",
        "10",
        "--evaluation-games",
        "8",
        "--evaluation-seed-start",
        "1729",
        "--max-actions-per-game",
        "48",
        "--history-modes",
        "intact",
        "amnesic",
        "shuffled",
        "--save-model",
        cwd=CHECKOUT,
    )

    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    EXPORT_ROOT.mkdir(parents=True)
    for relative in (
        Path("data/manifest.json"),
        Path("pretrained/summary.json"),
        Path("pretrained/model"),
    ):
        source = OUTPUT_ROOT / relative
        destination = EXPORT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    summary = json.loads(
        (OUTPUT_ROOT / "pretrained" / "summary.json").read_text(encoding="utf-8")
    )
    (EXPORT_ROOT / "KAGGLE_RUN_RESULT.json").write_text(
        json.dumps(
            {
                "status": summary.get("status"),
                "device": summary.get("device"),
                "gates": summary.get("gates"),
                "claim_scope": summary.get("claim_scope"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((EXPORT_ROOT / "KAGGLE_RUN_RESULT.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
