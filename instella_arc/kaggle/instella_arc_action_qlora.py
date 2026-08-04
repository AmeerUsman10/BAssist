"""Private Kaggle script for the gated Instella action QLoRA experiment."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


WORKING = Path("/kaggle/working")
REPOSITORY = WORKING / "BAssist-instella-action-qlora"


def run(*arguments: str | Path, cwd: Path | None = None) -> None:
    command = [str(argument) for argument in arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def main() -> None:
    run(sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "transformers==4.57.1",
        "accelerate>=1.2,<2",
        "huggingface_hub>=0.34,<1",
        "safetensors>=0.5,<1",
        "bitsandbytes>=0.45,<1",
        "peft>=0.14,<1",
    )

    if REPOSITORY.exists():
        shutil.rmtree(REPOSITORY)
    run(
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "instella-arc",
        "https://github.com/AmeerUsman10/BAssist.git",
        REPOSITORY,
    )

    run(
        sys.executable,
        "-m",
        "instella_arc.action_qlora_entry",
        "--checkpoint",
        "dpo",
        "--train-seed-base",
        "960000",
        "--train-games",
        "32",
        "--validation-games",
        "8",
        "--test-games",
        "32",
        "--epochs",
        "1",
        "--learning-rate",
        "0.0002",
        "--gradient-accumulation",
        "8",
        "--rank",
        "8",
        "--alpha",
        "16",
        "--dropout",
        "0.05",
        "--warmup-ratio",
        "0.05",
        "--max-grad-norm",
        "1.0",
        "--max-context-tokens",
        "2048",
        "--seed",
        "20260803",
        "--output-dir",
        WORKING / "instella_action_qlora",
        cwd=REPOSITORY,
    )


if __name__ == "__main__":
    main()
