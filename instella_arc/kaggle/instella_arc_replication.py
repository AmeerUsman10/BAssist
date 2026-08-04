"""Private Kaggle script for the 64-world frozen Instella replication."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


WORKING = Path("/kaggle/working")
REPOSITORY = WORKING / "BAssist-instella-arc-replication"


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
        "instella_arc.kaggle_action_replication",
        "--checkpoint",
        "think",
        "--quantizations",
        "int4",
        "int8",
        "--seed-base",
        "940000",
        "--games",
        "64",
        "--max-context-tokens",
        "6144",
        "--output-dir",
        WORKING / "instella_action_replication",
        cwd=REPOSITORY,
    )


if __name__ == "__main__":
    main()
