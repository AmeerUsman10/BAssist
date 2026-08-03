"""Kaggle entry point for the first real Instella-MoE ARC frozen run.

Create a Kaggle notebook/script with Internet and a GPU enabled, then run this
file. It installs the pinned runtime, checks out the exact research branch,
loads one quantized checkpoint, and writes all evidence under
`/kaggle/working/instella_arc_results`.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


WORKING = Path("/kaggle/working")
REPOSITORY = WORKING / "BAssist-instella-arc"


def run(*arguments: str) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


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
        str(REPOSITORY),
    )

    # The first run is deliberately bounded: one hidden-action game, all three
    # evidence modes, one checkpoint, and no full-precision fallback. It is a
    # load/quality gate before spending the weekly GPU quota on goals/mechanics.
    run(
        sys.executable,
        "-m",
        "instella_arc.kaggle_runner",
        "--checkpoint",
        "think",
        "--quantizations",
        "int4",
        "int8",
        "--tasks",
        "action",
        "--games-per-task",
        "1",
        "--seed-base",
        "930000",
        "--max-context-tokens",
        "6144",
        "--output-dir",
        str(WORKING / "instella_arc_results"),
    )


if __name__ == "__main__":
    # `python -m` needs the checked-out repository on sys.path. Re-execute from
    # the repository after cloning when this entry file is run standalone.
    if REPOSITORY.exists():
        sys.path.insert(0, str(REPOSITORY))
    main()
