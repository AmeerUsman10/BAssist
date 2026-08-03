"""One-checkpoint Kaggle runner for quantized frozen Instella ARC evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import traceback
from typing import Any

from .backend import LoadPlan, TransformersBackend
from .benchmark import run_benchmark
from .catalog import CHECKPOINTS
from .prompts import TaskKind, build_prompt, extract_final_json
from .smoke_benchmark import run_smoke_benchmark


def gpu_inventory() -> list[dict[str, Any]]:
    import torch

    inventory: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "index": index,
                "name": properties.name,
                "total_bytes": properties.total_memory,
                "total_gib": properties.total_memory / (1024.0**3),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return inventory


def kaggle_max_memory(gpus: list[dict[str, Any]], cpu_gib: int = 24) -> dict[int | str, str]:
    memory: dict[int | str, str] = {}
    for gpu in gpus:
        # Leave headroom for activations, candidate-scoring logits, and allocator fragmentation.
        usable = max(4, int(float(gpu["total_gib"]) - 2.0))
        memory[int(gpu["index"])] = f"{usable}GiB"
    memory["cpu"] = f"{cpu_gib}GiB"
    return memory


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-30:],
    }


def load_with_fallbacks(
    *,
    checkpoint: str,
    quantizations: list[str],
    max_context_tokens: int,
    allow_fp16_offload: bool,
) -> tuple[TransformersBackend, list[dict[str, Any]]]:
    import torch

    gpus = gpu_inventory()
    if not gpus:
        raise RuntimeError("No CUDA GPU is visible. Enable a Kaggle GPU accelerator.")
    max_memory = kaggle_max_memory(gpus)
    attempts: list[dict[str, Any]] = []
    modes = list(quantizations)
    if allow_fp16_offload and "none" not in modes:
        modes.append("none")

    for mode in modes:
        attempt: dict[str, Any] = {
            "quantization": mode,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            backend = TransformersBackend.from_plan(
                LoadPlan(
                    checkpoint_key=checkpoint,
                    quantization=mode,
                    dtype="float16",
                    max_memory=max_memory,
                    offload_folder="/kaggle/working/instella_offload",
                    max_context_tokens=max_context_tokens,
                )
            )
            attempt["status"] = "success"
            attempt["backend_metadata"] = backend.metadata
            attempts.append(attempt)
            return backend, attempts
        except Exception as exc:
            attempt["status"] = "failure"
            attempt["error"] = _error(exc)
            attempts.append(attempt)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("Every requested Instella loading mode failed")


def sanity_generation(backend: TransformersBackend) -> dict[str, Any]:
    prompt = build_prompt(
        TaskKind.INFER_ACTION,
        """Rows increase toward the south and columns increase toward the east.
I applied A3. A color-2 cell left row 4 column 4 and appeared at row 4 column 5.
No other persistent cell changed. The environment did not report success.""",
        query="Which direction does A3 represent?",
        output_schema={
            "action": "A3",
            "possibilities": [{"meaning": "north|south|west|east", "probability": "0..1"}],
        },
    )
    output = backend.generate(prompt, max_new_tokens=192, temperature=0.0)
    try:
        parsed = extract_final_json(output)
    except ValueError:
        parsed = None
    return {"output": output, "parsed": parsed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), default="think")
    parser.add_argument(
        "--quantizations",
        nargs="+",
        choices=("int4", "int8", "none"),
        default=["int4", "int8"],
    )
    parser.add_argument("--allow-fp16-offload", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("action", "goal", "contact"),
        default=["action"],
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--games-per-task", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=930_000)
    parser.add_argument("--max-context-tokens", type=int, default=6144)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/instella_arc_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema": "instella_arc.kaggle_run.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "gpu_inventory": gpu_inventory(),
        "status": "running",
    }
    status_path = args.output_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        backend, attempts = load_with_fallbacks(
            checkpoint=args.checkpoint,
            quantizations=list(args.quantizations),
            max_context_tokens=args.max_context_tokens,
            allow_fp16_offload=args.allow_fp16_offload,
        )
        status["load_attempts"] = attempts
        status["sanity_generation"] = sanity_generation(backend)
        if args.profile == "smoke":
            report = run_smoke_benchmark(
                backend,
                tasks=args.tasks,
                seed_base=args.seed_base,
            )
        else:
            report = run_benchmark(
                backend,
                tasks=args.tasks,
                seed_base=args.seed_base,
                games_per_task=args.games_per_task,
            )
        report_path = args.output_dir / "frozen_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        status.update(
            status="success",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            report=str(report_path),
            report_summary=report["summary"],
        )
    except Exception as exc:
        status.update(
            status="failure",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=_error(exc),
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
