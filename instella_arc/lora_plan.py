"""No-weight LoRA/QLoRA parameter planner for Instella-MoE.

The planner instantiates the official custom architecture on the meta device and
counts exact matrix shapes. It does not download model weights and does not
claim that bitsandbytes supports every module; the first Kaggle load remains the
runtime compatibility gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .catalog import CHECKPOINTS


ATTENTION_SUFFIXES = {
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
}
MLP_SUFFIXES = {"gate_proj", "up_proj", "down_proj"}


@dataclass(frozen=True)
class LinearShape:
    name: str
    suffix: str
    in_features: int
    out_features: int
    category: str


@dataclass(frozen=True)
class AdapterPlan:
    name: str
    rank: int
    module_count: int
    trainable_parameters: int
    adapter_mib_fp16: float
    training_state_mib_estimate: float
    target_suffixes: tuple[str, ...]
    includes_routed_experts: bool
    note: str


def classify_linear(name: str, suffix: str) -> str:
    lowered = name.lower()
    if suffix in ATTENTION_SUFFIXES:
        return "attention"
    if suffix in MLP_SUFFIXES:
        if "shared_expert" in lowered:
            return "shared_expert_mlp"
        if ".experts." in lowered or ".experts[" in lowered:
            return "routed_expert_mlp"
        return "dense_mlp"
    if suffix == "lm_head":
        return "lm_head"
    return "other"


def inventory_linear_shapes(model: Any) -> tuple[LinearShape, ...]:
    import torch

    shapes: list[LinearShape] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        shapes.append(
            LinearShape(
                name=name,
                suffix=suffix,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                category=classify_linear(name, suffix),
            )
        )
    return tuple(shapes)


def lora_parameter_count(shapes: Iterable[LinearShape], rank: int) -> int:
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    return sum(rank * (shape.in_features + shape.out_features) for shape in shapes)


def _plan(
    name: str,
    shapes: tuple[LinearShape, ...],
    rank: int,
    selector: Callable[[LinearShape], bool],
    *,
    note: str,
) -> AdapterPlan:
    selected = tuple(shape for shape in shapes if selector(shape))
    parameters = lora_parameter_count(selected, rank)
    # FP16 adapter weights are 2 bytes. A rough Adam training-state estimate is
    # 12 bytes/parameter: weights, gradients, and two FP32 moments. Activations
    # and dequantization workspaces are intentionally excluded.
    return AdapterPlan(
        name=name,
        rank=rank,
        module_count=len(selected),
        trainable_parameters=parameters,
        adapter_mib_fp16=parameters * 2 / (1024.0**2),
        training_state_mib_estimate=parameters * 12 / (1024.0**2),
        target_suffixes=tuple(sorted({shape.suffix for shape in selected})),
        includes_routed_experts=any(
            shape.category == "routed_expert_mlp" for shape in selected
        ),
        note=note,
    )


def build_plans(shapes: tuple[LinearShape, ...], ranks: Iterable[int]) -> list[AdapterPlan]:
    plans: list[AdapterPlan] = []
    for rank in ranks:
        plans.extend(
            (
                _plan(
                    "attention_q_o",
                    shapes,
                    rank,
                    lambda shape: shape.category == "attention"
                    and shape.suffix in {"q_proj", "o_proj"},
                    note="Lowest-risk first QLoRA target; router and all experts stay frozen.",
                ),
                _plan(
                    "attention_all",
                    shapes,
                    rank,
                    lambda shape: shape.category == "attention",
                    note="All Gated-MLA projection matrices; router and experts stay frozen.",
                ),
                _plan(
                    "attention_plus_dense_mlp",
                    shapes,
                    rank,
                    lambda shape: shape.category in {"attention", "dense_mlp"},
                    note="Adds only non-MoE dense MLP blocks to attention adaptation.",
                ),
                _plan(
                    "attention_plus_shared_experts",
                    shapes,
                    rank,
                    lambda shape: shape.category
                    in {"attention", "shared_expert_mlp"},
                    note="Adapts attention and shared experts while leaving routed experts and router frozen.",
                ),
                _plan(
                    "all_projection_matrices",
                    shapes,
                    rank,
                    lambda shape: shape.category
                    in {
                        "attention",
                        "dense_mlp",
                        "shared_expert_mlp",
                        "routed_expert_mlp",
                    },
                    note="Upper-bound plan. Usually inappropriate for a free-GPU first experiment because thousands of routed-expert adapters are created.",
                ),
            )
        )
    return plans


def audit_lora(checkpoint_key: str, ranks: Iterable[int]) -> dict[str, Any]:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    spec = CHECKPOINTS[checkpoint_key]
    config = AutoConfig.from_pretrained(
        spec.repository_id,
        revision=spec.revision,
        trust_remote_code=True,
    )
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    shapes = inventory_linear_shapes(model)
    category_counts: dict[str, int] = {}
    for shape in shapes:
        category_counts[shape.category] = category_counts.get(shape.category, 0) + 1
    return {
        "schema": "instella_arc.lora_plan.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": asdict(spec),
        "total_linear_modules": len(shapes),
        "linear_categories": dict(sorted(category_counts.items())),
        "plans": [asdict(plan) for plan in build_plans(shapes, ranks)],
        "router_policy": {
            "train_router": False,
            "reason": (
                "The initial ARC adaptation isolates representation learning. "
                "Router parameters and load-balancing behavior remain frozen until "
                "an attention-only adapter passes locked evidence controls."
            ),
        },
        "representative_modules": [asdict(shape) for shape in shapes[:30]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), default="dpo")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_lora(args.checkpoint, args.ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": report["checkpoint"], "plans": report["plans"]}, indent=2))


if __name__ == "__main__":
    main()
