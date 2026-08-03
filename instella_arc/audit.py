"""No-weight architecture, tokenizer, and hardware audit for Instella-MoE."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import traceback
from typing import Any

from .catalog import CHECKPOINTS, catalog_payload, estimate_weight_memory, stable_sha256
from .prompts import TaskKind, build_prompt


def _exception(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-20:],
    }


def _file_size(sibling: Any) -> int:
    for name in ("size", "blob_size"):
        value = getattr(sibling, name, None)
        if value is not None:
            return int(value)
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict) and lfs.get("size") is not None:
        return int(lfs["size"])
    return 0


def _tiny_config(config: Any, torch: Any) -> Any:
    tiny = copy.deepcopy(config)
    updates = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_head_dim": 32,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "kv_lora_rank": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "vocab_size": 512,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 128,
        "rope_scaling": None,
        "farskip": False,
        "ep_size": 1,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 0,
        "use_cache": True,
        "torch_dtype": torch.float32,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }
    for name, value in updates.items():
        setattr(tiny, name, value)
    return tiny


def _module_inventory(model: Any) -> dict[str, Any]:
    import torch

    linear_names: list[str] = []
    leaf_types: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
        if not any(True for _ in module.children()):
            key = type(module).__name__
            leaf_types[key] = leaf_types.get(key, 0) + 1
    suffixes = sorted({name.rsplit(".", 1)[-1] for name in linear_names})
    likely_lora_targets = [
        suffix
        for suffix in suffixes
        if any(
            token in suffix
            for token in (
                "q_",
                "kv_",
                "k_",
                "v_",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
    ]
    return {
        "linear_module_count": len(linear_names),
        "linear_suffixes": suffixes,
        "likely_lora_target_suffixes": likely_lora_targets,
        "leaf_module_types": dict(sorted(leaf_types.items())),
        "no_split_modules": getattr(model, "_no_split_modules", None),
    }


def _sample_prompts() -> dict[str, str]:
    evidence = """INITIAL GRID
The grid has 5 rows and 5 columns.
Row 0: 0 0 0 0 0.
Row 1: 0 2 0 3 0.
Row 2: 0 0 0 0 0.
Row 3: 0 0 0 0 0.
Row 4: 0 0 0 0 0.

OBSERVATION 1
I applied A3.
Exactly 2 grid cells changed:
- Row 1, column 1 changed from color 2 to color 0.
- Row 1, column 2 changed from color 0 to color 2.
The environment reported a terminal success: no."""
    return {
        "action": build_prompt(
            TaskKind.INFER_ACTION,
            evidence,
            query="Which cardinal movement does A3 represent?",
        ).plain_text(),
        "transition": build_prompt(
            TaskKind.PREDICT_TRANSITION,
            evidence,
            query="Predict the result of applying A3 again.",
        ).plain_text(),
        "experiment": build_prompt(
            TaskKind.PROPOSE_EXPERIMENT,
            evidence,
            legal_actions=("A1", "A2", "A3", "A4"),
        ).plain_text(),
    }


def audit_checkpoint(key: str) -> dict[str, Any]:
    spec = CHECKPOINTS[key]
    report: dict[str, Any] = {"spec": asdict(spec), "phases": {}}
    try:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from huggingface_hub import HfApi
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        report["phases"]["imports"] = {"status": "failure", "error": _exception(exc)}
        return report

    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    try:
        info = HfApi().model_info(spec.repository_id, files_metadata=True)
        shards = [
            {
                "name": sibling.rfilename,
                "bytes": _file_size(sibling),
            }
            for sibling in info.siblings
            if sibling.rfilename.endswith(".safetensors")
        ]
        total_weight_bytes = sum(item["bytes"] for item in shards)
        report["hub"] = {
            "revision": info.sha,
            "last_modified": (
                info.last_modified.isoformat() if info.last_modified else None
            ),
            "private": info.private,
            "gated": info.gated,
            "library_name": info.library_name,
            "pipeline_tag": info.pipeline_tag,
            "tags": sorted(info.tags or []),
            "safetensor_shards": shards,
            "total_safetensor_bytes": total_weight_bytes,
            "memory_estimates": {
                name: asdict(value)
                for name, value in estimate_weight_memory(total_weight_bytes).items()
            },
        }
        report["phases"]["hub_metadata"] = {"status": "success"}
    except Exception as exc:
        report["phases"]["hub_metadata"] = {
            "status": "failure",
            "error": _exception(exc),
        }

    try:
        config = AutoConfig.from_pretrained(
            spec.repository_id, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repository_id, trust_remote_code=True
        )
        config_dict = config.to_dict()
        report["config"] = {
            "class": type(config).__name__,
            "architectures": config_dict.get("architectures"),
            "model_type": config_dict.get("model_type"),
            "hidden_size": config_dict.get("hidden_size"),
            "num_hidden_layers": config_dict.get("num_hidden_layers"),
            "num_attention_heads": config_dict.get("num_attention_heads"),
            "n_routed_experts": config_dict.get("n_routed_experts"),
            "n_shared_experts": config_dict.get("n_shared_experts"),
            "num_experts_per_tok": config_dict.get("num_experts_per_tok"),
            "moe_intermediate_size": config_dict.get("moe_intermediate_size"),
            "max_position_embeddings": config_dict.get("max_position_embeddings"),
            "rope_scaling": config_dict.get("rope_scaling"),
            "torch_dtype": str(config_dict.get("torch_dtype")),
            "auto_map": config_dict.get("auto_map"),
            "config_sha256": stable_sha256(config_dict),
        }
        prompt_counts = {
            name: len(tokenizer.encode(text, add_special_tokens=False))
            for name, text in _sample_prompts().items()
        }
        report["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "sample_prompt_tokens": prompt_counts,
        }
        report["phases"]["config_and_tokenizer"] = {"status": "success"}
    except Exception as exc:
        report["phases"]["config_and_tokenizer"] = {
            "status": "failure",
            "error": _exception(exc),
        }
        return report

    try:
        with init_empty_weights(include_buffers=True):
            meta_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )
        total_parameters = sum(parameter.numel() for parameter in meta_model.parameters())
        report["meta_model"] = {
            "class": type(meta_model).__name__,
            "total_parameters": total_parameters,
            "parameter_gib_bf16": total_parameters * 2 / (1024.0**3),
            "module_inventory": _module_inventory(meta_model),
        }
        report["phases"]["full_meta_instantiation"] = {"status": "success"}
    except Exception as exc:
        report["phases"]["full_meta_instantiation"] = {
            "status": "failure",
            "error": _exception(exc),
        }

    try:
        tiny_config = _tiny_config(config, torch)
        tiny_model = AutoModelForCausalLM.from_config(
            tiny_config, trust_remote_code=True
        )
        tiny_model.eval()
        input_ids = torch.tensor([[0, 7, 11, 13, 17]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = tiny_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            generated = tiny_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=2,
                pad_token_id=2,
                eos_token_id=1,
            )
        cache_shapes: list[list[int]] = []
        past = getattr(output, "past_key_values", None)
        if past is not None:
            try:
                for layer in past:
                    for tensor in layer:
                        if hasattr(tensor, "shape"):
                            cache_shapes.append(list(tensor.shape))
            except TypeError:
                cache_shapes.append([int(getattr(past, "get_seq_length")())])
        report["tiny_model"] = {
            "class": type(tiny_model).__name__,
            "parameters": sum(parameter.numel() for parameter in tiny_model.parameters()),
            "logits_shape": list(output.logits.shape),
            "generated_shape": list(generated.shape),
            "cache_tensor_shapes": cache_shapes,
            "module_inventory": _module_inventory(tiny_model),
        }
        report["phases"]["tiny_forward_and_generation"] = {"status": "success"}
    except Exception as exc:
        report["phases"]["tiny_forward_and_generation"] = {
            "status": "failure",
            "error": _exception(exc),
        }

    return report


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Instella-MoE ARC architecture audit",
        "",
        f"Recorded: `{payload['recorded_at_utc']}`",
        "",
        "This audit downloads configuration, tokenizer, and remote model code only. It does not download the 16B weights.",
        "",
    ]
    for key, report in payload["checkpoints"].items():
        lines.extend((f"## {key}", ""))
        hub = report.get("hub", {})
        config = report.get("config", {})
        lines.append(f"- Repository: `{report['spec']['repository_id']}`")
        lines.append(f"- Revision: `{hub.get('revision')}`")
        if hub.get("total_safetensor_bytes"):
            lines.append(
                f"- Safetensor bytes: `{hub['total_safetensor_bytes']:,}`"
            )
        lines.append(f"- Config class: `{config.get('class')}`")
        lines.append(
            f"- Full meta model: `{report['phases'].get('full_meta_instantiation', {}).get('status')}`"
        )
        lines.append(
            f"- Tiny forward/generation: `{report['phases'].get('tiny_forward_and_generation', {}).get('status')}`"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=tuple(CHECKPOINTS),
        default=["base", "dpo", "think"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "schema": "instella_arc.audit.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": catalog_payload(),
        "checkpoints": {
            key: audit_checkpoint(key) for key in args.checkpoints
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({
        key: value["phases"] for key, value in payload["checkpoints"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
