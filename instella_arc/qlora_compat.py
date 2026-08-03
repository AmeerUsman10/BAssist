"""Compatibility bridge for PEFT training of the custom Instella MoE.

``prepare_model_for_kbit_training`` promotes some non-quantized parameters to
FP32. In Instella/DeepSeek-V3 routing this can make top-k routing weights Float
while the expert accumulator is Half, causing ``index_add_`` to fail during the
first backward-enabled forward pass. Frozen inference is unaffected.

The bridge leaves parameters and router decisions unchanged. It casts only
floating router *outputs* back to the router input activation dtype through a
forward hook. The cast is differentiable, but router parameters remain frozen by
the QLoRA safety contract.
"""

from __future__ import annotations

from typing import Any


_PATCHED = False


def is_moe_router_module(module: Any) -> bool:
    name = type(module).__name__.lower()
    if "moegate" in name or "topkrouter" in name or "moerouter" in name:
        return True
    return bool(
        hasattr(module, "top_k")
        and (
            hasattr(module, "scoring_func")
            or hasattr(module, "n_routed_experts")
            or hasattr(module, "num_experts")
        )
    )


def _first_floating_dtype(values: Any):
    import torch

    if isinstance(values, torch.Tensor):
        return values.dtype if values.is_floating_point() else None
    if isinstance(values, (list, tuple)):
        for value in values:
            dtype = _first_floating_dtype(value)
            if dtype is not None:
                return dtype
    if isinstance(values, dict):
        for value in values.values():
            dtype = _first_floating_dtype(value)
            if dtype is not None:
                return dtype
    return None


def cast_floating_tree(value: Any, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(cast_floating_tree(item, dtype) for item in value)
    if isinstance(value, list):
        return [cast_floating_tree(item, dtype) for item in value]
    if isinstance(value, dict):
        return {
            key: cast_floating_tree(item, dtype)
            for key, item in value.items()
        }
    return value


def install_router_output_dtype_hooks(model: Any) -> int:
    handles = []

    def hook(module, inputs, output):
        del module
        dtype = _first_floating_dtype(inputs)
        if dtype is None:
            return output
        return cast_floating_tree(output, dtype)

    for module in model.modules():
        if is_moe_router_module(module):
            handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError(
            "No Instella MoE router module was found for the dtype bridge"
        )
    # Retain handles for the lifetime of the model and expose a non-secret audit
    # count without changing the model's state dict.
    setattr(model, "_instella_router_dtype_hook_handles", handles)
    setattr(model, "_instella_router_dtype_hook_count", len(handles))
    return len(handles)


def patch_peft_prepare_model_for_kbit_training() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import peft

    original = peft.prepare_model_for_kbit_training

    def patched(*args, **kwargs):
        model = original(*args, **kwargs)
        install_router_output_dtype_hooks(model)
        return model

    peft.prepare_model_for_kbit_training = patched
    _PATCHED = True
