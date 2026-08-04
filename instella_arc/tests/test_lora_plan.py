from __future__ import annotations

from instella_arc.lora_plan import (
    LinearShape,
    build_plans,
    classify_linear,
    lora_parameter_count,
)


def test_linear_classification_separates_attention_shared_and_routed_experts() -> None:
    assert classify_linear("model.layers.0.self_attn.q_proj", "q_proj") == "attention"
    assert classify_linear("model.layers.1.mlp.shared_experts.up_proj", "up_proj") == "shared_expert_mlp"
    assert classify_linear("model.layers.1.mlp.experts.7.down_proj", "down_proj") == "routed_expert_mlp"
    assert classify_linear("model.layers.0.mlp.gate_proj", "gate_proj") == "dense_mlp"
    assert classify_linear("lm_head", "lm_head") == "lm_head"


def test_lora_count_matches_rank_times_input_plus_output() -> None:
    shapes = (
        LinearShape("a.q_proj", "q_proj", 128, 256, "attention"),
        LinearShape("a.o_proj", "o_proj", 256, 128, "attention"),
    )
    assert lora_parameter_count(shapes, 8) == 8 * (128 + 256 + 256 + 128)


def test_low_risk_plans_exclude_routed_experts() -> None:
    shapes = (
        LinearShape("a.q_proj", "q_proj", 128, 128, "attention"),
        LinearShape("a.o_proj", "o_proj", 128, 128, "attention"),
        LinearShape("a.mlp.shared_experts.up_proj", "up_proj", 128, 64, "shared_expert_mlp"),
        LinearShape("a.mlp.experts.0.up_proj", "up_proj", 128, 64, "routed_expert_mlp"),
    )
    plans = {plan.name: plan for plan in build_plans(shapes, (4,))}
    assert not plans["attention_q_o"].includes_routed_experts
    assert not plans["attention_plus_shared_experts"].includes_routed_experts
    assert plans["all_projection_matrices"].includes_routed_experts
