from __future__ import annotations

from instella_arc.action_qlora import (
    EXPECTED_RANK8_TRAINABLE_PARAMETERS,
    trainable_inventory,
)


class _Parameter:
    def __init__(self, count: int, *, requires_grad: bool = True):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _Model:
    def __init__(self, parameters):
        self._parameters = parameters

    def named_parameters(self):
        return iter(self._parameters)


def test_trainable_inventory_accepts_only_q_and_o_lora_parameters() -> None:
    model = _Model(
        [
            ("base.layers.0.self_attn.q_proj.lora_A.default.weight", _Parameter(10)),
            ("base.layers.0.self_attn.q_proj.lora_B.default.weight", _Parameter(20)),
            ("base.layers.0.self_attn.o_proj.lora_A.default.weight", _Parameter(30)),
            ("base.layers.0.self_attn.o_proj.lora_B.default.weight", _Parameter(40)),
            ("base.layers.0.mlp.experts.0.up_proj.weight", _Parameter(999, requires_grad=False)),
        ]
    )
    inventory = trainable_inventory(model)
    assert inventory["trainable_parameters"] == 100
    assert inventory["invalid_trainable_names"] == []
    assert inventory["router_or_expert_trainable"] is False


def test_trainable_inventory_detects_router_or_expert_escape() -> None:
    model = _Model(
        [
            ("base.layers.0.mlp.experts.0.q_proj.lora_A.default.weight", _Parameter(10)),
            ("base.layers.0.mlp.router.weight", _Parameter(20)),
        ]
    )
    inventory = trainable_inventory(model)
    assert inventory["router_or_expert_trainable"] is True
    assert "base.layers.0.mlp.router.weight" in inventory["invalid_trainable_names"]


def test_audited_rank8_parameter_count_is_pinned() -> None:
    assert EXPECTED_RANK8_TRAINABLE_PARAMETERS == 1_769_472
