from __future__ import annotations

from types import SimpleNamespace

from arcgpt2.official_mapping_memory import OfficialGPT2MappingMemory
from arcgpt2.phase0_hidden_action import Direction


class FakeInner:
    def __init__(self, action_name: str) -> None:
        self.action_name = action_name
        self.observed = []
        self.resets = 0

    def reset(self) -> None:
        self.observed.clear()
        self.resets += 1

    def observe(self, record):
        self.observed.append(record)
        return SimpleNamespace(
            observation_number=len(self.observed),
            loss_before_update=1.0,
            gradient_norm=2.0,
            prefix_norm=3.0,
        )

    def score(self, spec, history, action, current_grid):
        del spec, history, action, current_grid
        return {
            Direction.UP: 4.0,
            Direction.DOWN: 3.0,
            Direction.LEFT: 2.0,
            Direction.RIGHT: 1.0,
        }


def _memory():
    created = {}

    def factory(action_name):
        created[action_name] = FakeInner(action_name)
        return created[action_name]

    return OfficialGPT2MappingMemory(None, None, None, inner_factory=factory), created


def test_first_two_cell_transition_adapts_only_its_action() -> None:
    memory, created = _memory()
    before = ((0, 0, 0), (0, 2, 0), (0, 0, 0))
    after = ((0, 0, 0), (0, 0, 2), (0, 0, 0))

    receipt = memory.observe("ACTION2", before, after, "IN_PROGRESS")

    assert receipt["eligible"] is True
    assert receipt["adapted"] is True
    assert receipt["crop_shape"] == [3, 3]
    assert len(receipt["crop_sha256"]) == 64
    assert len(created["ACTION2"].observed) == 1
    assert not created["ACTION1"].observed
    assert memory.direction_scores("ACTION2") == {
        "UP": 4.0,
        "DOWN": 3.0,
        "LEFT": 2.0,
        "RIGHT": 1.0,
    }


def test_one_observation_contract_and_reset_are_explicit() -> None:
    memory, created = _memory()
    before = ((2, 0), (0, 0))
    after = ((0, 2), (0, 0))
    assert memory.observe("ACTION1", before, after, "IN_PROGRESS")["adapted"]
    second = memory.observe("ACTION1", before, after, "IN_PROGRESS")
    assert second["adapted"] is False
    assert second["reason"] == "single_observation_contract_already_used"

    memory.reset()
    assert created["ACTION1"].resets == 1
    assert memory.direction_scores("ACTION1") == {}
    assert memory.observe("ACTION1", before, after, "IN_PROGRESS")["adapted"]


def test_ambiguous_large_or_complex_transitions_do_not_adapt() -> None:
    memory, created = _memory()
    unchanged = ((0, 0), (0, 0))
    assert memory.observe("ACTION3", unchanged, unchanged, "IN_PROGRESS")["reason"] == (
        "requires_exactly_two_persistent_cell_changes"
    )
    assert memory.observe("ACTION6", unchanged, unchanged, "IN_PROGRESS")["reason"] == (
        "not_a_simple_cardinal_action"
    )
    assert not any(inner.observed for inner in created.values())


def test_teleport_like_transition_outside_local_crop_is_rejected() -> None:
    memory, _ = _memory()
    before = tuple(tuple(2 if (row, column) == (0, 0) else 0 for column in range(10)) for row in range(10))
    after = tuple(tuple(2 if (row, column) == (9, 9) else 0 for column in range(10)) for row in range(10))
    receipt = memory.observe("ACTION4", before, after, "IN_PROGRESS")
    assert receipt["adapted"] is False
    assert receipt["reason"] == "local_transition_crop_exceeds_8x8"
