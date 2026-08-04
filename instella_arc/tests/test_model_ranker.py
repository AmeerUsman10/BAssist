from __future__ import annotations

from arcgpt2.official_observation import OfficialFrameSequence

from instella_arc.action_belief import ActionBeliefState
from instella_arc.backend import MockBackend
from instella_arc.model_ranker import InstellaPlanRanker, parse_choice
from instella_arc.navigation import (
    AgentHypothesis,
    NavigationPlan,
    NavigationTarget,
)
from instella_arc.world_state import grid_facts


def _observation():
    return OfficialFrameSequence(
        game_id="test",
        state="IN_PROGRESS",
        levels_completed=0,
        win_levels=1,
        full_reset=False,
        available_actions=("ACTION1", "ACTION2"),
        rendered_frames=(((0, 2, 0, 3),),),
    )


def _plan(target_color: int, actions):
    return NavigationPlan(
        agent=AgentHypothesis(
            color=2,
            shape=((0, 0),),
            confidence=1.0,
            support=4,
        ),
        target=NavigationTarget(
            component_index=target_color,
            color=target_color,
            cells=((0, 3),),
            relation="overlap",
            heuristic_priority=1.0,
        ),
        start_top_left=(0, 1),
        destination_top_left=(0, 3),
        action_sequence=tuple(actions),
        position_sequence=tuple((0, 1 + index) for index in range(len(actions) + 1)),
        assumed_passable_colors=(0,),
    )


def test_parse_choice_uses_last_valid_final_object() -> None:
    text = (
        '<FINAL>{"choice":0,"reason":"first"}</FINAL>\n'
        '<FINAL>{"choice":2,"reason":"revised"}</FINAL>'
    )
    assert parse_choice(text, 3) == 2


def test_parse_choice_rejects_out_of_range_or_missing_choice() -> None:
    for text in (
        '<FINAL>{"choice":5}</FINAL>',
        "choice 0",
    ):
        try:
            parse_choice(text, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model choice must fail")


def test_model_ranker_maps_shortlist_choice_to_original_plan_index() -> None:
    backend = MockBackend(
        output='<FINAL>{"choice":1,"reason":"target evidence"}</FINAL>'
    )
    ranker = InstellaPlanRanker(backend=backend, max_candidates=2)
    plans = (_plan(3, ("ACTION1",)), _plan(4, ("ACTION2", "ACTION2")))
    observation = _observation()
    result = ranker.rank(
        observation=observation,
        facts=grid_facts(observation.final_grid),
        belief=ActionBeliefState(),
        plans=plans,
    )
    assert result.index == 1
    assert ranker.receipts[-1].parsed
    assert not ranker.receipts[-1].used_fallback


def test_model_ranker_falls_back_safely_on_malformed_output() -> None:
    backend = MockBackend(output="unfinished reasoning")
    ranker = InstellaPlanRanker(backend=backend)
    plans = (_plan(3, ("ACTION1",)), _plan(4, ("ACTION2", "ACTION2")))
    observation = _observation()
    result = ranker.rank(
        observation=observation,
        facts=grid_facts(observation.final_grid),
        belief=ActionBeliefState(),
        plans=plans,
    )
    assert result.index == 0
    receipt = ranker.receipts[-1]
    assert receipt.used_fallback
    assert not receipt.parsed
    assert receipt.error is not None
