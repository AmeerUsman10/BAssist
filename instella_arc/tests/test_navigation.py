from __future__ import annotations

from instella_arc.action_belief import ActionBeliefState
from instella_arc.navigation import (
    agent_hypotheses_from_current_and_bindings,
    generate_navigation_plans,
    infer_movement_bindings,
    locate_agent_components,
    plausible_targets,
    shortest_navigation_plan,
)
from instella_arc.receipts import RichActionObservation
from instella_arc.action_belief import EffectSignature
from instella_arc.world_state import grid_facts


def _movement_observation(action: str, vector, color=2):
    before = ((0, 0),)
    effect = EffectSignature(
        unchanged=False,
        changed_cell_count=2,
        translation_vectors=(vector,),
        colors_added=(),
        colors_removed=(),
        level_progress=0,
        terminal_state="IN_PROGRESS",
    )
    metadata = {
        "moved_components": [
            {
                "color": color,
                "normalized_shape": [[0, 0]],
                "delta_row": vector[0],
                "delta_column": vector[1],
            }
        ]
    }
    return RichActionObservation(
        action=action,
        coordinate=None,
        before_sha256="before-" + action,
        after_sha256="after-" + action,
        effect=effect,
        metadata=metadata,
        receipt_sha256="receipt-" + action,
    )


def _belief_with_cardinal_bindings():
    belief = ActionBeliefState()
    for action, vector in (
        ("ACTION1", (-1, 0)),
        ("ACTION2", (1, 0)),
        ("ACTION3", (0, -1)),
        ("ACTION4", (0, 1)),
    ):
        belief.add_transition(_movement_observation(action, vector))
    return belief


def test_movement_bindings_are_inferred_only_from_observed_translations() -> None:
    bindings = infer_movement_bindings(_belief_with_cardinal_bindings())
    assert {
        binding.direction: binding.action for binding in bindings
    } == {
        "north": "ACTION1",
        "south": "ACTION2",
        "west": "ACTION3",
        "east": "ACTION4",
    }


def test_agent_hypothesis_uses_moved_component_metadata() -> None:
    facts = grid_facts(
        (
            (0, 0, 0, 0),
            (0, 2, 0, 3),
            (0, 0, 0, 0),
        )
    )
    hypotheses = agent_hypotheses_from_current_and_bindings(
        facts, _belief_with_cardinal_bindings()
    )
    assert hypotheses[0].color == 2
    assert hypotheses[0].shape == ((0, 0),)
    located = locate_agent_components(facts, hypotheses[0])
    assert len(located) == 1
    assert located[0].cells == ((1, 1),)


def test_shortest_navigation_plan_reaches_target_overlap() -> None:
    facts = grid_facts(
        (
            (0, 0, 0, 0, 0),
            (0, 2, 0, 0, 3),
            (0, 0, 0, 0, 0),
        )
    )
    belief = _belief_with_cardinal_bindings()
    bindings = infer_movement_bindings(belief)
    agent = agent_hypotheses_from_current_and_bindings(facts, belief)[0]
    component = locate_agent_components(facts, agent)[0]
    target = next(
        target
        for target in plausible_targets(facts, component)
        if target.color == 3 and target.relation == "overlap"
    )
    plan = shortest_navigation_plan(
        facts,
        agent_component=component,
        agent_hypothesis=agent,
        target=target,
        bindings=bindings,
    )
    assert plan is not None
    assert plan.action_sequence == ("ACTION4", "ACTION4", "ACTION4")
    assert plan.position_sequence == ((1, 1), (1, 2), (1, 3), (1, 4))


def test_navigation_routes_around_blocking_component() -> None:
    facts = grid_facts(
        (
            (0, 0, 0, 0, 0),
            (0, 2, 8, 0, 3),
            (0, 0, 0, 0, 0),
        )
    )
    belief = _belief_with_cardinal_bindings()
    bindings = infer_movement_bindings(belief)
    agent = agent_hypotheses_from_current_and_bindings(facts, belief)[0]
    component = locate_agent_components(facts, agent)[0]
    target = next(
        target
        for target in plausible_targets(facts, component)
        if target.color == 3 and target.relation == "overlap"
    )
    plan = shortest_navigation_plan(
        facts,
        agent_component=component,
        agent_hypothesis=agent,
        target=target,
        bindings=bindings,
    )
    assert plan is not None
    assert plan.length == 5
    assert plan.action_sequence[0] in {"ACTION1", "ACTION2"}
    assert plan.action_sequence[-1] in {"ACTION1", "ACTION2"}


def test_generate_navigation_plans_requires_multiple_bindings() -> None:
    facts = grid_facts(((0, 2, 0, 3),))
    belief = ActionBeliefState()
    belief.add_transition(_movement_observation("ACTION4", (0, 1)))
    assert generate_navigation_plans(facts, belief) == ()
