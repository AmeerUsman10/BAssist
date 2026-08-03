from __future__ import annotations

from instella_arc.action_belief import (
    ActionBeliefState,
    ActionObservation,
    ActionProfile,
    EffectSignature,
    candidate_coordinates,
    choose_probe,
    probe_score,
)
from instella_arc.world_state import grid_facts, transition_facts


def _observation(action: str, before, after, *, coordinate=None, progress=0):
    facts = transition_facts(before, after)
    return ActionObservation(
        action=action,
        coordinate=coordinate,
        before_sha256=facts.before_sha256,
        after_sha256=facts.after_sha256,
        effect=EffectSignature.from_transition(
            facts,
            level_progress=progress,
            terminal_state="IN_PROGRESS",
        ),
    )


def test_action_profile_tracks_consistency_and_translation_vectors() -> None:
    before = ((0, 2, 0),)
    after = ((0, 0, 2),)
    profile = ActionProfile("A1")
    profile.add(_observation("A1", before, after))
    profile.add(_observation("A1", before, after))
    assert profile.trials == 2
    assert profile.consistency == 1.0
    assert profile.translation_vectors == ((0, 1),)
    assert profile.no_change_rate == 0.0


def test_action_profile_rejects_mismatched_action() -> None:
    profile = ActionProfile("A1")
    try:
        profile.add(_observation("A2", ((0,),), ((0,),)))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched action must fail")


def test_candidate_coordinates_prioritize_center_and_rare_components() -> None:
    facts = grid_facts(
        (
            (0, 0, 0, 0, 0),
            (0, 2, 0, 3, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        )
    )
    coordinates = candidate_coordinates(facts)
    assert coordinates[0] == (2, 2)
    assert (1, 1) in coordinates
    assert (1, 3) in coordinates


def test_probe_score_prefers_untried_non_reset_action() -> None:
    untried = ActionProfile("A1")
    tried = ActionProfile("A2")
    tried.add(_observation("A2", ((0,),), ((0,),)))
    untried_score, _ = probe_score(
        profile=untried,
        coordinate_trials=0,
        current_visit_count=0,
        is_reset=False,
        recent_same_action=False,
    )
    tried_score, _ = probe_score(
        profile=tried,
        coordinate_trials=1,
        current_visit_count=0,
        is_reset=False,
        recent_same_action=False,
    )
    reset_score, _ = probe_score(
        profile=ActionProfile("RESET"),
        coordinate_trials=0,
        current_visit_count=0,
        is_reset=True,
        recent_same_action=False,
    )
    assert untried_score > tried_score
    assert reset_score < tried_score


def test_choose_probe_uses_untried_action_and_avoids_reset() -> None:
    belief = ActionBeliefState()
    facts = grid_facts(((0, 2, 0), (0, 0, 0)))
    belief.observe_state(facts.sha256)
    choice = choose_probe(
        belief,
        legal_actions=("RESET", "ACTION1", "ACTION2"),
        current_facts=facts,
    )
    assert choice.action in {"ACTION1", "ACTION2"}
    assert choice.action != "RESET"


def test_complex_probe_uses_distinct_untested_coordinate() -> None:
    belief = ActionBeliefState()
    grid = (
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 0),
    )
    facts = grid_facts(grid)
    first = choose_probe(
        belief,
        legal_actions=("ACTION6",),
        current_facts=facts,
        complex_actions=("ACTION6",),
    )
    belief.add_transition(
        _observation(
            "ACTION6",
            grid,
            grid,
            coordinate=first.coordinate,
        )
    )
    second = choose_probe(
        belief,
        legal_actions=("ACTION6",),
        current_facts=facts,
        complex_actions=("ACTION6",),
    )
    assert second.coordinate != first.coordinate
