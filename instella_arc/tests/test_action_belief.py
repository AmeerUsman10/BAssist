from __future__ import annotations

from types import SimpleNamespace

from instella_arc.action_belief import (
    ActionBeliefState,
    ActionObservation,
    ActionProfile,
    EffectSignature,
    candidate_coordinates,
    choose_probe,
    observation_is_informative,
    probe_score,
    sequence_candidate_coordinates,
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


def _rich_observation(
    action: str,
    *,
    coordinate,
    before_state: str,
    after_state: str,
    animation=False,
):
    return SimpleNamespace(
        action=action,
        coordinate=coordinate,
        before_sha256="persistent-before",
        after_sha256="persistent-after",
        effect=EffectSignature(
            unchanged=True,
            changed_cell_count=0,
            translation_vectors=(),
            colors_added=(),
            colors_removed=(),
            level_progress=0,
            terminal_state="IN_PROGRESS",
        ),
        metadata={
            "observation_before_sha256": before_state,
            "observation_after_sha256": after_state,
            "animation_deltas": ["delta"] if animation else [],
            "rendered_frame_count": 2 if animation else 1,
        },
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


def test_animation_only_observation_counts_as_informative() -> None:
    observation = _rich_observation(
        "ACTION6",
        coordinate=(3, 4),
        before_state="state-a",
        after_state="state-b",
        animation=True,
    )
    profile = ActionProfile("ACTION6")
    profile.add(observation)
    assert observation.effect.unchanged
    assert observation_is_informative(observation)
    assert profile.informative_rate == 1.0


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


def test_animation_sequence_contributes_intermediate_frame_candidates() -> None:
    baseline = (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    highlighted = (
        (0, 0, 0, 0, 0),
        (0, 0, 7, 0, 0),
        (0, 0, 0, 0, 0),
    )
    coordinates = sequence_candidate_coordinates((baseline, highlighted))
    assert (1, 2) in coordinates


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


def test_complex_probe_uses_distinct_untested_coordinate_in_same_state() -> None:
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
        current_state_id="state-a",
        coordinate_candidates=((1, 1), (0, 0)),
    )
    belief.add_transition(
        _rich_observation(
            "ACTION6",
            coordinate=first.coordinate,
            before_state="state-a",
            after_state="state-a",
        )
    )
    second = choose_probe(
        belief,
        legal_actions=("ACTION6",),
        current_facts=facts,
        complex_actions=("ACTION6",),
        current_state_id="state-a",
        coordinate_candidates=((1, 1), (0, 0)),
    )
    assert first.coordinate == (1, 1)
    assert second.coordinate == (0, 0)


def test_coordinate_history_reopens_after_observation_state_changes() -> None:
    belief = ActionBeliefState()
    facts = grid_facts(((0, 0), (0, 0)))
    belief.add_transition(
        _rich_observation(
            "ACTION6",
            coordinate=(1, 1),
            before_state="state-a",
            after_state="state-b",
            animation=True,
        )
    )
    choice = choose_probe(
        belief,
        legal_actions=("ACTION6",),
        current_facts=facts,
        complex_actions=("ACTION6",),
        current_state_id="state-b",
        coordinate_candidates=((1, 1), (0, 0)),
    )
    # The same coordinate has never been tried from state-b and retains priority.
    assert choice.coordinate == (1, 1)


def test_equal_probe_scores_preserve_candidate_priority_not_numeric_maximum() -> None:
    belief = ActionBeliefState()
    facts = grid_facts(((0, 0), (0, 0)))
    choice = choose_probe(
        belief,
        legal_actions=("ACTION6",),
        current_facts=facts,
        complex_actions=("ACTION6",),
        current_state_id="new-state",
        coordinate_candidates=((0, 0), (1, 1)),
    )
    assert choice.coordinate == (0, 0)
