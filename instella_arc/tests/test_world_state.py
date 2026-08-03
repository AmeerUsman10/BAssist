from __future__ import annotations

from instella_arc.world_state import (
    changed_cells,
    connected_components,
    grid_facts,
    match_component_translations,
    transition_facts,
)


def test_connected_components_are_color_exact_and_four_connected() -> None:
    grid = (
        (0, 2, 0, 2),
        (0, 2, 0, 0),
        (3, 0, 3, 0),
    )
    components = connected_components(grid)
    color_two = [component for component in components if component.color == 2]
    color_three = [component for component in components if component.color == 3]
    assert [component.area for component in color_two] == [2, 1]
    assert [component.area for component in color_three] == [1, 1]


def test_grid_facts_are_stable() -> None:
    grid = ((0, 1), (1, 1))
    first = grid_facts(grid)
    second = grid_facts(grid)
    assert first == second
    assert first.color_counts == ((0, 1), (1, 3))
    assert len(first.sha256) == 64


def test_changed_cells_report_exact_before_and_after_values() -> None:
    before = ((0, 2, 0), (0, 0, 0))
    after = ((0, 0, 2), (0, 0, 0))
    changes = changed_cells(before, after)
    assert [(c.row, c.column, c.before, c.after) for c in changes] == [
        (0, 1, 2, 0),
        (0, 2, 0, 2),
    ]


def test_component_translation_detects_exact_shape_motion() -> None:
    before = (
        (0, 2, 2, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    after = (
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 0, 0),
    )
    translations = match_component_translations(before, after)
    target = next(translation for translation in translations if translation.color == 2)
    assert (target.delta_row, target.delta_column) == (1, 1)
    assert len(target.before_cells) == 2


def test_transition_facts_separate_motion_and_color_count_changes() -> None:
    before = (
        (0, 2, 0),
        (0, 0, 0),
    )
    after = (
        (0, 0, 2),
        (3, 0, 0),
    )
    facts = transition_facts(before, after)
    assert not facts.unchanged
    assert facts.changed_cell_count == 3
    assert (0, 1) in facts.translation_vectors
    assert facts.colors_added == ((3, 1),)
    assert facts.colors_removed == ((0, 1),)


def test_unchanged_transition_is_explicit() -> None:
    grid = ((0, 1), (0, 0))
    facts = transition_facts(grid, grid)
    assert facts.unchanged
    assert facts.changed_cells == ()
    assert facts.translation_vectors == ()
