"""Exact, model-free state and transition facts for ARC-AGI-3.

This module does not decide what an object *means*. It extracts only reversible
or directly testable facts from integer grids: changed cells, same-color
connected components, exact component translations, repeated states, and
terminal/level progress. Those facts form the deterministic evidence layer for
one Instella checkpoint to reason over.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

from arcgpt2.codec import Grid, normalize_grid


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class CellChange:
    row: int
    column: int
    before: int
    after: int


@dataclass(frozen=True)
class Component:
    color: int
    cells: tuple[Coordinate, ...]
    top: int
    left: int
    bottom: int
    right: int

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def normalized_shape(self) -> tuple[Coordinate, ...]:
        return tuple((row - self.top, column - self.left) for row, column in self.cells)

    @property
    def centroid(self) -> tuple[float, float]:
        return (
            sum(row for row, _ in self.cells) / self.area,
            sum(column for _, column in self.cells) / self.area,
        )

    @property
    def signature(self) -> str:
        payload = {
            "color": self.color,
            "shape": self.normalized_shape,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ComponentTranslation:
    color: int
    before_cells: tuple[Coordinate, ...]
    after_cells: tuple[Coordinate, ...]
    delta_row: int
    delta_column: int


@dataclass(frozen=True)
class GridFacts:
    height: int
    width: int
    colors: tuple[int, ...]
    color_counts: tuple[tuple[int, int], ...]
    components: tuple[Component, ...]
    sha256: str


@dataclass(frozen=True)
class TransitionFacts:
    before_sha256: str
    after_sha256: str
    changed_cells: tuple[CellChange, ...]
    translations: tuple[ComponentTranslation, ...]
    colors_added: tuple[tuple[int, int], ...]
    colors_removed: tuple[tuple[int, int], ...]
    unchanged: bool

    @property
    def changed_cell_count(self) -> int:
        return len(self.changed_cells)

    @property
    def translation_vectors(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                {
                    (translation.delta_row, translation.delta_column)
                    for translation in self.translations
                }
            )
        )


def grid_sha256(grid: Sequence[Sequence[int]]) -> str:
    normalized = normalize_grid(grid)
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def connected_components(
    grid: Sequence[Sequence[int]],
    *,
    include_colors: Iterable[int] | None = None,
) -> tuple[Component, ...]:
    frame = normalize_grid(grid)
    height = len(frame)
    width = len(frame[0])
    allowed = set(include_colors) if include_colors is not None else None
    visited: set[Coordinate] = set()
    components: list[Component] = []

    for row in range(height):
        for column in range(width):
            coordinate = (row, column)
            if coordinate in visited:
                continue
            color = frame[row][column]
            if allowed is not None and color not in allowed:
                visited.add(coordinate)
                continue
            queue: deque[Coordinate] = deque([coordinate])
            visited.add(coordinate)
            cells: list[Coordinate] = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (
                        current_row + delta_row,
                        current_column + delta_column,
                    )
                    next_row, next_column = neighbor
                    if (
                        next_row < 0
                        or next_row >= height
                        or next_column < 0
                        or next_column >= width
                        or neighbor in visited
                        or frame[next_row][next_column] != color
                    ):
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            ordered = tuple(sorted(cells))
            components.append(
                Component(
                    color=color,
                    cells=ordered,
                    top=min(cell[0] for cell in ordered),
                    left=min(cell[1] for cell in ordered),
                    bottom=max(cell[0] for cell in ordered),
                    right=max(cell[1] for cell in ordered),
                )
            )
    return tuple(
        sorted(
            components,
            key=lambda component: (
                component.color,
                component.top,
                component.left,
                component.area,
                component.cells,
            ),
        )
    )


def grid_facts(grid: Sequence[Sequence[int]]) -> GridFacts:
    frame = normalize_grid(grid)
    counts: dict[int, int] = {}
    for row in frame:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return GridFacts(
        height=len(frame),
        width=len(frame[0]),
        colors=tuple(sorted(counts)),
        color_counts=tuple(sorted(counts.items())),
        components=connected_components(frame),
        sha256=grid_sha256(frame),
    )


def changed_cells(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> tuple[CellChange, ...]:
    first = normalize_grid(before)
    second = normalize_grid(after)
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        raise ValueError("transition grids must have the same shape")
    return tuple(
        CellChange(row, column, first[row][column], second[row][column])
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    )


def _component_translation(
    before: Component,
    after: Component,
) -> ComponentTranslation | None:
    if before.color != after.color or before.normalized_shape != after.normalized_shape:
        return None
    delta_row = after.top - before.top
    delta_column = after.left - before.left
    if delta_row == 0 and delta_column == 0:
        return None
    translated = tuple(
        sorted(
            (row + delta_row, column + delta_column)
            for row, column in before.cells
        )
    )
    if translated != after.cells:
        return None
    return ComponentTranslation(
        color=before.color,
        before_cells=before.cells,
        after_cells=after.cells,
        delta_row=delta_row,
        delta_column=delta_column,
    )


def _color_positions(grid: Grid) -> dict[int, frozenset[Coordinate]]:
    positions: dict[int, set[Coordinate]] = {}
    for row, values in enumerate(grid):
        for column, color in enumerate(values):
            positions.setdefault(color, set()).add((row, column))
    return {color: frozenset(cells) for color, cells in positions.items()}


def match_component_translations(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> tuple[ComponentTranslation, ...]:
    first = normalize_grid(before)
    second = normalize_grid(after)
    changes = changed_cells(first, second)
    changed_positions = frozenset((change.row, change.column) for change in changes)
    if not changed_positions:
        return ()

    before_positions = _color_positions(first)
    after_positions = _color_positions(second)
    before_components = connected_components(first)
    after_components = connected_components(second)
    candidates: list[
        tuple[int, int, int, int, ComponentTranslation]
    ] = []
    for before_index, source in enumerate(before_components):
        source_cells = frozenset(source.cells)
        removed = source_cells - after_positions.get(source.color, frozenset())
        if not removed or not removed.issubset(changed_positions):
            continue
        for after_index, target in enumerate(after_components):
            target_cells = frozenset(target.cells)
            added = target_cells - before_positions.get(target.color, frozenset())
            if not added or not added.issubset(changed_positions):
                continue
            translation = _component_translation(source, target)
            if translation is None:
                continue
            changed_support = len(removed) + len(added)
            distance = abs(translation.delta_row) + abs(translation.delta_column)
            candidates.append(
                (
                    -changed_support,
                    distance,
                    before_index,
                    after_index,
                    translation,
                )
            )

    # Prefer candidates explaining more exact changed cells, then shorter
    # translations. A component participates in at most one motion receipt.
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[4].color,
            item[2],
            item[3],
        )
    )
    used_before: set[int] = set()
    used_after: set[int] = set()
    selected: list[ComponentTranslation] = []
    for _, _, before_index, after_index, translation in candidates:
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index)
        used_after.add(after_index)
        selected.append(translation)
    return tuple(
        sorted(
            selected,
            key=lambda translation: (
                translation.color,
                translation.before_cells,
                translation.after_cells,
            ),
        )
    )


def transition_facts(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> TransitionFacts:
    first = normalize_grid(before)
    second = normalize_grid(after)
    changes = changed_cells(first, second)
    before_counts = dict(grid_facts(first).color_counts)
    after_counts = dict(grid_facts(second).color_counts)
    colors = sorted(set(before_counts) | set(after_counts))
    added = tuple(
        (color, after_counts.get(color, 0) - before_counts.get(color, 0))
        for color in colors
        if after_counts.get(color, 0) > before_counts.get(color, 0)
    )
    removed = tuple(
        (color, before_counts.get(color, 0) - after_counts.get(color, 0))
        for color in colors
        if before_counts.get(color, 0) > after_counts.get(color, 0)
    )
    return TransitionFacts(
        before_sha256=grid_sha256(first),
        after_sha256=grid_sha256(second),
        changed_cells=changes,
        translations=match_component_translations(first, second),
        colors_added=added,
        colors_removed=removed,
        unchanged=not changes,
    )


def concise_grid_facts(facts: GridFacts, *, max_components: int = 24) -> str:
    lines = [
        f"GRID SHA={facts.sha256} SIZE={facts.height}x{facts.width}",
        "COLOR_COUNTS "
        + " ".join(f"{color}:{count}" for color, count in facts.color_counts),
    ]
    for index, component in enumerate(facts.components[:max_components]):
        centroid_row, centroid_column = component.centroid
        lines.append(
            f"COMPONENT {index} COLOR={component.color} AREA={component.area} "
            f"BOX={component.top},{component.left}-{component.bottom},{component.right} "
            f"CENTROID={centroid_row:.2f},{centroid_column:.2f} "
            f"SHAPE={component.normalized_shape}"
        )
    if len(facts.components) > max_components:
        lines.append(f"COMPONENTS_OMITTED {len(facts.components) - max_components}")
    return "\n".join(lines)


def concise_transition_facts(facts: TransitionFacts, *, max_changes: int = 48) -> str:
    lines = [
        f"TRANSITION {facts.before_sha256}->{facts.after_sha256} "
        f"CHANGED={facts.changed_cell_count} UNCHANGED={facts.unchanged}",
    ]
    for change in facts.changed_cells[:max_changes]:
        lines.append(
            f"CELL {change.row},{change.column} {change.before}->{change.after}"
        )
    if len(facts.changed_cells) > max_changes:
        lines.append(f"CHANGES_OMITTED {len(facts.changed_cells) - max_changes}")
    for translation in facts.translations:
        lines.append(
            f"TRANSLATION COLOR={translation.color} "
            f"DELTA={translation.delta_row},{translation.delta_column} "
            f"AREA={len(translation.before_cells)}"
        )
    if facts.colors_added:
        lines.append(
            "COLORS_ADDED "
            + " ".join(f"{color}:{count}" for color, count in facts.colors_added)
        )
    if facts.colors_removed:
        lines.append(
            "COLORS_REMOVED "
            + " ".join(f"{color}:{count}" for color, count in facts.colors_removed)
        )
    return "\n".join(lines)
