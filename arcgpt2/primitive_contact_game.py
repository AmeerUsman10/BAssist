"""Multi-level hidden-action and hidden-contact environment for Gate C.

Every game randomizes two independent latent variables:

1. the permutation from A1..A4 to cardinal movement;
2. what contact with one special color does.

Both persist across levels. The first level is a controlled laboratory where all
four actions can be tested without termination and the special object can then
be approached safely. Later levels vary geometry and walls. This creates a
strict progression beyond the Phase-0 mapping-only task while remaining small
enough for exact version-space evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

from .codec import Grid, normalize_grid
from .mechanics_v2 import (
    ActionMove,
    ContactMode,
    MechanicsObservationV2,
    MechanicsProgramV2,
    MechanicsStatus,
    enumerate_candidate_programs_v2,
    execute_v2,
    shortest_plan_v2,
)
from .phase0_hidden_action import Action, DIRECTION_DELTA, Direction


@dataclass(frozen=True)
class ContactPalette:
    background: int
    wall: int
    agent: int
    goal: int
    interaction: int


@dataclass(frozen=True)
class ContactLevelSpec:
    height: int
    width: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    interactions: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class ContactGameSpec:
    game_seed: int
    action_to_direction: Mapping[Action, Direction]
    contact_mode: ContactMode
    palette: ContactPalette
    levels: tuple[ContactLevelSpec, ...]
    probe_order: tuple[Action, ...]


@dataclass(frozen=True)
class ContactStepRecord:
    level_index: int
    step_index: int
    before: Grid
    action: Action
    after: Grid
    status: str
    moved: bool
    blocked: bool
    contacted_color: int | None

    def as_observation(self) -> MechanicsObservationV2:
        return MechanicsObservationV2(
            before=self.before,
            action=self.action,
            after=self.after,
            terminal=self.status in {"LEVEL_WIN", "GAME_WIN"},
        )


def render_contact_level(
    level: ContactLevelSpec,
    palette: ContactPalette,
) -> Grid:
    grid = [
        [palette.background for _ in range(level.width)]
        for _ in range(level.height)
    ]
    for row, column in level.walls:
        grid[row][column] = palette.wall
    for row, column in level.interactions:
        grid[row][column] = palette.interaction
    goal_row, goal_column = level.goal
    grid[goal_row][goal_column] = palette.goal
    start_row, start_column = level.start
    grid[start_row][start_column] = palette.agent
    return normalize_grid(grid)


def program_from_contact_spec(spec: ContactGameSpec) -> MechanicsProgramV2:
    return MechanicsProgramV2(
        moving_color=spec.palette.agent,
        background_color=spec.palette.background,
        blocking_colors=(spec.palette.wall,),
        goal_colors=(spec.palette.goal,),
        interaction_color=spec.palette.interaction,
        contact_mode=spec.contact_mode,
        moves=tuple(
            ActionMove(action, *DIRECTION_DELTA[direction])
            for action, direction in spec.action_to_direction.items()
        ),
    )


def enumerate_contact_programs(
    spec: ContactGameSpec,
) -> tuple[MechanicsProgramV2, ...]:
    return enumerate_candidate_programs_v2(
        moving_color=spec.palette.agent,
        background_color=spec.palette.background,
        blocking_colors=(spec.palette.wall,),
        goal_colors=(spec.palette.goal,),
        interaction_color=spec.palette.interaction,
    )


def _laboratory_level(rng: random.Random) -> ContactLevelSpec:
    height = width = 7
    start = (3, 3)
    interaction = (3, 5)
    candidates = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if (row, column) not in {start, interaction}
        and abs(row - start[0]) + abs(column - start[1]) >= 5
    ]
    return ContactLevelSpec(
        height=height,
        width=width,
        walls=frozenset(),
        start=start,
        goal=rng.choice(candidates),
        interactions=frozenset({interaction}),
    )


def _random_level(
    rng: random.Random,
    *,
    level_index: int,
    palette: ContactPalette,
    mapping: Mapping[Action, Direction],
    mode: ContactMode,
) -> ContactLevelSpec:
    size = 7 + min(level_index, 2)
    for _ in range(2_000):
        cells = [(row, column) for row in range(size) for column in range(size)]
        start, goal, interaction = rng.sample(cells, 3)
        protected = {start, goal, interaction}
        density = 0.08 + 0.04 * level_index
        walls = frozenset(
            cell
            for cell in cells
            if cell not in protected and rng.random() < density
        )
        level = ContactLevelSpec(
            height=size,
            width=size,
            walls=walls,
            start=start,
            goal=goal,
            interactions=frozenset({interaction}),
        )
        candidate_spec = ContactGameSpec(
            game_seed=-1,
            action_to_direction=mapping,
            contact_mode=mode,
            palette=palette,
            levels=(level,),
            probe_order=tuple(Action),
        )
        program = program_from_contact_spec(candidate_spec)
        if shortest_plan_v2(program, render_contact_level(level, palette), max_depth=96):
            return level
    raise RuntimeError("failed to generate a solvable contact-mechanics level")


def generate_contact_game(
    game_seed: int,
    *,
    levels: int = 3,
) -> ContactGameSpec:
    if levels < 2:
        raise ValueError("contact-mechanics games require at least two levels")
    rng = random.Random(game_seed)
    directions = list(Direction)
    rng.shuffle(directions)
    mapping = {
        action: direction
        for action, direction in zip(tuple(Action), directions, strict=True)
    }
    colors = list(range(16))
    rng.shuffle(colors)
    palette = ContactPalette(
        background=colors[0],
        wall=colors[1],
        agent=colors[2],
        goal=colors[3],
        interaction=colors[4],
    )
    mode = rng.choice(tuple(ContactMode))
    probe_order = list(Action)
    rng.shuffle(probe_order)
    level_specs = [_laboratory_level(rng)]
    for level_index in range(1, levels):
        level_specs.append(
            _random_level(
                rng,
                level_index=level_index,
                palette=palette,
                mapping=mapping,
                mode=mode,
            )
        )
    return ContactGameSpec(
        game_seed=game_seed,
        action_to_direction=mapping,
        contact_mode=mode,
        palette=palette,
        levels=tuple(level_specs),
        probe_order=tuple(probe_order),
    )


class PrimitiveContactGame:
    def __init__(self, spec: ContactGameSpec):
        self.spec = spec
        self.program = program_from_contact_spec(spec)
        self.level_index = 0
        self.step_index = 0
        self.finished = False
        self._grid = render_contact_level(spec.levels[0], spec.palette)

    @property
    def frame(self) -> Grid:
        return self._grid

    @property
    def level(self) -> ContactLevelSpec:
        return self.spec.levels[self.level_index]

    def step(self, action: Action) -> ContactStepRecord:
        if self.finished:
            raise RuntimeError("cannot act after the contact game is finished")
        before = self._grid
        execution = execute_v2(self.program, before, action)
        status = "ACTIVE"
        if execution.status is MechanicsStatus.WIN:
            if self.level_index + 1 == len(self.spec.levels):
                status = "GAME_WIN"
                self.finished = True
            else:
                status = "LEVEL_WIN"

        record = ContactStepRecord(
            level_index=self.level_index,
            step_index=self.step_index,
            before=before,
            action=action,
            after=execution.after,
            status=status,
            moved=execution.moved,
            blocked=execution.blocked,
            contacted_color=execution.contacted_color,
        )
        self.step_index += 1
        self._grid = execution.after

        if status == "LEVEL_WIN":
            self.level_index += 1
            self.step_index = 0
            self._grid = render_contact_level(
                self.spec.levels[self.level_index], self.spec.palette
            )
        return record


def simulate_truth_plan(
    spec: ContactGameSpec,
    *,
    max_actions: int = 256,
) -> tuple[ContactStepRecord, ...]:
    """Generate an offline complete trajectory using the declared true program."""

    game = PrimitiveContactGame(spec)
    records: list[ContactStepRecord] = []
    for _ in range(max_actions):
        plan = shortest_plan_v2(game.program, game.frame, max_depth=96)
        if not plan:
            raise RuntimeError("truth program could not find a plan in a generated level")
        record = game.step(plan[0])
        records.append(record)
        if game.finished:
            return tuple(records)
    raise RuntimeError("truth-plan simulator exceeded the action budget")
