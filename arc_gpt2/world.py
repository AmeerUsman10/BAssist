from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

try:
    from .protocol import build_prompt, encode_delta, initial_memory
except ImportError:  # direct script execution
    from protocol import build_prompt, encode_delta, initial_memory

Direction = str
Position = tuple[int, int]

DIRECTIONS: dict[Direction, tuple[int, int]] = {
    "N": (0, -1),
    "S": (0, 1),
    "W": (-1, 0),
    "E": (1, 0),
}
DIRECTION_TOKENS = {"N": "<N>", "S": "<S>", "W": "<W>", "E": "<E>"}


@dataclass(frozen=True)
class Transition:
    action: int
    before: Position
    after: Position
    moved: bool
    won: bool
    before_grid: list[list[int]]
    after_grid: list[list[int]]


@dataclass
class HiddenRuleGrid:
    width: int
    height: int
    player: Position
    goal: Position
    walls: set[Position]
    background_color: int
    player_color: int
    goal_color: int
    wall_color: int
    action_to_direction: dict[int, Direction]
    won: bool = False

    @classmethod
    def generate(
        cls,
        seed: int,
        *,
        min_size: int = 5,
        max_size: int = 8,
        wall_fraction: float = 0.13,
    ) -> "HiddenRuleGrid":
        rng = random.Random(seed)
        width = rng.randint(min_size, max_size)
        height = rng.randint(min_size, max_size)

        positions = [(x, y) for y in range(height) for x in range(width)]
        player = rng.choice(positions)
        distant = [
            position
            for position in positions
            if abs(position[0] - player[0]) + abs(position[1] - player[1]) >= 3
        ]
        goal = rng.choice(distant or [position for position in positions if position != player])

        candidate_walls = [position for position in positions if position not in {player, goal}]
        target_wall_count = int(width * height * wall_fraction)
        walls: set[Position] = set()
        rng.shuffle(candidate_walls)
        for position in candidate_walls:
            if len(walls) >= target_wall_count:
                break
            trial = set(walls)
            trial.add(position)
            if cls._has_path(width, height, player, goal, trial):
                walls = trial

        palette = rng.sample(range(1, 16), 3)
        directions = list(DIRECTIONS)
        rng.shuffle(directions)
        action_to_direction = {
            action: direction for action, direction in zip((1, 2, 3, 4), directions)
        }

        return cls(
            width=width,
            height=height,
            player=player,
            goal=goal,
            walls=walls,
            background_color=0,
            player_color=palette[0],
            goal_color=palette[1],
            wall_color=palette[2],
            action_to_direction=action_to_direction,
        )

    @staticmethod
    def _has_path(
        width: int,
        height: int,
        start: Position,
        goal: Position,
        walls: set[Position],
    ) -> bool:
        frontier = deque([start])
        visited = {start}
        while frontier:
            current = frontier.popleft()
            if current == goal:
                return True
            for dx, dy in DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    0 <= nxt[0] < width
                    and 0 <= nxt[1] < height
                    and nxt not in walls
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    frontier.append(nxt)
        return False

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return (1, 2, 3, 4)

    def clone(self) -> "HiddenRuleGrid":
        return copy.deepcopy(self)

    def render(self) -> list[list[int]]:
        grid = [
            [self.background_color for _ in range(self.width)]
            for _ in range(self.height)
        ]
        for x, y in self.walls:
            grid[y][x] = self.wall_color
        goal_x, goal_y = self.goal
        grid[goal_y][goal_x] = self.goal_color
        player_x, player_y = self.player
        grid[player_y][player_x] = self.player_color
        return grid

    def _destination(self, action: int) -> Position:
        direction = self.action_to_direction[action]
        dx, dy = DIRECTIONS[direction]
        candidate = (self.player[0] + dx, self.player[1] + dy)
        if (
            not 0 <= candidate[0] < self.width
            or not 0 <= candidate[1] < self.height
            or candidate in self.walls
        ):
            return self.player
        return candidate

    def step(self, action: int) -> Transition:
        if action not in self.legal_actions:
            raise ValueError(f"Illegal action: {action}")
        before = self.player
        before_grid = self.render()
        if not self.won:
            self.player = self._destination(action)
            self.won = self.player == self.goal
        after_grid = self.render()
        return Transition(
            action=action,
            before=before,
            after=self.player,
            moved=before != self.player,
            won=self.won,
            before_grid=before_grid,
            after_grid=after_grid,
        )

    def counterfactual(self, action: int) -> Transition:
        cloned = self.clone()
        return cloned.step(action)

    def shortest_path_directions(self) -> list[Direction]:
        frontier: deque[Position] = deque([self.player])
        parent: dict[Position, tuple[Position, Direction] | None] = {self.player: None}
        while frontier:
            current = frontier.popleft()
            if current == self.goal:
                break
            for direction, (dx, dy) in DIRECTIONS.items():
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    0 <= nxt[0] < self.width
                    and 0 <= nxt[1] < self.height
                    and nxt not in self.walls
                    and nxt not in parent
                ):
                    parent[nxt] = (current, direction)
                    frontier.append(nxt)
        if self.goal not in parent:
            return []
        path: list[Direction] = []
        current = self.goal
        while parent[current] is not None:
            previous, direction = parent[current]  # type: ignore[misc]
            path.append(direction)
            current = previous
        path.reverse()
        return path

    def distance_to_goal(self, start: Position | None = None) -> int:
        origin = self.player if start is None else start
        frontier: deque[tuple[Position, int]] = deque([(origin, 0)])
        visited = {origin}
        while frontier:
            current, distance = frontier.popleft()
            if current == self.goal:
                return distance
            for dx, dy in DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    0 <= nxt[0] < self.width
                    and 0 <= nxt[1] < self.height
                    and nxt not in self.walls
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    frontier.append((nxt, distance + 1))
        return 10_000


@dataclass
class MappingBelief:
    mapping: dict[int, Direction | None]

    @classmethod
    def empty(cls, actions: Sequence[int] = (1, 2, 3, 4)) -> "MappingBelief":
        return cls(mapping={action: None for action in actions})

    def update(self, action: int, before: Position, after: Position) -> None:
        dx = after[0] - before[0]
        dy = after[1] - before[1]
        observed = {
            (0, -1): "N",
            (0, 1): "S",
            (-1, 0): "W",
            (1, 0): "E",
        }.get((dx, dy))
        if observed is not None:
            self.mapping[action] = observed

    def memory(self) -> str:
        items = []
        for action in sorted(self.mapping):
            direction = self.mapping[action]
            items.append(
                f"<A{action}>" + (DIRECTION_TOKENS[direction] if direction else "<UNK>")
            )
        return "<MEM><MAP>" + "".join(items) + "</MEM>"


def counterfactual_text(environment: HiddenRuleGrid) -> str:
    parts = ["<CF>"]
    for action in environment.legal_actions:
        transition = environment.counterfactual(action)
        if transition.won:
            outcome = "<WIN>"
        elif transition.moved:
            outcome = "<MOVE>"
        else:
            outcome = "<NOOP>"
        parts.extend(
            (
                f"<A{action}>",
                outcome,
                f"<X{transition.after[0]}>",
                f"<Y{transition.after[1]}>",
            )
        )
    parts.append("</CF>")
    return "".join(parts)


def choose_oracle_action(
    environment: HiddenRuleGrid,
    belief: MappingBelief,
    rng: random.Random,
) -> int:
    path = environment.shortest_path_directions()
    if path:
        desired_direction = path[0]
        known_inverse = {
            direction: action
            for action, direction in belief.mapping.items()
            if direction is not None
        }
        if desired_direction in known_inverse:
            return known_inverse[desired_direction]

    unknown_actions = [
        action for action, direction in belief.mapping.items() if direction is None
    ]
    informative: list[tuple[int, int]] = []
    for action in unknown_actions:
        transition = environment.counterfactual(action)
        if transition.moved:
            informative.append((environment.distance_to_goal(transition.after), action))
    if informative:
        informative.sort()
        best_distance = informative[0][0]
        tied = [action for distance, action in informative if distance == best_distance]
        return rng.choice(tied)

    moving_known: list[tuple[int, int]] = []
    for action, direction in belief.mapping.items():
        if direction is None:
            continue
        transition = environment.counterfactual(action)
        if transition.moved:
            moving_known.append((environment.distance_to_goal(transition.after), action))
    if moving_known:
        moving_known.sort()
        return moving_known[0][1]

    if unknown_actions:
        return rng.choice(unknown_actions)
    return rng.choice(list(environment.legal_actions))


def generate_episode(seed: int, max_steps: int = 32) -> list[dict[str, object]]:
    environment = HiddenRuleGrid.generate(seed)
    rng = random.Random(seed ^ 0xA5A5A5A5)
    belief = MappingBelief.empty(environment.legal_actions)
    carried_memory = initial_memory(environment.legal_actions)

    previous_action: int | None = None
    previous_position: Position | None = None
    previous_grid: list[list[int]] | None = None
    records: list[dict[str, object]] = []

    for step_index in range(max_steps):
        current_grid = environment.render()
        previous_delta = (
            encode_delta(previous_grid, current_grid) if previous_grid is not None else None
        )
        memory_input = carried_memory

        if previous_action is not None and previous_position is not None:
            belief.update(previous_action, previous_position, environment.player)
        memory_output = belief.memory()

        action = choose_oracle_action(environment, belief, rng)
        prompt = build_prompt(
            memory=memory_input,
            grid=current_grid,
            legal_actions=environment.legal_actions,
            state="RUN",
            previous_action=previous_action,
            previous_delta=previous_delta,
        )
        completion = (
            memory_output
            + counterfactual_text(environment)
            + f"<ACT><A{action}></ACT><END>"
        )

        before_position = environment.player
        before_grid = current_grid
        transition = environment.step(action)
        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "seed": seed,
                    "step": step_index,
                    "action": action,
                    "true_direction": environment.action_to_direction[action],
                    "moved": transition.moved,
                    "completed": transition.won,
                    "width": environment.width,
                    "height": environment.height,
                },
            }
        )

        carried_memory = memory_output
        previous_action = action
        previous_position = before_position
        previous_grid = before_grid
        if transition.won:
            break

    return records
