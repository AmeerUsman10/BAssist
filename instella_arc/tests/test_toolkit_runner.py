from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from instella_arc.controller import ClosedLoopController
from instella_arc.toolkit_runner import ToolkitControllerRunner, write_run_result


class FakeAction:
    def __init__(self, name: str, complex_action: bool = False):
        self.name = name
        self._complex = complex_action

    def is_complex(self):
        return self._complex


class FakeEnvironment:
    def __init__(self, *, complex_action=False, complete_after_step=False):
        self.action = FakeAction(
            "ACTION6" if complex_action else "ACTION1",
            complex_action=complex_action,
        )
        self.action_space = [self.action]
        self.calls = []
        self.complete_after_step = complete_after_step
        self.observation_space = self._frame(((0, 2), (0, 0)), levels=0)

    def _frame(self, grid, *, levels):
        return SimpleNamespace(
            game_id="test",
            state="WIN" if levels else "IN_PROGRESS",
            levels_completed=levels,
            win_levels=1,
            guid="guid",
            full_reset=False,
            available_actions=[self.action],
            frame=grid,
        )

    def step(self, action, *, data, reasoning):
        self.calls.append(
            {"action": action, "data": dict(data), "reasoning": dict(reasoning)}
        )
        levels = 1 if self.complete_after_step else 0
        return self._frame(((0, 0), (0, 2)), levels=levels)


def test_toolkit_runner_maps_complex_row_column_to_x_y() -> None:
    environment = FakeEnvironment(complex_action=True)
    runner = ToolkitControllerRunner(
        environment=environment,
        controller=ClosedLoopController(),
        max_actions=1,
    )
    result = runner.run()
    assert result.actions == 1
    call = environment.calls[0]
    coordinate = result.trace[0].coordinate_row_column
    assert coordinate is not None
    row, column = coordinate
    assert call["data"] == {"x": column, "y": row}
    assert call["reasoning"]["source"] == "probe"


def test_toolkit_runner_stops_after_terminal_level_progress() -> None:
    environment = FakeEnvironment(complete_after_step=True)
    runner = ToolkitControllerRunner(
        environment=environment,
        controller=ClosedLoopController(),
        max_actions=10,
    )
    result = runner.run()
    assert result.actions == 1
    assert result.levels_completed == 1
    assert result.terminal_state == "WIN"
    assert len(environment.calls) == 1


def test_toolkit_runner_honors_action_budget() -> None:
    environment = FakeEnvironment()
    runner = ToolkitControllerRunner(
        environment=environment,
        controller=ClosedLoopController(),
        max_actions=3,
    )
    result = runner.run()
    assert result.actions == 3
    assert len(environment.calls) == 3


def test_write_run_result_is_deterministic_for_written_bytes(tmp_path: Path) -> None:
    environment = FakeEnvironment(complete_after_step=True)
    result = ToolkitControllerRunner(
        environment=environment,
        controller=ClosedLoopController(),
        max_actions=2,
    ).run()
    path = tmp_path / "result.json"
    first = write_run_result(result, path)
    second = write_run_result(result, path)
    assert first == second
    assert len(first) == 64
