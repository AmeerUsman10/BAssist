"""Strict prompts and parsers for Instella ARC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Iterable, Mapping, Sequence


class TaskKind(str, Enum):
    PREDICT_TRANSITION = "predict_transition"
    INFER_ACTION = "infer_action_meaning"
    INFER_GOAL = "infer_goal"
    INFER_MECHANICS = "infer_mechanics"
    PROPOSE_EXPERIMENT = "propose_experiment"
    PROPOSE_PROGRAM = "propose_program"
    REPAIR_PROGRAM = "repair_program"
    CHOOSE_ACTION = "choose_action"


FINAL_PATTERN = re.compile(r"<FINAL>\s*(\{.*?\})\s*</FINAL>", re.DOTALL)
ACTION_PATTERN = re.compile(r"\b(?:A|ACTION)([0-7])\b", re.IGNORECASE)


SYSTEM_TEXT = """You are operating an unfamiliar deterministic grid world.
Use only the exact interaction record supplied by the environment. Keep multiple
hypotheses when the evidence does not distinguish them. Never infer a rule from
an action number, color name, or superficial layout alone. Predictions must be
concrete enough to compare against the next exact grid. Put the machine-readable
answer in the final <FINAL>{...}</FINAL> block. Do not put another JSON object
after that block."""


TASK_INSTRUCTIONS: dict[TaskKind, str] = {
    TaskKind.PREDICT_TRANSITION: (
        "Predict the exact persistent state change and whether the environment "
        "will report terminal success after the candidate action."
    ),
    TaskKind.INFER_ACTION: (
        "Score or state the possible meanings of the queried action. Preserve "
        "uncertainty for meanings not eliminated by observed transitions."
    ),
    TaskKind.INFER_GOAL: (
        "Infer terminal predicates consistent with every terminal and "
        "non-terminal observation."
    ),
    TaskKind.INFER_MECHANICS: (
        "Infer executable mechanics clauses that exactly replay the observed "
        "state transitions."
    ),
    TaskKind.PROPOSE_EXPERIMENT: (
        "Choose a legal intervention that safely distinguishes leading hidden-"
        "world hypotheses. Prefer information gain before planning to a goal."
    ),
    TaskKind.PROPOSE_PROGRAM: (
        "Write one typed executable world program. Do not add a clause that is "
        "not supported by an observation or required to explain a transition."
    ),
    TaskKind.REPAIR_PROGRAM: (
        "Repair the candidate program using the exact replay contradiction while "
        "preserving every earlier transition that already passed."
    ),
    TaskKind.CHOOSE_ACTION: (
        "Choose one legal action from the supplied set. Use an information-seeking "
        "action while decisive rules remain unknown; otherwise pursue the most "
        "supported terminal plan."
    ),
}


@dataclass(frozen=True)
class Prompt:
    task: TaskKind
    messages: tuple[dict[str, str], ...]
    legal_actions: tuple[str, ...]

    def plain_text(self) -> str:
        sections: list[str] = []
        for message in self.messages:
            sections.append(f"[{message['role'].upper()}]\n{message['content']}")
        return "\n\n".join(sections) + "\n\n[ASSISTANT]\n"


def normalize_actions(actions: Iterable[str | int]) -> tuple[str, ...]:
    normalized: list[str] = []
    for action in actions:
        text = str(action).strip().upper()
        match = ACTION_PATTERN.search(text)
        if match is None:
            raise ValueError(f"invalid ARC action: {action!r}")
        value = int(match.group(1))
        canonical = f"A{value}"
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def build_prompt(
    task: TaskKind | str,
    evidence: str,
    *,
    legal_actions: Sequence[str | int] = (),
    query: str = "",
    output_schema: Mapping[str, Any] | None = None,
) -> Prompt:
    kind = TaskKind(task)
    actions = normalize_actions(legal_actions)
    schema = dict(output_schema or default_schema(kind))
    user_sections = [
        f"TASK: {kind.value}",
        TASK_INSTRUCTIONS[kind],
        "EXACT EVIDENCE\n" + evidence.strip(),
    ]
    if query.strip():
        user_sections.append("QUERY\n" + query.strip())
    if actions:
        user_sections.append("LEGAL ACTIONS\n" + " ".join(actions))
    user_sections.append(
        "FINAL JSON SCHEMA\n" + json.dumps(schema, sort_keys=True, separators=(",", ":"))
    )
    user_sections.append(
        "Return the schema-compatible object inside <FINAL> and </FINAL>."
    )
    return Prompt(
        task=kind,
        messages=(
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ),
        legal_actions=actions,
    )


def default_schema(task: TaskKind) -> dict[str, Any]:
    if task in {TaskKind.CHOOSE_ACTION, TaskKind.PROPOSE_EXPERIMENT}:
        return {
            "action": "A0..A7",
            "x": "integer or null",
            "y": "integer or null",
            "confidence": "0..1",
            "purpose": "brief testable statement",
        }
    if task is TaskKind.PREDICT_TRANSITION:
        return {
            "changed_cells": [["row", "column", "old", "new"]],
            "terminal": "yes|no|unknown",
            "confidence": "0..1",
        }
    if task is TaskKind.INFER_ACTION:
        return {
            "action": "A0..A7",
            "possibilities": [{"meaning": "string", "probability": "0..1"}],
        }
    if task is TaskKind.INFER_GOAL:
        return {
            "possibilities": [{"goal": "typed predicate", "probability": "0..1"}]
        }
    if task is TaskKind.INFER_MECHANICS:
        return {
            "possibilities": [{"mechanics": "typed clauses", "probability": "0..1"}]
        }
    if task in {TaskKind.PROPOSE_PROGRAM, TaskKind.REPAIR_PROGRAM}:
        return {"program": "typed ARC DSL", "confidence": "0..1"}
    raise AssertionError(f"unhandled task: {task}")


def extract_final_json(text: str) -> dict[str, Any]:
    matches = list(FINAL_PATTERN.finditer(text))
    if not matches:
        raise ValueError("model output did not contain a <FINAL> JSON block")
    raw = matches[-1].group(1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid final JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("final JSON must be an object")
    return value


def extract_legal_action(text: str, legal_actions: Sequence[str | int]) -> tuple[str, int | None, int | None]:
    allowed = set(normalize_actions(legal_actions))
    if not allowed:
        raise ValueError("at least one legal action is required")

    try:
        payload = extract_final_json(text)
    except ValueError:
        payload = {}

    candidate = str(payload.get("action", "")).upper()
    match = ACTION_PATTERN.search(candidate)
    if match is not None:
        action = f"A{int(match.group(1))}"
        if action in allowed:
            return action, _optional_int(payload.get("x")), _optional_int(payload.get("y"))

    for match in reversed(list(ACTION_PATTERN.finditer(text))):
        action = f"A{int(match.group(1))}"
        if action in allowed:
            return action, None, None
    raise ValueError("model output did not contain a legal action")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("coordinate may not be boolean")
    return int(value)
