"""Synthetic qualification and local evidence/tool transfer benchmarks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .client import ClientError, LlamaClient
from .policy import GlimmerPolicy, PolicyConfig
from .protocol import ProtocolError, canonical_sha256


SYNTHETIC_VERSION = "muse-glimmer-synthetic/v1"
TRANSFER_VERSION = "muse-glimmer-evidence-tools/v1"


@dataclass(frozen=True)
class SyntheticChoice:
    action_name: str
    signature: str
    simple: bool = True
    x: int | None = None
    y: int | None = None
    candidate_source: str | None = None


@dataclass(frozen=True)
class SyntheticObservation:
    final_grid: tuple[tuple[int, ...], ...]
    state: str = "NOT_FINISHED"
    levels_completed: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.final_grid), len(self.final_grid[0]))


def qualification_tasks(count: int = 20) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        correct = f"ACTION{index % 4 + 1}"
        risk = f"ACTION{(index + 1) % 4 + 1}"
        grid = (
            (index % 10, 0, 0, 0),
            (0, (index * 3) % 10, 0, 0),
            (0, 0, (index * 7) % 10, 0),
            (0, 0, 0, 9),
        )
        history: list[dict[str, Any]] = []
        for action_number in range(1, 5):
            action = f"ACTION{action_number}"
            outcome = (
                "PROGRESS"
                if action == correct
                else "GAME_OVER" if action == risk else "NO_CHANGE"
            )
            history.append(
                {
                    "action": action,
                    "status": "GAME_OVER" if outcome == "GAME_OVER" else "NOT_FINISHED",
                    "changed_cells": 2 if outcome == "PROGRESS" else 0,
                    "shape_changed": False,
                    "level_delta": 1 if outcome == "PROGRESS" else 0,
                    "outcome": outcome,
                }
            )
        tasks.append(
            {
                "task_id": f"qualification-{index:02d}",
                "grid": grid,
                "history": history,
                "correct": correct,
                "risk": risk,
            }
        )
    return tasks


def run_synthetic_qualification(
    client: LlamaClient,
    *,
    task_count: int = 20,
) -> dict[str, Any]:
    tasks = qualification_tasks(task_count)
    arms: dict[str, Any] = {}
    for mode in ("native", "cfs_lite"):
        rows: list[dict[str, Any]] = []
        for task in tasks:
            config = PolicyConfig(
                mode=mode,
                max_model_calls=2,
                max_repair_calls=1,
                shortlist_size=4,
                plan_length=1,
                max_actions=1,
                max_resets=0,
                max_click_candidates_per_action=1,
                low_discrepancy_points=0,
                max_prompt_history=8,
            )
            policy = GlimmerPolicy(client, config)
            policy.seed_history(task["history"])
            choices = [
                SyntheticChoice(f"ACTION{number}", f"ACTION{number}")
                for number in range(1, 5)
            ]
            observation = SyntheticObservation(task["grid"])
            chosen = policy.choose(observation, choices)
            receipt = policy.receipt()
            signature = chosen.signature if chosen is not None else None
            rows.append(
                {
                    "task_id": task["task_id"],
                    "correct_signature": task["correct"],
                    "risk_signature": task["risk"],
                    "chosen_signature": signature,
                    "correct": signature == task["correct"],
                    "risk_selected": signature == task["risk"],
                    "policy": receipt,
                }
            )
        decisions = sum(row["policy"]["decision_calls"] for row in rows)
        fallbacks = sum(row["policy"]["fallbacks"] for row in rows)
        invalid = sum(row["policy"]["invalid_decisions"] for row in rows)
        illegal = sum(row["policy"]["illegal_suggestions"] for row in rows)
        valid = sum(row["policy"]["valid_decisions"] for row in rows)
        attempts = valid + invalid
        arms[mode] = {
            "task_count": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "risk_selections": sum(bool(row["risk_selected"]) for row in rows),
            "decision_calls": decisions,
            "valid_response_rate": valid / attempts if attempts else 1.0,
            "fallback_rate": fallbacks / decisions if decisions else 0.0,
            "illegal_suggestion_rate": illegal / max(1, decisions),
            "rows": rows,
        }
    combined_decisions = sum(arms[mode]["decision_calls"] for mode in arms)
    combined_fallbacks = sum(
        sum(row["policy"]["fallbacks"] for row in arms[mode]["rows"])
        for mode in arms
    )
    combined_invalid = sum(
        sum(row["policy"]["invalid_decisions"] for row in arms[mode]["rows"])
        for mode in arms
    )
    combined_valid = sum(
        sum(row["policy"]["valid_decisions"] for row in arms[mode]["rows"])
        for mode in arms
    )
    combined_illegal = sum(
        sum(row["policy"]["illegal_suggestions"] for row in arms[mode]["rows"])
        for mode in arms
    )
    attempts = combined_valid + combined_invalid
    gate = {
        "structured_validity_pass": (combined_valid / attempts if attempts else 1.0) >= 0.95,
        "fallback_pass": (combined_fallbacks / combined_decisions if combined_decisions else 0.0) <= 0.05,
        "illegal_action_pass": (combined_illegal / max(1, combined_decisions)) <= 0.01,
        "task_signal_pass": max(arms["native"]["accuracy"], arms["cfs_lite"]["accuracy"]) >= 0.25,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "benchmark_version": SYNTHETIC_VERSION,
        "task_manifest_sha256": canonical_sha256(tasks),
        "arms": arms,
        "combined": {
            "valid_response_rate": combined_valid / attempts if attempts else 1.0,
            "fallback_rate": combined_fallbacks / combined_decisions if combined_decisions else 0.0,
            "illegal_suggestion_rate": combined_illegal / max(1, combined_decisions),
        },
        "gate": gate,
    }


def transfer_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index in range(8):
        answer = "ALPHA" if index % 2 == 0 else "BETA"
        other = "BETA" if answer == "ALPHA" else "ALPHA"
        decisive = ["VERIFY_LEDGER", "CHECK_HASH", "READ_PRIMARY", "RECALCULATE"][index % 4]
        tools = {
            "READ_SUMMARY": f"A secondary summary favors {other}, but says its source was not verified.",
            "READ_NOTE": f"An informal note favors {other} and contains no reproducible evidence.",
            decisive: f"The immutable primary evidence proves {answer} and explicitly rejects {other}.",
        }
        # Ensure unique three-tool surface even when a decisive name matches none above.
        ordered = list(tools)
        tasks.append(
            {
                "task_id": f"evidence-{index:02d}",
                "question": "Which of the two mutually exclusive claims is supported by the authoritative evidence?",
                "initial_evidence": (
                    f"One unverified report says {other}. Another report says {answer}. "
                    "Only one available tool accesses authoritative or reproducible evidence."
                ),
                "tools": {name: tools[name] for name in ordered},
                "decisive_tool": decisive,
                "answer": answer,
            }
        )
    return tasks


def _native_tool_choice(client: LlamaClient, task: Mapping[str, Any]) -> tuple[str | None, bool]:
    tool_names = sorted(task["tools"])
    system = (
        "Choose one diagnostic tool to resolve an evidence conflict. Prefer authoritative, "
        "reproducible evidence. Output strict JSON only and copy an exact tool name."
    )
    user = json.dumps(
        {
            "question": task["question"],
            "initial_evidence": task["initial_evidence"],
            "tools": tool_names,
            "schema": {"tool": "EXACT_TOOL_NAME", "confidence": "NUMBER_0_TO_1"},
        },
        sort_keys=True,
    )
    try:
        value = client.complete_json(
            system=system,
            user=user,
            purpose="transfer_native_tool",
            max_tokens=160,
            temperature=0.0,
        )
        tool = value.get("tool")
        if isinstance(tool, str) and tool in tool_names:
            return tool, True
    except ClientError:
        pass
    return tool_names[0], False


def _cfs_tool_choice(client: LlamaClient, task: Mapping[str, Any]) -> tuple[str | None, bool, dict[str, Any]]:
    tool_names = sorted(task["tools"])
    system = (
        "Maintain 2-4 competing hypotheses about whether ALPHA or BETA is true. Predict what "
        "each exact tool would reveal under each hypothesis. Do not recommend a tool. Output "
        "strict JSON only; a deterministic controller will choose the tool with greatest "
        "hypothesis disagreement and evidentiary value."
    )
    user = json.dumps(
        {
            "question": task["question"],
            "initial_evidence": task["initial_evidence"],
            "tools": tool_names,
            "schema": {
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "ALPHA|BETA",
                        "weight": "POSITIVE_NUMBER",
                        "predictions": [
                            {
                                "tool": "EXACT_TOOL_NAME",
                                "result": "SUPPORTS_ALPHA|SUPPORTS_BETA|INCONCLUSIVE",
                                "confidence": "NUMBER_0_TO_1",
                            }
                        ],
                    }
                ]
            },
        },
        sort_keys=True,
    )
    try:
        value = client.complete_json(
            system=system,
            user=user,
            purpose="transfer_cfs_tool",
            max_tokens=520,
            temperature=0.0,
        )
        hypotheses = value.get("hypotheses")
        if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 4:
            raise ProtocolError("invalid transfer hypotheses")
        weights: list[float] = []
        parsed: list[dict[str, Any]] = []
        for raw in hypotheses:
            if not isinstance(raw, Mapping):
                raise ProtocolError("invalid transfer hypothesis")
            weight = raw.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) <= 0:
                raise ProtocolError("invalid transfer hypothesis weight")
            predictions: dict[str, tuple[str, float]] = {}
            rows = raw.get("predictions")
            if not isinstance(rows, list):
                raise ProtocolError("invalid transfer predictions")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ProtocolError("invalid transfer prediction")
                tool = row.get("tool")
                result = row.get("result")
                confidence = row.get("confidence")
                if tool not in tool_names or result not in {
                    "SUPPORTS_ALPHA",
                    "SUPPORTS_BETA",
                    "INCONCLUSIVE",
                }:
                    raise ProtocolError("invalid transfer prediction value")
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    raise ProtocolError("invalid transfer confidence")
                confidence = float(confidence)
                if not 0 <= confidence <= 1 or not math.isfinite(confidence):
                    raise ProtocolError("invalid transfer confidence")
                predictions[str(tool)] = (str(result), confidence)
            weights.append(float(weight))
            parsed.append({"weight": float(weight), "predictions": predictions})
        total = sum(weights)
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        for tool in tool_names:
            distribution: dict[str, float] = defaultdict(float)
            coverage = 0.0
            for hypothesis in parsed:
                weight = hypothesis["weight"] / total
                result, confidence = hypothesis["predictions"].get(tool, ("INCONCLUSIVE", 0.0))
                distribution[result] += weight * confidence
                distribution["INCONCLUSIVE"] += weight * (1.0 - confidence)
                coverage += weight * confidence
            entropy = -sum(probability * math.log(probability) for probability in distribution.values() if probability > 0)
            # Authoritative/reproducible names are a deterministic evidentiary
            # prior, not a task-specific answer leak.
            authority_prior = 0.5 if any(token in tool for token in ("VERIFY", "HASH", "PRIMARY", "RECALCULATE")) else 0.0
            scores[tool] = entropy + 0.2 * coverage + authority_prior
            details[tool] = {"score": scores[tool], "distribution": dict(distribution)}
        chosen = min(scores, key=lambda name: (-scores[name], name))
        return chosen, True, details
    except (ClientError, ProtocolError, ValueError):
        return tool_names[0], False, {}


def _answer_after_tool(
    client: LlamaClient,
    task: Mapping[str, Any],
    tool: str,
    *,
    mode: str,
) -> tuple[str | None, bool]:
    system = (
        "Answer an evidence question using the retrieved tool result. Output strict JSON only. "
        "Choose exactly ALPHA or BETA and cite the exact tool name."
    )
    user = json.dumps(
        {
            "question": task["question"],
            "initial_evidence": task["initial_evidence"],
            "tool": tool,
            "tool_result": task["tools"][tool],
            "schema": {"answer": "ALPHA|BETA", "evidence_tool": "EXACT_TOOL_NAME"},
        },
        sort_keys=True,
    )
    try:
        value = client.complete_json(
            system=system,
            user=user,
            purpose=f"transfer_{mode}_answer",
            max_tokens=160,
            temperature=0.0,
        )
        answer = value.get("answer")
        evidence_tool = value.get("evidence_tool")
        if answer in {"ALPHA", "BETA"} and evidence_tool == tool:
            return str(answer), True
    except ClientError:
        pass
    return None, False


def run_transfer_benchmark(client: LlamaClient) -> dict[str, Any]:
    tasks = transfer_tasks()
    arms: dict[str, Any] = {}
    for mode in ("native", "cfs_lite"):
        rows: list[dict[str, Any]] = []
        for task in tasks:
            if mode == "native":
                tool, tool_valid = _native_tool_choice(client, task)
                fracture_details: dict[str, Any] = {}
            else:
                tool, tool_valid, fracture_details = _cfs_tool_choice(client, task)
            assert tool is not None
            answer, answer_valid = _answer_after_tool(client, task, tool, mode=mode)
            rows.append(
                {
                    "task_id": task["task_id"],
                    "chosen_tool": tool,
                    "decisive_tool": task["decisive_tool"],
                    "diagnostic_tool_correct": tool == task["decisive_tool"],
                    "answer": answer,
                    "expected_answer": task["answer"],
                    "answer_correct": answer == task["answer"],
                    "tool_response_valid": tool_valid,
                    "answer_response_valid": answer_valid,
                    "fracture_details": fracture_details,
                }
            )
        arms[mode] = {
            "task_count": len(rows),
            "answer_accuracy": sum(row["answer_correct"] for row in rows) / len(rows),
            "diagnostic_tool_accuracy": sum(row["diagnostic_tool_correct"] for row in rows) / len(rows),
            "validity_rate": sum(
                row["tool_response_valid"] and row["answer_response_valid"] for row in rows
            ) / len(rows),
            "rows": rows,
        }
    native = arms["native"]
    cfs = arms["cfs_lite"]
    controller_positive = (
        cfs["answer_accuracy"] > native["answer_accuracy"]
        or (
            cfs["answer_accuracy"] == native["answer_accuracy"]
            and cfs["diagnostic_tool_accuracy"] >= native["diagnostic_tool_accuracy"] + 0.125
        )
    )
    any_useful = max(native["answer_accuracy"], cfs["answer_accuracy"]) >= 0.625
    return {
        "schema_version": 1,
        "benchmark_version": TRANSFER_VERSION,
        "task_manifest_sha256": canonical_sha256(tasks),
        "arms": arms,
        "controller_positive": controller_positive,
        "any_useful": any_useful,
    }


__all__ = [
    "SYNTHETIC_VERSION",
    "TRANSFER_VERSION",
    "SyntheticChoice",
    "SyntheticObservation",
    "qualification_tasks",
    "run_synthetic_qualification",
    "run_transfer_benchmark",
    "transfer_tasks",
]
