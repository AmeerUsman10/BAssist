"""Strict JSON/action protocol helpers for the Muse Glimmer utility gate.

The module deliberately has no model or ARC SDK dependency.  It owns only
canonical serialization, exact candidate validation, compact grid rendering,
and deterministic CFS-Lite scoring.  Hidden model reasoning is never required
or persisted; only final JSON objects are accepted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "muse-glimmer-action-protocol/v1"
PROMPT_VERSION = "muse-glimmer-prompts/2026-08-10-v1"
OUTCOMES = ("PROGRESS", "CHANGE", "NO_CHANGE", "GAME_OVER", "UNKNOWN")


class ProtocolError(ValueError):
    """A fail-closed protocol violation."""


@dataclass(frozen=True)
class Candidate:
    signature: str
    action_name: str
    simple: bool
    x: int | None = None
    y: int | None = None
    source: str | None = None

    def as_prompt_object(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "signature": self.signature,
            "action": self.action_name,
            "kind": "simple" if self.simple else "coordinate",
        }
        if not self.simple:
            value.update({"x": self.x, "y": self.y, "source": self.source})
        return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one top-level JSON object from a model response.

    A single Markdown fence is tolerated because several chat templates add it
    despite explicit JSON-only instructions.  Prose before or after the object
    is rejected: accepting it would make protocol-validity metrics ambiguous.
    """

    if not isinstance(text, str):
        raise ProtocolError("model content is not text")
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProtocolError("model content is not one strict JSON value") from exc
    if not isinstance(value, dict):
        raise ProtocolError("model JSON root must be an object")
    # Round-trip here also rejects NaN/Infinity under Python's permissive parser.
    canonical_json(value)
    return value


def encode_grid(grid: Sequence[Sequence[int]] | None) -> str:
    if grid is None:
        return "<NO_GRID>"
    rows = [tuple(int(cell) for cell in row) for row in grid]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ProtocolError("grid must be a non-empty rectangle")
    if any(cell < 0 or cell > 255 for row in rows for cell in row):
        raise ProtocolError("grid cells must be integers in [0,255]")
    return "\n".join(" ".join(str(cell) for cell in row) for row in rows)


def grid_sha256(grid: Sequence[Sequence[int]] | None) -> str | None:
    if grid is None:
        return None
    return sha256_text(encode_grid(grid))


def candidate_from_choice(choice: Any) -> Candidate:
    signature = str(getattr(choice, "signature"))
    action_name = str(getattr(choice, "action_name"))
    simple = bool(getattr(choice, "simple"))
    x = getattr(choice, "x", None)
    y = getattr(choice, "y", None)
    source = getattr(choice, "candidate_source", None)
    if not signature or not action_name:
        raise ProtocolError("candidate signature/action is empty")
    if simple:
        x = y = None
    elif not isinstance(x, int) or not isinstance(y, int):
        raise ProtocolError("coordinate candidate requires integer x/y")
    return Candidate(signature, action_name, simple, x, y, str(source) if source else None)


def validate_signature_list(
    raw: Any,
    *,
    allowed: Iterable[str],
    maximum: int,
) -> list[str]:
    if not isinstance(raw, list):
        raise ProtocolError("plan must be a list")
    allowed_set = set(allowed)
    output: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProtocolError("plan entries must be objects")
        signature = item.get("signature")
        if not isinstance(signature, str) or signature not in allowed_set:
            raise ProtocolError("plan contains a non-candidate signature")
        confidence = item.get("confidence", 0.5)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ProtocolError("plan confidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ProtocolError("plan confidence must lie in [0,1]")
        if signature not in output:
            output.append(signature)
        if len(output) >= maximum:
            break
    if not output:
        raise ProtocolError("plan contains no legal candidate")
    return output


def parse_native_plan(
    value: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    maximum: int,
) -> list[str]:
    if set(value) - {"plan", "uncertainty"}:
        raise ProtocolError("native response has unknown top-level fields")
    return validate_signature_list(value.get("plan"), allowed=allowed, maximum=maximum)


def _prediction_rows(raw: Any, allowed: set[str]) -> dict[str, tuple[str, float]]:
    if not isinstance(raw, list):
        raise ProtocolError("hypothesis predictions must be a list")
    output: dict[str, tuple[str, float]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ProtocolError("prediction entries must be objects")
        signature = row.get("signature")
        outcome = row.get("outcome")
        confidence = row.get("confidence")
        if not isinstance(signature, str) or signature not in allowed:
            raise ProtocolError("prediction references a non-candidate signature")
        if not isinstance(outcome, str) or outcome.upper() not in OUTCOMES:
            raise ProtocolError("prediction outcome is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ProtocolError("prediction confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ProtocolError("prediction confidence must lie in [0,1]")
        if signature in output:
            raise ProtocolError("hypothesis predicts one candidate twice")
        output[signature] = (outcome.upper(), confidence)
    return output


def parse_cfs_packet(
    value: Mapping[str, Any],
    *,
    allowed: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if set(value) - {"hypotheses", "unsafe_signatures", "uncertainty"}:
        raise ProtocolError("CFS response has unknown top-level fields")
    raw_hypotheses = value.get("hypotheses")
    if not isinstance(raw_hypotheses, list) or not 2 <= len(raw_hypotheses) <= 5:
        raise ProtocolError("CFS response requires 2-5 hypotheses")
    allowed_set = set(allowed)
    seen_ids: set[str] = set()
    hypotheses: list[dict[str, Any]] = []
    total_weight = 0.0
    for raw in raw_hypotheses:
        if not isinstance(raw, Mapping):
            raise ProtocolError("hypothesis must be an object")
        hid = raw.get("id")
        weight = raw.get("weight")
        if not isinstance(hid, str) or not hid or hid in seen_ids:
            raise ProtocolError("hypothesis id is missing or duplicated")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ProtocolError("hypothesis weight must be numeric")
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ProtocolError("hypothesis weight must be finite and positive")
        predictions = _prediction_rows(raw.get("predictions"), allowed_set)
        if not predictions:
            raise ProtocolError("hypothesis has no predictions")
        seen_ids.add(hid)
        total_weight += weight
        hypotheses.append({"id": hid, "weight": weight, "predictions": predictions})
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ProtocolError("hypothesis weights are invalid")
    for hypothesis in hypotheses:
        hypothesis["weight"] = hypothesis["weight"] / total_weight

    unsafe: dict[str, float] = {}
    raw_unsafe = value.get("unsafe_signatures", [])
    if not isinstance(raw_unsafe, list):
        raise ProtocolError("unsafe_signatures must be a list")
    for item in raw_unsafe:
        if not isinstance(item, Mapping):
            raise ProtocolError("unsafe signature entries must be objects")
        signature = item.get("signature")
        probability = item.get("probability")
        if not isinstance(signature, str) or signature not in allowed_set:
            raise ProtocolError("unsafe entry references a non-candidate")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ProtocolError("unsafe probability must be numeric")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ProtocolError("unsafe probability must lie in [0,1]")
        unsafe[signature] = max(unsafe.get(signature, 0.0), probability)
    return hypotheses, unsafe


def _entropy(probabilities: Iterable[float]) -> float:
    values = [float(value) for value in probabilities if value > 0.0]
    return -sum(value * math.log(value) for value in values)


def cfs_action_scores(
    hypotheses: Sequence[Mapping[str, Any]],
    unsafe: Mapping[str, float],
    candidates: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Score actions by model-predicted hypothesis fracture plus progress/risk.

    The model supplies only declarative hypotheses.  The deterministic shell
    computes disagreement and selects the action, preventing a persuasive final
    recommendation from bypassing the intended epistemic controller.
    """

    progress_value = {
        "PROGRESS": 2.5,
        "CHANGE": 0.7,
        "NO_CHANGE": -0.15,
        "GAME_OVER": -3.0,
        "UNKNOWN": 0.0,
    }
    output: dict[str, dict[str, float]] = {}
    for signature in candidates:
        distribution: dict[str, float] = defaultdict(float)
        expected_progress = 0.0
        coverage = 0.0
        model_game_over = 0.0
        for hypothesis in hypotheses:
            weight = float(hypothesis["weight"])
            prediction = hypothesis["predictions"].get(signature)
            if prediction is None:
                outcome, confidence = "UNKNOWN", 0.0
            else:
                outcome, confidence = prediction
            # Confidence leaves the residual mass as UNKNOWN rather than
            # pretending a low-confidence point estimate is certain.
            confident_mass = weight * float(confidence)
            distribution[str(outcome)] += confident_mass
            distribution["UNKNOWN"] += weight - confident_mass
            expected_progress += confident_mass * progress_value[str(outcome)]
            model_game_over += confident_mass * float(str(outcome) == "GAME_OVER")
            coverage += confident_mass
        entropy = _entropy(distribution.values())
        max_entropy = math.log(max(2, len([value for value in distribution.values() if value > 0])))
        normalized_disagreement = entropy / max_entropy if max_entropy > 0 else 0.0
        risk = max(float(unsafe.get(signature, 0.0)), model_game_over)
        score = 2.0 * normalized_disagreement + expected_progress - 2.5 * risk + 0.1 * coverage
        output[signature] = {
            "score": score,
            "disagreement": normalized_disagreement,
            "expected_progress": expected_progress,
            "risk": risk,
            "coverage": coverage,
        }
    return output


def choose_cfs_action(
    hypotheses: Sequence[Mapping[str, Any]],
    unsafe: Mapping[str, float],
    candidates: Sequence[str],
) -> tuple[str, dict[str, dict[str, float]]]:
    scores = cfs_action_scores(hypotheses, unsafe, candidates)
    if not scores:
        raise ProtocolError("CFS packet produced no candidate scores")
    chosen = min(scores, key=lambda signature: (-scores[signature]["score"], signature))
    return chosen, scores


def summarize_delta(
    before: Sequence[Sequence[int]] | None,
    after: Sequence[Sequence[int]] | None,
) -> dict[str, Any]:
    if before is None or after is None:
        return {"comparable": False, "changed_cells": None, "shape_changed": before != after}
    before_rows = [tuple(row) for row in before]
    after_rows = [tuple(row) for row in after]
    if not before_rows or not after_rows or len(before_rows) != len(after_rows) or len(before_rows[0]) != len(after_rows[0]):
        return {"comparable": False, "changed_cells": None, "shape_changed": True}
    changed = sum(
        int(left != right)
        for left_row, right_row in zip(before_rows, after_rows)
        for left, right in zip(left_row, right_row)
    )
    return {"comparable": True, "changed_cells": changed, "shape_changed": False}


__all__ = [
    "Candidate",
    "OUTCOMES",
    "PROMPT_VERSION",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "candidate_from_choice",
    "canonical_json",
    "canonical_sha256",
    "cfs_action_scores",
    "choose_cfs_action",
    "encode_grid",
    "extract_json_object",
    "grid_sha256",
    "parse_cfs_packet",
    "parse_native_plan",
    "sha256_text",
    "summarize_delta",
]
