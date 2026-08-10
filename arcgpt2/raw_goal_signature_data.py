"""Deterministic data contract for :mod:`RAW_GOAL_SIGNATURE_V1`.

This module contains no learned code.  It constructs balanced counterfactual
groups, executes the existing mechanics and Goal DSL, and exposes deliberately
narrow text serializers.  Identifiers and labels live in metadata only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence

from .codec import Grid, normalize_grid
from .dsl import MoveColorRule, Program, execute
from .goal_dsl import (
    ColorAbsent,
    ColorCount,
    ColorsTouch,
    Comparison,
    Connectivity,
    ContactedColor,
    GoalProgram,
    evaluate_goal,
)
from .natural_protocol import grid_text
from .phase0_hidden_action import Action


PROTOCOL = "raw_goal_signature_v1"
FAMILY_NAMES = ("CONTACT", "ABSENT", "COUNT", "TOUCH")
SIGNATURES = ((False, False), (False, True), (True, False), (True, True))
SEMANTIC_MASKS = tuple(itertools.combinations(range(4), 2))
EVIDENCE_ORDERS = ((0, 1), (1, 0))
SEMANTIC_COLORS = (2, 3, 4, 5, 6, 7)
ROLE_NAMES = ("CONTACT", "ABSENT", "COUNT", "TOUCH_LEFT", "TOUCH_RIGHT")


class Family(str, Enum):
    CONTACT = "CONTACT"
    ABSENT = "ABSENT"
    COUNT = "COUNT"
    TOUCH = "TOUCH"


FAMILIES = tuple(Family(name) for name in FAMILY_NAMES)


@dataclass(frozen=True)
class GoalSignatureTrial:
    before: Grid
    action: Action
    after: Grid
    contacted_colors: tuple[int, ...]
    terminal_by_family: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class GoalSignatureWorld:
    family: Family
    identification_terminal: tuple[bool, bool]
    semantic_terminal: bool


@dataclass(frozen=True)
class GoalSignatureGroup:
    group_id: str
    level_identity: str
    mechanics: Program
    candidates: tuple[GoalProgram, GoalProgram, GoalProgram, GoalProgram]
    candidate_order: tuple[Family, Family, Family, Family]
    role_colors: tuple[int, int, int, int, int]
    signature_assignment: tuple[int, int, int, int]
    identification_trials: tuple[GoalSignatureTrial, GoalSignatureTrial]
    semantic_trial: GoalSignatureTrial
    worlds: tuple[GoalSignatureWorld, GoalSignatureWorld, GoalSignatureWorld, GoalSignatureWorld]
    surface_sha256: str
    evidence_orders: tuple[tuple[int, int], tuple[int, int]] = EVIDENCE_ORDERS


@dataclass(frozen=True)
class RejectedSurface:
    reason: str
    canonical_payload: Mapping[str, Any]


@dataclass(frozen=True)
class SplitManifest:
    name: str
    groups: tuple[GoalSignatureGroup, ...]
    rejected_surfaces: tuple[RejectedSurface, ...] = ()


_SPLIT_REPETITIONS = {"train": 5, "validation": 1, "locked_test": 3}
_MASK_SCHEDULES = {
    "train": (3,1,1,3,2,2,2,1,5,1,5,0,4,2,4,4,4,3,3,3,0,5,0,5,0,1,2,1,5,3,5,0,5,0,0,1,2,3,0,0,3,1,0,4,2,1,0,2,5,5,3,4,1,0,1,4,3,5,4,2,0,2,2,4,0,2,1,2,2,2,5,4,0,2,5,3,2,5,1,3,3,1,1,3,5,0,1,4,3,1,1,4,5,5,4,4,4,2,4,5,2,4,1,5,3,4,5,0,4,1,3,0,4,3,0,0,3,3,2,5),
    "validation": (4,5,1,5,0,1,1,0,4,4,2,4,2,0,3,1,5,3,3,5,2,0,3,2),
    "locked_test": (4,4,0,2,3,1,1,1,1,5,0,2,4,3,3,0,1,5,3,1,2,5,3,2,5,5,4,0,4,4,4,5,4,3,4,5,3,1,1,5,2,2,3,0,0,3,4,3,2,0,1,2,0,4,2,2,1,5,2,3,5,4,1,0,5,0,5,3,1,0,2,0),
}

# The model sees candidate order, semantic colors, and inert nuisance geometry.
# These schedules retain all exact marginals while ensuring that no
# (signature assignment, visible factor) pair recurs across data splits.  Thus
# a train-time lookup from any one visible factor to the hidden signature is
# invalid on validation and locked test.
_CANDIDATE_ORDER_SCHEDULES = {
    "train": tuple(
        (7 * permutation_index + offset) % 24
        for offset in (0, 5, 11, 17, 23)
        for permutation_index in range(24)
    ),
    "validation": (1,23,0,3,12,14,13,5,16,7,10,19,8,15,20,21,4,17,18,22,2,9,11,6),
    "locked_test": (3,10,21,13,16,1,22,3,20,11,1,0,22,8,14,23,10,18,13,2,8,11,4,15,7,3,4,22,8,23,7,2,11,10,4,14,14,5,6,18,0,0,21,9,5,5,12,1,16,9,17,7,20,18,20,19,12,23,16,6,15,21,9,19,17,15,2,19,6,17,13,12),
}

_ROLE_COLOR_TUPLES = (
    (2,3,4,5,6), (3,4,5,6,7), (4,5,6,7,2),
    (5,6,7,2,3), (6,7,2,3,4), (7,2,3,4,5),
    (2,7,6,5,4), (3,2,7,6,5), (4,3,2,7,6),
    (5,4,3,2,7), (6,5,4,3,2), (7,6,5,4,3),
)
_ROLE_COLOR_SCHEDULES = {
    "train": (4,3,0,0,2,6,1,7,8,7,11,5,5,8,7,8,4,2,2,6,11,5,2,0,3,2,5,3,9,4,8,4,2,0,0,8,1,11,5,11,9,11,0,3,9,9,8,6,11,11,9,8,10,5,4,3,6,2,4,10,2,6,4,10,2,7,7,8,3,8,1,1,1,10,1,4,3,0,10,8,0,9,5,11,9,1,10,3,0,4,9,10,6,10,10,9,6,5,7,5,0,11,11,6,7,5,3,9,10,4,3,7,1,1,6,1,7,2,6,7),
    "validation": (2,0,4,6,7,1,3,0,3,1,9,4,8,10,8,2,10,9,11,7,5,6,5,11),
    "locked_test": (9,6,8,1,11,8,9,5,5,6,6,3,0,0,11,6,6,10,3,0,1,1,3,5,5,4,6,9,8,7,5,9,9,10,10,2,7,7,2,0,7,3,8,11,10,7,7,2,8,1,3,11,1,3,2,10,10,11,8,1,4,2,9,5,11,0,4,2,4,4,0,4),
}

_NUISANCE_OFFSETS = {
    "train": (0, 1, 2, 3, 4),
    "validation": (5,),
    "locked_test": (6, 7, 8),
}


def _mechanics() -> Program:
    return Program(tuple(MoveColorRule(action, 1, 0, 1, 0) for action in (Action.A1, Action.A2, Action.A3)))


def _programs(colors: tuple[int, int, int, int, int]) -> tuple[GoalProgram, ...]:
    contact, absent, count, touch_left, touch_right = colors
    return (
        GoalProgram(ContactedColor(contact)),
        GoalProgram(ColorAbsent(absent)),
        GoalProgram(ColorCount(count, Comparison.EQ, 1)),
        GoalProgram(ColorsTouch(touch_left, touch_right, Connectivity.FOUR)),
    )


def _trial(
    truth: Sequence[bool], colors: tuple[int, int, int, int, int], trial_index: int,
    mechanics: Program, *, nuisance_position: tuple[int, int], height: int = 12, width: int = 12,
) -> GoalSignatureTrial:
    """Synthesize a real transition whose four DSL truths equal ``truth``."""

    if len(truth) != 4 or sum(bool(value) for value in truth) != 2:
        raise ValueError("every trial must make exactly two families true")
    contact, absent, count, touch_left, touch_right = colors
    start_row = 1 + trial_index
    start_col = 0
    cells: dict[tuple[int, int], int] = {(start_row, start_col): 1}
    if truth[0]:
        cells[(start_row, start_col + 1)] = contact
    else:
        cells[(height - 1, width - 1)] = contact
    if not truth[1]:
        cells[(height - 2, width - 2)] = absent
    cells[(height - 3, 1)] = count
    if not truth[2]:
        cells[(height - 3, 3)] = count
    cells[(height - 5, width - 4)] = touch_left
    cells[(height - 5, width - (3 if truth[3] else 2))] = touch_right
    # One of 24 repeatedly reused inert nuisance layouts.  The balanced layout
    # multiset is shuffled independently of every semantic schedule; unlike an
    # index watermark, no layout identifies a group or split.
    cells[nuisance_position] = 8
    canvas = [[0 for _ in range(width)] for _ in range(height)]
    for (row, column), color in cells.items():
        if canvas[row][column] != 0:
            raise RuntimeError("synthetic semantic roles overlap")
        canvas[row][column] = color
    before = normalize_grid(canvas)
    action = (Action.A1, Action.A2, Action.A3)[trial_index]
    execution = execute(mechanics, before, action)
    programs = _programs(colors)
    actual = tuple(evaluate_goal(mechanics, goal, before, action)[0] for goal in programs)
    if actual != tuple(truth):
        raise RuntimeError(f"Goal-DSL synthesis mismatch: wanted {tuple(truth)}, got {actual}")
    return GoalSignatureTrial(before, action, execution.after, execution.contacted_colors, actual)  # type: ignore[arg-type]


def _surface_payload(
    mechanics: Program, candidates: Sequence[GoalProgram],
    colors: Sequence[int], trials: Sequence[GoalSignatureTrial],
) -> dict[str, Any]:
    """Canonical surface excludes candidate order and every terminal label."""

    return {
        "protocol": PROTOCOL,
        "mechanics": mechanics.canonical_text(),
        "candidate_programs": sorted(goal.canonical_text() for goal in candidates),
        "role_colors": list(colors),
        "trials": [
            {
                "dimensions": [len(trial.before), len(trial.before[0])],
                "before": trial.before,
                "action": trial.action.value,
                "after": trial.after,
                "changed_cells": _changed_cells(trial),
            }
            for trial in trials
        ],
    }


def canonical_json_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _build_group(
    split: str,
    index: int,
    signature_perm: tuple[int, ...],
    order_perm: tuple[int, ...],
    mask_index: int,
    colors: tuple[int, int, int, int, int],
    nuisance_position: tuple[int, int],
) -> GoalSignatureGroup:
    mechanics = _mechanics()
    candidates = _programs(colors)
    trial_truths = [tuple(SIGNATURES[signature_perm[f]][bit] for f in range(4)) for bit in range(2)]
    semantic_truth = tuple(f in SEMANTIC_MASKS[mask_index] for f in range(4))
    trials = tuple(_trial(bits, colors, i, mechanics, nuisance_position=nuisance_position) for i, bits in enumerate((*trial_truths, semantic_truth)))
    level_identity = f"{split}-synthetic-reset-{index:03d}"
    payload = _surface_payload(mechanics, candidates, colors, trials)
    worlds = tuple(
        GoalSignatureWorld(FAMILIES[f], (trial_truths[0][f], trial_truths[1][f]), semantic_truth[f])
        for f in range(4)
    )
    return GoalSignatureGroup(
        group_id=f"{split}-{index:03d}",
        level_identity=level_identity,
        mechanics=mechanics,
        candidates=candidates,  # family order; candidate_order controls scoring order
        candidate_order=tuple(FAMILIES[f] for f in order_perm),  # type: ignore[arg-type]
        role_colors=colors,  # type: ignore[arg-type]
        signature_assignment=signature_perm,  # type: ignore[arg-type]
        identification_trials=(trials[0], trials[1]),
        semantic_trial=trials[2],
        worlds=worlds,  # type: ignore[arg-type]
        surface_sha256=canonical_json_sha256(payload),
    )


@lru_cache(maxsize=1)
def build_manifests() -> tuple[SplitManifest, SplitManifest, SplitManifest]:
    permutations = tuple(itertools.permutations(range(4)))
    nuisance_layouts = tuple((row, column) for row in (4, 5) for column in range(12))
    manifests: list[SplitManifest] = []
    for split in ("train", "validation", "locked_test"):
        groups: list[GoalSignatureGroup] = []
        repetitions = _SPLIT_REPETITIONS[split]
        masks = _MASK_SCHEDULES[split]
        candidate_schedule = _CANDIDATE_ORDER_SCHEDULES[split]
        color_schedule = _ROLE_COLOR_SCHEDULES[split]
        nuisance_schedule = tuple(
            nuisance_layouts[
                (5 * permutation_index + _NUISANCE_OFFSETS[split][repetition]) % 24
            ]
            for repetition in range(repetitions)
            for permutation_index in range(24)
        )
        if len(nuisance_schedule) != repetitions * 24:
            raise RuntimeError(f"invalid frozen nuisance schedule for {split}")
        if Counter(nuisance_schedule) != Counter({layout: repetitions for layout in nuisance_layouts}):
            raise RuntimeError(f"unbalanced frozen nuisance schedule for {split}")
        if not (
            len(candidate_schedule)
            == len(color_schedule)
            == len(masks)
            == repetitions * 24
        ):
            raise RuntimeError(f"invalid frozen visible-factor schedule for {split}")
        for repetition in range(repetitions):
            for permutation_index, signature_perm in enumerate(permutations):
                index = repetition * 24 + permutation_index
                groups.append(
                    _build_group(
                        split,
                        index,
                        signature_perm,
                        permutations[candidate_schedule[index]],
                        masks[index],
                        _ROLE_COLOR_TUPLES[color_schedule[index]],
                        nuisance_schedule[index],
                    )
                )
        # The runtime builder performs no search or rejection.  The field is
        # retained in the manifest schema and is therefore empty by design.
        manifests.append(SplitManifest(split, tuple(groups), rejected_surfaces=()))
    audit_manifests(tuple(manifests), raise_on_error=True)
    return tuple(manifests)  # type: ignore[return-value]


def _changed_cells(trial: GoalSignatureTrial) -> list[tuple[int, int, int, int]]:
    return [
        (r, c, trial.before[r][c], trial.after[r][c])
        for r in range(len(trial.before)) for c in range(len(trial.before[0]))
        if trial.before[r][c] != trial.after[r][c]
    ]


def raw_support_parts(world: GoalSignatureWorld, trial: GoalSignatureTrial, *, neutral_status: bool = False) -> tuple[str, str]:
    """Return raw-NLL prompt/target; only the terminal word varies by world."""

    prompt = grid_text(trial.before) + f"\nI applied {trial.action.value}.\nOUTCOME:"
    changes = _changed_cells(trial)
    lines = [f" Exactly {len(changes)} grid cells changed."]
    lines.extend(f" Row {r}, column {c}: color {old} became color {new}." for r, c, old, new in changes)
    if neutral_status:
        status = "nil"
    else:
        # The trial itself carries all four truths, so metadata family selects
        # the environment report without exposing that family in text.
        status = "yes" if trial.terminal_by_family[FAMILIES.index(world.family)] else "no"
    lines.append(f" Terminal success: {status}.")
    return prompt, "".join(lines)


def serialize_raw_support(world: GoalSignatureWorld, trial: GoalSignatureTrial, *, neutral_status: bool = False) -> str:
    prompt, target = raw_support_parts(world, trial, neutral_status=neutral_status)
    return prompt + target


def goal_query_text(group: GoalSignatureGroup) -> str:
    del group
    return "Infer the fixed terminal rule from the observed outcomes. The terminal rule is:"


def candidate_completion_text(group: GoalSignatureGroup) -> tuple[str, str, str, str]:
    contact, absent, count, touch_left, touch_right = group.role_colors
    # All candidates use exactly the same fixed-slot surface.  The runtime
    # tokenizer audit still fails closed: structural symmetry is not treated as
    # proof about a particular tokenizer revision.
    by_family = {
        Family.CONTACT: f" rule contact, primary color {contact}, secondary color {contact}, value 0, relation none.",
        Family.ABSENT: f" rule absent, primary color {absent}, secondary color {absent}, value 0, relation none.",
        Family.COUNT: f" rule count, primary color {count}, secondary color {count}, value 1, relation equal.",
        Family.TOUCH: f" rule touch, primary color {touch_left}, secondary color {touch_right}, value 0, relation four.",
    }
    return tuple(by_family[family] for family in group.candidate_order)  # type: ignore[return-value]


def semantic_query_text(group: GoalSignatureGroup) -> str:
    trial = group.semantic_trial
    prompt, target = raw_support_parts(group.worlds[0], trial, neutral_status=True)
    # Trial 3 receives no report or update; strip the neutral terminal field.
    outcome = target.rsplit(" Terminal success:", 1)[0]
    return prompt + outcome + " Terminal success:"


def terminal_completion_text() -> tuple[str, str, str]:
    return (" yes.", " no.", " nil.")


def audit_token_contract(group: GoalSignatureGroup, tokenizer: Any) -> dict[str, Any]:
    """Fail-closed token boundary audit for the pinned runtime tokenizer.

    The data module remains tokenizer agnostic; callers must provide the exact
    tokenizer used by the experiment.  Both Hugging Face ``encode`` and a
    minimal callable returning ``input_ids`` are supported.
    """

    def encode(text: str) -> tuple[int, ...]:
        if hasattr(tokenizer, "encode"):
            values = tokenizer.encode(text, add_special_tokens=False)
        else:
            values = tokenizer(text, add_special_tokens=False)["input_ids"]
        return tuple(int(value) for value in values)

    candidate_lengths = tuple(len(encode(text)) for text in candidate_completion_text(group))
    terminal_lengths = tuple(len(encode(text)) for text in terminal_completion_text())
    support_checks = []
    for trial in group.identification_trials:
        prompts_targets = [raw_support_parts(world, trial) for world in group.worlds]
        prompt_ids = [encode(prompt) for prompt, _ in prompts_targets]
        # Audit yes, no, and the neutral status as complete strings.  BPE can
        # merge a preceding space into the status token, so separately encoding
        # a text prefix ending in whitespace would be invalid.
        representative = prompts_targets[0][1]
        if representative.endswith("yes."):
            stem = representative[:-4]
        elif representative.endswith("no."):
            stem = representative[:-3]
        else:
            raise RuntimeError("raw support target lacks a terminal status")
        variants = (stem + "yes.", stem + "no.", stem + "nil.")
        target_ids = [encode(target) for target in variants]
        equal_length = len({len(ids) for ids in target_ids}) == 1
        differing_positions = tuple(
            index for index in range(min(map(len, target_ids)))
            if len({ids[index] for ids in target_ids}) > 1
        )
        contiguous_difference = bool(differing_positions) and differing_positions == tuple(
            range(differing_positions[0], differing_positions[-1] + 1)
        )
        outside_equal = all(
            len({ids[index] for ids in target_ids}) == 1
            for index in range(min(map(len, target_ids)))
            if index not in differing_positions
        )
        support_checks.append({
            "prompt_ids_equal": len(set(prompt_ids)) == 1,
            "target_lengths_equal": equal_length,
            "terminal_difference_contiguous": contiguous_difference,
            "ids_equal_outside_terminal_difference": outside_equal,
            "differing_positions": differing_positions,
        })
    passed = (
        len(set(candidate_lengths)) == 1
        and len(set(terminal_lengths)) == 1
        and all(all(value for key, value in check.items() if key != "differing_positions") for check in support_checks)
    )
    return {
        "passed": passed,
        "candidate_lengths": candidate_lengths,
        "terminal_lengths": terminal_lengths,
        "support_checks": support_checks,
    }


def audit_counterfactual_bytes(group: GoalSignatureGroup) -> dict[str, Any]:
    checks = []
    for trial in group.identification_trials:
        texts = [serialize_raw_support(world, trial) for world in group.worlds]
        normalized = [text.replace("Terminal success: yes.", "Terminal success: STATUS.").replace("Terminal success: no.", "Terminal success: STATUS.") for text in texts]
        checks.append({"normalized_equal": len(set(normalized)) == 1, "distinct_texts": len(set(texts)) == 2})
    forbidden = (group.group_id, group.level_identity, "signature_assignment", "candidate_index")
    model_texts = [goal_query_text(group), semantic_query_text(group), *candidate_completion_text(group)]
    model_texts.extend(serialize_raw_support(world, trial) for world in group.worlds for trial in group.identification_trials)
    return {
        "support_checks": checks,
        "metadata_absent": not any(token in text for token in forbidden for text in model_texts),
    }


def audit_manifest(manifest: SplitManifest) -> dict[str, Any]:
    n = len(manifest.groups)
    family_signature = Counter()
    family_position = Counter()
    family_color = Counter()
    named_role_color = Counter()
    mask_counts = Counter()
    signature_orders = Counter()
    candidate_orders = Counter()
    joint = Counter()
    role_color_tuples = Counter()
    nuisance_layouts = Counter()
    signature_candidate_edges: set[tuple[Any, ...]] = set()
    signature_color_edges: set[tuple[Any, ...]] = set()
    signature_nuisance_edges: set[tuple[Any, ...]] = set()
    errors: list[str] = []
    for group in manifest.groups:
        sig = tuple(group.signature_assignment)
        order = tuple(FAMILIES.index(family) for family in group.candidate_order)
        nuisance_cells = tuple(
            (row, column)
            for row, values in enumerate(group.identification_trials[0].before)
            for column, color in enumerate(values)
            if color == 8
        )
        if len(nuisance_cells) != 1:
            errors.append(f"{group.group_id}: expected exactly one inert nuisance cell")
            nuisance = (-1, -1)
        else:
            nuisance = nuisance_cells[0]
        signature_orders[sig] += 1
        candidate_orders[order] += 1
        joint[(sig, order)] += 1
        role_color_tuples[group.role_colors] += 1
        nuisance_layouts[nuisance] += 1
        signature_candidate_edges.add((sig, order))
        signature_color_edges.add((sig, group.role_colors))
        signature_nuisance_edges.add((sig, nuisance))
        mask = tuple(i for i, value in enumerate(group.semantic_trial.terminal_by_family) if value)
        mask_counts[mask] += 1
        for f, family in enumerate(FAMILIES):
            truth = int(group.semantic_trial.terminal_by_family[f])
            family_signature[(family.value, group.signature_assignment[f], truth)] += 1
            family_position[(family.value, group.candidate_order.index(family), truth)] += 1
            family_color[(family.value, group.role_colors[f], truth)] += 1
            actual_signature = tuple(trial.terminal_by_family[f] for trial in group.identification_trials)
            if actual_signature != SIGNATURES[group.signature_assignment[f]]:
                errors.append(f"{group.group_id}: DSL signature mismatch for {family.value}")
        # TOUCH has two independently named color roles; both are conditioned
        # on the TOUCH semantic target in the leakage audit.
        family_color[("TOUCH_RIGHT", group.role_colors[4], int(group.semantic_trial.terminal_by_family[3]))] += 1
        for role, color in zip(ROLE_NAMES, group.role_colors):
            named_role_color[(role, color)] += 1
        byte_audit = audit_counterfactual_bytes(group)
        if not all(item["normalized_equal"] and item["distinct_texts"] for item in byte_audit["support_checks"]):
            errors.append(f"{group.group_id}: counterfactual support leakage")
        if not byte_audit["metadata_absent"]:
            errors.append(f"{group.group_id}: metadata entered model text")
    expected_repetitions = n // 24
    if set(signature_orders.values()) != {expected_repetitions}:
        errors.append("signature permutations are not exactly balanced")
    if set(candidate_orders.values()) != {expected_repetitions}:
        errors.append("candidate orders are not exactly balanced")
    if set(mask_counts.values()) != {n // 6}:
        errors.append("semantic masks are not exactly balanced")
    if all(sig == order for sig, order in joint):
        errors.append("signature and candidate-order schedules are identically paired")
    if len(signature_candidate_edges) != n:
        errors.append("signature/candidate-order edges repeat within the split")
    if len(signature_color_edges) != n:
        errors.append("signature/role-color edges repeat within the split")
    if len(signature_nuisance_edges) != n:
        errors.append("signature/nuisance edges repeat within the split")
    if set(role_color_tuples.values()) != {n // len(_ROLE_COLOR_TUPLES)}:
        errors.append("full role-color tuples are not exactly balanced")
    if set(nuisance_layouts.values()) != {n // 24}:
        errors.append("nuisance layouts are not exactly balanced")
    for table, marginal, label in ((family_signature, n // 8, "signature"), (family_position, n // 8, "position"), (family_color, n // 12, "color")):
        if set(table.values()) != {marginal}:
            errors.append(f"family x {label} semantic truth is not exactly 50/50")
    if set(named_role_color.values()) != {n // 6}:
        errors.append("named semantic-role colors are not exactly balanced")
    return {
        "name": manifest.name,
        "groups": n,
        "errors": errors,
        "signature_permutation_counts": dict(signature_orders),
        "candidate_order_counts": dict(candidate_orders),
        "semantic_mask_counts": dict(mask_counts),
        "family_signature_truth": dict(family_signature),
        "family_position_truth": dict(family_position),
        "family_color_truth": dict(family_color),
        "named_role_color_counts": dict(named_role_color),
        "joint_schedule_unique_pairs": len(joint),
        "joint_schedule_max_multiplicity": max(joint.values(), default=0),
        "role_color_tuple_counts": dict(role_color_tuples),
        "nuisance_layout_counts": dict(nuisance_layouts),
        "signature_candidate_edge_count": len(signature_candidate_edges),
        "signature_color_edge_count": len(signature_color_edges),
        "signature_nuisance_edge_count": len(signature_nuisance_edges),
    }


def audit_manifests(manifests: Sequence[SplitManifest], *, raise_on_error: bool = False) -> dict[str, Any]:
    reports = {manifest.name: audit_manifest(manifest) for manifest in manifests}
    surfaces = {manifest.name: {group.surface_sha256 for group in manifest.groups} for manifest in manifests}
    overlaps = {
        f"{left}:{right}": sorted(surfaces[left].intersection(surfaces[right]))
        for left, right in itertools.combinations(surfaces, 2)
    }
    def edge_sets(manifest: SplitManifest) -> dict[str, set[str]]:
        result = {"candidate_order": set(), "role_colors": set(), "nuisance": set()}
        for group in manifest.groups:
            signature = list(group.signature_assignment)
            order = [FAMILIES.index(family) for family in group.candidate_order]
            nuisance = [
                [row, column]
                for row, values in enumerate(group.identification_trials[0].before)
                for column, color in enumerate(values)
                if color == 8
            ]
            result["candidate_order"].add(canonical_json_sha256({"signature": signature, "factor": order}))
            result["role_colors"].add(canonical_json_sha256({"signature": signature, "factor": list(group.role_colors)}))
            result["nuisance"].add(canonical_json_sha256({"signature": signature, "factor": nuisance}))
        return result

    edges = {manifest.name: edge_sets(manifest) for manifest in manifests}
    edge_overlaps = {
        factor: {
            f"{left}:{right}": sorted(edges[left][factor].intersection(edges[right][factor]))
            for left, right in itertools.combinations(edges, 2)
        }
        for factor in ("candidate_order", "role_colors", "nuisance")
    }
    errors = [f"{name}: {error}" for name, report in reports.items() for error in report["errors"]]
    errors.extend(f"surface overlap {pair}" for pair, values in overlaps.items() if values)
    errors.extend(
        f"signature/{factor} edge overlap {pair}"
        for factor, pairs in edge_overlaps.items()
        for pair, values in pairs.items()
        if values
    )
    result = {
        "splits": reports,
        "surface_overlaps": overlaps,
        "signature_factor_overlaps": edge_overlaps,
        "errors": errors,
    }
    if errors and raise_on_error:
        raise RuntimeError("; ".join(errors))
    return result


def canonical_manifest_payload(manifests: Sequence[SplitManifest]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "splits": [
            {
                "name": manifest.name,
                "groups": [asdict(group) for group in manifest.groups],
                "rejected_surfaces": [asdict(item) for item in manifest.rejected_surfaces],
            }
            for manifest in manifests
        ],
    }


def canonical_manifest_digest(manifests: Sequence[SplitManifest]) -> str:
    return canonical_json_sha256(canonical_manifest_payload(manifests))


def model_visible_surface_payload(group: GoalSignatureGroup) -> dict[str, Any]:
    """Reconstruct the exact status/order/identifier-free physical surface."""

    return _surface_payload(
        group.mechanics,
        group.candidates,
        group.role_colors,
        (*group.identification_trials, group.semantic_trial),
    )
