"""One-Instella ranker for executable navigation hypotheses.

The model is never asked to invent a legal action sequence from scratch here.
Deterministic code supplies executable candidates. Instella receives exact state
facts, action-effect evidence, and candidate plans, then selects the hypothesis
most likely to yield level progress. A strict parser and deterministic fallback
prevent malformed generations from controlling the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Sequence

from arcgpt2.official_observation import OfficialFrameSequence

from .action_belief import ActionBeliefState
from .backend import GenerationBackend
from .controller import HeuristicPlanRanker, RankedPlan
from .navigation import NavigationPlan
from .prompts import Prompt, TaskKind
from .world_state import GridFacts, concise_grid_facts


CHOICE_PATTERN = re.compile(
    r"<FINAL>\s*\{[^{}]*[\"']choice[\"']\s*:\s*(\d+)[^{}]*\}\s*</FINAL>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ModelRankReceipt:
    prompt_sha256: str
    candidate_count: int
    selected_index: int | None
    parsed: bool
    used_fallback: bool
    raw_output_tail: str
    error: str | None


def parse_choice(text: str, candidate_count: int) -> int:
    matches = list(CHOICE_PATTERN.finditer(text))
    if not matches:
        raise ValueError("model output lacks a valid <FINAL> choice object")
    choice = int(matches[-1].group(1))
    if choice < 0 or choice >= candidate_count:
        raise ValueError(
            f"model choice {choice} is outside 0..{candidate_count - 1}"
        )
    return choice


def _receipt_lines(belief: ActionBeliefState, *, limit: int = 16) -> list[str]:
    lines: list[str] = []
    for observation in belief.transitions[-limit:]:
        metadata = getattr(observation, "metadata", {})
        changed = metadata.get("changed_cells", []) if isinstance(metadata, dict) else []
        moved = metadata.get("moved_components", []) if isinstance(metadata, dict) else []
        lines.append(
            "OBSERVED "
            f"ACTION={observation.action} COORD={observation.coordinate} "
            f"UNCHANGED={observation.effect.unchanged} "
            f"CHANGED={observation.effect.changed_cell_count} "
            f"TRANSLATIONS={observation.effect.translation_vectors} "
            f"COLORS_ADDED={observation.effect.colors_added} "
            f"COLORS_REMOVED={observation.effect.colors_removed} "
            f"PROGRESS={observation.effect.level_progress} "
            f"TERMINAL={observation.effect.terminal_state} "
            f"MOVED={json.dumps(moved, sort_keys=True, separators=(',', ':'))} "
            f"CELLS={json.dumps(changed[:24], sort_keys=True, separators=(',', ':'))}"
        )
    return lines


def _diverse_plans(
    plans: Sequence[NavigationPlan],
    *,
    limit: int,
) -> tuple[tuple[int, NavigationPlan], ...]:
    """Keep short candidates while preserving target/relation diversity."""

    selected: list[tuple[int, NavigationPlan]] = []
    seen: set[tuple[int, int, str, tuple[int, int]]] = set()
    for index, plan in enumerate(plans):
        key = (
            plan.agent.color,
            plan.target.color,
            plan.target.relation,
            plan.destination_top_left,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append((index, plan))
        if len(selected) >= limit:
            break
    if not selected and plans:
        selected.append((0, plans[0]))
    return tuple(selected)


def build_plan_prompt(
    *,
    observation: OfficialFrameSequence,
    facts: GridFacts,
    belief: ActionBeliefState,
    candidates: Sequence[tuple[int, NavigationPlan]],
) -> Prompt:
    sections = [
        "TASK: select one executable goal hypothesis for an unfamiliar deterministic grid world.",
        (
            "Use only the exact evidence below. Each candidate is already legal and "
            "collision-checked under its stated assumptions. Select the candidate "
            "most likely to increase LEVELS_COMPLETED. Do not choose by action "
            "number or color convention. Treat target relation as a hypothesis, "
            "not a fact. Prefer plans supported by observed progress or repeated "
            "mechanics. Return exactly <FINAL>{\"choice\":N,\"reason\":\"brief\"}</FINAL>."
        ),
        (
            f"OBSERVATION GAME={observation.game_id} STATE={observation.state} "
            f"LEVELS={observation.levels_completed} WIN_LEVELS={observation.win_levels} "
            f"AVAILABLE_ACTIONS={observation.available_actions}"
        ),
        "CURRENT EXACT FACTS\n" + concise_grid_facts(facts),
        "ACTION EFFECT POSTERIOR\n" + belief.action_summary(observation.available_actions),
    ]
    receipt_lines = _receipt_lines(belief)
    sections.append(
        "RECENT EXACT RECEIPTS\n" + ("\n".join(receipt_lines) if receipt_lines else "NONE")
    )
    candidate_lines: list[str] = []
    for choice, (_, plan) in enumerate(candidates):
        candidate_lines.append(
            f"CHOICE {choice} AGENT_COLOR={plan.agent.color} "
            f"AGENT_SHAPE={plan.agent.shape} TARGET_COLOR={plan.target.color} "
            f"TARGET_CELLS={plan.target.cells} RELATION={plan.target.relation} "
            f"START={plan.start_top_left} DESTINATION={plan.destination_top_left} "
            f"LENGTH={plan.length} ACTIONS={plan.action_sequence} "
            f"PASSABLE_COLORS={plan.assumed_passable_colors} "
            f"HEURISTIC_PRIORITY={plan.target.heuristic_priority:.6f}"
        )
    sections.append("EXECUTABLE CANDIDATES\n" + "\n".join(candidate_lines))
    return Prompt(
        task=TaskKind.CHOOSE_ACTION,
        messages=(
            {
                "role": "system",
                "content": (
                    "You infer latent goals in exact deterministic environments. "
                    "Keep hypotheses separate from observations. Output only one "
                    "machine-readable final choice after any private reasoning."
                ),
            },
            {"role": "user", "content": "\n\n".join(sections)},
        ),
        legal_actions=(),
    )


@dataclass
class InstellaPlanRanker:
    backend: GenerationBackend
    max_candidates: int = 12
    max_new_tokens: int = 768
    fallback: HeuristicPlanRanker = field(default_factory=HeuristicPlanRanker)
    receipts: list[ModelRankReceipt] = field(default_factory=list)

    def rank(
        self,
        *,
        observation: OfficialFrameSequence,
        facts: GridFacts,
        belief: ActionBeliefState,
        plans: Sequence[NavigationPlan],
    ) -> RankedPlan:
        if not plans:
            raise ValueError("cannot rank an empty plan list")
        shortlisted = _diverse_plans(plans, limit=self.max_candidates)
        prompt = build_plan_prompt(
            observation=observation,
            facts=facts,
            belief=belief,
            candidates=shortlisted,
        )
        import hashlib

        prompt_sha = hashlib.sha256(
            prompt.plain_text().encode("utf-8")
        ).hexdigest()
        raw = ""
        try:
            raw = self.backend.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
            local_choice = parse_choice(raw, len(shortlisted))
            original_index = shortlisted[local_choice][0]
            self.receipts.append(
                ModelRankReceipt(
                    prompt_sha256=prompt_sha,
                    candidate_count=len(shortlisted),
                    selected_index=original_index,
                    parsed=True,
                    used_fallback=False,
                    raw_output_tail=raw[-4000:],
                    error=None,
                )
            )
            return RankedPlan(
                index=original_index,
                score=1.0,
                reason="Instella selected an executable latent-goal hypothesis",
            )
        except Exception as exc:
            fallback = self.fallback.rank(
                observation=observation,
                facts=facts,
                belief=belief,
                plans=plans,
            )
            self.receipts.append(
                ModelRankReceipt(
                    prompt_sha256=prompt_sha,
                    candidate_count=len(shortlisted),
                    selected_index=fallback.index,
                    parsed=False,
                    used_fallback=True,
                    raw_output_tail=raw[-4000:],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return fallback
