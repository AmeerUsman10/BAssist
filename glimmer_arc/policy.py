"""Bounded Muse Glimmer policies and ARC-compatible game runner."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence

from .client import ClientError, LlamaClient
from .protocol import (
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    candidate_from_choice,
    canonical_sha256,
    choose_cfs_action,
    encode_grid,
    grid_sha256,
    parse_cfs_packet,
    parse_native_plan,
    summarize_delta,
)


POLICY_VERSION = "muse-glimmer-policy/v1"
MODES = {"native", "cfs_lite"}


@dataclass(frozen=True)
class PolicyConfig:
    mode: str
    max_model_calls: int = 8
    max_repair_calls: int = 1
    shortlist_size: int = 16
    plan_length: int = 4
    max_actions: int = 96
    max_resets: int = 2
    max_click_candidates_per_action: int = 24
    low_discrepancy_points: int = 16
    max_prompt_history: int = 8
    representation: str = "exact_text_grid"
    reasoning_strength: str = "high"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown policy mode: {self.mode}")
        for name in (
            "max_model_calls",
            "max_repair_calls",
            "shortlist_size",
            "plan_length",
            "max_actions",
            "max_resets",
            "max_click_candidates_per_action",
            "low_discrepancy_points",
            "max_prompt_history",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.shortlist_size < 2 or self.plan_length < 1:
            raise ValueError("shortlist_size/plan_length are too small")
        if self.low_discrepancy_points > self.max_click_candidates_per_action:
            raise ValueError("low_discrepancy_points exceeds click candidate bound")
        if self.representation != "exact_text_grid":
            raise ValueError("V1 freezes exact_text_grid for scored play")


class GlimmerPolicy:
    """One per-game model adapter with bounded calls and strict fallbacks."""

    def __init__(self, client: LlamaClient, config: PolicyConfig) -> None:
        config.validate()
        self.client = client
        self.config = config
        self._plan: deque[str] = deque()
        self._transitions: list[dict[str, Any]] = []
        self._model_calls = 0
        self._repair_calls = 0
        self._decision_calls = 0
        self._valid_decisions = 0
        self._invalid_decisions = 0
        self._illegal_suggestions = 0
        self._fallbacks = 0
        self._chosen: list[dict[str, Any]] = []
        self._decision_receipts: list[dict[str, Any]] = []
        self._last_levels: int | None = None
        self._last_state: str | None = None
        self._last_grid: Any = None
        self._client_start = len(client.receipts)

    # Existing TransitionFrontierAgent provider seam.
    def reset(self) -> None:
        self._plan.clear()
        self._transitions.clear()
        self._last_levels = None
        self._last_state = None
        self._last_grid = None

    def observe(
        self,
        action_name: str,
        before_grid: Any,
        after_grid: Any,
        status: str,
    ) -> dict[str, Any]:
        delta = summarize_delta(before_grid, after_grid)
        item = {
            "index": len(self._transitions) + 1,
            "action": str(action_name),
            "before_grid_sha256": grid_sha256(before_grid),
            "after_grid_sha256": grid_sha256(after_grid),
            "status": str(status),
            **delta,
        }
        self._transitions.append(item)
        if str(action_name).upper() == "RESET" or str(status).rsplit(".", 1)[-1].upper() == "GAME_OVER":
            self._plan.clear()
        self._last_grid = after_grid
        self._last_state = str(status)
        return item

    def direction_scores(self, action_name: str) -> Mapping[str, float]:
        # The custom runner performs one batched choice call across all simple
        # and coordinate actions.  This legacy method remains deterministic so
        # the provider satisfies the existing protocol without extra inference.
        return {"UNSPECIFIED": 0.0}

    def seed_history(self, history: Sequence[Mapping[str, Any]]) -> None:
        """Synthetic-test seam for controlled prior transition evidence."""
        self._transitions = [dict(item) for item in history]

    @property
    def call_budget_remaining(self) -> int:
        return max(0, self.config.max_model_calls - self._model_calls)

    def _history_for_prompt(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in self._transitions[-self.config.max_prompt_history :]:
            output.append(
                {
                    "action": item.get("action"),
                    "status": item.get("status"),
                    "changed_cells": item.get("changed_cells"),
                    "shape_changed": item.get("shape_changed"),
                    "level_delta": item.get("level_delta"),
                    "outcome": item.get("outcome"),
                }
            )
        return output

    @staticmethod
    def _state_fields(observation: Any) -> tuple[Any, str, int | None, tuple[int, int] | None]:
        grid = getattr(observation, "final_grid", None)
        state = str(getattr(observation, "state", "UNKNOWN"))
        levels = getattr(observation, "levels_completed", None)
        levels = levels if isinstance(levels, int) and not isinstance(levels, bool) else None
        shape = getattr(observation, "shape", None)
        if shape is None and grid is not None:
            try:
                shape = (len(grid), len(grid[0]))
            except Exception:
                shape = None
        if shape is not None:
            shape = tuple(int(value) for value in shape)
        return grid, state, levels, shape

    @staticmethod
    def _choice_sort_key(choice: Any, tried: set[str], uses: Mapping[str, int]) -> tuple[Any, ...]:
        return (
            0 if str(getattr(choice, "signature")) not in tried else 1,
            0 if bool(getattr(choice, "simple")) else 1,
            0 if getattr(choice, "candidate_source", None) == "non_modal_component" else 1,
            int(uses.get(str(getattr(choice, "signature")), 0)),
            str(getattr(choice, "signature")),
        )

    def _shortlist(self, choices: Sequence[Any], agent: Any | None) -> list[Any]:
        tried: set[str] = set()
        uses: Mapping[str, int] = {}
        if agent is not None:
            try:
                from arcgpt2.official_score_agent import _policy_state_sha256

                node = agent._nodes[_policy_state_sha256(agent.current)]
                tried = set(node.tried_signatures)
                uses = agent._signature_uses
            except Exception:
                tried = set()
                uses = {}
        ordered = sorted(choices, key=lambda choice: self._choice_sort_key(choice, tried, uses))
        # Preserve at least one candidate of every action name before filling by
        # the deterministic frontier order.  This avoids a coordinate-heavy
        # action crowding all opaque simple interventions out of the prompt.
        output: list[Any] = []
        seen_actions: set[str] = set()
        for choice in ordered:
            name = str(getattr(choice, "action_name"))
            if name not in seen_actions:
                output.append(choice)
                seen_actions.add(name)
                if len(output) == self.config.shortlist_size:
                    return output
        for choice in ordered:
            if choice not in output:
                output.append(choice)
                if len(output) == self.config.shortlist_size:
                    break
        return output

    def _system_prompt(self) -> str:
        mode_instruction = (
            "Return a short exact action plan."
            if self.config.mode == "native"
            else (
                "Return competing hypotheses and predictions only. The deterministic "
                "controller, not you, will choose the action."
            )
        )
        return (
            "You control an unknown visual environment through opaque legal actions. "
            "Infer from the exact grid and observed transitions. Aim to complete levels, "
            "prefer diagnostic changes over repeated no-ops, and avoid GAME_OVER risk. "
            "Use only exact candidate signatures copied character-for-character. "
            "Never invent actions or coordinates. Output one strict JSON object and no prose. "
            f"Reasoning strength: {self.config.reasoning_strength}. {mode_instruction}"
        )

    def _user_prompt(self, observation: Any, shortlist: Sequence[Any], *, repair: bool) -> str:
        grid, state, levels, shape = self._state_fields(observation)
        candidates = [candidate_from_choice(choice).as_prompt_object() for choice in shortlist]
        common = {
            "protocol": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "mode": self.config.mode,
            "state": state,
            "levels_completed": levels,
            "shape": list(shape) if shape else None,
            "grid": encode_grid(grid),
            "recent_transitions": self._history_for_prompt(),
            "candidates": candidates,
        }
        if self.config.mode == "native":
            common["required_schema"] = {
                "plan": [
                    {
                        "signature": "EXACT_CANDIDATE_SIGNATURE",
                        "confidence": "NUMBER_0_TO_1",
                        "expected": "progress|change|probe|no_change|risk",
                    }
                ],
                "uncertainty": "optional short label",
            }
            common["constraints"] = {
                "max_plan_actions": self.config.plan_length,
                "all_signatures_must_be_candidates": True,
            }
        else:
            common["required_schema"] = {
                "hypotheses": [
                    {
                        "id": "H1",
                        "weight": "POSITIVE_NUMBER",
                        "predictions": [
                            {
                                "signature": "EXACT_CANDIDATE_SIGNATURE",
                                "outcome": "PROGRESS|CHANGE|NO_CHANGE|GAME_OVER|UNKNOWN",
                                "confidence": "NUMBER_0_TO_1",
                            }
                        ],
                    }
                ],
                "unsafe_signatures": [
                    {"signature": "EXACT_CANDIDATE_SIGNATURE", "probability": "NUMBER_0_TO_1"}
                ],
                "uncertainty": "optional short label",
            }
            common["constraints"] = {
                "hypothesis_count_min": 2,
                "hypothesis_count_max": 5,
                "predict_multiple_candidates_per_hypothesis": True,
            }
        if repair:
            common["repair"] = (
                "A previous response failed schema validation. Return only the required "
                "object using exact candidate signatures."
            )
        return json.dumps(common, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _request_decision(self, observation: Any, shortlist: Sequence[Any]) -> tuple[list[str], dict[str, Any]]:
        allowed = [str(getattr(choice, "signature")) for choice in shortlist]
        last_error: str | None = None
        attempts = 1 + min(self.config.max_repair_calls, self.call_budget_remaining - 1)
        attempts = max(1, attempts)
        for attempt in range(attempts):
            if self.call_budget_remaining <= 0:
                break
            repair = attempt > 0
            self._model_calls += 1
            if repair:
                self._repair_calls += 1
            purpose = f"arc_{self.config.mode}_{'repair' if repair else 'decision'}"
            try:
                value = self.client.complete_json(
                    system=self._system_prompt(),
                    user=self._user_prompt(observation, shortlist, repair=repair),
                    purpose=purpose,
                    max_tokens=700 if self.config.mode == "cfs_lite" else 320,
                    temperature=0.0,
                    timeout=120.0,
                )
                if self.config.mode == "native":
                    plan = parse_native_plan(
                        value, allowed=allowed, maximum=self.config.plan_length
                    )
                    receipt = {
                        "mode": self.config.mode,
                        "parsed_sha256": canonical_sha256(value),
                        "plan": list(plan),
                        "cfs_scores": None,
                        "repair": repair,
                    }
                    return plan, receipt
                hypotheses, unsafe = parse_cfs_packet(value, allowed=allowed)
                chosen, scores = choose_cfs_action(hypotheses, unsafe, allowed)
                ordered = sorted(scores, key=lambda signature: (-scores[signature]["score"], signature))
                plan = ordered[: self.config.plan_length]
                if chosen != plan[0]:
                    raise ProtocolError("CFS controller ordering mismatch")
                receipt = {
                    "mode": self.config.mode,
                    "parsed_sha256": canonical_sha256(value),
                    "plan": list(plan),
                    "cfs_scores": scores,
                    "hypothesis_count": len(hypotheses),
                    "repair": repair,
                }
                return list(plan), receipt
            except (ClientError, ProtocolError, ValueError) as exc:
                last_error = type(exc).__name__
        raise ProtocolError(last_error or "decision_failed")

    def choose(self, observation: Any, choices: Sequence[Any], *, agent: Any | None = None) -> Any | None:
        self._decision_calls += 1
        by_signature = {str(getattr(choice, "signature")): choice for choice in choices}
        while self._plan:
            signature = self._plan.popleft()
            choice = by_signature.get(signature)
            if choice is not None:
                self._chosen.append(
                    {
                        "decision": self._decision_calls,
                        "signature": signature,
                        "source": "cached_plan",
                    }
                )
                return choice
            self._illegal_suggestions += 1
        if not choices:
            return None

        shortlist = self._shortlist(choices, agent)
        decision_receipt: dict[str, Any]
        if self.call_budget_remaining > 0:
            try:
                plan, decision_receipt = self._request_decision(observation, shortlist)
                self._valid_decisions += 1
                self._plan.extend(plan)
                self._decision_receipts.append(
                    {
                        "index": len(self._decision_receipts) + 1,
                        "candidate_count": len(shortlist),
                        **decision_receipt,
                    }
                )
                while self._plan:
                    signature = self._plan.popleft()
                    choice = by_signature.get(signature)
                    if choice is not None:
                        self._chosen.append(
                            {
                                "decision": self._decision_calls,
                                "signature": signature,
                                "source": "model",
                            }
                        )
                        return choice
                    self._illegal_suggestions += 1
            except ProtocolError:
                self._invalid_decisions += 1

        self._fallbacks += 1
        fallback = self._shortlist(choices, agent)[0]
        self._chosen.append(
            {
                "decision": self._decision_calls,
                "signature": str(getattr(fallback, "signature")),
                "source": "deterministic_fallback",
            }
        )
        return fallback

    def receipt(self) -> dict[str, Any]:
        total_decision_attempts = self._valid_decisions + self._invalid_decisions
        return {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "config": asdict(self.config),
            "model_calls": self._model_calls,
            "repair_calls": self._repair_calls,
            "decision_calls": self._decision_calls,
            "valid_decisions": self._valid_decisions,
            "invalid_decisions": self._invalid_decisions,
            "valid_response_rate": (
                self._valid_decisions / total_decision_attempts
                if total_decision_attempts
                else 1.0
            ),
            "illegal_suggestions": self._illegal_suggestions,
            "fallbacks": self._fallbacks,
            "fallback_rate": self._fallbacks / self._decision_calls if self._decision_calls else 0.0,
            "chosen": list(self._chosen),
            "decision_receipts": list(self._decision_receipts),
            "transition_count": len(self._transitions),
            "client_calls": self.client.receipt_slice(self._client_start),
            "hidden_reasoning_persisted": False,
        }


def _state_token(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].upper()


def run_glimmer_game(
    environment: Any,
    *,
    provider: GlimmerPolicy,
    monotonic_deadline: float | None = None,
    monotonic_clock=time.monotonic,
) -> dict[str, Any]:
    """Run one official-SDK-compatible game with a custom bounded chooser.

    The existing exact observation/transition recorder is reused verbatim; only
    action selection is replaced.  This keeps scorecard reconciliation and
    sticky-GAME_OVER evidence compatible with the frozen official driver.
    """

    from arcgpt2.official_score_agent import AgentConfig, TransitionFrontierAgent

    cfg = provider.config
    agent = TransitionFrontierAgent(
        AgentConfig(
            max_actions=cfg.max_actions,
            max_resets=cfg.max_resets,
            max_click_candidates_per_action=cfg.max_click_candidates_per_action,
            low_discrepancy_points=cfg.low_discrepancy_points,
        ),
        provider,
    )
    initial = getattr(environment, "observation_space", None)
    if initial is None:
        initial = environment.reset()
        if initial is None:
            raise RuntimeError("environment_reset_none")
        agent.begin(initial, bootstrap_reset=True)
    else:
        agent.begin(initial)

    while agent.actions_taken < cfg.max_actions:
        if monotonic_deadline is not None and monotonic_clock() >= monotonic_deadline:
            receipt = agent.finish("runtime_deadline")
            receipt["provider"]["glimmer"] = provider.receipt()
            return receipt
        state = _state_token(agent.current.state)
        if state in {"WIN", "WON", "GAME_WIN"}:
            receipt = agent.finish("win")
            receipt["provider"]["glimmer"] = provider.receipt()
            return receipt
        if state in {"GAME_OVER", "NOT_PLAYED", "NOT_STARTED"}:
            if not agent.can_reset():
                receipt = agent.finish("game_over" if state == "GAME_OVER" else "not_started")
                receipt["provider"]["glimmer"] = provider.receipt()
                return receipt
            observation = environment.reset()
            if observation is None:
                raise RuntimeError("environment_reset_none")
            agent.record_reset(observation)
            continue

        choices = agent._choices(getattr(environment, "action_space", ()))
        choice = provider.choose(agent.current, choices, agent=agent)
        if choice is None:
            if agent.can_reset():
                observation = environment.reset()
                if observation is None:
                    raise RuntimeError("environment_reset_none")
                agent.record_reset(observation)
                continue
            receipt = agent.finish("frontier_exhausted")
            receipt["provider"]["glimmer"] = provider.receipt()
            return receipt
        before_levels = agent.current.levels_completed
        observation = environment.step(choice.action, data=choice.data)
        if observation is None:
            raise RuntimeError("environment_step_none")
        agent.record_step(choice, observation)
        after_levels = agent.current.levels_completed
        if provider._transitions:
            provider._transitions[-1]["level_delta"] = (
                after_levels - before_levels
                if isinstance(before_levels, int) and isinstance(after_levels, int)
                else None
            )
            provider._transitions[-1]["outcome"] = (
                "PROGRESS"
                if isinstance(before_levels, int)
                and isinstance(after_levels, int)
                and after_levels > before_levels
                else (
                    "GAME_OVER"
                    if _state_token(agent.current.state) == "GAME_OVER"
                    else (
                        "CHANGE"
                        if provider._transitions[-1].get("changed_cells")
                        else "NO_CHANGE"
                    )
                )
            )
            if provider._transitions[-1]["outcome"] in {"PROGRESS", "GAME_OVER"}:
                provider._plan.clear()

    receipt = agent.finish(
        "win" if _state_token(agent.current.state) in {"WIN", "WON", "GAME_WIN"} else "action_budget_exhausted"
    )
    receipt["provider"]["glimmer"] = provider.receipt()
    receipt["provider"]["policy_use"] = (
        "bounded exact-candidate plan" if cfg.mode == "native" else "declarative hypotheses plus deterministic fracture score"
    )
    return receipt


def run_official_arm(
    *,
    client: LlamaClient,
    config: PolicyConfig,
    manifest_path: str,
    temp_root: str,
    monotonic_deadline: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one exact public-development scorecard with a temporary driver seam."""

    import arcgpt2.official_public_score as official

    providers: list[GlimmerPolicy] = []

    def factory() -> GlimmerPolicy:
        provider = GlimmerPolicy(client, config)
        providers.append(provider)
        return provider

    def custom_run_one_game(
        environment: Any,
        *,
        direction_score_provider: Any = None,
        monotonic_deadline: float | None = None,
        abort_callback: Any = None,
        monotonic_clock: Any = None,
    ) -> dict[str, Any]:
        if abort_callback is not None and abort_callback():
            raise official.OfficialPublicScoreError("runtime_deadline", "abort requested")
        if not isinstance(direction_score_provider, GlimmerPolicy):
            raise official.OfficialPublicScoreError("provider_invalid", "Glimmer provider missing")
        return run_glimmer_game(
            environment,
            provider=direction_score_provider,
            monotonic_deadline=monotonic_deadline,
            monotonic_clock=monotonic_clock or time.monotonic,
        )

    original = official.run_one_game
    official.run_one_game = custom_run_one_game
    try:
        receipt = official.run_official_public_score(
            manifest_path=manifest_path,
            provider_factory=factory,
            temp_root=temp_root,
            monotonic_deadline=monotonic_deadline,
        )
    finally:
        official.run_one_game = original

    provider_receipts = [provider.receipt() for provider in providers]
    calls = sum(item["model_calls"] for item in provider_receipts)
    decisions = sum(item["decision_calls"] for item in provider_receipts)
    fallbacks = sum(item["fallbacks"] for item in provider_receipts)
    invalid = sum(item["invalid_decisions"] for item in provider_receipts)
    metrics = {
        "arm": config.mode,
        "providers_created": len(providers),
        "model_calls": calls,
        "decision_calls": decisions,
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / decisions if decisions else 0.0,
        "invalid_decisions": invalid,
        "provider_receipts": provider_receipts,
    }
    return receipt, metrics


__all__ = [
    "GlimmerPolicy",
    "MODES",
    "POLICY_VERSION",
    "PolicyConfig",
    "run_glimmer_game",
    "run_official_arm",
]
