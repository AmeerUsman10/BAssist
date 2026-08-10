from __future__ import annotations

from dataclasses import dataclass

from glimmer_arc.policy import GlimmerPolicy, PolicyConfig
from glimmer_arc.synthetic import SyntheticChoice, SyntheticObservation


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.receipts = []

    def complete_json(self, **kwargs):
        self.receipts.append({"purpose": kwargs["purpose"]})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def receipt_slice(self, start_index=0):
        return list(self.receipts[start_index:])


def observation():
    return SyntheticObservation(((1, 0), (0, 9)))


def choices():
    return [SyntheticChoice("ACTION1", "ACTION1"), SyntheticChoice("ACTION2", "ACTION2")]


def test_native_policy_selects_exact_model_plan() -> None:
    client = FakeClient([
        {"plan": [{"signature": "ACTION2", "confidence": 0.9, "expected": "progress"}]}
    ])
    policy = GlimmerPolicy(
        client,
        PolicyConfig(mode="native", max_model_calls=1, shortlist_size=2, plan_length=1),
    )
    chosen = policy.choose(observation(), choices())
    assert chosen.signature == "ACTION2"
    assert policy.receipt()["fallbacks"] == 0


def test_cfs_policy_controller_selects_from_hypotheses() -> None:
    client = FakeClient([
        {
            "hypotheses": [
                {
                    "id": "H1",
                    "weight": 1,
                    "predictions": [
                        {"signature": "ACTION1", "outcome": "NO_CHANGE", "confidence": 0.9},
                        {"signature": "ACTION2", "outcome": "PROGRESS", "confidence": 0.9},
                    ],
                },
                {
                    "id": "H2",
                    "weight": 1,
                    "predictions": [
                        {"signature": "ACTION1", "outcome": "NO_CHANGE", "confidence": 0.8},
                        {"signature": "ACTION2", "outcome": "CHANGE", "confidence": 0.8},
                    ],
                },
            ],
            "unsafe_signatures": [],
        }
    ])
    policy = GlimmerPolicy(
        client,
        PolicyConfig(mode="cfs_lite", max_model_calls=1, shortlist_size=2, plan_length=1),
    )
    chosen = policy.choose(observation(), choices())
    assert chosen.signature == "ACTION2"
    receipt = policy.receipt()
    assert receipt["valid_decisions"] == 1
    assert receipt["decision_receipts"][0]["hypothesis_count"] == 2


def test_invalid_response_fails_closed_to_deterministic_candidate() -> None:
    client = FakeClient([{"plan": [{"signature": "INVENTED", "confidence": 1.0}]}])
    policy = GlimmerPolicy(
        client,
        PolicyConfig(
            mode="native",
            max_model_calls=1,
            max_repair_calls=0,
            shortlist_size=2,
            plan_length=1,
        ),
    )
    chosen = policy.choose(observation(), choices())
    assert chosen.signature == "ACTION1"
    receipt = policy.receipt()
    assert receipt["fallbacks"] == 1
    assert receipt["invalid_decisions"] == 1
