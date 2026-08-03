from __future__ import annotations

from instella_arc.benchmark import build_action_cases
from instella_arc.smoke_benchmark import select_smoke_cases


def test_smoke_selection_keeps_one_no_evidence_case_and_matched_controls() -> None:
    all_cases = build_action_cases((770_001,))
    selected = select_smoke_cases(all_cases)
    assert len(selected) == 4
    assert sum(case.evidence_depth == 0 for case in selected) == 1
    maximum = max(case.evidence_depth for case in selected)
    controls = [case for case in selected if case.evidence_depth == maximum]
    assert {case.mode for case in controls} == {"intact", "amnesic", "shuffled"}
    identities = {case.case_id.rsplit(":", 1)[0] for case in controls}
    assert len(identities) == 1
