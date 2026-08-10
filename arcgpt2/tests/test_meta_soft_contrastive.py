from __future__ import annotations

from arcgpt2.meta_soft_contrastive import (
    counterfactual_probe_records,
    infer_probe_colors,
    outcome_only_target,
)
from arcgpt2.phase0_hidden_action import Direction, HiddenActionGame, generate_game


def _first_probe(seed: int):
    spec = generate_game(seed)
    game = HiddenActionGame(spec)
    return spec, game.step(spec.probe_order[0])


def test_probe_counterfactuals_cover_all_four_distinct_cardinal_moves() -> None:
    spec, record = _first_probe(130_001)
    moving, background, source = infer_probe_colors(record)
    assert moving == spec.palette.agent
    assert background == spec.palette.background
    assert record.before[source[0]][source[1]] == moving

    candidates = counterfactual_probe_records(record)
    assert len(candidates) == 4
    assert all(candidate.moved for candidate in candidates)
    assert all(candidate.status == "ACTIVE" for candidate in candidates)
    assert len({candidate.after for candidate in candidates}) == 4
    assert record.after in {candidate.after for candidate in candidates}


def test_exact_observation_selects_one_counterfactual() -> None:
    _, record = _first_probe(130_002)
    candidates = counterfactual_probe_records(record)
    matching = [candidate for candidate in candidates if candidate.after == record.after]
    assert len(matching) == 1
    texts = [outcome_only_target(candidate) for candidate in candidates]
    assert len(set(texts)) == 4
    assert outcome_only_target(record) in texts
