from collections import Counter
import itertools

from arcgpt2.goal_dsl import evaluate_goal
from arcgpt2.raw_goal_signature_data import (
    EVIDENCE_ORDERS,
    FAMILIES,
    ROLE_NAMES,
    SIGNATURES,
    audit_counterfactual_bytes,
    audit_token_contract,
    audit_manifests,
    build_manifests,
    candidate_completion_text,
    canonical_manifest_digest,
    canonical_manifest_payload,
    canonical_json_sha256,
    model_visible_surface_payload,
    raw_support_parts,
    semantic_query_text,
    terminal_completion_text,
)


def test_exact_split_sizes_and_reproducible_digest():
    first = build_manifests()
    second = build_manifests()
    assert [len(item.groups) for item in first] == [120, 24, 72]
    assert canonical_manifest_digest(first) == canonical_manifest_digest(second)
    assert canonical_manifest_digest(first) == "02e59b60dab038e16f45fbfe03dc9dadece1c0d1fc8082210834965e7634ab04"
    assert canonical_manifest_payload(first) == canonical_manifest_payload(second)


def test_balances_independence_and_disjoint_surfaces():
    manifests = build_manifests()
    audit = audit_manifests(manifests)
    assert audit["errors"] == []
    assert all(not overlap for overlap in audit["surface_overlaps"].values())
    assert all(
        not overlap
        for factor_pairs in audit["signature_factor_overlaps"].values()
        for overlap in factor_pairs.values()
    )
    physical = []
    for manifest in manifests:
        hashes = [canonical_json_sha256(model_visible_surface_payload(group)) for group in manifest.groups]
        assert len(hashes) == len(set(hashes))
        assert hashes == [group.surface_sha256 for group in manifest.groups]
        physical.append(set(hashes))
    assert all(not physical[left].intersection(physical[right]) for left, right in itertools.combinations(range(3), 2))
    for manifest, repetitions in zip(manifests, (5, 1, 3)):
        report = audit["splits"][manifest.name]
        assert set(report["signature_permutation_counts"].values()) == {repetitions}
        assert set(report["candidate_order_counts"].values()) == {repetitions}
        assert set(report["semantic_mask_counts"].values()) == {len(manifest.groups) // 6}
        assert set(report["named_role_color_counts"].values()) == {len(manifest.groups) // 6}
        assert report["joint_schedule_unique_pairs"] == len(manifest.groups)
        assert report["joint_schedule_max_multiplicity"] <= 1
        assert report["signature_candidate_edge_count"] == len(manifest.groups)
        assert report["signature_color_edge_count"] == len(manifest.groups)
        assert report["signature_nuisance_edge_count"] == len(manifest.groups)
        assert set(report["role_color_tuple_counts"].values()) == {2 * repetitions}
        assert set(report["nuisance_layout_counts"].values()) == {repetitions}


def test_every_identification_signature_is_replayed_not_relabelled():
    for manifest in build_manifests():
        for group in manifest.groups:
            assert group.evidence_orders == EVIDENCE_ORDERS
            for family_index, goal in enumerate(group.candidates):
                actual = tuple(
                    evaluate_goal(group.mechanics, goal, trial.before, trial.action)[0]
                    for trial in group.identification_trials
                )
                assert actual == SIGNATURES[group.signature_assignment[family_index]]
                assert actual == group.worlds[family_index].identification_terminal
            actual_semantic = tuple(
                evaluate_goal(group.mechanics, goal, group.semantic_trial.before, group.semantic_trial.action)[0]
                for goal in group.candidates
            )
            assert actual_semantic == group.semantic_trial.terminal_by_family
            assert sum(actual_semantic) == 2


def test_trial_three_truth_is_exactly_independent_of_balanced_axes():
    for manifest in build_manifests():
        for axis in ("signature", "position", "color"):
            counts = Counter()
            for group in manifest.groups:
                for f, family in enumerate(FAMILIES):
                    key = (
                        group.signature_assignment[f] if axis == "signature" else
                        group.candidate_order.index(family) if axis == "position" else
                        group.role_colors[f]
                    )
                    counts[(family, key, group.semantic_trial.terminal_by_family[f])] += 1
            paired = {(family, key) for family, key, _ in counts}
            assert all(counts[(family, key, False)] == counts[(family, key, True)] for family, key in paired)
        touch_right = Counter()
        for group in manifest.groups:
            touch_right[(group.role_colors[4], group.semantic_trial.terminal_by_family[3])] += 1
        assert all(touch_right[(color, False)] == touch_right[(color, True)] for color in range(2, 8))


def test_all_five_named_role_colors_are_balanced():
    for manifest in build_manifests():
        counts = Counter(
            (role, color)
            for group in manifest.groups
            for role, color in zip(ROLE_NAMES, group.role_colors)
        )
        assert set(counts.values()) == {len(manifest.groups) // 6}


def test_nuisance_geometry_is_reused_and_split_balanced_not_an_id_watermark():
    manifests = build_manifests()
    expected = (5, 1, 3)
    positions_by_split = []
    for manifest, repetitions in zip(manifests, expected):
        positions = Counter()
        for group in manifest.groups:
            marker = tuple(
                (row, column)
                for row, values in enumerate(group.identification_trials[0].before)
                for column, value in enumerate(values)
                if value == 8
            )
            assert len(marker) == 1
            positions[marker[0]] += 1
        assert len(positions) == 24
        assert set(positions.values()) == {repetitions}
        positions_by_split.append(set(positions))
    assert positions_by_split[0] == positions_by_split[1] == positions_by_split[2]


def test_counterfactual_supports_have_only_terminal_status_difference():
    for manifest in build_manifests():
        for group in manifest.groups:
            audit = audit_counterfactual_bytes(group)
            assert audit["metadata_absent"]
            assert all(check == {"normalized_equal": True, "distinct_texts": True} for check in audit["support_checks"])
            for trial in group.identification_trials:
                prompts_targets = [raw_support_parts(world, trial) for world in group.worlds]
                assert len({prompt for prompt, _ in prompts_targets}) == 1
                targets = [target for _, target in prompts_targets]
                normalized = [target.replace("yes", "STATUS").replace("no", "STATUS") for target in targets]
                assert len(set(normalized)) == 1


def test_model_text_omits_metadata_and_trial_three_has_no_status_target():
    for manifest in build_manifests():
        for group in manifest.groups:
            semantic = semantic_query_text(group)
            candidates = candidate_completion_text(group)
            assert group.group_id not in semantic
            assert group.level_identity not in semantic
            assert semantic.endswith("Terminal success:")
            assert "yes." not in semantic and "no." not in semantic and "nil." not in semantic
            assert len(candidates) == 4 and len(set(candidates)) == 4
    assert terminal_completion_text() == (" yes.", " no.", " nil.")


def test_token_contract_has_a_fail_closed_runtime_audit():
    class WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [sum(map(ord, word)) for word in text.split()]

    group = build_manifests()[0].groups[0]
    audit = audit_token_contract(group, WordTokenizer())
    assert audit["passed"]
    assert len(set(audit["candidate_lengths"])) == 1
    assert len(set(audit["terminal_lengths"])) == 1


def test_candidate_and_signature_permutations_are_complete():
    all_permutations = set(itertools.permutations(range(4)))
    for manifest in build_manifests():
        signatures = {group.signature_assignment for group in manifest.groups}
        orders = {tuple(FAMILIES.index(family) for family in group.candidate_order) for group in manifest.groups}
        assert signatures == all_permutations
        assert orders == all_permutations
