import json

import pytest

from aox_g3.eval_metrics import (
    coefficient_summary,
    delta_metrics,
    derive_group_pairs,
    load_pairs,
    resolve_explicit_pairs,
    spearman,
)


def test_spearman_perfect_and_inverted():
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_handles_ties_and_degenerate_input():
    assert spearman([1.0, 2.0, 2.0, 3.0], [1.0, 2.5, 2.5, 4.0]) == pytest.approx(1.0)
    assert spearman([1.0], [1.0]) is None
    assert spearman([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None


def test_derive_group_pairs_only_within_groups():
    rows = [
        {"case": "run_1", "group_id": "a"},
        {"case": "run_2", "group_id": "b"},
        {"case": "run_3", "group_id": "a"},
        {"case": "run_4", "group_id": None},
        {"case": "run_5", "group_id": "a"},
    ]
    pairs = derive_group_pairs(rows)
    assert pairs == [(0, 2), (0, 4), (2, 4)]
    assert derive_group_pairs(rows) == pairs


def test_resolve_explicit_pairs_matches_numeric_and_string_keys():
    rows = [{"run": 14}, {"run": "20"}]
    resolved = resolve_explicit_pairs(rows, [{"baseline": "14", "variant": 20}], case_key="run")
    assert resolved == [(0, 1)]
    with pytest.raises(ValueError, match="run_99"):
        resolve_explicit_pairs(
            [{"case": "run_1"}], [{"baseline": "run_1", "variant": "run_99"}]
        )


def test_delta_metrics_mae_and_direction():
    rows = [
        {"true_cd": 0.30, "pred_cd": 0.31},
        {"true_cd": 0.28, "pred_cd": 0.30},
        {"true_cd": 0.31, "pred_cd": 0.305},
    ]
    metrics = delta_metrics(rows, [(0, 1), (0, 2)], "true_cd", "pred_cd")
    # pair 1: true -0.02, pred -0.01 (correct); pair 2: true +0.01, pred -0.005 (wrong)
    assert metrics["pairs"] == 2
    assert metrics["mae"] == pytest.approx((0.01 + 0.015) / 2)
    assert metrics["direction_pairs"] == 2
    assert metrics["direction_accuracy"] == pytest.approx(0.5)


def test_delta_metrics_tolerance_and_zero_prediction():
    rows = [
        {"true_cd": 0.300, "pred_cd": 0.30},
        {"true_cd": 0.3005, "pred_cd": 0.30},
        {"true_cd": 0.320, "pred_cd": 0.30},
    ]
    # |true Δ| = 0.0005 is below tolerance -> excluded; pred Δ = 0 vs true Δ = 0.02 -> wrong
    metrics = delta_metrics(rows, [(0, 1), (0, 2)], "true_cd", "pred_cd", direction_tolerance=0.001)
    assert metrics["direction_pairs"] == 1
    assert metrics["direction_accuracy"] == 0.0


def test_coefficient_summary_with_group_pairs():
    rows = [
        {"run": 1, "group_id": "g", "true_cd": 0.30, "pred_cd": 0.32, "true_cl": 0.10, "pred_cl": 0.10},
        {"run": 2, "group_id": "g", "true_cd": 0.28, "pred_cd": 0.31, "true_cl": 0.12, "pred_cl": 0.13},
        {"run": 3, "group_id": "h", "true_cd": 0.40, "pred_cd": 0.35, "true_cl": 0.00, "pred_cl": 0.02},
    ]
    summary = coefficient_summary(rows, case_key="run")
    assert summary["cases"] == 3
    assert summary["cd_mae"] == pytest.approx((0.02 + 0.03 + 0.05) / 3)
    assert summary["pair_source"] == "group"
    assert summary["delta_cd"]["pairs"] == 1
    # true Δ -0.02, pred Δ -0.01: direction correct
    assert summary["delta_cd"]["direction_accuracy"] == 1.0


def test_coefficient_summary_explicit_pairs_and_empty(tmp_path):
    rows = [
        {"run": 1, "true_cd": 0.30, "pred_cd": 0.29, "true_cl": 0.0, "pred_cl": 0.0},
        {"run": 2, "true_cd": 0.33, "pred_cd": 0.35, "true_cl": 0.0, "pred_cl": 0.0},
    ]
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": [{"baseline": 1, "variant": 2}]}))
    summary = coefficient_summary(rows, pairs=load_pairs(manifest), case_key="run")
    assert summary["pair_source"] == "explicit"
    assert summary["delta_cd"]["pairs"] == 1
    assert summary["delta_cd"]["direction_accuracy"] == 1.0

    no_groups = coefficient_summary(rows, case_key="run")
    assert no_groups["pair_source"] is None
    assert no_groups["delta_cd"] is None


def test_load_pairs_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"baseline": 1}]))
    with pytest.raises(ValueError, match="baseline"):
        load_pairs(bad)
