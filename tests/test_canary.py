import json

from aox_g3.canary import ShadowConfig, ShadowDispatcher, compare_results, should_sample


def test_shadow_sampling_is_deterministic():
    payload = b"solid deterministic"
    assert should_sample(payload, 1.0)
    assert not should_sample(payload, 0.0)
    assert should_sample(payload, 0.37) == should_sample(payload, 0.37)


def test_compare_results_records_drift():
    row = compare_results(
        {
            "model": "v6.pt",
            "coefficient_expert": "g2_su2_clean",
            "drag_coefficient": 0.20,
            "lift_coefficient": 0.05,
            "elapsed_seconds": 2.0,
            "ood_score": 0.4,
        },
        {
            "model": "v7.pt",
            "coefficient_expert": "g2_su2_clean",
            "drag_coefficient": 0.22,
            "lift_coefficient": 0.04,
            "elapsed_seconds": 3.0,
            "ood_score": 0.5,
        },
    )
    assert abs(row["delta_cd"] - 0.02) < 1e-12
    assert abs(row["delta_cl"] + 0.01) < 1e-12
    assert row["latency_ratio"] == 1.5


def test_audit_is_compact_jsonl(tmp_path):
    config = ShadowConfig(audit_path=str(tmp_path / "audit.jsonl"))
    dispatcher = ShadowDispatcher(config)
    dispatcher._write_audit({"status": "ok", "delta_cd": 0.01})
    row = json.loads((tmp_path / "audit.jsonl").read_text())
    assert row == {"status": "ok", "delta_cd": 0.01}

