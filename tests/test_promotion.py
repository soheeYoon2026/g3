from aox_g3.promotion import evaluate_offline_gates, evaluate_shadow_gates


def test_offline_gate_allows_small_regression_only():
    production = {"metrics": {"cd": {"mae": 0.10}, "cp": {"mae": 0.08}}}
    challenger = {"metrics": {"cd": {"mae": 0.102}, "cp": {"mae": 0.07}}}
    report = evaluate_offline_gates(
        production,
        challenger,
        {
            "metrics.cd.mae": {"max_regression_ratio": 1.03},
            "metrics.cp.mae": {"max_regression_ratio": 1.00},
        },
    )
    assert report["passed"]


def test_shadow_gate_requires_runtime_evidence():
    rows = [
        {
            "status": "ok",
            "challenger_model": "v7.pt",
            "delta_cd": 0.01,
            "delta_cl": -0.01,
            "latency_ratio": 1.1,
        }
        for _ in range(20)
    ]
    report = evaluate_shadow_gates(rows, challenger_model="v7.pt", min_samples=20)
    assert report["passed"]

    report = evaluate_shadow_gates(rows[:19], challenger_model="v7.pt", min_samples=20)
    assert not report["passed"]

