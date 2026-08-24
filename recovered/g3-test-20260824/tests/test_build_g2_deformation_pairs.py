from scripts.build_g2_deformation_pairs import available_jobs, case_by_source


def test_case_by_source_selects_and_orders_accepted_rbf_cases():
    manifest = {"cases": [
        {"run": 3, "accepted": True, "source": {"source_id": "job:RBF_DSN_003"}},
        {"run": 1, "accepted": True, "source": {"source_id": "job:RBF_DSN_001"}},
        {"run": 2, "accepted": False, "source": {"source_id": "job:RBF_DSN_002"}},
        {"run": 4, "accepted": True, "source": {"source_id": "other:RBF_DSN_001"}},
    ]}
    assert [row["run"] for row in case_by_source(manifest, "job")] == [1, 3]


def test_available_jobs_requires_two_accepted_rbf_cases():
    manifest = {"cases": [
        {"accepted": True, "source": {"source_id": "a:RBF_DSN_001"}},
        {"accepted": True, "source": {"source_id": "a:RBF_DSN_002"}},
        {"accepted": True, "source": {"source_id": "b:RBF_DSN_001"}},
        {"accepted": False, "source": {"source_id": "b:RBF_DSN_002"}},
    ]}
    assert available_jobs(manifest) == ["a"]
