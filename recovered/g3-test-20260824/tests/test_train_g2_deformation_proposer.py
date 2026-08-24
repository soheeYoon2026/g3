from scripts.train_g2_deformation_proposer import best_pair_per_job, split_jobs


def test_best_pair_per_job_selects_lowest_target_cd():
    rows = best_pair_per_job({"pairs": [
        {"job_uid": "a", "target_cd": 0.3},
        {"job_uid": "a", "target_cd": 0.2},
        {"job_uid": "b", "target_cd": 0.4},
    ]})
    assert [(row["job_uid"], row["target_cd"]) for row in rows] == [("a", 0.2), ("b", 0.4)]


def test_split_jobs_has_no_job_leakage():
    split = split_jobs([{"job_uid": str(i)} for i in range(20)], seed=1)
    groups = [set(row["job_uid"] for row in split[name]) for name in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(map(len, groups)) == 20
