import numpy as np

from aox_g3.optimize import (
    FeasGP,
    NoisyGP,
    expected_improvement,
    optimise,
    parametrise,
    propose_cei,
)


def _grid_surface(n=21, extent=1000.0):
    xs = np.linspace(0, extent, n)
    ys = np.linspace(-extent / 2, extent / 2, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    return vertices, normals


def _scales_of(param, deformed):
    """가짜 서로게이트용: 변형된 형상에서 s 를 되읽는다 (평면 격자, 법선 +z)."""
    return np.array([-(deformed[idx][2]) / param.delta for idx in param.vertex_ids])


def test_parametrise_picks_points_and_inward_normals():
    vertices, normals = _grid_surface()
    points = np.array([[100.0, 300.0, 0.0], [500.0, 0.0, 0.0], [900.0, -300.0, 0.0]])
    param = parametrise(vertices, normals, points, max_points=6, delta_fraction=0.02, symmetric=False)
    assert param.size == 3
    assert np.allclose(param.inward, [[0, 0, -1]] * 3)
    assert np.isclose(param.delta, 0.02 * 1000.0)


def test_deform_is_identity_at_zero_and_pushes_by_scale_times_delta():
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 300.0, 0.0]]), delta_fraction=0.02, symmetric=False)
    assert np.allclose(param.deform(vertices, np.zeros(1)), vertices)
    assert np.isclose(param.deform(vertices, np.array([1.0]))[param.vertex_ids[0]][2], -param.delta)
    assert np.isclose(param.deform(vertices, np.array([-0.5]))[param.vertex_ids[0]][2], +0.5 * param.delta)


def test_deform_mirrors_when_symmetric():
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 300.0, 0.0]]), symmetric=True)
    pushed = param.deform(vertices, np.array([1.0]))
    mirror_idx = np.argmin(((vertices - [500.0, -300.0, 0.0]) ** 2).sum(axis=1))
    assert np.isclose(pushed[mirror_idx][2], -param.delta)


def test_noisy_gp_interpolates_and_is_uncertain_far_away():
    x = np.array([[0.1], [0.5], [0.9]])
    y = np.array([0.30, 0.25, 0.31])
    gp = NoisyGP().fit(x, y, np.full(3, 1e-4))
    mu, sd = gp.predict(np.array([[0.5], [0.3]]))
    assert abs(mu[0] - 0.25) < 1e-2
    assert sd[1] > sd[0]


def test_expected_improvement_prefers_low_mean_and_high_uncertainty():
    ei = expected_improvement(np.array([0.30, 0.30, 0.31]), np.array([0.001, 0.01, 0.01]), best=0.30)
    assert ei[1] > ei[0]
    assert ei[1] > ei[2]


def test_feasibility_gp_is_trivial_without_failures_and_low_near_a_failure():
    feas = FeasGP().fit(np.array([[0.2, 0.2], [0.8, 0.8]]), np.array([1.0, 1.0]))
    assert np.all(feas.p(np.array([[0.5, 0.5]])) == 1.0)
    feas = FeasGP().fit(np.array([[0.2, 0.2], [0.8, 0.8]]), np.array([1.0, 0.0]))
    p_near_fail, p_far = feas.p(np.array([[0.8, 0.8], [0.1, 0.1]]))
    assert p_near_fail < 0.5 < p_far


def test_propose_cei_stays_at_the_frontier_instead_of_the_failed_corner():
    rng = np.random.default_rng(0)
    x = rng.random((16, 2))
    y = 0.30 - 0.02 * x[:, 0] + rng.normal(0, 0.002, 16)  # x0 이 클수록 좋아 보이지만 …
    ok = (x[:, 0] < 0.7).astype(float)  # … x0 > 0.7 은 전부 실패 (6/16)
    # 목적 GP 는 실현가능한 관측만 본다 — 실패한 설계에는 Cd 가 없다(optimise() 와 동일).
    gp = NoisyGP().fit(x[ok == 1], y[ok == 1], np.full(int(ok.sum()), 2e-3))
    plain = FeasGP().fit(x, np.ones(16))
    feas = FeasGP().fit(x, ok)
    seeds = range(10)
    # 순수 EI 는 외삽이 낙관적인 실패 지대(x0≈0.9)로 간다. cEI 는 P(feasible) 을 곱해
    # 실현가능 관측 안(x0≈0.45)에 머문다 — 실패 지대는 P 가 바닥(0.01) 근처다.
    plain_x0 = np.array([propose_cei(gp, plain, 2, np.random.default_rng(s))[0] for s in seeds])
    cei_x0 = np.array([propose_cei(gp, feas, 2, np.random.default_rng(s))[0] for s in seeds])
    assert np.mean(plain_x0 > 0.8) >= 0.8
    assert np.mean(cei_x0 < 0.85) >= 0.8
    assert cei_x0.mean() < plain_x0.mean()


def test_optimise_finds_a_better_design_than_the_initial_sample():
    vertices, normals = _grid_surface()
    points = np.array([[200.0, 200.0, 0.0], [500.0, -200.0, 0.0], [800.0, 200.0, 0.0]])
    param = parametrise(vertices, normals, points, symmetric=False)
    target = np.array([0.6, -0.4, 0.2])

    def evaluate(deformed):
        scales = _scales_of(param, deformed)
        return {"cd": 0.30 + 0.05 * float(np.sum((scales - target) ** 2)), "cl": -0.02, "ok": True}

    result = optimise(evaluate, param, vertices, budget=28, top=3, seed=1)
    assert result.evaluated == 28 and result.failed == 0
    assert result.baseline is not None and result.baseline["scales"] == [0.0, 0.0, 0.0]
    init_best = min(h["cd"] for h in result.history if h["kind"] in ("baseline", "init"))
    assert result.best[0]["cd"] < init_best
    assert np.linalg.norm(np.array(result.best[0]["scales"]) - target) < 0.5
    assert len(result.best[0]["moves"]) == 3 and result.best[0]["gp_std"] >= 0.0


def test_optimise_records_failures_and_keeps_going():
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 0.0, 0.0]]), symmetric=False)
    calls = {"n": 0}

    def flaky(deformed):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise RuntimeError("boom")
        s = _scales_of(param, deformed)[0]
        if s > 0.8:
            return {"cd": None, "cl": None, "ok": False, "error": "out of distribution"}
        return {"cd": 0.3 + 0.02 * s * s, "cl": 0.0, "ok": True}

    result = optimise(flaky, param, vertices, budget=12, top=2)
    assert result.evaluated == 12
    assert result.failed >= 3
    assert all(h["error"] for h in result.history if not h["ok"])
    assert all(h["ok"] for h in result.history if h["cd"] is not None and h["error"] is None)
    assert len(result.best) == 2 and all(b["cd"] is not None for b in result.best)


def test_optimise_returns_designs_that_differ_from_each_other():
    """BO 는 한 점에 수렴한다 — 상위 카드가 전부 같은 설계면 고를 수가 없다."""
    vertices, normals = _grid_surface()
    points = np.array([[200.0, 200.0, 0.0], [500.0, -200.0, 0.0], [800.0, 200.0, 0.0]])
    param = parametrise(vertices, normals, points, symmetric=False)
    target = np.array([0.6, -0.4, 0.2])

    def evaluate(deformed):
        scales = _scales_of(param, deformed)
        return {"cd": 0.30 + 0.05 * float(np.sum((scales - target) ** 2)), "cl": -0.02, "ok": True}

    result = optimise(evaluate, param, vertices, budget=30, top=5, seed=3)
    picks = np.array([b["scales"] for b in result.best])
    assert len(picks) == 5
    gaps = [
        float(np.linalg.norm(picks[i] - picks[j])) for i in range(len(picks)) for j in range(i + 1, len(picks))
    ]
    assert min(gaps) > 0.2


def test_influence_radius_grows_with_delta_so_bumps_stay_smooth():
    vertices, normals = _grid_surface()
    points = np.array([[500.0, 0.0, 0.0]])
    narrow = parametrise(vertices, normals, points, delta_fraction=0.02, symmetric=False)
    wide = parametrise(vertices, normals, points, delta_fraction=0.10, symmetric=False)
    assert narrow.radius < wide.radius
    # 기울기 대리지표: δ / 반경 이 커지면 뿔이 된다.
    assert wide.delta / wide.radius <= 0.45


def test_geometry_rejects_do_not_eat_the_surrogate_budget():
    """기하 기각은 1 ms 다 — 예산(서로게이트 호출)을 깎으면 표본만 잃는다."""
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 0.0, 0.0]]), symmetric=False)

    def picky(deformed):
        s = _scales_of(param, deformed)[0]
        if s > 0.3:  # 절반 넘게 기하에서 걸리는 설정
            return {"cd": None, "cl": None, "ok": False, "error": "surface folded", "rejected": True}
        return {"cd": 0.3 + 0.02 * s * s, "cl": 0.0, "ok": True}

    result = optimise(picky, param, vertices, budget=10, top=2)
    assert result.evaluated == 10  # 서로게이트는 예산만큼 다 불렀다
    assert result.rejected > 0
    assert result.failed == 0  # 기하 기각은 서로게이트 실패가 아니다
    assert len(result.history) == result.evaluated + result.rejected


def test_optimise_stops_when_every_shape_is_rejected():
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 0.0, 0.0]]), symmetric=False)

    def always_rejected(_deformed):
        return {"cd": None, "cl": None, "ok": False, "error": "volume changed +80%", "rejected": True}

    result = optimise(always_rejected, param, vertices, budget=10, top=3)
    assert result.evaluated == 0
    assert result.rejected <= 10 * 6 + 1  # ATTEMPT_LIMIT 로 막힌다
    assert result.best == []


def test_baseline_is_not_offered_as_a_recommendation():
    vertices, normals = _grid_surface()
    param = parametrise(vertices, normals, np.array([[500.0, 0.0, 0.0]]), symmetric=False)

    def only_baseline_survives(deformed):
        s = _scales_of(param, deformed)[0]
        if abs(s) > 1e-9:
            return {"cd": None, "cl": None, "ok": False, "error": "surface folded", "rejected": True}
        return {"cd": 0.3, "cl": 0.0, "ok": True}

    result = optimise(only_baseline_survives, param, vertices, budget=8, top=3)
    assert result.baseline is not None and result.baseline["cd"] == 0.3
    assert result.best == []  # 안 바꾼 차를 추천이라고 내보내지 않는다
