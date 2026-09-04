"""BO+GP 형상 추천 — 제어점 변위 공간에서 서로게이트 Cd 를 최소화한다.

v1(recommend.py)은 점마다 한 방향으로 한 번 밀어 보는 스크리닝이었다. 여기는
최적화다: 제어점 K개의 부호 있는 변위 s ∈ [-1, 1]^K (s_i·δ 만큼 법선 안쪽으로,
음수면 바깥으로)를 설계변수로 두고, 가우시안 프로세스로 Cd(s) 를 근사하며
기대개선(EI)이 큰 곳을 골라 서로게이트에 묻는다.

GP·획득함수·실현가능성 GP 는 LES 솔버 BO(BO_자료: wing_opt.py / body_bo.py,
2026-09-04)에서 그대로 옮겼다 — 순수 numpy, 등방 RBF, 길이척도는 작은 격자에서
주변우도 최대화, 관측별 잡음, 실패한 설계는 버리지 않고 실현가능성 GP 에 넣어
획득함수를 EI × P(feasible) 로 곱한다(Gelbart+2014 cEI). 그쪽에서 검증된 값
(GT-R 4단계 −20.7%, CAS-A −5.5%)이 있어 새로 짓지 않는다.

왜 미분이 아니라 BO 인가: 서로게이트는 STL 을 numpy 로 점구름/SDF 텐서로 바꿔
넣어서 정점에 대한 미분이 거기서 끊긴다. 평가 한 번이 0.5~1초라 40번이면
1분 — 비동기 잡으로 돌린다.

δ = 전장의 5%, 제어점 최대 12개 (2026-09-04).
rs6(전장 4.97 m) 실측: δ 2% → ΔCd −0.5%, 5% → −1.0%, 10% → −1.6%. 10% 는
50 cm 변형이라 반경을 같이 키워도 차 비율이 무너져 5% 에서 멈췄다. 카드 10장이
전부 같아 보이던 건 제어점이 6개뿐이고 상위 10개를 그냥 자른 탓이라, 제어점을
12개로 늘리고 다양성 선택(_diverse)으로 고쳤다. 어느 δ 에서도 개선은 서로게이트 오차(±0.008) 안이라
순위 매기기용이지 Cd 감소 주장이 아니다. 제어점은 최대 6개 — 이 GP 는 저차원
(자료 기준 5~7) 전제다.

여기 함수들은 평가 함수를 주입받는다 — 추론을 직접 부르지 않아 가짜 목적함수로
테스트한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

from .recommend import (
    RADIUS_FRACTION,
    WIDTH_AXIS,
    bump_weights,
    nearest_vertex,
    select_candidates,
    vehicle_frame,
)

DELTA_FRACTION = 0.05
MAX_POINTS = 12
# 봉우리가 가파르면 차가 아니라 뿔이 된다. 가우시안 봉우리(σ = 반경/2)의 최대
# 기울기는 대략 0.6·h/σ 라, 높이 h = δ 를 키우면 반경도 같이 키워야 매끈하다.
# δ 10% 에 반경 8%(고정) 로 돌렸더니 지붕과 펜더에 뿔이 솟았다(2026-09-04).
# 반경 ≥ 2.5·δ 로 묶으면 기울기가 0.5 아래로 떨어져 차체 형상이 유지된다.
RADIUS_PER_DELTA = 2.5
BUDGET = 40
TOP = 5
# 서로게이트는 결정적이다 — 같은 입력이면 같은 출력. 잡음항은 수치 안정용.
SURROGATE_NOISE = 1e-4
# 자료(body_bo.py)의 값 그대로: 0.25~1.0 스윕에서 0.50 이 최선, 바닥 0 이면 영원히 못 돌아온다.
FEAS_ELL = 0.50
FEAS_NOISE = 0.10
FEAS_FLOOR = 0.01


# ── 파라미터화: 제어점 변위 → 형상 ───────────────────────────────────────────


@dataclass
class Parametrisation:
    """설계변수 → 형상. 제어점 K개, 각각 원점·안쪽 단위법선·최대 이동 δ."""

    control_ids: list[int]
    vertex_ids: list[int]
    origins: np.ndarray  # (K, 3)
    inward: np.ndarray  # (K, 3) 단위벡터
    delta: float
    radius: float
    symmetric: bool
    width_centre: float

    @property
    def size(self) -> int:
        return len(self.control_ids)

    def _bumps(self, vertices: np.ndarray):
        """점마다 (원점, 안쪽방향, 가중치)와 거울 자리(없으면 None). 가중치는 원본 정점 기준."""
        out = []
        for origin, inward in zip(self.origins, self.inward):
            out.append((origin, inward, bump_weights(vertices, origin, self.radius)))
            if self.symmetric and abs(float(origin[WIDTH_AXIS]) - self.width_centre) > self.radius / 4.0:
                mirror = origin.copy()
                mirror[WIDTH_AXIS] = 2.0 * self.width_centre - mirror[WIDTH_AXIS]
                mirror_inward = inward.copy()
                mirror_inward[WIDTH_AXIS] = -mirror_inward[WIDTH_AXIS]
                out.append((mirror, mirror_inward, bump_weights(vertices, mirror, self.radius)))
            else:
                out.append(None)
        return out

    def deform(self, vertices: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """s_i·δ 만큼 각 점을 민 정점. 밀기는 원본 정점 위에 선형으로 겹친다."""
        scales = np.asarray(scales, dtype=float)
        result = vertices.copy()
        bumps = self._bumps(vertices)
        for i, s in enumerate(scales):
            if s == 0.0:
                continue
            for entry in (bumps[2 * i], bumps[2 * i + 1]):
                if entry is None:
                    continue
                _, inward, weight = entry
                result += weight[:, None] * (s * self.delta * inward)[None, :]
        return result

    def weights(self, vertices: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """렌더 색칠용: 각 점의 |s_i| 를 곱한 가중치의 최댓값."""
        scales = np.asarray(scales, dtype=float)
        total = np.zeros(len(vertices))
        bumps = self._bumps(vertices)
        for i, s in enumerate(scales):
            for entry in (bumps[2 * i], bumps[2 * i + 1]):
                if entry is None:
                    continue
                total = np.maximum(total, abs(float(s)) * entry[2])
        return np.clip(total, 0.0, 1.0)

    def moves(self, scales: np.ndarray) -> list[dict]:
        """편집기가 적용할 수 있는 형태: 제어점별 위치·변위(원본 좌표)."""
        scales = np.asarray(scales, dtype=float)
        out = []
        for control_id, origin, inward, s in zip(self.control_ids, self.origins, self.inward, scales):
            displacement = s * self.delta * inward
            out.append({
                "control_id": int(control_id),
                "position": [float(v) for v in origin],
                "displacement": [float(v) for v in displacement],
                "scale": float(s),
            })
        return out


def parametrise(
    vertices: np.ndarray,
    normals: np.ndarray,
    control_points: np.ndarray,
    *,
    max_points: int = MAX_POINTS,
    delta_fraction: float = DELTA_FRACTION,
    symmetric: bool = True,
) -> Parametrisation:
    length, width_centre = vehicle_frame(vertices)
    chosen = select_candidates(np.asarray(control_points, dtype=float), max_points)
    vertex_ids, origins, inward = [], [], []
    for control_id in chosen:
        idx = nearest_vertex(vertices, np.asarray(control_points[control_id], dtype=float))
        normal = np.asarray(normals[idx], dtype=float)
        norm_ = float(np.linalg.norm(normal))
        normal = normal / norm_ if norm_ > 0 else np.array([0.0, 0.0, 1.0])
        vertex_ids.append(idx)
        origins.append(vertices[idx].copy())
        inward.append(-normal)
    return Parametrisation(
        control_ids=[int(c) for c in chosen],
        vertex_ids=vertex_ids,
        origins=np.asarray(origins, dtype=float),
        inward=np.asarray(inward, dtype=float),
        delta=delta_fraction * length,
        radius=max(RADIUS_FRACTION, RADIUS_PER_DELTA * delta_fraction) * length,
        symmetric=symmetric,
        width_centre=width_centre,
    )


# ── GP / 획득함수 (BO_자료 wing_opt.py · body_bo.py 이식) ────────────────────


def _rbf(a: np.ndarray, b: np.ndarray, ell: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / (ell * ell))


class NoisyGP:
    """등방 RBF + 관측별 잡음(heteroscedastic) GP. 입력은 [0,1]^K. 저차원 전제."""

    def fit(self, x: np.ndarray, y: np.ndarray, sig: np.ndarray) -> NoisyGP:
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        sig = np.asarray(sig, float)
        self.mu0 = float(self.y.mean())
        yc = self.y - self.mu0
        self.sf2 = max(float(yc.var()), 1e-8)
        best = None
        # 길이척도는 소규모 격자에서 주변우도 최대화 — 과공학 불필요(자료 원문).
        for ell in (0.15, 0.25, 0.4, 0.6, 1.0):
            K = self.sf2 * _rbf(self.x, self.x, ell) + np.diag(sig**2 + 1e-8)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            al = np.linalg.solve(L.T, np.linalg.solve(L, yc))
            nll = 0.5 * yc @ al + np.log(np.diag(L)).sum()
            if best is None or nll < best[0]:
                best = (nll, ell, L, al)
        if best is None:  # 전부 특이 — 잡음을 키워 한 번 더
            return self.fit(x, y, np.maximum(sig, 1e-3) * 10.0)
        _, self.ell, self.L, self.alpha = best
        return self

    def predict(self, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xq = np.asarray(xq, float)
        k = self.sf2 * _rbf(xq, self.x, self.ell)
        mu = self.mu0 + k @ self.alpha
        v = np.linalg.solve(self.L, k.T)
        var = np.maximum(self.sf2 - (v**2).sum(0), 1e-10)
        return mu, np.sqrt(var)


class FeasGP:
    """실현가능성 GP — 실패(예외·분포밖)한 설계를 라벨 0 으로 회귀해 P(feasible) 로 쓴다.

    사전평균은 1.0: 모르는 곳은 실현가능하다고 본다. mean(ok) 을 쓰면 데이터에서
    먼 곳이 전부 같은 값이 되어 획득함수에 상수를 곱하는 꼴이 된다(자료 원문).
    """

    def __init__(self, ell: float = FEAS_ELL, noise: float = FEAS_NOISE):
        self.ell, self.noise = ell, noise
        self.trivial = True

    def fit(self, x: np.ndarray, ok: np.ndarray) -> FeasGP:
        ok = np.asarray(ok, float)
        self.x = np.asarray(x, float).reshape(len(ok), -1)
        if ok.size == 0 or ok.min() > 0.5:
            self.trivial = True
            return self
        self.trivial = False
        self.mu0 = 1.0
        K = _rbf(self.x, self.x, self.ell) + np.eye(len(ok)) * self.noise**2
        self.L = np.linalg.cholesky(K)
        self.al = np.linalg.solve(self.L.T, np.linalg.solve(self.L, ok - self.mu0))
        return self

    def p(self, xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, float)
        if self.trivial:
            return np.ones(len(xq))
        k = _rbf(xq, self.x, self.ell)
        return np.clip(self.mu0 + k @ self.al, FEAS_FLOOR, 1.0)


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """최소화 기준 EI (자료 원문 그대로)."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (best - np.asarray(mu, float)) / sd
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return sd * (z * cdf + pdf)


def propose_cei(gp: NoisyGP, feas: FeasGP, n_dim: int, rng: np.random.Generator, n_cand: int = 4096) -> np.ndarray:
    """획득함수 = EI × P(feasible). [0,1]^K 에 후보를 뿌려 argmax."""
    xq = rng.random((n_cand, n_dim))
    mu_obs, _ = gp.predict(gp.x)
    mu, sd = gp.predict(xq)
    acq = expected_improvement(mu, sd, float(mu_obs.min())) * feas.p(xq)
    return xq[int(np.argmax(acq))]


def _to_scales(u: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(u, float) - 1.0


def _to_unit(scales: np.ndarray) -> np.ndarray:
    return (np.asarray(scales, float) + 1.0) / 2.0


# ── 루프 ─────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    history: list[dict] = field(default_factory=list)
    best: list[dict] = field(default_factory=list)
    evaluated: int = 0
    failed: int = 0

    @property
    def baseline(self) -> dict | None:
        return next((h for h in self.history if h["kind"] == "baseline"), None)


def _diverse(order: np.ndarray, points: list[np.ndarray], top: int) -> list[int]:
    """cd 순서에서 서로 떨어진 설계를 고른다 — 카드가 전부 같아 보이면 못 고른다.

    BO 는 한 점에 수렴한다. 그래서 상위 10개를 그냥 자르면 사실상 같은 설계 10장이
    나온다(2026-09-04 사용자 확인). 최선은 항상 넣고, 그 다음부터는 이미 고른
    설계와 u-공간 거리가 임계 이상인 것만 넣는다. 임계를 못 채우면 남는 자리는
    cd 순으로 채운다 — 후보가 적을 때 빈 카드를 만들지 않는다.
    """
    if top <= 0:
        return []
    dim = len(points[0]) if points else 1
    # 임계는 차원에 비례한다. [0,1]^K 에서 무작위 두 점의 평균 거리 ≈ 0.4·sqrt(K).
    threshold = 0.25 * np.sqrt(dim)
    picked: list[int] = []
    for i in order:
        if len(picked) >= top:
            break
        if picked and min(float(np.linalg.norm(points[i] - points[j])) for j in picked) < threshold:
            continue
        picked.append(int(i))
    for i in order:  # 임계로 못 채운 자리
        if len(picked) >= top:
            break
        if int(i) not in picked:
            picked.append(int(i))
    return picked


def optimise(
    evaluate,
    param: Parametrisation,
    vertices: np.ndarray,
    *,
    budget: int = BUDGET,
    n_init: int | None = None,
    top: int = TOP,
    seed: int = 0,
    on_progress=None,
) -> Result:
    """evaluate(vertices) -> {"cd", "cl", "ok", "cd_std"?}. 예외나 ok=False 는 실현불가로 기록한다.

    첫 평가는 항상 s=0(baseline). 그 다음 n_init 개는 무작위, 이후는 cEI.
    """
    dim = param.size
    rng = np.random.default_rng(seed)
    # 자료의 기본값 8 — 7모드에서 검증된 초기표본 수. 예산이 작으면 그 안에서.
    n_init = n_init if n_init is not None else min(8, max(2 * dim, 4))
    n_init = min(n_init, budget)
    result = Result()
    X, Y, S = [], [], []  # 성공한 관측
    XA, OK = [], []  # 모든 관측의 실현가능성 라벨

    def run(u: np.ndarray, kind: str) -> dict:
        scales = _to_scales(u)
        record = {"scales": [float(v) for v in scales], "kind": kind, "cd": None, "cl": None, "ok": False, "error": None}
        try:
            out = evaluate(param.deform(vertices, scales))
            ok = bool(out.get("ok", True)) and out.get("cd") is not None
            record["cd"] = None if out.get("cd") is None else float(out["cd"])
            record["cl"] = None if out.get("cl") is None else float(out["cl"])
            record["ok"] = ok
            if not ok:
                record["error"] = out.get("error") or "infeasible"
        except Exception as exc:  # 한 설계의 실패가 잡을 죽이면 안 된다
            record["error"] = str(exc)
        result.history.append(record)
        result.evaluated += 1
        XA.append(u)
        OK.append(1.0 if record["ok"] else 0.0)
        if record["ok"]:
            X.append(u)
            Y.append(record["cd"])
            S.append(float(out.get("cd_std") or SURROGATE_NOISE))
        else:
            result.failed += 1
        if on_progress:
            on_progress(result.evaluated, budget)
        return record

    run(np.full(dim, 0.5), "baseline")
    while result.evaluated < budget:
        if result.evaluated < n_init or len(X) < 2:
            u = rng.random(dim)
            kind = "init"
        else:
            gp = NoisyGP().fit(np.array(X), np.array(Y), np.array(S))
            feas = FeasGP().fit(np.array(XA), np.array(OK))
            u = propose_cei(gp, feas, dim, rng)
            kind = "bo"
        run(u, kind)

    # 결과: 성공한 설계를 cd 오름차순으로, GP 의 표준편차를 붙여서
    valid = [h for h in result.history if h["ok"]]
    if valid:
        sd = np.zeros(len(valid))
        if len(X) >= 2:
            gp = NoisyGP().fit(np.array(X), np.array(Y), np.array(S))
            _, sd = gp.predict(np.array([_to_unit(h["scales"]) for h in valid]))
        order = np.argsort([h["cd"] for h in valid])
        chosen = _diverse(order, [_to_unit(h["scales"]) for h in valid], max(int(top), 0))
        for rank_, i in enumerate(chosen):
            h = valid[i]
            result.best.append({
                "rank": rank_ + 1,
                "scales": h["scales"],
                "cd": h["cd"],
                "cl": h["cl"],
                "gp_std": float(sd[i]),
                "moves": param.moves(np.asarray(h["scales"])),
            })
    return result
