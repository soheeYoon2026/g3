"""표면 밀기 추천 — 서로게이트로 후보를 실제 평가해 ΔCd 순으로 돌려준다.

플랫폼의 "Recommend lower-Cd shapes" 가 부르는 /v1/recommend 의 계산부.
편집기가 준 제어점(원본 STL 좌표, ≤72개) 각각에서 표면을 법선 반대 방향으로
조금 밀어 본 형상을 만들고, 그 형상을 /v1/infer 와 같은 경로로 추론해
baseline 대비 ΔCd 를 잰다. 민감도 해석기가 아니라 **유한차분 스크리닝**이다.

v1 의 가정 (응답 limitations 에 그대로 적는다):
- 후보는 길이축으로 고르게 최대 MAX_CANDIDATES 개만 평가한다 — 한 번 추론에
  수 초가 걸려 72개 전부는 요청 타임아웃을 넘긴다.
- 방향은 안쪽(−법선) 하나뿐이다. 바깥으로 미는 후보는 보지 않는다.
- 밀기 크기와 영향 반경은 전장 비율로 고정한다.
- 순위는 ΔCd 만 본다. ΔCl 은 같이 돌려주되 순위에 쓰지 않는다.

여기 함수들은 numpy 만 쓰고 추론을 부르지 않는다 — 테스트할 수 있게.
"""

from __future__ import annotations

import numpy as np

LENGTH_AXIS = 0
WIDTH_AXIS = 1

# 전장 대비 비율. 4.6 m 차에서 밀기 ≈ 28 mm, 반경 ≈ 370 mm.
PUSH_FRACTION = 0.006
RADIUS_FRACTION = 0.08
MAX_CANDIDATES = 12


def vehicle_frame(vertices: np.ndarray) -> tuple[float, float]:
    """(전장, 폭 중심 좌표). 폭축은 WIDTH_AXIS 로 고정한다 — flow_axis=+x 전제."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    length = float(hi[LENGTH_AXIS] - lo[LENGTH_AXIS])
    width_centre = float((lo[WIDTH_AXIS] + hi[WIDTH_AXIS]) / 2.0)
    return length, width_centre


def select_candidates(points: np.ndarray, max_candidates: int = MAX_CANDIDATES) -> list[int]:
    """길이축으로 정렬해 고르게 뽑은 제어점 인덱스. 코부터 꼬리까지 골고루 본다."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= max_candidates:
        return list(range(n))
    order = np.argsort(pts[:, LENGTH_AXIS], kind="stable")
    picks = np.linspace(0, n - 1, max_candidates).round().astype(int)
    return [int(order[i]) for i in picks]


def nearest_vertex(vertices: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(((vertices - point) ** 2).sum(axis=1)))


def gaussian_bump(
    vertices: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    magnitude: float,
    radius: float,
) -> np.ndarray:
    """centre 에서 magnitude 만큼, radius 밖에서는 사실상 0 으로 감쇠하는 밀기."""
    d2 = ((vertices - centre) ** 2).sum(axis=1)
    sigma = max(radius, 1e-9) / 2.0
    weight = np.exp(-d2 / (2.0 * sigma * sigma))
    return vertices + weight[:, None] * (magnitude * direction)[None, :]


def push_inward(
    vertices: np.ndarray,
    normals: np.ndarray,
    point: np.ndarray,
    magnitude: float,
    radius: float,
    *,
    symmetric: bool,
    width_centre: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """제어점에 가장 가까운 정점에서 법선 반대 방향으로 민 정점 배열.

    반환: (새 정점들, 눌린 정점 인덱스, 그 점의 변위 벡터).
    symmetric 이면 폭 중심면에 대칭인 자리도 같이 민다 — 제어점이 중심면에
    거의 붙어 있으면(반경의 1/4 안) 두 밀기가 겹쳐 두 배가 되므로 건너뛴다.
    """
    idx = nearest_vertex(vertices, np.asarray(point, dtype=float))
    normal = np.asarray(normals[idx], dtype=float)
    norm = float(np.linalg.norm(normal))
    normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    direction = -normal
    origin = vertices[idx]
    pushed = gaussian_bump(vertices, origin, direction, magnitude, radius)

    if symmetric and abs(float(origin[WIDTH_AXIS]) - width_centre) > radius / 4.0:
        mirror_origin = origin.copy()
        mirror_origin[WIDTH_AXIS] = 2.0 * width_centre - mirror_origin[WIDTH_AXIS]
        mirror_direction = direction.copy()
        mirror_direction[WIDTH_AXIS] = -mirror_direction[WIDTH_AXIS]
        pushed = gaussian_bump(pushed, mirror_origin, mirror_direction, magnitude, radius)

    return pushed, idx, direction * magnitude


def rank(results: list[dict], top: int) -> list[dict]:
    """ΔCd 오름차순 상위 top. 값이 없는(None) 후보는 뒤로 보낸다."""
    def key(item: dict) -> float:
        value = item.get("delta_cd")
        return float("inf") if value is None else float(value)

    return sorted(results, key=key)[: max(int(top), 0)]
