"""추천 후보 미리보기 PNG — 차 전체에 밀린 자리를 색으로, 미는 방향을 화살표로.

카드 하나만 보고 "어디를 어느 쪽으로"가 읽혀야 한다. 점 몇백 개짜리 도식은
그걸 못 했다(2026-09-04, 사용자가 "전혀 못 알아보겠다"). 그래서 실제 형상을
렌더한다 — /v1/infer 의 pressure PNG 와 같은 PyVista 오프스크린 경로다.

- 차체는 밝은 회색, 밀리는 영역은 가중치(0→1)에 따라 주황→빨강.
- 화살표는 표면 밖에서 시작해 표면 안쪽으로 향한다 — "여기를 안으로".
- 카메라는 밀린 점이 있는 쪽(좌/우, 앞/뒤)에서 3/4 로 본다. 반대편에서 보면
  가려져서 아무것도 안 보인다.
- 배경은 카드와 같은 어두운 색이라 이미지 경계가 안 보인다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BACKGROUND = "#111820"
BODY = "#d9dde3"
HIGHLIGHT = [BODY, "#f59e0b", "#ef4444"]
WINDOW = (720, 405)


def _polydata(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    faces = np.asarray(faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=float), cells)


def render_candidate_png(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    out_path: str | Path,
    *,
    length_axis: int = 0,
    width_axis: int = 1,
    fixed_view: bool = False,
) -> Path:
    """가중치로 색칠한 차와 밀기 화살표를 out_path 에 PNG 로 쓴다."""
    import pyvista as pv

    vertices = np.asarray(vertices, dtype=float)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.array([0.0, 0.0, -1.0])

    mesh = _polydata(vertices, faces)
    mesh.point_data["push"] = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)

    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    centre = (lo + hi) / 2.0
    length = max(float(hi[length_axis] - lo[length_axis]), 1e-9)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        mesh,
        scalars="push",
        cmap=HIGHLIGHT,
        clim=(0.0, 1.0),
        show_scalar_bar=False,
        smooth_shading=True,
        specular=0.25,
    )

    # 화살표: 표면 밖(법선 쪽)에서 출발해 안쪽으로. 길이는 전장의 9% — 한눈에 보이게.
    arrow_length = length * 0.09
    start = origin - direction * arrow_length
    arrow = pv.Arrow(
        start=start.tolist(),
        direction=direction.tolist(),
        scale=arrow_length,
        tip_length=0.3,
        tip_radius=0.12,
        shaft_radius=0.045,
    )
    # 빨간 점 위에 빨간 화살표는 묻힌다 — 노란색이 회색·빨강 모두에서 살아남는다.
    plotter.add_mesh(arrow, color="#ffd60a", smooth_shading=True)

    # 밀린 점이 있는 쪽에서 본다. 반대편이면 차체에 가려 아무것도 안 보인다.
    # 최적화 카드는 예외 — 점 여러 개를 한꺼번에 움직이므로 "밀린 쪽"이 없고,
    # 카드끼리 비교하려면 시점이 같아야 한다(2026-09-04: 카드마다 각도가 달라
    # 무엇이 다른지 안 보였다). 그래서 앞왼쪽 3/4 고정.
    side = -1.0 if (not fixed_view and origin[width_axis] < centre[width_axis]) else 1.0
    front = -1.0 if (fixed_view or origin[length_axis] < centre[length_axis]) else 1.0
    view = np.array([front * 0.9, side * 1.0, 0.65])
    view /= np.linalg.norm(view)
    position = centre + view * length * 1.7
    up = [0.0, 0.0, 0.0]
    up[3 - length_axis - width_axis] = 1.0
    plotter.camera_position = [position.tolist(), centre.tolist(), up]
    # 차체와 화살표 시작점이 모두 들어오게 맞춘다 — 코 끝 후보는 화살표가 프레임 밖으로 나갔다.
    frame_lo = np.minimum(lo, start)
    frame_hi = np.maximum(hi, start)
    if fixed_view:  # 차체만으로 맞춘다 — 설계마다 화살표 위치가 달라 크기가 흔들린다
        frame_lo, frame_hi = lo, hi
    plotter.reset_camera(bounds=tuple(np.column_stack([frame_lo, frame_hi]).ravel()))
    plotter.camera.zoom(1.25)

    out_path = Path(out_path)
    plotter.show(screenshot=str(out_path), auto_close=True)
    return out_path
