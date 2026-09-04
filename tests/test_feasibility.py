import numpy as np

from aox_g3.feasibility import geometry_ok, summarise


def _box(length=4.0, width=2.0, height=1.0):
    """축정렬 직육면체 — 길이축 x, 폭축 y, 높이축 z. 바닥은 z=0."""
    x0, x1 = 0.0, length
    y0, y1 = -width / 2, width / 2
    z0, z1 = 0.0, height
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # 바닥
        [4, 5, 6], [4, 6, 7],  # 천장
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])
    return vertices, faces


def test_unchanged_geometry_passes():
    vertices, faces = _box()
    ok, why = geometry_ok(vertices, vertices.copy(), faces)
    assert ok and why == ""


def test_small_push_passes():
    vertices, faces = _box()
    moved = vertices.copy()
    moved[4:, 2] -= 0.02  # 천장을 2 cm 낮춘다
    ok, why = geometry_ok(vertices, moved, faces)
    assert ok, why


def test_nan_vertices_fail():
    vertices, faces = _box()
    moved = vertices.copy()
    moved[0, 0] = np.nan
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok and "NaN" in why


def test_folded_surface_fails():
    vertices, faces = _box()
    moved = vertices.copy()
    moved[4:, 2] = -0.5  # 천장을 바닥 아래로 — 옆면이 뒤집힌다
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok
    assert "folded" in why or "volume" in why


def test_volume_blow_up_fails():
    vertices, faces = _box()
    moved = vertices.copy()
    moved[4:, 2] += 1.0  # 높이 2배 → 부피 +100%
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok and ("volume" in why or "frontal" in why)


def test_frontal_area_change_fails():
    vertices, faces = _box(length=4.0, width=2.0, height=1.0)
    moved = vertices.copy()
    moved[:, 1] *= 1.4  # 폭만 40% 넓힘 → 전면적 +40%, 부피도 +40%
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok and ("frontal" in why or "volume" in why)


def test_sinking_below_the_floor_fails():
    vertices, faces = _box(length=4.0)
    moved = vertices.copy()
    moved[:, 2] -= 0.2  # 전장의 5% 만큼 노면 아래로
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok and "floor" in why


def test_collapsed_geometry_fails():
    vertices, faces = _box()
    moved = vertices.copy()
    moved[:, 2] = 0.0  # 납작하게 눌림
    ok, why = geometry_ok(vertices, moved, faces)
    assert not ok


def test_summarise_groups_reasons_by_kind():
    counts = summarise([
        "volume changed +73%",
        "volume changed +41%",
        "surface folded (0.2% of faces)",
        "body sinks 1.1% of length below the floor",
    ])
    assert counts == {"volume changed": 2, "surface folded": 1, "body sinks": 1}
    assert list(counts)[0] == "volume changed"  # 많은 것부터
