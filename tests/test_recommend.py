import numpy as np
import pytest

from aox_g3.recommend import (
    gaussian_bump,
    push_inward,
    rank,
    select_candidates,
    vehicle_frame,
)


def _grid_surface(n=21, extent=1000.0):
    """z=0 평면에 n×n 정점. 법선은 전부 +z."""
    xs = np.linspace(0, extent, n)
    ys = np.linspace(-extent / 2, extent / 2, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    return vertices, normals


def test_vehicle_frame_reads_length_and_width_centre():
    vertices = np.array([[0, -900, 0], [4600, 900, 1400], [2000, 0, 700]], dtype=float)
    length, width_centre = vehicle_frame(vertices)
    assert length == 4600
    assert width_centre == 0


def test_select_candidates_keeps_everything_under_the_cap():
    points = np.random.default_rng(0).random((5, 3))
    assert select_candidates(points, 12) == [0, 1, 2, 3, 4]


def test_select_candidates_spreads_along_the_length_axis():
    # x 가 0..71 인 72개 점. 12개를 뽑으면 첫 점과 끝 점을 포함하고 x 가 단조 증가한다.
    points = np.array([[x, 0.0, 0.0] for x in range(72)])
    picks = select_candidates(points, 12)
    assert len(picks) == 12
    xs = [points[i][0] for i in picks]
    assert xs[0] == 0 and xs[-1] == 71
    assert xs == sorted(xs)


def test_gaussian_bump_moves_the_centre_fully_and_far_points_hardly():
    vertices, _ = _grid_surface()
    centre = np.array([500.0, 0.0, 0.0])
    pushed = gaussian_bump(vertices, centre, np.array([0.0, 0.0, -1.0]), 30.0, 200.0)
    at_centre = pushed[np.argmin(((vertices - centre) ** 2).sum(axis=1))]
    assert np.isclose(at_centre[2], -30.0)
    far = pushed[np.argmin(((vertices - [0.0, -500.0, 0.0]) ** 2).sum(axis=1))]
    assert abs(far[2]) < 1e-6


def test_push_inward_goes_against_the_normal_and_reports_the_displacement():
    vertices, normals = _grid_surface()
    pushed, idx, displacement = push_inward(
        vertices, normals, np.array([500.0, 300.0, 0.0]), 30.0, 200.0,
        symmetric=False, width_centre=0.0,
    )
    assert np.allclose(displacement, [0.0, 0.0, -30.0])
    assert np.isclose(pushed[idx][2], -30.0)


def test_push_inward_mirrors_across_the_width_centre_when_symmetric():
    vertices, normals = _grid_surface()
    point = np.array([500.0, 300.0, 0.0])
    pushed, _, _ = push_inward(vertices, normals, point, 30.0, 200.0, symmetric=True, width_centre=0.0)
    mirror = np.array([500.0, -300.0, 0.0])
    at_mirror = pushed[np.argmin(((vertices - mirror) ** 2).sum(axis=1))]
    assert np.isclose(at_mirror[2], -30.0)


def test_push_inward_skips_the_mirror_on_the_centre_plane():
    # 중심면 위의 점을 대칭으로 두 번 밀면 두 배가 된다. 건너뛰어야 한다.
    vertices, normals = _grid_surface()
    point = np.array([500.0, 0.0, 0.0])
    pushed, idx, _ = push_inward(vertices, normals, point, 30.0, 200.0, symmetric=True, width_centre=0.0)
    assert np.isclose(pushed[idx][2], -30.0)


def test_rank_orders_by_delta_cd_and_pushes_missing_values_last():
    results = [
        {"control_id": 1, "delta_cd": 0.002},
        {"control_id": 2, "delta_cd": None},
        {"control_id": 3, "delta_cd": -0.004},
        {"control_id": 4, "delta_cd": -0.001},
    ]
    assert [r["control_id"] for r in rank(results, 3)] == [3, 4, 1]
    assert rank(results, 0) == []

def test_overview_points_are_normalised_by_half_length_and_centred():
    from aox_g3.recommend import overview_points
    vertices = np.array([[0, -900, 0], [4600, 900, 1400]], dtype=float)
    pts = np.array(overview_points(vertices))
    assert pts.shape == (2, 3)
    assert np.allclose(pts[:, 0], [-1, 1])          # 길이축이 [-1, 1]
    assert np.allclose(pts[:, 2], [-1400 / 4600, 1400 / 4600], atol=1e-4)  # 같은 배율 (4자리 반올림)


def test_overview_points_are_subsampled_to_the_cap():
    from aox_g3.recommend import overview_points
    vertices = np.random.default_rng(1).random((5000, 3)) * 1000
    assert len(overview_points(vertices, count=480)) == 480


def test_preview_points_are_local_and_within_the_unit_ball():
    from aox_g3.recommend import preview_points
    vertices, _ = _grid_surface()
    pts = np.array(preview_points(vertices, np.array([500.0, 0.0, 0.0]), 200.0))
    assert len(pts) > 0
    assert np.all(np.linalg.norm(pts, axis=1) <= 1.0 + 1e-9)
    assert np.allclose(pts.mean(axis=0), 0, atol=0.25)  # 눌린 점 주변에 모여 있다


def test_bump_weights_are_one_at_the_centre_and_vanish_far_away():
    from aox_g3.recommend import bump_weights
    vertices, _ = _grid_surface()
    w = bump_weights(vertices, np.array([500.0, 0.0, 0.0]), 200.0)
    assert np.isclose(w.max(), 1.0)
    assert w[np.argmin(((vertices - [0.0, -500.0, 0.0]) ** 2).sum(axis=1))] < 1e-6


def test_highlight_weights_include_the_mirror_when_symmetric():
    from aox_g3.recommend import highlight_weights
    vertices, _ = _grid_surface()
    point = np.array([500.0, 300.0, 0.0])
    w = highlight_weights(vertices, point, 200.0, symmetric=True, width_centre=0.0)
    mirror_idx = np.argmin(((vertices - [500.0, -300.0, 0.0]) ** 2).sum(axis=1))
    assert np.isclose(w[mirror_idx], 1.0)
    w_one = highlight_weights(vertices, point, 200.0, symmetric=False, width_centre=0.0)
    assert w_one[mirror_idx] < 1e-6


def test_render_candidate_png_writes_an_image(tmp_path):
    pv = pytest.importorskip("pyvista")
    from aox_g3.preview import render_candidate_png
    # 작은 닫힌 상자 하나면 충분하다 — 그림이 예쁜지가 아니라 파이프라인이 도는지를 본다.
    box = pv.Box(bounds=(0, 4, -1, 1, 0, 1)).triangulate()
    vertices = np.asarray(box.points, float)
    faces = np.asarray(box.faces).reshape(-1, 4)[:, 1:]
    weights = np.zeros(len(vertices)); weights[0] = 1.0
    out = tmp_path / "candidate.png"
    try:
        render_candidate_png(vertices, faces, weights, vertices[0], np.array([0.0, 0.0, -1.0]), out)
    except RuntimeError as exc:  # 헤드리스 CI 에 OpenGL 이 없을 수 있다
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    data = out.read_bytes()
    assert data.startswith(bytes([0x89]) + b"PNG") and len(data) > 1000
