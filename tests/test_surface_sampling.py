import numpy as np

from aox_g3.geometry.surface_sampling import (
    interpolate_vertex_values,
    normalize_points,
    sample_triangle_surface,
)


def test_sampling_is_deterministic_and_area_weighted():
    vertices = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [2, 0, 0], [5, 0, 0], [2, 3, 0],
    ], dtype=float)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])
    first = sample_triangle_surface(vertices, faces, 10_000, seed=7)
    second = sample_triangle_surface(vertices, faces, 10_000, seed=7)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.face_indices, second.face_indices)
    ratio = np.count_nonzero(first.face_indices == 1) / np.count_nonzero(first.face_indices == 0)
    assert 8.0 < ratio < 10.0
    np.testing.assert_allclose(np.linalg.norm(first.normals, axis=1), 1.0)


def test_barycentric_interpolation_and_normalization():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2]])
    samples = sample_triangle_surface(vertices, faces, 100, seed=3)
    scalar = interpolate_vertex_values(np.asarray([0.0, 1.0, 2.0]), faces, samples)
    expected = samples.barycentric @ np.asarray([0.0, 1.0, 2.0])
    np.testing.assert_allclose(scalar, expected)
    vector = interpolate_vertex_values(vertices, faces, samples)
    np.testing.assert_allclose(vector, samples.points)
    normalized, center, scale = normalize_points(vertices)
    np.testing.assert_allclose(center, [0.5, 0.5, 0.0])
    assert scale == 1.0
    np.testing.assert_allclose(normalized.min(0), [-0.5, -0.5, 0.0])


def test_degenerate_faces_are_ignored():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 1, 3]])
    samples = sample_triangle_surface(vertices, faces, 100, seed=0)
    assert np.all(samples.face_indices == 0)
