import numpy as np
import pyvista as pv

from scripts.recommend_domino_deformations import _deform


def test_symmetric_deformation_moves_both_sides():
    mesh = pv.Box(bounds=(0, 10, -1, 1, 0, 2)).triangulate()
    origin = np.array([0.0, 1.0, 0.0])
    displacement = np.array([0.0, 0.1, 0.0])
    changed = _deform(mesh, origin, displacement, radius=1.5, lateral_axis=1, symmetric=True)
    points = np.asarray(changed.points)
    assert points[:, 1].max() > 1.0
    assert points[:, 1].min() < -1.0


def test_asymmetric_deformation_moves_only_selected_side():
    mesh = pv.Box(bounds=(0, 10, -1, 1, 0, 2)).triangulate()
    origin = np.array([0.0, 1.0, 0.0])
    displacement = np.array([0.0, 0.1, 0.0])
    changed = _deform(mesh, origin, displacement, radius=1.5, lateral_axis=1, symmetric=False)
    points = np.asarray(changed.points)
    assert points[:, 1].max() > 1.0
    assert np.isclose(points[:, 1].min(), -1.0)
