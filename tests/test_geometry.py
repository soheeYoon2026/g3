"""Unit tests for the geometry path (run: pytest g3/tests)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.geometry.ffd import FFD
from aox_g3.data.dataset import SyntheticAeroDataset, pooled_features
from aox_g3.config import SampleConfig


def _unit_box_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, 3))


def test_ffd_identity_is_exact():
    pts = _unit_box_points()
    ffd = FFD(pts, dims=(4, 3, 3))
    out = ffd.deform(np.zeros((ffd.n_control, 3)))
    # Linear precision => Bernstein volume reproduces embedded points exactly.
    assert np.allclose(out, pts, atol=1e-9)


def test_ffd_control_point_moves_points():
    pts = _unit_box_points()
    ffd = FFD(pts, dims=(4, 3, 3))
    delta = np.zeros((ffd.n_control, 3))
    delta[0] = [0.5, 0.0, 0.0]
    out = ffd.deform(delta)
    assert np.abs(out - pts).max() > 1e-3
    # Displacement must stay bounded by the control-point move (partition of unity).
    assert np.abs(out - pts).max() <= 0.5 + 1e-9


def test_ffd_jacobian_matches_finite_difference():
    pts = _unit_box_points(n=50)
    ffd = FFD(pts, dims=(3, 3, 3))
    J = ffd.jacobian()  # (N, P), same for every xyz component
    base = ffd.deform(np.zeros((ffd.n_control, 3)))
    eps = 1e-4
    for cp in (0, 5, ffd.n_control - 1):
        d = np.zeros((ffd.n_control, 3)); d[cp, 0] = eps
        fd = (ffd.deform(d)[:, 0] - base[:, 0]) / eps
        assert np.allclose(fd, J[:, cp], atol=1e-6)


def test_synthetic_dataset_shapes_and_signal():
    ds = SyntheticAeroDataset(n=20, cfg=SampleConfig(n_surface_points=256), seed=3)
    samples = list(ds)
    assert len(samples) == 20
    s0 = samples[0]
    assert s0.cloud.points.shape == (256, 3)
    assert s0.cloud.normals.shape == (256, 3)
    assert s0.targets.shape == (2,)
    # Pooled features are fixed-length regardless of point count.
    f = pooled_features(s0.cloud)
    assert f.ndim == 1 and np.isfinite(f).all()
    # Targets vary across the dataset (there is signal to learn).
    cds = np.array([s.targets[0] for s in samples])
    assert cds.std() > 1e-3


def test_pooled_features_permutation_invariant():
    ds = SyntheticAeroDataset(n=1, cfg=SampleConfig(n_surface_points=300), seed=0)
    cloud = list(ds)[0].cloud
    f1 = pooled_features(cloud)
    perm = np.random.default_rng(0).permutation(len(cloud.points))
    cloud.points = cloud.points[perm]
    cloud.normals = cloud.normals[perm]
    f2 = pooled_features(cloud)
    assert np.allclose(f1, f2, atol=1e-9)
