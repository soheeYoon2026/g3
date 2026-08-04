"""Training samples for the G3 surrogate.

Two sources:

* :class:`SyntheticAeroDataset` — parametric Ahmed-like bodies with an analytic
  pseudo-drag law. It has real, learnable signal, so the smoke test can prove
  the whole pipeline (sampling -> features -> regression) end-to-end **today**,
  before a single CFD label exists. It is NOT physically accurate; it exists to
  exercise the plumbing.

* :class:`ManifestDataset` — the real path: a JSON manifest of
  ``{"stl": path, "cd": float, "cl": float}`` rows, where the labels come from
  G1/G2/G4 (see :mod:`aox_g3.data.label_interface`) or a public CC-BY-SA dataset
  (DrivAerML / AhmedML / WindsorML).

Both yield :class:`Sample` objects with a normalised surface point cloud and a
target vector, so downstream code does not care which source it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..config import SampleConfig, DEFAULT_SAMPLE, TARGETS
from ..geometry.stl_sampler import stl_to_pointcloud, PointCloud


@dataclass
class Sample:
    cloud: PointCloud
    targets: np.ndarray  # shape (len(TARGETS),), order = config.TARGETS
    meta: dict


def pooled_features(cloud: PointCloud) -> np.ndarray:
    """Collapse a variable point cloud into a fixed global descriptor.

    This is what the light-weight sklearn baseline regresses on (a real
    permutation-invariant network like PointNet consumes the raw points
    instead). The descriptor is deliberately aero-flavoured: bounding-box
    proportions, coordinate moments, and a coarse frontal-area / slant proxy
    from the surface-normal distribution — the geometric quantities drag
    actually depends on.
    """
    p = cloud.points
    n = cloud.normals
    lo, hi = p.min(0), p.max(0)
    extent = hi - lo

    feats = [
        extent,                          # bbox proportions (3)
        p.mean(0), p.std(0),             # coordinate moments (6)
        np.abs(n).mean(0),               # mean |normal| per axis (3)
        [(n[:, 0] > 0.5).mean()],        # forward-facing fraction ~ frontal area (1)
        [(n[:, 2] < -0.5).mean()],       # downward-facing fraction ~ underbody (1)
        [np.percentile(p[:, 2], 90) - np.percentile(p[:, 2], 10)],  # height span (1)
    ]
    return np.concatenate([np.ravel(f) for f in feats]).astype(np.float64)


class SyntheticAeroDataset:
    """Parametric Ahmed-like boxes with an analytic pseudo-drag / -lift law.

    A box of length/width/height with a rear slant. "Drag" rises with frontal
    area and with slant angle near the real Ahmed critical angle (~30 deg);
    "lift" grows with slant. Pure geometry -> pure signal, so a model that
    learns it proves the features carry information end to end.
    """

    def __init__(self, n: int = 400, cfg: SampleConfig = DEFAULT_SAMPLE, seed: int = 0):
        self.n = n
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._params = self._sample_params(n)

    def _sample_params(self, n):
        return dict(
            length=self.rng.uniform(0.6, 1.2, n),
            width=self.rng.uniform(0.25, 0.45, n),
            height=self.rng.uniform(0.2, 0.35, n),
            slant_deg=self.rng.uniform(0.0, 45.0, n),
        )

    @staticmethod
    def _pseudo_coeffs(length, width, height, slant_deg):
        # SCALE-INVARIANT by construction: the surrogate sees point clouds
        # normalised to a unit box, so absolute size is gone (as it should be —
        # Cd is dimensionless). The law therefore depends only on proportions
        # (aspect ratios, recoverable from the normalised bbox) and slant angle
        # (recoverable from the rear-roof surface normals).
        aspect_w = width / length
        aspect_h = height / length
        slant = np.deg2rad(slant_deg)
        # Drag: bluffer (taller/wider) bodies drag more; plus an Ahmed-style
        # bump peaking near the ~30 deg critical slant angle.
        cd = (0.20 + 0.55 * aspect_h + 0.45 * aspect_w
              + 0.12 * np.exp(-((slant_deg - 30.0) ** 2) / (2 * 8.0**2)))
        # Lift: grows with rear-roof slant, mild coupling to body height.
        cl = -0.05 + 0.25 * np.sin(slant) - 0.15 * aspect_h
        return np.array([cd, cl], dtype=np.float64)

    def _build_box(self, length, width, height, slant_deg) -> PointCloud:
        """Sample a clean slanted box with correct per-face outward normals.

        Six axis-aligned faces; the rear third of the roof is tilted down by
        ``slant_deg`` (an Ahmed-body slant), and those points get a normal
        rotated about the y-axis so the slant is legible from the normals.
        """
        rng = self.rng
        m = self.cfg.n_surface_points
        face = rng.integers(0, 6, m)   # 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z(top)
        a = rng.random(m)
        b = rng.random(m)

        pts = np.zeros((m, 3))
        nrm = np.zeros((m, 3))
        # x-faces (front/back): span (y, z)
        for f, xval, nx in ((0, 0.0, -1.0), (1, length, 1.0)):
            sel = face == f
            pts[sel] = np.stack([np.full(sel.sum(), xval), a[sel] * width, b[sel] * height], 1)
            nrm[sel] = [nx, 0.0, 0.0]
        # y-faces (sides): span (x, z)
        for f, yval, ny in ((2, 0.0, -1.0), (3, width, 1.0)):
            sel = face == f
            pts[sel] = np.stack([a[sel] * length, np.full(sel.sum(), yval), b[sel] * height], 1)
            nrm[sel] = [0.0, ny, 0.0]
        # z-faces (bottom/top): span (x, y)
        for f, zval, nz in ((4, 0.0, -1.0), (5, height, 1.0)):
            sel = face == f
            pts[sel] = np.stack([a[sel] * length, b[sel] * width, np.full(sel.sum(), zval)], 1)
            nrm[sel] = [0.0, 0.0, nz]

        # Slant the rear third of the roof (top face, x > 2/3 L).
        slant = np.deg2rad(slant_deg)
        roof_rear = (face == 5) & (pts[:, 0] > (2.0 / 3.0) * length)
        drop = np.tan(slant) * (pts[roof_rear, 0] - (2.0 / 3.0) * length)
        pts[roof_rear, 2] -= drop
        # normal of a plane tilted by `slant` about +y: (sin, 0, cos)
        nrm[roof_rear] = np.array([np.sin(slant), 0.0, np.cos(slant)])

        pts += rng.normal(0, 0.001, pts.shape)  # tiny jitter avoids degeneracy

        center = np.zeros(3)
        scale = 1.0
        if self.cfg.normalize:
            lo, hi = pts.min(0), pts.max(0)
            center = (lo + hi) / 2
            scale = float((hi - lo).max()) or 1.0
            pts = (pts - center) / scale
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        return PointCloud(points=pts, normals=nrm, center=center, scale=scale)

    def __len__(self):
        return self.n

    def __getitem__(self, i) -> Sample:
        p = {k: self._params[k][i] for k in self._params}
        cloud = self._build_box(**p)
        targets = self._pseudo_coeffs(**p)
        return Sample(cloud=cloud, targets=targets, meta={"synthetic": True, **p})

    def __iter__(self):
        for i in range(self.n):
            yield self[i]


class ManifestDataset:
    """Real STL + label rows from a JSON manifest.

    Manifest format (list of dicts)::

        [{"stl": "cars/rs6_0001.stl", "cd": 0.301, "cl": -0.12}, ...]

    Labels come from G1/G2/G4 (:mod:`aox_g3.data.label_interface`) or a
    commercial-clean public dataset. STLs are sampled through the exact same
    :func:`stl_to_pointcloud` used at inference.
    """

    def __init__(self, manifest_path: str, cfg: SampleConfig = DEFAULT_SAMPLE):
        with open(manifest_path) as f:
            self.rows = json.load(f)
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i) -> Sample:
        row = self.rows[i]
        cloud = stl_to_pointcloud(row["stl"], self.cfg)
        targets = np.array([row.get(t, np.nan) for t in TARGETS], dtype=np.float64)
        return Sample(cloud=cloud, targets=targets, meta=row)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
