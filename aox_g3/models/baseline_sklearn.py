"""Torch-free baseline surrogate: pooled global features -> MLP regression.

This is the Phase 0 workhorse. It is NOT the production model — it throws away
per-point structure by pooling to a global descriptor — but it needs only numpy
and scikit-learn, so it runs in the current environment and lets us validate the
entire data path before committing to the GPU stack.

When ``torch`` is available, swap this for :class:`PointNetRegressor`, which
keeps per-point structure and is what the real point-based operator (DoMINO)
generalises. The public API (``fit`` / ``predict`` on point clouds) is the same
so :mod:`aox_g3.train` and :mod:`aox_g3.infer` are model-agnostic.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..config import TARGETS
from ..data.dataset import pooled_features
from ..geometry.stl_sampler import PointCloud


class PooledMLPRegressor:
    """Pooled-feature MLP over TARGETS (Cd, Cl)."""

    def __init__(self, hidden=(128, 128), alpha=1e-3, max_iter=2000, seed=0):
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                                 max_iter=max_iter, random_state=seed)),
        ])
        self.target_names = list(TARGETS)

    # --- feature extraction -------------------------------------------------
    @staticmethod
    def _featurize(clouds: list[PointCloud]) -> np.ndarray:
        return np.stack([pooled_features(c) for c in clouds])

    # --- sklearn-style API on point clouds ----------------------------------
    def fit(self, clouds: list[PointCloud], targets: np.ndarray) -> "PooledMLPRegressor":
        X = self._featurize(clouds)
        y = np.asarray(targets, dtype=np.float64)
        # Standardise the targets. Cd (~0.4-0.9) and Cl (~-0.1-0.15) live on
        # different scales; a single MLP loss over raw targets underfits the
        # smaller one. Store the transform to invert at predict time.
        self.y_mean = y.mean(0)
        self.y_std = y.std(0)
        self.y_std[self.y_std == 0] = 1.0
        self.model.fit(X, (y - self.y_mean) / self.y_std)
        return self

    def predict(self, clouds: list[PointCloud]) -> np.ndarray:
        X = self._featurize(clouds)
        pred = self.model.predict(X)
        pred = np.atleast_2d(pred).reshape(len(clouds), len(self.target_names))
        return pred * self.y_std + self.y_mean

    # --- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PooledMLPRegressor":
        with open(path, "rb") as f:
            return pickle.load(f)
