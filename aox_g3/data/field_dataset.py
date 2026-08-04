"""Datasets for G2-derived pressure and three-dimensional velocity fields.

Each case is stored as a compressed NPZ produced by
``scripts/prepare_g2_fields.py``. Coordinates share one body-based transform:
``x_normalized = (x_raw - center) / scale``. Velocity is divided by the
freestream speed and pressure is represented as pressure coefficient ``Cp``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONDITION_NAMES = (
    "u_x", "u_y", "u_z", "density", "viscosity",
    "temperature", "ref_length", "ref_area",
)


@dataclass(frozen=True)
class FieldCase:
    case_id: str
    group_id: str
    path: Path
    conditions: np.ndarray
    metadata: dict


def load_field_manifest(path: str | Path) -> list[FieldCase]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    rows = payload["cases"] if isinstance(payload, dict) else payload
    result = []
    for row in rows:
        npz = Path(row["npz"])
        if not npz.is_absolute():
            npz = (manifest_path.parent / npz).resolve()
        conditions = np.asarray(
            [float(row["conditions"].get(name, 0.0)) for name in CONDITION_NAMES],
            dtype=np.float32,
        )
        result.append(FieldCase(
            case_id=str(row["case_id"]),
            group_id=str(row.get("group_id", row["case_id"])),
            path=npz,
            conditions=conditions,
            metadata=row,
        ))
    return result


def condition_stats(cases: list[FieldCase]) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([case.conditions for case in cases])
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


class G2FieldDataset:
    """Lazy NPZ dataset that samples geometry and query points per access."""

    def __init__(
        self,
        cases: list[FieldCase],
        geometry_points: int = 4096,
        volume_queries: int = 8192,
        surface_queries: int = 2048,
        condition_mean: np.ndarray | None = None,
        condition_std: np.ndarray | None = None,
        log_cd_mean: float = 0.0,
        log_cd_std: float = 1.0,
        cl_mean: float = 0.0,
        cl_std: float = 1.0,
        seed: int = 0,
    ):
        self.cases = list(cases)
        self.geometry_points = geometry_points
        self.volume_queries = volume_queries
        self.surface_queries = surface_queries
        self.condition_mean = np.zeros(len(CONDITION_NAMES), np.float32) if condition_mean is None else condition_mean
        self.condition_std = np.ones(len(CONDITION_NAMES), np.float32) if condition_std is None else condition_std
        self.log_cd_mean = float(log_cd_mean)
        self.log_cd_std = max(float(log_cd_std), 1e-8)
        self.cl_mean = float(cl_mean)
        self.cl_std = max(float(cl_std), 1e-8)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.cases)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _choice(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
        return rng.choice(n, size=count, replace=n < count)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        case = self.cases[index]
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        with np.load(case.path) as data:
            gi = self._choice(rng, len(data["geometry_points"]), self.geometry_points)
            vi = self._choice(rng, len(data["volume_points"]), self.volume_queries)
            si = self._choice(rng, len(data["surface_points"]), self.surface_queries)
            geometry = np.concatenate(
                [data["geometry_points"][gi], data["geometry_normals"][gi]], axis=1
            ).astype(np.float32)
            volume_points = data["volume_points"][vi].astype(np.float32)
            volume_targets = np.concatenate(
                [data["volume_cp"][vi, None], data["volume_velocity"][vi]], axis=1
            ).astype(np.float32)
            surface_points = data["surface_points"][si].astype(np.float32)
            surface_cp = data["surface_cp"][si, None].astype(np.float32)

        conditions = ((case.conditions - self.condition_mean) / self.condition_std).astype(np.float32)
        coefficients = case.metadata.get("coefficients", {})
        try:
            cd = float(coefficients.get("cd", np.nan))
        except (TypeError, ValueError):
            cd = np.nan
        try:
            cl = float(coefficients.get("cl", np.nan))
        except (TypeError, ValueError):
            cl = np.nan
        has_cd = np.isfinite(cd) and cd > 0.0
        has_cl = np.isfinite(cl)
        global_targets = np.asarray([
            (np.log(cd) - self.log_cd_mean) / self.log_cd_std if has_cd else 0.0,
            (cl - self.cl_mean) / self.cl_std if has_cl else 0.0,
        ], dtype=np.float32)
        return {
            "geometry": geometry,
            "conditions": conditions,
            "volume_points": volume_points,
            "volume_targets": volume_targets,
            "surface_points": surface_points,
            "surface_cp": surface_cp,
            "global_targets": global_targets,
            "global_values": np.asarray([cd, cl], dtype=np.float32),
            "global_mask": np.asarray([has_cd, has_cl], dtype=np.bool_),
        }


class G1SurfaceDataset:
    """G1 surface fields plus one signed or positive global coefficient."""

    def __init__(
        self,
        cases: list[FieldCase],
        geometry_points: int = 4096,
        surface_queries: int = 4096,
        condition_mean: np.ndarray | None = None,
        condition_std: np.ndarray | None = None,
        log_cd_mean: float = 0.0,
        log_cd_std: float = 1.0,
        coefficient_key: str = "cd_final",
        coefficient_transform: str = "log",
        coefficient_mean: float | None = None,
        coefficient_std: float | None = None,
        seed: int = 0,
    ):
        self.cases = list(cases)
        self.geometry_points = geometry_points
        self.surface_queries = surface_queries
        self.condition_mean = np.zeros(len(CONDITION_NAMES), np.float32) if condition_mean is None else condition_mean
        self.condition_std = np.ones(len(CONDITION_NAMES), np.float32) if condition_std is None else condition_std
        self.log_cd_mean = float(log_cd_mean)
        self.log_cd_std = max(float(log_cd_std), 1e-8)
        if coefficient_transform not in {"log", "standard"}:
            raise ValueError("coefficient_transform must be 'log' or 'standard'")
        self.coefficient_key = coefficient_key
        self.coefficient_transform = coefficient_transform
        self.coefficient_mean = float(log_cd_mean if coefficient_mean is None else coefficient_mean)
        self.coefficient_std = max(float(log_cd_std if coefficient_std is None else coefficient_std), 1e-8)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.cases)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        case = self.cases[index]
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        with np.load(case.path) as data:
            n_geometry = len(data["geometry_points"])
            n_surface = len(data["surface_points"])
            gi = rng.choice(n_geometry, self.geometry_points, replace=n_geometry < self.geometry_points)
            si = rng.choice(n_surface, self.surface_queries, replace=n_surface < self.surface_queries)
            geometry = np.concatenate(
                [data["geometry_points"][gi], data["geometry_normals"][gi]], axis=1
            ).astype(np.float32)
            surface_points = data["surface_points"][si].astype(np.float32)
            surface_targets = np.concatenate([
                data["surface_cp"][si, None], data["surface_velocity"][si]
            ], axis=1).astype(np.float32)
            coefficient = float(data[self.coefficient_key])
        conditions = ((case.conditions - self.condition_mean) / self.condition_std).astype(np.float32)
        transformed = np.log(max(coefficient, 1e-8)) if self.coefficient_transform == "log" else coefficient
        target = (transformed - self.coefficient_mean) / self.coefficient_std
        result = {
            "geometry": geometry,
            "conditions": conditions,
            "surface_points": surface_points,
            "surface_targets": surface_targets,
            "coefficient_target": np.asarray([target], dtype=np.float32),
            "coefficient": np.asarray([coefficient], dtype=np.float32),
        }
        # Preserve the v2 public sample contract for existing drag consumers.
        if self.coefficient_key == "cd_final" and self.coefficient_transform == "log":
            result["log_cd"] = result["coefficient_target"]
            result["cd"] = result["coefficient"]
        return result
