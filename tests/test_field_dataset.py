import json
from pathlib import Path

import numpy as np

from aox_g3.data.field_dataset import (
    G1SurfaceDataset,
    G2FieldDataset,
    condition_stats,
    load_field_manifest,
)


def test_field_npz_contract_and_sampling(tmp_path: Path):
    rng = np.random.default_rng(4)
    np.savez_compressed(
        tmp_path / "case.npz",
        geometry_points=rng.normal(size=(100, 3)).astype(np.float32),
        geometry_normals=rng.normal(size=(100, 3)).astype(np.float32),
        volume_points=rng.normal(size=(200, 3)).astype(np.float32),
        volume_velocity=rng.normal(size=(200, 3)).astype(np.float32),
        volume_cp=rng.normal(size=200).astype(np.float32),
        surface_points=rng.normal(size=(100, 3)).astype(np.float32),
        surface_cp=rng.normal(size=100).astype(np.float32),
    )
    conditions = {
        "u_x": 30.0, "u_y": 0.0, "u_z": 0.0, "density": 1.225,
        "viscosity": 1.7894e-5, "temperature": 288.15,
        "ref_length": 5.0, "ref_area": 2.2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({
        "cases": [{"case_id": "a", "group_id": "g", "npz": "case.npz",
                   "conditions": conditions}]
    }))
    cases = load_field_manifest(tmp_path / "manifest.json")
    mean, std = condition_stats(cases)
    sample = G2FieldDataset(
        cases, geometry_points=64, volume_queries=80, surface_queries=32,
        condition_mean=mean, condition_std=std,
    )[0]
    assert sample["geometry"].shape == (64, 6)
    assert sample["volume_points"].shape == (80, 3)
    assert sample["volume_targets"].shape == (80, 4)
    assert sample["surface_points"].shape == (32, 3)
    assert np.isfinite(sample["conditions"]).all()


def test_g1_surface_contract_and_drag_scaling(tmp_path: Path):
    rng = np.random.default_rng(7)
    np.savez_compressed(
        tmp_path / "g1.npz",
        geometry_points=rng.normal(size=(100, 3)).astype(np.float32),
        geometry_normals=rng.normal(size=(100, 3)).astype(np.float32),
        surface_points=rng.normal(size=(100, 3)).astype(np.float32),
        surface_cp=rng.normal(size=100).astype(np.float32),
        surface_velocity=rng.normal(size=(100, 3)).astype(np.float32),
        cd_final=np.float32(0.3),
    )
    conditions = {name: 1.0 for name in (
        "u_x", "u_y", "u_z", "density", "viscosity",
        "temperature", "ref_length", "ref_area",
    )}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "cases": [{"case_id": "g1", "group_id": "project", "npz": "g1.npz",
                   "conditions": conditions}]
    }))
    cases = load_field_manifest(tmp_path / "manifest.json")
    sample = G1SurfaceDataset(
        cases, geometry_points=64, surface_queries=32,
        log_cd_mean=np.log(0.3), log_cd_std=1.0,
    )[0]
    assert sample["geometry"].shape == (64, 6)
    assert sample["surface_points"].shape == (32, 3)
    assert sample["surface_targets"].shape == (32, 4)
    assert np.isclose(sample["log_cd"][0], 0.0, atol=1e-6)
    assert np.isclose(sample["cd"][0], 0.3)
