import json
from pathlib import Path

import numpy as np
import pytest

from aox_g3.geometry.stl_sampler import load_mesh, sample_surface, stl_to_pointcloud
from aox_g3.geometry.surface_sampling import (
    GEOMETRY_PREPROCESSING_VERSION,
    sample_triangle_surface,
)


def test_stl_and_shared_sampler_are_identical(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    stl = tmp_path / "box.stl"
    mesh.export(stl)
    loaded = load_mesh(str(stl))
    points, normals = sample_surface(loaded, 512, seed=11)
    shared = sample_triangle_surface(
        np.asarray(loaded.vertices), np.asarray(loaded.faces), 512, seed=11
    )
    np.testing.assert_array_equal(points, shared.points)
    np.testing.assert_array_equal(normals, shared.normals)
    cloud = stl_to_pointcloud(str(stl))
    lo, hi = np.asarray(loaded.bounds)
    np.testing.assert_allclose(cloud.center, (lo + hi) / 2.0)
    assert cloud.scale == pytest.approx(float(np.max(hi - lo)))


def _write_case(path: Path, case_index: int) -> dict:
    rng = np.random.default_rng(100 + case_index)
    geometry = rng.uniform(-0.5, 0.5, size=(64, 3)).astype(np.float32)
    normals = rng.normal(size=(64, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    volume_points = rng.uniform(-1.0, 1.0, size=(96, 3)).astype(np.float32)
    volume_cp = (0.1 * volume_points[:, 0] + 0.01 * case_index).astype(np.float32)
    volume_velocity = np.column_stack([
        np.ones(96), 0.05 * volume_points[:, 1], 0.05 * volume_points[:, 2]
    ]).astype(np.float32)
    surface_cp = (0.1 * geometry[:, 0]).astype(np.float32)
    np.savez_compressed(
        path,
        geometry_points=geometry,
        geometry_normals=normals,
        volume_points=volume_points,
        volume_cp=volume_cp,
        volume_velocity=volume_velocity,
        surface_points=geometry,
        surface_cp=surface_cp,
        geometry_preprocessing_version=np.asarray(GEOMETRY_PREPROCESSING_VERSION),
    )
    return {
        "case_id": f"case-{case_index}",
        "group_id": f"group-{case_index}",
        "npz": path.name,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "conditions": {
            "u_x": 30.0, "u_y": 0.0, "u_z": 0.0,
            "density": 1.225, "viscosity": 1.7894e-5,
            "temperature": 288.15, "ref_length": 4.0, "ref_area": 2.0,
        },
        "coefficients": {"cd": 0.2 + 0.01 * case_index, "cl": 0.01 * case_index},
    }


def test_tiny_train_checkpoint_and_stl_inference(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    from aox_g3 import field_io
    from aox_g3.infer_fields import main as infer_main
    from aox_g3.train_fields import main as train_main

    rows = [_write_case(tmp_path / f"case-{index}.npz", index) for index in range(2)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "cases": rows,
    }))
    checkpoint = tmp_path / "tiny.pt"
    train_main([
        "--manifest", str(manifest), "--out", str(checkpoint),
        "--epochs", "1", "--batch-size", "1", "--steps-per-epoch", "1",
        "--geometry-points", "16", "--volume-queries", "16",
        "--surface-queries", "16", "--device", "cpu",
    ])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["geometry_preprocessing_version"] == GEOMETRY_PREPROCESSING_VERSION

    stl = tmp_path / "box.stl"
    trimesh.creation.box(extents=(4.0, 2.0, 1.0)).export(stl)

    def fake_volume(path, *args, **kwargs):
        Path(path).write_bytes(b"volume")
        return object()

    def fake_surface(path, *args, **kwargs):
        Path(path).write_bytes(b"surface")

    def fake_streamlines(volume, path, *args, **kwargs):
        Path(path).write_bytes(b"streamlines")

    monkeypatch.setattr(field_io, "write_volume_vti", fake_volume)
    monkeypatch.setattr(field_io, "write_surface_vtp", fake_surface)
    monkeypatch.setattr(field_io, "write_streamlines", fake_streamlines)
    output = tmp_path / "prediction"
    summary = infer_main([
        "--stl", str(stl), "--model", str(checkpoint),
        "--out-dir", str(output), "--grid", "8", "8", "8",
        "--geometry-points", "16", "--chunk-size", "128", "--device", "cpu",
        "--ref-length", "4.0", "--ref-area", "2.0",
    ])
    assert summary["geometry_preprocessing_version"] == GEOMETRY_PREPROCESSING_VERSION
    assert np.isfinite(summary["drag_coefficient"])
    assert np.isfinite(summary["lift_coefficient"])
    assert (output / "prediction.json").is_file()

    quick_output = tmp_path / "prediction-quick"
    quick = infer_main([
        "--stl", str(stl), "--model", str(checkpoint),
        "--out-dir", str(quick_output), "--geometry-points", "16",
        "--device", "cpu", "--ref-length", "4.0", "--ref-area", "2.0",
        "--coefficients-only",
    ])
    assert quick["mode"] == "coefficients-only"
    assert np.isfinite(quick["drag_coefficient"])
    assert quick["mesh"]["faces"] == 12
    assert (quick_output / "prediction.json").is_file()
    assert not (quick_output / "volume_field.vti").exists()
