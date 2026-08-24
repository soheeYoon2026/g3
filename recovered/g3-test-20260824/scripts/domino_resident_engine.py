"""Long-lived DoMINO model and geometry-input cache for STL inference."""
from __future__ import annotations

import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from huggingface_hub import snapshot_download
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper
from physicsnemo.models.domino.utils import unnormalize

from scripts.evaluate_domino_v3 import forward_surface
from scripts.infer_domino_stl import g2_reference_area, orient


class ResidentDominoEngine:
    def __init__(self, checkpoint: Path, device: str = "cuda:0", cache_size: int = 8):
        self.device = device
        self.cache_size = cache_size
        snapshot = Path(snapshot_download("nvidia/domino_drivaerml", local_files_only=True))
        surface = snapshot / "domino_drivaerml_surface_checkpoint"
        self.wrapper = DominoWrapper().load(
            checkpoint_path=str(surface / "DoMINO.0.501.mdlus"),
            stats_path=str(surface / "scaling_factors.pkl"),
            device=device,
            domino_config=str(surface / "config.yaml"),
        )
        self.model = self.wrapper._model
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.eval()
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def _prepare(self, stl_path: Path, flow_axis: str) -> dict:
        mesh = pv.read(stl_path).extract_surface().triangulate().clean()
        if mesh.n_cells < 100 or not np.isfinite(mesh.points).all():
            raise ValueError("invalid or too-small STL surface")
        mesh.points = orient(np.asarray(mesh.points, dtype=np.float64), flow_axis)
        automatic_area = g2_reference_area(mesh)
        if not np.isfinite(automatic_area) or automatic_area <= 0:
            raise ValueError("could not determine positive reference area")
        with tempfile.TemporaryDirectory(prefix="domino-resident-") as temp_dir:
            run = Path(temp_dir) / "run_1"
            run.mkdir()
            mesh.cell_data["pMeanTrim"] = np.zeros(mesh.n_cells, dtype=np.float32)
            mesh.cell_data["wallShearStressMeanTrim"] = np.zeros((mesh.n_cells, 3), dtype=np.float32)
            mesh.save(run / "boundary_1.vtp")
            mesh.save(run / "drivaer_1.stl")
            adapter = DrivAerMLAdapter(root=temp_dir)
            data = self.wrapper.prepare_inputs(adapter.load_case("run_1"))["data_dict"]
        return {"data": data, "automatic_area": automatic_area}

    def predict(
        self,
        stl_path: Path,
        geometry_key: str,
        flow_axis: str,
        density: float,
        reference_area: float | None,
    ) -> dict:
        cache_key = f"{geometry_key}:{flow_axis}"
        cache_hit = cache_key in self.cache
        prepare_started = time.perf_counter()
        if cache_hit:
            prepared = self.cache.pop(cache_key)
            self.cache[cache_key] = prepared
        else:
            prepared = self._prepare(stl_path, flow_axis)
            self.cache[cache_key] = prepared
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        preparation_seconds = time.perf_counter() - prepare_started

        result = self.predict_prepared(prepared, density, reference_area)
        return {
            **result,
            "cache_hit": cache_hit,
            "preparation_seconds": preparation_seconds,
        }

    def prepare(self, stl_path: Path, flow_axis: str) -> dict:
        return self._prepare(stl_path, flow_axis)

    def predict_prepared(
        self,
        prepared: dict,
        density: float,
        reference_area: float | None,
    ) -> dict:
        data = prepared["data"]
        area = float(reference_area or prepared["automatic_area"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            prediction = forward_surface(self.model, data)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        integration_started = time.perf_counter()
        fields = unnormalize(
            prediction, self.wrapper._surf_factors[0], self.wrapper._surf_factors[1]
        ) * (2.0 * density)
        centers = data["surface_mesh_centers"][0]
        normals = data["surface_normals"][0]
        surface_areas = data["surface_areas"][0]
        if torch.sum(torch.sum(centers * normals, dim=1) * surface_areas) < 0:
            normals = -normals
        cp, cf = fields[:, 0], fields[:, 1:4]
        force = (
            torch.sum((-cp[:, None] * normals) * surface_areas[:, None], dim=0)
            - torch.sum(cf * surface_areas[:, None], dim=0)
        ) / area
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        integration_seconds = time.perf_counter() - integration_started
        return {
            "cd": float(force[0]),
            "cl": float(force[2]),
            "reference_area_m2": area,
            "reference_area_source": "argument" if reference_area else "G2 bounding-box frontal area",
            "inference_seconds": inference_seconds,
            "coefficient_integration_seconds": integration_seconds,
        }

    def predict_prepared_batch(
        self,
        prepared_items: list[dict],
        density: float,
        reference_areas: list[float | None] | None = None,
    ) -> list[dict]:
        """Run equally sampled prepared geometries in one GPU forward pass."""
        if not prepared_items:
            return []
        reference_areas = reference_areas or [None] * len(prepared_items)
        if len(reference_areas) != len(prepared_items):
            raise ValueError("reference area count does not match prepared inputs")

        data_items = [item["data"] for item in prepared_items]
        keys = data_items[0].keys()
        data = {}
        for key in keys:
            values = [item[key] for item in data_items]
            if not all(torch.is_tensor(value) for value in values):
                raise TypeError(f"cannot batch non-tensor input {key}")
            data[key] = torch.cat(values, dim=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            predictions = forward_surface(self.model, data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        results = []
        for index, (prepared, reference_area) in enumerate(
            zip(prepared_items, reference_areas, strict=True)
        ):
            item_data = prepared["data"]
            area = float(reference_area or prepared["automatic_area"])
            fields = unnormalize(
                predictions[index], self.wrapper._surf_factors[0], self.wrapper._surf_factors[1]
            ) * (2.0 * density)
            centers = item_data["surface_mesh_centers"][0]
            normals = item_data["surface_normals"][0]
            surface_areas = item_data["surface_areas"][0]
            if torch.sum(torch.sum(centers * normals, dim=1) * surface_areas) < 0:
                normals = -normals
            cp, cf = fields[:, 0], fields[:, 1:4]
            force = (
                torch.sum((-cp[:, None] * normals) * surface_areas[:, None], dim=0)
                - torch.sum(cf * surface_areas[:, None], dim=0)
            ) / area
            results.append({
                "cd": float(force[0]),
                "cl": float(force[2]),
                "reference_area_m2": area,
                "reference_area_source": "argument" if reference_area else "G2 bounding-box frontal area",
                "inference_seconds": inference_seconds / len(prepared_items),
                "coefficient_integration_seconds": 0.0,
            })
        return results
