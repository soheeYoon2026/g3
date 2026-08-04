"""Generate training labels by reusing the existing G1/G2/G4 solvers.

This is the whole point of building G3 *inside* AOX rather than from scratch:
the platform already owns three solvers that turn an STL into (Cd, Cl [, surface
fields]). That is exactly the label a surrogate needs, so the usual ML-CFD data
bottleneck is already paid for.

This module is the adapter seam. The functions below are thin, documented shells
around the real solver CLIs — fill in the argument wiring for your deployment
(container image, resource tier, S3 paths). They are intentionally NOT invoked
by the smoke test; the smoke test runs on synthetic data so it needs no solver.

Solver entry points, for reference (paths relative to SU2_work/):
  * G2  SU2 RANS adjoint : g2-docker-image/su2_optimization_pipeline_v8.py
  * G4  XLB LBM          : g4-docker-image/run_stl.py --mode eval --output run.json
  * G1  OpenFOAM         : (legacy pipeline; wrap similarly)

Recommended label source by role:
  * bulk pretraining volume -> G2 (cheap, adjoint-quality Cd) or public DrivAerML
  * high-fidelity fine-tune -> G1 / G4 on the exact target geometry family
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import TARGETS


@dataclass
class Label:
    stl: str
    cd: float
    cl: float
    engine: str
    surface_field: str | None = None  # path to surface p / tau artifact, if any
    extra: dict | None = None


def label_from_g4(stl_path: str, run_stl: str, python_bin: str,
                  extra_args: list[str] | None = None, timeout: int = 3600) -> Label:
    """Run G4 (XLB LBM) in eval mode on one STL and parse Cd/Cl.

    Mirrors the documented G4 recipe (``run_stl.py --mode eval ... --output``).
    This is a *template*: confirm the flags/JSON keys against your g4 image.
    """
    out = Path(stl_path).with_suffix(".g4.json")
    cmd = [python_bin, run_stl, "--stl", stl_path, "--mode", "eval",
           "--output", str(out), *(extra_args or [])]
    subprocess.run(cmd, check=True, timeout=timeout)
    data = json.loads(out.read_text())
    return Label(stl=stl_path, cd=float(data["cd"]), cl=float(data.get("cl", float("nan"))),
                 engine="g4", extra=data)


def label_from_g2(stl_path: str, pipeline: str, python_bin: str,
                  extra_args: list[str] | None = None, timeout: int = 7200) -> Label:
    """Run G2 (SU2 RANS adjoint) as a single evaluation on one STL.

    Template only. Wire to your su2_optimization_pipeline eval/analysis mode and
    read Cd/Cl from its result JSON.
    """
    out = Path(stl_path).with_suffix(".g2.json")
    cmd = [python_bin, pipeline, "--stl", stl_path, "--mode", "eval",
           "--output", str(out), *(extra_args or [])]
    subprocess.run(cmd, check=True, timeout=timeout)
    data = json.loads(out.read_text())
    return Label(stl=stl_path, cd=float(data["cd"]), cl=float(data.get("cl", float("nan"))),
                 engine="g2", extra=data)


def build_manifest(labels: list[Label], manifest_path: str) -> None:
    """Write labels to the JSON manifest consumed by ``ManifestDataset``."""
    rows = []
    for lab in labels:
        d = asdict(lab)
        rows.append({"stl": d["stl"], **{t: d[t] for t in TARGETS},
                     "engine": d["engine"], "surface_field": d["surface_field"]})
    Path(manifest_path).write_text(json.dumps(rows, indent=2))
