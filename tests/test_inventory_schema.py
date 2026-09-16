import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


import pytest

# scripts/prepare_g2_fields.py 가 import 시점에 VTK 를 요구한다. 이 환경에는 없다.
# 조용히 건너뛰지 않고 이유를 찍어 둔다 — 안 도는 시험은 통과가 아니다.
try:
    import vtk  # noqa: F401
    HAVE_VTK = True
except Exception:
    HAVE_VTK = False

needs_vtk = pytest.mark.skipif(not HAVE_VTK, reason="VTK 미설치 (pip install vtk) — 이 시험은 안 돈 것이다")


def load_script(name):
    path = ROOT / "scripts" / name
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@needs_vtk
def test_g2_all_buckets_inventory_schema():
    module = load_script("prepare_smoke_g2_s3.py")
    row = module.normalize_inventory_row({
        "status": "succeeded",
        "output_s3_key": "tenant/t/team/a/project/p/job/j/output/",
        "project_uid": "p",
        "job_uid": "j",
    })
    assert row["job_status"] == "succeeded"
    assert row["s3_output_prefix"].endswith("/output/")
    assert row["group_key"] == "p"
    assert row["source_kind"] == "inventory"


def test_g4_smoke_schema_is_preserved():
    module = load_script("prepare_smoke_g4_geometry.py")
    row = module.normalize_inventory_row({
        "job_status": "succeeded",
        "s3_output_prefix": "smoke/test/job/output/",
        "test_case": "test002",
        "job_uid": "j",
    })
    assert row["job_status"] == "succeeded"
    assert row["s3_output_prefix"] == "smoke/test/job/output/"
    assert row["group_key"] == "test002"
    assert row["source_kind"] == "smoke"
