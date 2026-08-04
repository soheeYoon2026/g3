import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
