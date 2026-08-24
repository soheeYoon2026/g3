import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_case_identity.py"
SPEC = importlib.util.spec_from_file_location("verify_case_identity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_history(path):
    path.write_text(
        '"Inner_Iter",     "CD"     ,     "CL"\n'
        "0, 0.5000000, 0.10\n"
        "1, 0.3300000, 0.11\n"
        "2, 0.3210000, 0.12\n"
        "3, 0.3207000, 0.13\n"
    )
    return path


def test_summarize_history_reads_steps_and_locates_cd(tmp_path):
    summary = MODULE.summarize_history(
        write_history(tmp_path / "history.csv"),
        step=2, find_cd=0.3207, find_tolerance=5e-5, tail=2,
    )
    assert summary["steps"] == 4
    assert summary["iteration_column"] == "Inner_Iter"
    assert summary["last_step"] == 3
    assert summary["cd_last"] == 0.3207
    assert summary["cl_last"] == 0.13
    assert summary["cd_at_step"] == 0.321
    assert summary["cd_tail_steps"] == 2
    assert summary["find_cd"]["matches"] == 1
    assert summary["find_cd"]["hits"] == [{"step": 3, "cd": 0.3207}]


def test_summarize_history_without_cd_column(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text('"Inner_Iter","rms[P]"\n0, -3.0\n')
    summary = MODULE.summarize_history(path)
    assert summary["note"] == "no CD column or no data rows"


def test_relative_check_statuses():
    assert MODULE.relative_check("x", 1.0, 1.0)["status"] == "ok"
    assert MODULE.relative_check("x", 1.0, 1.1)["status"] == "mismatch"
    assert MODULE.relative_check("x", None, 1.0)["status"] == "skipped"


def test_build_checks_locates_conditions_cd():
    conditions = {"su2_cd": 0.3207}
    histories = [{
        "path": "/case/history.csv",
        "find_cd": {"target": 0.3207, "tolerance": 5e-5, "matches": 1,
                    "hits": [{"step": 3, "cd": 0.3207}]},
    }]
    checks = {c["name"]: c for c in MODULE.build_checks(None, None, None, conditions, histories, None)}
    assert checks["conditions_cd_in_history"]["status"] == "ok"
    assert "history.csv@3" in checks["conditions_cd_in_history"]["detail"]

    histories[0]["find_cd"]["hits"] = []
    checks = {c["name"]: c for c in MODULE.build_checks(None, None, None, conditions, histories, None)}
    assert checks["conditions_cd_in_history"]["status"] == "mismatch"
