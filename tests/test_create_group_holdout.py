import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_group_holdout.py"
SPEC = importlib.util.spec_from_file_location("create_group_holdout", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stable_group_order_is_deterministic():
    groups = {"a": [], "b": [], "c": []}
    assert MODULE.stable_group_order(groups, 7) == MODULE.stable_group_order(groups, 7)
    assert sorted(MODULE.stable_group_order(groups, 7)) == ["a", "b", "c"]


def test_group_name_requires_group():
    assert MODULE.group_name({"case_id": "x", "group_id": "family"}) == "family"
    try:
        MODULE.group_name({"case_id": "x"})
    except ValueError as exc:
        assert "no group_id" in str(exc)
    else:
        raise AssertionError("missing group must fail")
