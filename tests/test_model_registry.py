from aox_g3.model_registry import ModelRegistry


def test_promote_and_rollback_are_atomic_pointer_changes(tmp_path):
    v6 = tmp_path / "v6.pt"
    v7 = tmp_path / "v7.pt"
    v6.write_bytes(b"v6")
    v7.write_bytes(b"v7")
    registry = ModelRegistry(tmp_path / "registry")

    registry.initialize(v6)
    registry.stage(v7)
    state = registry.promote("offline and shadow gates passed")
    assert state["production"] == str(v7.resolve())
    assert (registry.root / "production.pt").resolve() == v7.resolve()

    state = registry.rollback("runtime regression")
    assert state["production"] == str(v6.resolve())
    assert (registry.root / "production.pt").resolve() == v6.resolve()
