"""Portable mesh checkpoints and explicit controller outcomes."""
import os
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from . import ledger
from .run_state import write_json


def mesh_checkpoint(directory, name, identity, build, force=False, stl=False):
    directory = Path(directory)
    data = directory / f"{name}.npz"
    manifest = directory / f"{name}.json"
    output = directory / "input_mm.stl" if stl else None
    try:
        import json
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        valid = (not force and saved["identity"] == identity
                 and ledger.file_hash(data) == saved["hash"]
                 and (output is None or ledger.file_hash(output) == saved["stl_hash"]))
        if valid:
            with np.load(data, allow_pickle=False) as arrays:
                mesh = trimesh.Trimesh(arrays["vertices"], arrays["faces"], process=False)
            return mesh, True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    mesh = build()
    temporary = data.with_suffix(".tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, vertices=mesh.vertices, faces=mesh.faces)
    os.replace(temporary, data)
    saved = {"identity": identity, "hash": ledger.file_hash(data)}
    if output is not None:
        mesh.export(output)
        saved["stl_hash"] = ledger.file_hash(output)
    write_json(manifest, saved)
    return mesh, False


def outcome(plan, requested="done"):
    if requested == "failed":
        return "failed", 2
    if any("answer" not in q and not q.get("assumed") for q in plan["questions"]):
        return "waiting_for_answers", 3
    if plan.get("status") == "needs_customer" or any(
            not c["ok"] and not c.get("resolved") for c in plan["checks"]):
        return "needs_review", 4
    if not plan["deliverables"] or any(
            not Path(p).is_file() or Path(p).stat().st_size == 0 for p in plan["deliverables"]):
        return "failed", 2
    return "done", 0


def preparation_outcome(summary, required):
    failed = [name for name in required
              if summary["stages"].get(name, {}).get("status") not in ("ok", "unchanged", "hollow")]
    summary["required_stages"] = required
    summary["failed_required_stages"] = failed
    review = any(summary["stages"].get(name, {}).get("status") == "hollow" for name in required)
    summary["status"] = "failed" if failed else "needs_review" if review else "done"
    return 2 if failed else 4 if review else 0
