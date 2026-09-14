"""A ledger of completed steps, so a run can continue instead of starting over.

The controller used to rebuild plan.json from scratch on every start. On 2026-09-14 a page
restarted a finished GT-R run with an empty form; the run re-read its own converted file,
applied the units twice and turned a 4.6 m car into a 117 m one. Nothing recorded which
steps were already done, or on which bytes.

A step is identified by the tool, its arguments and the content hash of every input file.
If that identity is already in steps.json and the outputs it names still exist, the step is
done: the caller can skip it and reuse the outputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LEDGER_NAME = "steps.json"


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """sha1 of the file's bytes, short form. Reading 60 MB costs about 0.2 s."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def _norm(value, root: Path):
    """Arguments become comparable: paths relative to the run, numbers as numbers."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        p = Path(value)
        if p.exists() and p.is_file():
            try:
                return f"file:{p.resolve().relative_to(root.resolve())}" if root in p.resolve().parents else f"file:{p.name}"
            except ValueError:
                return f"file:{p.name}"
    return value


def identity(tool: str, args, inputs, root: Path) -> str:
    """One string that says: this tool, these arguments, these exact input bytes."""
    parts = [tool, json.dumps([_norm(a, root) for a in args], ensure_ascii=False, sort_keys=True)]
    for path in sorted(Path(p) for p in inputs):
        parts.append(f"{path.name}:{file_hash(path) if path.exists() else 'missing'}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load(run_dir: Path) -> list:
    path = Path(run_dir) / LEDGER_NAME
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def find(run_dir: Path, ident: str):
    """The completed step with this identity, if every output it named is still there."""
    for step in load(run_dir):
        if step.get("identity") != ident or step.get("status") != "done":
            continue
        if all((Path(run_dir) / o["path"]).exists() or Path(o["path"]).exists() for o in step.get("outputs", [])):
            return step
    return None


def record(run_dir: Path, tool: str, args, inputs, outputs, ident: str,
           seconds: float, status: str = "done", metrics=None, note: str = "") -> dict:
    run_dir = Path(run_dir)
    entries = [s for s in load(run_dir) if s.get("identity") != ident]
    out = []
    for path in outputs:
        p = Path(path)
        if p.exists():
            out.append({"path": str(p if not p.is_absolute() else p.relative_to(run_dir) if run_dir.resolve() in p.resolve().parents else p),
                        "hash": file_hash(p), "bytes": p.stat().st_size})
    step = {"identity": ident, "tool": tool,
            "args": [str(a) for a in args],
            "inputs": [{"path": str(p), "hash": file_hash(Path(p))} for p in inputs if Path(p).exists()],
            "outputs": out, "metrics": metrics or {}, "seconds": round(seconds, 1),
            "status": status, "note": note, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    entries.append(step)
    (run_dir / LEDGER_NAME).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    return step
