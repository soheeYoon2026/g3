#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
updated = 0
for row in manifest.get("cases", []):
    run = str(row["run"])
    labels = row.get("integrated") or {}
    cd, cl = float(labels.get("su2_cd", math.nan)), float(labels.get("su2_cl", math.nan))
    if not math.isfinite(cd) or not math.isfinite(cl):
        raise ValueError(f"run_{run} has non-finite manifest labels")
    path = root / f"run_{run}" / f"conditions_{run}.json"
    data = json.loads(path.read_text())
    data.update(su2_cd=cd, su2_cl=cl)
    temp = path.with_name(f".{path.name}.{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)
    updated += 1
print(json.dumps({"updated": updated, "root": str(root)}))
