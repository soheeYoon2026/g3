"""Run the read-only measurements on the known cases and say what moved.

    regression_check.py [--cases benchmarks/geometry_cases.json] [--only NAME] [--update]

Without this, "did the change help?" is answered by looking at pictures. Each case names a
reference, a candidate and the numbers that must hold, and the tools that produce them are
the same ones the model may call. --update rewrites the expected values from this run, which
is how a deliberate improvement is recorded.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import geometry_tools

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cases", type=Path, default=HERE.parent / "benchmarks" / "geometry_cases.json")
ap.add_argument("--only")
ap.add_argument("--update", action="store_true", help="store what this run measured as the expectation")
ap.add_argument("--out", type=Path, help="write the measured numbers here as JSON")
args = ap.parse_args()

spec = json.loads(args.cases.read_text())
rows, results = [], {}
t_all = time.time()
for case in spec["cases"]:
    if args.only and args.only != case["name"]:
        continue
    ref = str(Path(case["reference"]).expanduser()) if case.get("reference") else None
    cand = str(Path(case["candidate"]).expanduser())
    missing = [p for p in ([ref] if ref else []) + [cand] if not Path(p).exists()]
    if missing:
        rows.append((case["name"], "파일 없음", "-", "-", "건너뜀"))
        print(f"[{case['name']}] 파일이 없어 건너뜁니다: {missing}")
        continue
    cache = {}
    for chk in case["checks"]:
        tool = chk["tool"]
        call_args = dict(chk.get("args") or {})
        if tool in ("measure_mesh", "list_openings"):
            call_args["path"] = cand if chk.get("on", "candidate") == "candidate" else ref
        else:
            call_args.update({"candidate": cand, "reference": ref})
        key = (tool, json.dumps(call_args, sort_keys=True))
        if key not in cache:
            cache[key] = geometry_tools.call(tool, call_args)
        out = cache[key]
        got = out.get(chk["key"], out.get("error"))
        want = chk.get("expect")
        if isinstance(want, bool) or isinstance(got, bool) or want is None:
            ok = got == want
            delta = ""
        else:
            tol = chk.get("tol", abs(want) * chk.get("tol_frac", 0.0))
            ok = isinstance(got, (int, float)) and abs(got - want) <= tol
            delta = f"{got - want:+.3g}" if isinstance(got, (int, float)) else "-"
        rows.append((case["name"], f"{tool}.{chk['key']}", want, got, ("통과" if ok else "실패") + (f" ({delta})" if delta else "")))
        results.setdefault(case["name"], {})[f"{tool}.{chk['key']}"] = got
        if args.update:
            chk["expect"] = got

width = max(len(r[1]) for r in rows) if rows else 10
print(f"\n{'케이스':16s} {'항목':{width}s} {'기대':>12s} {'측정':>12s}  판정")
for name, item, want, got, verdict in rows:
    w = f"{want:.4g}" if isinstance(want, float) else str(want)
    g = f"{got:.4g}" if isinstance(got, float) else str(got)
    print(f"{name:16s} {item:{width}s} {w:>12s} {g:>12s}  {verdict}")
failed = [r for r in rows if r[4].startswith("실패")]
print(f"\n{len(rows)}개 중 실패 {len(failed)}개 · {time.time()-t_all:.0f} s")
if args.update:
    args.cases.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print(f"기대값을 이번 측정으로 갱신했습니다: {args.cases}")
if args.out:
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
sys.exit(1 if failed else 0)
