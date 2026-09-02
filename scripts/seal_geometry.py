#!/usr/bin/env python
"""Seal dirty STLs into watertight ones, and say exactly what happened.

    seal_geometry.py --in dirty.stl --out clean.stl
    seal_geometry.py --in shapes/ --out sealed/ --report sealed/report.json

Exits non-zero when a mesh cannot be sealed, so a pipeline stops rather than
handing a solver an open shell. The v8 LES campaign burned seven 4-hour runs on
exactly that failure, and the results looked plausible enough to survive review.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.seal import available_tools, seal_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="source", type=Path,
                        help="STL file, or a directory of them")
    parser.add_argument("--out", type=Path, help="output file or directory")
    parser.add_argument("--report", type=Path, help="write the per-file reports as json")
    parser.add_argument("--force", action="store_true",
                        help="seal even when a winding number would do")
    parser.add_argument("--require-watertight", action="store_true",
                        help="treat a below-threshold pass-through as a failure; use "
                             "this when the consumer genuinely needs a closed surface")
    parser.add_argument("--tools", action="store_true", help="report tool availability and exit")
    args = parser.parse_args()

    tools = available_tools()
    if args.tools:
        for name, ok in tools.items():
            print(f"  {name}: {'있음' if ok else '없음'}")
        return 0
    if args.source is None:
        parser.error("--in 이 필요합니다 (--tools 로 도구 확인만 할 수도 있습니다)")
    missing = [n for n, ok in tools.items() if not ok]
    if missing:
        # Not fatal - a lower tier may still work - but it must be visible, since
        # a silently skipped tier is what made the v8 results wrong.
        print(f"경고: 사용할 수 없는 단계 {missing} — 낮은 단계로 내려갑니다", file=sys.stderr)

    if args.source.is_dir():
        sources = sorted(args.source.glob("*.stl"))
        if not sources:
            print(f"{args.source} 에 STL이 없습니다", file=sys.stderr)
            return 2
        out_dir = args.out or args.source / "sealed"
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [out_dir / s.name for s in sources]
    else:
        sources = [args.source]
        targets = [args.out] if args.out else [None]

    print(f"{'파일':>28s} {'열림지표':>9s} {'방법':>16s} {'면 수':>18s} {'체적비':>7s} {'결과':>12s}")
    print("-" * 98)
    reports, failed = [], 0
    for source, target in zip(sources, targets):
        report, written = seal_file(source, target, force=args.force)
        reports.append({"file": str(source), "out": str(written) if written else None,
                        **report.as_dict()})
        # A pass-through is not a failure, but it is not a watertight file either.
        # Calling both "OK" would hand a caller an open shell under a green light -
        # the same class of mistake that cost the v8 campaign seven runs.
        if report.watertight_out or report.method == "already_watertight":
            status, ok = "수밀", True
        elif report.method == "below_threshold":
            status, ok = "통과(비수밀)", not args.require_watertight
        else:
            status, ok = "실패", False
        failed += not ok
        print(f"{source.name[:28]:>28s} {report.openness_in:9.2f} {report.method:>16s} "
              f"{f'{report.faces_in:,}->{report.faces_out:,}':>18s} "
              f"{report.volume_ratio:7.2f} {status:>12s}")
        for warning in report.warnings:
            print(f"{'':>28s}   · {warning}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"tools": tools, "files": reports}, indent=1) + "\n")
        print(f"\n리포트 -> {args.report}")

    print(f"\n{len(sources) - failed}/{len(sources)} 성공")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
