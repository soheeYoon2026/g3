#!/usr/bin/env python3
"""Audit raw source files referenced by a G3 preprocessing manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SOURCE_FIELDS = ("case_dir", "volume", "surface", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ready-list", type=Path)
    parser.add_argument("--ready-manifest", type=Path)
    return parser.parse_args()


def inspect_path(value: object, *, directory: bool = False) -> dict[str, object]:
    raw = str(value or "")
    path = Path(raw) if raw else None
    exists = bool(path and (path.is_dir() if directory else path.is_file()))
    result: dict[str, object] = {"path": raw, "exists": exists}
    if exists and not directory:
        result["size_bytes"] = path.stat().st_size
    return result


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    audited = []
    prefix_counts: Counter[str] = Counter()

    for index, case in enumerate(cases):
        source = case.get("source") or {}
        paths = {
            field: inspect_path(source.get(field), directory=(field == "case_dir"))
            for field in SOURCE_FIELDS
        }
        case_dir = str(source.get("case_dir") or "")
        parts = Path(case_dir).parts
        prefix_counts[str(Path(*parts[:4])) if parts else "<missing>"] += 1
        ready = all(paths[field]["exists"] for field in SOURCE_FIELDS)
        audited.append(
            {
                "index": index,
                "id": case.get("id") or case.get("case_id") or case.get("name"),
                "group": case.get("group"),
                "ready": ready,
                "paths": paths,
            }
        )

    report = {
        "manifest": str(args.manifest.resolve()),
        "total": len(audited),
        "ready": sum(item["ready"] for item in audited),
        "not_ready": sum(not item["ready"] for item in audited),
        "field_exists": {
            field: sum(item["paths"][field]["exists"] for item in audited)
            for field in SOURCE_FIELDS
        },
        "case_dir_prefixes": dict(prefix_counts.most_common()),
        "ready_by_group": dict(
            Counter(str(item["group"] or "<none>") for item in audited if item["ready"]).most_common()
        ),
        "not_ready_by_group": dict(
            Counter(str(item["group"] or "<none>") for item in audited if not item["ready"]).most_common()
        ),
        "cases": audited,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.ready_list:
        args.ready_list.parent.mkdir(parents=True, exist_ok=True)
        ready_ids = [str(item["id"]) for item in audited if item["ready"]]
        args.ready_list.write_text("\n".join(ready_ids) + ("\n" if ready_ids else ""), encoding="utf-8")
    if args.ready_manifest:
        args.ready_manifest.parent.mkdir(parents=True, exist_ok=True)
        ready_manifest = dict(manifest)
        ready_manifest["cases"] = [
            case for case, item in zip(cases, audited) if item["ready"]
        ]
        ready_manifest["source_audit"] = {
            "input_manifest": str(args.manifest),
            "selected_cases": len(ready_manifest["cases"]),
            "excluded_cases": report["not_ready"],
            "selection_rule": "all source paths exist on the audit host",
        }
        args.ready_manifest.write_text(
            json.dumps(ready_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        key: report[key]
        for key in (
            "total",
            "ready",
            "not_ready",
            "field_exists",
            "case_dir_prefixes",
            "ready_by_group",
            "not_ready_by_group",
        )
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report["ready"] == report["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
