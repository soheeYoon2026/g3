#!/usr/bin/env python3
"""Build the reviewed automotive-only DoMINO family split."""
import argparse, json
from pathlib import Path

FAMILIES={
 "generic_car_shell":[3,9,12,13,14,15,18,19,20,26,28,31,32,33,53,54,63,75,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,122,123,124,125,128,129,130],
 "civic_fl5":[59,64,65,66,68,111,112,113,114],
 "nissan_gtr_family":[60,61,62,67,69,70,126,127],
 "g21":[108,109], "car5_reverse":[104,131], "generic_component_car":[71,74],
 "aero_test_car":[57], "a90":[110], "eos_full":[76], "e423":[80],
}
SPLIT={"train":["generic_car_shell","g21","car5_reverse","generic_component_car","aero_test_car","a90","eos_full","e423"],
       "validation":["civic_fl5"],"test":["nissan_gtr_family"]}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
 m=json.loads(a.manifest.read_text()); by={int(r["run"]):r for r in m["cases"] if r.get("accepted")}
 assigned={run:name for name,runs in FAMILIES.items() for run in runs}
 if len(assigned)!=sum(map(len,FAMILIES.values())): raise SystemExit("run assigned to multiple families")
 missing=sorted(set(assigned)-set(by));
 if missing: raise SystemExit(f"curation references missing runs: {missing}")
 def rows(names): return [dict(by[run],shape_family=name) for name in names for run in FAMILIES[name]]
 payload={"schema_version":1,"source_manifest":str(a.manifest.resolve()),
  "review_method":"original uploaded STL set names + normalized surface-distance audit + representative render review",
  "families":FAMILIES,"train_families":SPLIT["train"],"validation_families":SPLIT["validation"],"test_families":SPLIT["test"],
  "train_cases":rows(SPLIT["train"]),"validation_cases":rows(SPLIT["validation"]),"test_cases":rows(SPLIT["test"]),
  "excluded_cases":[dict(row,exclusion_reason="non-automotive, component-only, incomplete, or unrelated benchmark geometry") for run,row in by.items() if run not in assigned]}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+"\n")
 print(json.dumps({"automotive_cases":len(assigned),"shape_families":len(FAMILIES),"train_cases":len(payload["train_cases"]),
  "train_families":len(SPLIT["train"]),"validation_cases":len(payload["validation_cases"]),"validation_families":1,
  "test_cases":len(payload["test_cases"]),"test_families":1,"excluded_cases":len(payload["excluded_cases"]),"family_overlap":0},indent=2))
if __name__=="__main__": main()
