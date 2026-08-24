#!/usr/bin/env python3
"""Scan complete static buckets for G2 output prefixes, including event orphans."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import boto3

def main():
    p=argparse.ArgumentParser(); p.add_argument("bucket",nargs="+"); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); rows=[]; summary={}
    for bucket in a.bucket:
        client=boto3.client("s3"); prefixes=set(); objects=0
        paginator=client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for item in page.get("Contents",[]):
                objects+=1; key=item["Key"]
                if not key.lower().endswith("/surface_flow.vtu") or "/output/" not in key:
                    continue
                prefixes.add(key.split("/output/",1)[0]+"/output")
        for prefix in sorted(prefixes):
            match=re.search(r"/job/([^/]+)", "/"+prefix.lstrip("/"))
            uid=match.group(1) if match else "orphan_"+str(abs(hash((bucket,prefix))))
            rows.append({"solver":"G2","job_status":"succeeded","job_uid":uid,"project_uid":uid,
                         "s3_bucket":bucket,"s3_output_prefix":prefix,"output_s3_key":prefix+"/"})
        summary[bucket]={"objects_scanned":objects,"g2_like_output_prefixes":len(prefixes)}
        print(json.dumps({bucket:summary[bucket]}),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fields=["solver","job_status","job_uid","project_uid","s3_bucket","s3_output_prefix","output_s3_key"]
    with a.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (a.out.with_suffix(".summary.json")).write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
