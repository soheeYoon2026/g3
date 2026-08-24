#!/usr/bin/env python3
"""Attach original uploaded STL names to accepted S3 DoMINO cases."""
import argparse, json, re
from pathlib import Path
import boto3
p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
m=json.loads(a.manifest.read_text()); clients={}; cache={}; rows=[]
for case in m["cases"]:
    source=case.get("source") or {}; bucket=source.get("bucket"); output=source.get("output_prefix")
    if not bucket or not output: continue
    project=output.split("/job/",1)[0]; key=(bucket,project)
    if key not in cache:
        client=clients.setdefault(bucket,boto3.client("s3")); uploaded=project.rstrip("/")+"/uploaded/"; names=[]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket,Prefix=uploaded):
            names += [item["Key"].rsplit("/",1)[-1] for item in page.get("Contents",[]) if item["Key"].lower().endswith(".stl")]
        cache[key]=sorted(names)
    rows.append({"run":case["run"],"bucket":bucket,"project_prefix":project,"uploaded_stls":cache[key]})
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps({"accepted_s3_cases":len(rows),"projects":len(cache),"projects_without_uploaded_stl":sum(not v for v in cache.values())},indent=2))
