#!/usr/bin/env python3
"""Cluster DoMINO cases by normalized STL surface geometry, not names or hashes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def points_for(root, run, count, seed):
    import pyvista as pv
    mesh=pv.read(root/f"run_{run}"/f"drivaer_{run}.stl").triangulate()
    tri=np.asarray(mesh.faces).reshape(-1,4)[:,1:]
    p=np.asarray(mesh.points,dtype=np.float64); a,b,c=p[tri[:,0]],p[tri[:,1]],p[tri[:,2]]
    area=np.linalg.norm(np.cross(b-a,c-a),axis=1)*.5
    good=np.isfinite(area)&(area>0); a,b,c,area=a[good],b[good],c[good],area[good]
    rng=np.random.default_rng(seed); pick=rng.choice(len(area),size=count,replace=True,p=area/area.sum())
    u=rng.random(count); v=rng.random(count); swap=u+v>1; u[swap]=1-u[swap]; v[swap]=1-v[swap]
    q=a[pick]+u[:,None]*(b[pick]-a[pick])+v[:,None]*(c[pick]-a[pick])
    lo,hi=np.quantile(q,[.002,.998],axis=0); center=(lo+hi)/2; scale=max(float(hi[0]-lo[0]),1e-9)
    return ((q-center)/scale).astype(np.float32), ((hi-lo)/scale).tolist()

def distance(a,b):
    da=cKDTree(a).query(b,k=1,workers=-1)[0]; db=cKDTree(b).query(a,k=1,workers=-1)[0]
    # Robust to small wings/lips, but still sensitive to a different base body.
    return float(max(np.quantile(da,.90),np.quantile(db,.90)))

def components(matrix, threshold):
    n=len(matrix); seen=set(); result=[]
    for start in range(n):
        if start in seen: continue
        stack=[start]; seen.add(start); group=[]
        while stack:
            i=stack.pop(); group.append(i)
            for j in range(n):
                if j not in seen and matrix[i,j]<=threshold:
                    seen.add(j); stack.append(j)
        result.append(sorted(group))
    return sorted(result,key=lambda x:(-len(x),x[0]))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--points",type=int,default=3000); p.add_argument("--seed",type=int,default=20260811)
    a=p.parse_args(); manifest=json.loads(a.manifest.read_text()); rows=[r for r in manifest["cases"] if r.get("accepted")]
    clouds=[]; extents=[]
    for i,row in enumerate(rows):
        cloud,extent=points_for(a.root,str(row["run"]),a.points,a.seed+int(row["run"]))
        clouds.append(cloud); extents.append(extent); print(f"sample {i+1}/{len(rows)} run_{row['run']}",flush=True)
    n=len(rows); matrix=np.zeros((n,n),np.float32)
    for i in range(n):
        for j in range(i): matrix[i,j]=matrix[j,i]=distance(clouds[i],clouds[j])
        print(f"distance {i+1}/{n}",flush=True)
    nearest=np.partition(matrix+np.eye(n,dtype=np.float32)*1e9,0,axis=1)[:,0]
    thresholds=[.005,.01,.015,.02,.03,.04,.05,.075,.1]
    candidates={str(t):[[int(rows[i]["run"]) for i in g] for g in components(matrix,t)] for t in thresholds}
    payload={"schema_version":1,"normalization":"bbox center and X-length; area-weighted surface points",
             "distance":"symmetric 90th-percentile nearest-surface distance / X-length",
             "runs":[int(r["run"]) for r in rows],"normalized_extents":extents,
             "nearest_distance_quantiles":{str(q):float(np.quantile(nearest,q)) for q in [0,.1,.25,.5,.75,.9,.95,1]},
             "threshold_cluster_counts":{k:len(v) for k,v in candidates.items()},"clusters":candidates}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+"\n")
    np.save(a.out.with_suffix(".distance.npy"),matrix); print(json.dumps({"nearest":payload["nearest_distance_quantiles"],"counts":payload["threshold_cluster_counts"]},indent=2))
if __name__=="__main__": main()
