#!/usr/bin/env python3
"""Run the fine-tuned DoMINO surface model from one STL and emit Cd/Cl."""
from __future__ import annotations
import argparse, json, shutil, tempfile, time
from pathlib import Path
import numpy as np, torch
try:
    from scripts.evaluate_domino_v3 import forward_surface
except ModuleNotFoundError:
    from evaluate_domino_v3 import forward_surface

def orient(points, axis):
    # Convert requested flow axis to the model's +X flow frame.
    maps={"+x":((0,1,2),(1,1,1)),"-x":((0,1,2),(-1,1,1)),
          "+y":((1,0,2),(1,-1,1)),"-y":((1,0,2),(-1,1,1)),
          "+z":((2,1,0),(1,1,-1)),"-z":((2,1,0),(-1,1,1))}
    order,sign=maps[axis]; return points[:,order]*np.asarray(sign)

def g2_reference_area(mesh):
    """Match G2: Y extent multiplied by Z extent for its +X flow frame."""
    points=np.asarray(mesh.points,dtype=np.float64)
    span=np.ptp(points,axis=0)
    return float(span[1]*span[2])

def main():
    total_started=time.perf_counter(); timings={}
    import pyvista as pv
    from huggingface_hub import snapshot_download
    from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
    from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper
    from physicsnemo.models.domino.utils import unnormalize
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--stl",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--speed",type=float,default=30.0); p.add_argument("--density",type=float,default=1.225)
    p.add_argument("--ref-area",type=float,default=None)
    p.add_argument("--flow-axis",choices=("+x","-x","+y","-y","+z","-z"),default="+x")
    p.add_argument("--device",default="cuda:0"); a=p.parse_args()
    step_started=time.perf_counter()
    mesh=pv.read(a.stl).extract_surface().triangulate().clean()
    if mesh.n_cells<100 or not np.isfinite(mesh.points).all(): raise ValueError("invalid or too-small STL surface")
    mesh.points=orient(np.asarray(mesh.points,dtype=np.float64),a.flow_axis)
    ref=float(a.ref_area) if a.ref_area is not None else g2_reference_area(mesh)
    if not np.isfinite(ref) or ref<=0: raise ValueError("could not determine positive reference area")
    timings["mesh_read_and_orient_seconds"]=time.perf_counter()-step_started
    step_started=time.perf_counter()
    snap=Path(snapshot_download("nvidia/domino_drivaerml")); surf=snap/"domino_drivaerml_surface_checkpoint"
    wrapper=DominoWrapper().load(checkpoint_path=str(surf/"DoMINO.0.501.mdlus"),stats_path=str(surf/"scaling_factors.pkl"),
                                 device=a.device,domino_config=str(surf/"config.yaml"))
    wrapper._model.load_state_dict(torch.load(a.checkpoint,map_location=a.device)); model=wrapper._model; model.eval()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    timings["model_load_seconds"]=time.perf_counter()-step_started
    with tempfile.TemporaryDirectory(prefix="domino-stl-") as td:
        step_started=time.perf_counter()
        root=Path(td); run=root/"run_1"; run.mkdir()
        mesh.cell_data["pMeanTrim"]=np.zeros(mesh.n_cells,dtype=np.float32)
        mesh.cell_data["wallShearStressMeanTrim"]=np.zeros((mesh.n_cells,3),dtype=np.float32)
        mesh.save(run/"boundary_1.vtp"); mesh.save(run/"drivaer_1.stl")
        adapter=DrivAerMLAdapter(root=str(root)); data=wrapper.prepare_inputs(adapter.load_case("run_1"))["data_dict"]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timings["domino_prepare_seconds"]=time.perf_counter()-step_started
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.no_grad(): pred=forward_surface(model,data)[0]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        timings["gpu_inference_seconds"]=elapsed
        step_started=time.perf_counter()
        fields=unnormalize(pred,wrapper._surf_factors[0],wrapper._surf_factors[1])*(2.0*a.density)
        centers=data["surface_mesh_centers"][0]; normals=data["surface_normals"][0]; areas=data["surface_areas"][0]
        if torch.sum(torch.sum(centers*normals,dim=1)*areas)<0: normals=-normals
        cp,cf=fields[:,0],fields[:,1:4]; force=(torch.sum((-cp[:,None]*normals)*areas[:,None],dim=0)-torch.sum(cf*areas[:,None],dim=0))/ref
        cd,cl=float(force[0]),float(force[2])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timings["coefficient_integration_seconds"]=time.perf_counter()-step_started
        predicted=mesh.copy()
        if predicted.n_cells==len(cp):
            predicted.cell_data["predicted_Cp"]=cp.detach().cpu().numpy().astype(np.float32)
            predicted.cell_data["predicted_Cf"]=cf.detach().cpu().numpy().astype(np.float32)
    step_started=time.perf_counter()
    a.out.parent.mkdir(parents=True,exist_ok=True); field_path=a.out.with_suffix(".vtp"); predicted.save(field_path)
    timings["output_write_seconds"]=time.perf_counter()-step_started
    timings["worker_total_seconds"]=time.perf_counter()-total_started
    result={"stl":str(a.stl.resolve()),"checkpoint":str(a.checkpoint.resolve()),"cd":cd,"cl":cl,"speed_mps":a.speed,
            "density_kg_m3":a.density,"reference_area_m2":ref,"reference_area_source":"argument" if a.ref_area is not None else "G2 bounding-box frontal area",
            "flow_axis":a.flow_axis,"inference_seconds":elapsed,"predicted_surface":str(field_path.resolve()),
            "timings":{key:round(value,6) for key,value in timings.items()},
            "limitations":["checkpoint is trained near its dataset conditions; speed is recorded but is not a broadly validated conditioning variable",
                           "STL units must be metres and the selected flow axis must be correct",
                           "automatic reference area matches G2: bounding-box width times height perpendicular to the flow axis"]}
    a.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
