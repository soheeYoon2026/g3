"""Display-only STEP preview; never heals or modifies CAD geometry."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from aox_g3 import cad, ledger
ap=argparse.ArgumentParser();ap.add_argument('--in',dest='src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
manifest=a.out/'step_preview.json';target=a.out/'viewer.stl'
identity={'source':ledger.file_hash(a.src),'cad':ledger.file_hash(Path(cad.__file__)),'version':1,'deflection_frac':0.003,'target_faces':150000}
try: previous=json.loads(manifest.read_text(encoding='utf-8'))
except (OSError,ValueError): previous={}
if previous.get('identity')==identity and previous.get('status')=='done' and target.exists() and previous.get('output_hash')==ledger.file_hash(target):sys.exit(0)
start=time.time()
def report(status,**extra):
 data={'identity':identity,'status':status,'elapsed_seconds':time.time()-start,**extra};tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');os.replace(tmp,manifest)
try:
 report('running');shape,r=cad.read_step(a.src)
 if shape is None:raise RuntimeError('STEP preview read failed')
 mesh,r=cad.tessellate(shape,r,deflection_frac=0.003)
 if mesh is None:raise RuntimeError('STEP preview contains no triangles')
 if len(mesh.faces)>150000:
  try:mesh=mesh.simplify_quadric_decimation(face_count=150000)
  except ImportError:r.warnings.append('Decimator unavailable; displaying coarse tessellation without further simplification')
 tmp=target.with_name('viewer.tmp.stl');mesh.export(tmp);os.replace(tmp,target)
 report('done',triangles=len(mesh.faces),output_hash=ledger.file_hash(target))
except Exception as exc:
 report('failed',error=str(exc));raise
