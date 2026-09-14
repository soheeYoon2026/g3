"""Local web page for the controller: questions with proposals and evidence, an
answer form that resumes plan_geometry.py, and a chat with the AI controller.

    plan_ui.py --in X.stp --dir var/runs/plan-x [--port 8765] [--model openai/gpt-5.6-terra]

The chat model sees plan.json (measurements, decisions, checks, open questions)
and helps the customer decide; it cannot run anything. When it proposes answers
it emits a JSON block {"answers": {...}} that the page loads into the form. The
rules in plan_geometry.py stay in charge of the tools. Model calls go through
the Prime inference API (~/.prime/config.json), stdlib only.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--dir", type=Path, required=True, help="the controller's --out directory")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--model", default=os.environ.get("PRIME_MODEL", "openai/gpt-5.6-terra"),
                help="Prime Inference model id; PRIME_MODEL overrides. Measured 2026-09-14: terra 2.5 s, gpt-oss-120b 5.2 s on the same question, both returned a valid answer block")
ap.add_argument("--no-chat", action="store_true", help="page without the model (no API calls)")
ap.add_argument("--knowledge", action="append", metavar="PATH|docs",
                help="reference document put into every system prompt verbatim; repeatable, "
                     "'docs' loads the tech doc and the pipeline doc. OFF by default because it is not free: "
                     "the two documents are 35.6 k input tokens, which is $0.09 a turn on terra and $0.013 on "
                     "gpt-oss-120b, re-sent every turn (Prime publishes no cache price). The text also leaves "
                     "this machine with every call")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DIR = args.dir.resolve()
DIR.mkdir(parents=True, exist_ok=True)
STATE = {"proc": None}
CFG = None
if not args.no_chat:
    try:
        CFG = json.load(open(os.path.expanduser("~/.prime/config.json")))
    except Exception:
        CFG = None

SYSTEM = """당신은 자동차 형상 정리 파이프라인의 AI 통제기입니다. 규칙 통제기(plan_geometry.py)가 측정하고 실행한 내용이
아래 plan.json 에 있습니다. 당신의 역할은 (1) 고객이 답해야 할 질문을 측정값을 근거로 쉬운 말로 설명하고,
(2) 고객의 선택을 도와 답을 정리하는 것입니다. 도구를 실행하거나 실행했다고 말하지 마세요. plan.json 에 없는
수치를 지어내지 마세요. 모르면 모른다고 하세요. 고객이 답을 정하면 마지막에 반드시 다음 형식의 JSON 블록을
붙이세요: ```json {"answers": {"질문id": 값}} ```. 값은 질문의 type 에 맞게(number → 숫자, bool → true/false,
choice → 선택지 문자열). 한국어로, 짧게, 결론부터."""

DEFAULT_KNOWLEDGE = [Path("var/docs/geometry_cleaning/GEOMETRY_CLEANING.md"), Path("docs/GEOMETRY_PIPELINE.md")]


def load_knowledge():
    """The reference documents, verbatim. No retrieval: the whole text goes in every call."""
    if not args.knowledge:
        return ""
    root = HERE.parent
    paths = []
    for item in args.knowledge:
        paths += [root / p for p in DEFAULT_KNOWLEDGE] if str(item) == "docs" else [Path(item)]
    parts = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"  참고 문서 없음: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        parts.append(f"----- 문서: {path.name} -----\n{text}")
        print(f"  참고 문서 {path.name}: {len(text):,}자")
    if not parts:
        return ""
    body = "\n\n".join(parts)
    # measured 2026-09-14: the two documents are 35.6 k input tokens on terra, about $0.09 a turn,
    # and Prime publishes no cache price, so every turn pays for them again
    per_turn = {"openai/gpt-5.6-terra": 2.5, "openai/gpt-5.6-terra-pro": 2.5,
                "openai/gpt-oss-120b": 0.35, "openai/gpt-oss-20b": 0.07}.get(args.model)
    # measured 2026-09-14: 68,050 characters of Korean documents came to 35,646 input tokens
    cost = f", {args.model} 기준 한 번에 약 ${len(body)/1.9*1e-6*per_turn:.3f}" if per_turn else ""
    print(f"  참고 문서 합계 {len(body):,}자 — 대화 한 번마다 다시 보냅니다{cost}")
    return ("\n\n아래는 이 파이프라인의 정본 문서입니다. 수치·규칙·이전 실측은 여기서 인용하고, "
            "여기에 없으면 모른다고 하세요. 문서와 plan.json 이 어긋나면 plan.json(이번 실행)이 우선입니다.\n" + body)


KNOWLEDGE = load_knowledge()


def digest():
    plan = json.loads((DIR / "plan.json").read_text()) if (DIR / "plan.json").exists() else {}
    keep = {k: plan.get(k) for k in ("status", "route", "diagnosis", "decisions", "checks", "questions", "assumptions", "deliverables", "warnings")}
    # terra takes a 1.05 M context, so the whole plan fits; the cap is only a guard
    return json.dumps(keep, ensure_ascii=False)[:120000]


def chat(history):
    if CFG is None:
        return "모델 연결이 없습니다(--no-chat 이거나 ~/.prime/config.json 없음).", None
    msgs = [{"role": "system", "content": SYSTEM + KNOWLEDGE + "\n\nplan.json:\n" + digest()}] + history[-12:]
    body = json.dumps({"model": args.model, "messages": msgs, "max_tokens": 1200, "temperature": 0.2}).encode()
    req = urllib.request.Request(CFG["inference_url"].rstrip("/") + "/chat/completions", data=body, headers={
        "Authorization": "Bearer " + CFG["api_key"], "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": "prime-cli/0.6.33"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    usage = d.get("usage") or {}
    print(f"  [대화] {args.model}  입력 {usage.get('prompt_tokens','?')} 출력 {usage.get('completion_tokens','?')} 토큰")
    text = d["choices"][0]["message"]["content"]
    suggested = None
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            suggested = json.loads(m.group(1)).get("answers")
        except Exception:
            suggested = None
    return text, suggested


def start_controller(answers=None, assume=False):
    if STATE["proc"] is not None and STATE["proc"].poll() is None:
        return False
    cmd = [sys.executable, str(HERE / "plan_geometry.py"), "--in", str(args.src), "--out", str(DIR), "--no-render"]
    if answers is not None:
        (DIR / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1))
        cmd += ["--answers", str(DIR / "answers.json")]
    if assume:
        cmd.append("--assume-defaults")
    STATE["proc"] = subprocess.Popen(cmd, stdout=open(DIR / "ui_run.txt", "a"), stderr=subprocess.STDOUT)
    return True


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>형상 정리 통제기</title>
<style>body{font-family:"Noto Sans CJK KR",sans-serif;margin:0;display:grid;grid-template-columns:1fr 420px;height:100vh}
main{padding:18px 24px;overflow:auto}aside{border-left:1px solid #ddd;display:flex;flex-direction:column;height:100vh}
h1{font-size:20px;margin:0 0 6px}h2{font-size:15px;margin:18px 0 6px;color:#333}.q{border:1px solid #ccd;border-radius:6px;padding:10px 12px;margin:8px 0;background:#f8f9fb}
.q b{display:block;margin-bottom:4px}.q small{color:#666}.q input,.q select{margin-top:6px;padding:4px 6px;font-size:14px}
.ev img{max-width:100%;border:1px solid #ddd;margin-top:6px}.ok{color:#2a7}.bad{color:#c33}#log{font-family:monospace;font-size:12px;white-space:pre-wrap;background:#111;color:#ddd;padding:8px;max-height:220px;overflow:auto}
#chat{flex:1;overflow:auto;padding:12px}.m{margin:6px 0;padding:8px 10px;border-radius:6px;white-space:pre-wrap;font-size:14px}.u{background:#e8f0fe}.a{background:#f1f3f4}
#in{display:flex;border-top:1px solid #ddd}#in textarea{flex:1;border:0;padding:10px;font-size:14px;resize:none;height:70px}#in button{width:70px;border:0;background:#2b5;color:#fff}
button.run{padding:8px 14px;font-size:14px;margin-right:8px}.st{display:inline-block;padding:2px 8px;border-radius:10px;background:#eee;font-size:12px}</style></head>
<body><main><h1>형상 정리 통제기 <span class="st" id="status"></span></h1><div id="input"></div>
<div id="viewer" style="width:100%;height:380px;border:1px solid #ccd;border-radius:6px;background:#f4f5f7;position:relative;margin:8px 0">
<div id="vhint" style="position:absolute;left:8px;top:6px;font-size:12px;color:#555;pointer-events:none">모델을 돌리고(드래그) 확대(휠)하세요. 표식을 누르면 해당 질문으로, 질문의 "위치 보기"를 누르면 그 자리로 갑니다.</div></div>
<h2>고객이 정할 것</h2><div id="questions"></div>
<div><button class="run" onclick="submitAnswers()">답 저장 후 이어서 실행</button><button class="run" onclick="runDefaults()">제안값으로 실행</button></div>
<h2>진단·결정·검증</h2><div id="plan"></div><h2>실행 로그</h2><div id="log"></div></main>
<aside><div id="chat"></div><div id="in"><textarea id="msg" placeholder="AI 통제기에게 묻기: 왜 이 높이인지, 무엇을 고르는 게 나은지…"></textarea><button onclick="send()">보내기</button></div></aside>
<script>
let hist=[];let Q=[];
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function refresh(){const r=await fetch('/api/state');const s=await r.json();const p=s.plan||{};
 document.getElementById('status').textContent=(p.status||'?')+(p.route?' · '+p.route:'');
 document.getElementById('input').textContent=p.input||'';
 Q=s.questions||[];const qd=document.getElementById('questions');
 if(!Q.length){qd.innerHTML='<i>지금은 물을 것이 없습니다.</i>'}else{qd.innerHTML=Q.map(q=>{let inp='';
  if(q.type==='bool')inp=`<select id="q_${q.id}"><option value="true" ${q.proposal?'selected':''}>예</option><option value="false" ${!q.proposal?'selected':''}>아니오</option></select>`;
  else if(q.type==='choice')inp=`<select id="q_${q.id}">${(q.choices||[]).map(c=>`<option ${c==q.proposal?'selected':''}>${c}</option>`).join('')}</select>`;
  else inp=`<input id="q_${q.id}" type="number" step="any" value="${q.proposal}"> ${q.unit||''}`;
  const ev=(q.evidence||[]).map(e=>e.match(/\\.png$/)?`<div class="ev"><img src="/files/${esc(e)}"></div>`:`<div><a href="/files/${esc(e)}" target="_blank">${esc(e)}</a></div>`).join('');
  const w=q.where||{};const hasWhere=(w.points&&w.points.length)||(w.plane_z!=null);
  const btn=hasWhere?` <button onclick="focusQ('${q.id}')">위치 보기</button>`:'';
  const live=(w.plane_z!=null&&q.type==='number')?` oninput="planeFromInput('${q.id}')"`:'';
  return `<div class="q" id="card_${q.id}"><b>${esc(q.question)}</b><small>제안 ${esc(q.proposal)} ${esc(q.unit||'')} — ${esc(q.reason)}</small><br>${inp.replace('<input ','<input '+live+' ')}${btn}${ev}</div>`}).join('');
 if(window.updateMarkers)window.updateMarkers(Q);}
 const pd=document.getElementById('plan');const d=p.diagnosis||{};
 pd.innerHTML='<b>진단</b> '+esc(JSON.stringify(d).slice(0,600))+'<br>'+(p.decisions||[]).map(x=>`<div>• ${esc(x.what)} <small>← ${esc(x.because)}</small></div>`).join('')
  +(p.checks||[]).map(c=>`<div class="${c.ok?'ok':'bad'}">${c.ok?'✓':'✗'} ${esc(c.name)}: ${esc(c.detail)}</div>`).join('')
  +((p.assumptions||[]).length?'<div><b>가정</b> '+p.assumptions.map(a=>esc(a.id+'='+a.value)).join(', ')+'</div>':'')
  +((p.deliverables||[]).length?'<div><b>산출물</b> '+p.deliverables.map(esc).join('<br>')+'</div>':'');
 document.getElementById('log').textContent=s.log||'';}
function collect(){const a={};for(const q of Q){const el=document.getElementById('q_'+q.id);if(!el)continue;let v=el.value;
 if(q.type==='bool')v=(v==='true');else if(q.type==='number')v=parseFloat(v);a[q.id]=v}return a}
async function submitAnswers(){await fetch('/api/answers',{method:'POST',body:JSON.stringify({answers:collect()})});setTimeout(refresh,1500)}
async function runDefaults(){await fetch('/api/run',{method:'POST',body:'{}'});setTimeout(refresh,1500)}
function add(role,t){const c=document.getElementById('chat');const d=document.createElement('div');d.className='m '+(role==='user'?'u':'a');d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight}
async function send(){const m=document.getElementById('msg');const t=m.value.trim();if(!t)return;m.value='';add('user',t);hist.push({role:'user',content:t});
 add('assistant','…');const r=await fetch('/api/chat',{method:'POST',body:JSON.stringify({history:hist})});const s=await r.json();
 document.getElementById('chat').lastChild.textContent=s.reply;hist.push({role:'assistant',content:s.reply});
 if(s.suggested){for(const [k,v] of Object.entries(s.suggested)){const el=document.getElementById('q_'+k);if(el)el.value=(typeof v==='boolean')?String(v):v}}}
refresh();setInterval(refresh,4000);
</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
const box=document.getElementById('viewer');const W=()=>box.clientWidth,Hh=()=>box.clientHeight;
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(W(),Hh());renderer.setPixelRatio(window.devicePixelRatio);box.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0xf4f5f7);
const camera=new THREE.PerspectiveCamera(40,W()/Hh(),1,100000);camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x667788,1.1));const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(1,-1,2);scene.add(dl);
let meshObj=null,bbox=null;const markers=new THREE.Group();scene.add(markers);let plane=null;
const loader=new STLLoader();
function loadMesh(){loader.load('/files/viewer.stl',g=>{if(meshObj)scene.remove(meshObj);g.computeVertexNormals();
 meshObj=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x8a97a8,metalness:0.1,roughness:0.75,side:THREE.DoubleSide}));scene.add(meshObj);
 g.computeBoundingBox();bbox=g.boundingBox;fit(bbox.getCenter(new THREE.Vector3()),bbox.getSize(new THREE.Vector3()).length()*0.6)},undefined,()=>{});}
function fit(center,dist){controls.target.copy(center);camera.position.set(center.x-dist*0.9,center.y-dist*1.1,center.z+dist*0.7);camera.lookAt(center);controls.update();}
function label(text,pos,r){const c=document.createElement('canvas');c.width=256;c.height=64;const x=c.getContext('2d');x.fillStyle='rgba(255,255,255,0.85)';x.fillRect(0,0,256,64);x.fillStyle='#c33';x.font='28px sans-serif';x.fillText(text,8,42);
 const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sp.position.copy(pos).add(new THREE.Vector3(0,0,r*1.3+40));sp.scale.set(r*2+240,(r*2+240)/4,1);return sp;}
const qOf=new Map();
window.updateMarkers=function(Q){markers.clear();qOf.clear();let pz=null;
 for(const q of Q){const w=q.where||{};for(const p of (w.points||[])){const r=Math.max(20,p[3]||60);
  const m=new THREE.Mesh(new THREE.SphereGeometry(r,24,16),new THREE.MeshStandardMaterial({color:0xe04a3f,transparent:true,opacity:0.45}));m.position.set(p[0],p[1],p[2]);markers.add(m);qOf.set(m.uuid,q.id);
  markers.add(label((p[4]||q.id)+'',m.position,r));}
  if(w.plane_z!=null&&pz==null)pz=w.plane_z;}
 setPlane(pz);}
function setPlane(z){if(plane){scene.remove(plane);plane=null}if(z==null||!bbox)return;const s=bbox.getSize(new THREE.Vector3());
 plane=new THREE.Mesh(new THREE.PlaneGeometry(s.x*1.1,s.y*1.1),new THREE.MeshBasicMaterial({color:0x2b6cff,transparent:true,opacity:0.35,side:THREE.DoubleSide}));
 const c=bbox.getCenter(new THREE.Vector3());plane.position.set(c.x,c.y,z);scene.add(plane);}
window.planeFromInput=function(qid){const el=document.getElementById('q_'+qid);if(el)setPlane(parseFloat(el.value));};
window.focusQ=function(qid){const q=Q.find(x=>x.id===qid);if(!q)return;const w=q.where||{};const pts=w.points||[];
 if(pts.length){const c=new THREE.Vector3();for(const p of pts)c.add(new THREE.Vector3(p[0],p[1],p[2]));c.multiplyScalar(1/pts.length);const r=Math.max(...pts.map(p=>p[3]||60));fit(c,Math.max(r*6,600));}
 else if(w.plane_z!=null&&bbox){const c=bbox.getCenter(new THREE.Vector3());c.z=w.plane_z;fit(c,bbox.getSize(new THREE.Vector3()).length()*0.5);setPlane(w.plane_z);}
 const card=document.getElementById('card_'+qid);if(card){card.scrollIntoView({behavior:'smooth',block:'center'});card.style.outline='2px solid #e04a3f';setTimeout(()=>card.style.outline='',2000);}};
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
renderer.domElement.addEventListener('click',e=>{const rc=renderer.domElement.getBoundingClientRect();mouse.x=((e.clientX-rc.left)/rc.width)*2-1;mouse.y=-((e.clientY-rc.top)/rc.height)*2+1;
 ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects(markers.children.filter(o=>o.isMesh));if(hit.length){const qid=qOf.get(hit[0].object.uuid);if(qid)window.focusQ(qid);}});
window.addEventListener('resize',()=>{camera.aspect=W()/Hh();camera.updateProjectionMatrix();renderer.setSize(W(),Hh());});
(function anim(){requestAnimationFrame(anim);controls.update();renderer.render(scene,camera);})();
loadMesh();setTimeout(()=>{if(window.updateMarkers&&Q)window.updateMarkers(Q)},1500);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/state":
            plan = json.loads((DIR / "plan.json").read_text()) if (DIR / "plan.json").exists() else {}
            qs = json.loads((DIR / "questions.json").read_text()) if (DIR / "questions.json").exists() else []
            if STATE["proc"] is not None and STATE["proc"].poll() is None:
                plan["status"] = "running"
            log = (DIR / "plan_log.txt").read_text()[-4000:] if (DIR / "plan_log.txt").exists() else ""
            return self._send(200, json.dumps({"plan": plan, "questions": qs, "log": log}, ensure_ascii=False))
        if self.path.startswith("/files/"):
            rel = urllib.request.unquote(self.path[len("/files/"):])
            f = Path(rel) if rel.startswith("/") else DIR / rel
            try:
                f = f.resolve()
                if not (str(f).startswith(str(DIR)) or str(f).startswith(str(HERE.parent / "var"))) or not f.exists():
                    return self._send(404, "not found", "text/plain")
                ctype = "image/png" if f.suffix == ".png" else "application/json" if f.suffix == ".json" else "text/plain; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype)
            except Exception as exc:
                return self._send(500, str(exc), "text/plain")
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/answers":
            ok = start_controller(answers=body.get("answers", {}))
            return self._send(200, json.dumps({"started": ok}))
        if self.path == "/api/run":
            ok = start_controller(assume=True)
            return self._send(200, json.dumps({"started": ok}))
        if self.path == "/api/chat":
            try:
                reply, suggested = chat(body.get("history", []))
            except Exception as exc:
                reply, suggested = f"모델 호출 실패: {type(exc).__name__}: {exc}", None
            return self._send(200, json.dumps({"reply": reply, "suggested": suggested}, ensure_ascii=False))
        return self._send(404, "not found", "text/plain")


print(f"http://127.0.0.1:{args.port}/   (dir {DIR}, model {'없음' if CFG is None else args.model})")
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
