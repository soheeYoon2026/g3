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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aox_g3.run_state import reconcile
from aox_g3.quality_display import healing_result
import geometry_tools

def read_state_text(path):
    """Read UTF-8 state and legacy Windows CP949 files without losing text."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp949")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--dir", type=Path, required=True, help="the controller's --out directory")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--model", default=os.environ.get("PRIME_MODEL", "openai/gpt-5.6-terra"),
                help="Prime Inference model id; PRIME_MODEL overrides. Measured 2026-09-14: terra 2.5 s, gpt-oss-120b 5.2 s on the same question, both returned a valid answer block")
ap.add_argument("--no-chat", action="store_true", help="page without the model (no API calls)")
ap.add_argument("--no-tools", action="store_true", help="do not let the model call the read-only measurements")
ap.add_argument("--max-tool-calls", type=int, default=6, help="measurements the model may run for one message")
ap.add_argument("--no-docs", action="store_true", help="never load the reference documents, whatever the provider")
ap.add_argument("--api", choices=["auto", "openai", "prime", "local"], default="auto",
                help="who serves the model. auto: LOCAL_LLM_URL, then OPENAI_API_KEY, then ~/.prime/config.json")
ap.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:11500/v1 for a local ollama")
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
OPENAI_FILE = {}


def openai_key():
    """The key from the environment, or from a file, so it never has to be pasted into a shell.
    Order: OPENAI_API_KEY, the file named by OPENAI_API_KEY_FILE, ~/.config/openai/key.
    The file may hold the bare key or NAME=value lines (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for path in [os.environ.get("OPENAI_API_KEY_FILE"), os.path.expanduser("~/.config/openai/key")]:
        if not path or not os.path.exists(path):
            continue
        if os.stat(path).st_mode & 0o077:
            print(f"  경고: {path} 를 다른 사용자도 읽을 수 있습니다 (chmod 600 을 권합니다)")
        text = open(path).read()
        if "=" in text:
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, _, value = line.partition("=")
                    OPENAI_FILE[name.strip()] = value.strip().strip('"\'')
            if OPENAI_FILE.get("OPENAI_API_KEY"):
                return OPENAI_FILE["OPENAI_API_KEY"]
        else:
            return text.strip()
    return ""


API = {"url": None, "key": "", "how": "없음"}
if not args.no_chat:
    # Provider order: an explicit --base-url, then a local server, then OpenAI, then Prime.
    # Prices per 1M tokens measured 2026-09-14: OpenAI terra $2 in / $12 out with $0.2 cached
    # input; Prime relays the same model at $2.5 / $15 and publishes no cache price, so the
    # documents cost ten times more per turn there. A local server costs nothing.
    if args.base_url:
        API = {"url": args.base_url.rstrip("/"), "key": os.environ.get("OPENAI_API_KEY", ""), "how": "직접 지정"}
    elif args.api in ("auto", "local") and os.environ.get("LOCAL_LLM_URL"):
        API = {"url": os.environ["LOCAL_LLM_URL"].rstrip("/"), "key": "", "how": "로컬 서버"}
    elif args.api in ("auto", "openai") and openai_key():
        API = {"url": os.environ.get("OPENAI_BASE_URL") or OPENAI_FILE.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
               "key": openai_key(), "how": "OpenAI 직접"}
        # Prime addresses models as "vendor/name"; OpenAI itself has no prefix
        if args.model.startswith("openai/"):
            args.model = OPENAI_FILE.get("OPENAI_MODEL") or args.model.split("/", 1)[1]
    elif args.api in ("auto", "prime"):
        try:
            cfg = json.load(open(os.path.expanduser("~/.prime/config.json")))
            API = {"url": cfg["inference_url"].rstrip("/"), "key": cfg["api_key"], "how": "Prime 경유(비쌈: 캐시 없음)"}
        except Exception:
            pass
    print(f"  모델 연결: {API['how']}  ({API['url'] or '없음'})")
CFG = API if API["url"] else None

SYSTEM = """당신은 자동차 형상 정리 파이프라인의 AI 통제기입니다. 규칙 통제기(plan_geometry.py)가 측정하고 실행한 내용이
아래 plan.json 에 있습니다. 당신의 역할은 (1) 고객이 답해야 할 질문을 측정값을 근거로 쉬운 말로 설명하고,
(2) 고객의 선택을 도와 답을 정리하는 것입니다. 도구를 실행하거나 실행했다고 말하지 마세요. plan.json 에 없는
수치를 지어내지 마세요. 모르면 모른다고 하세요. 고객이 답을 정하면 마지막에 반드시 다음 형식의 JSON 블록을
붙이세요: ```json {"answers": {"질문id": 값}} ```. 값은 질문의 type 에 맞게(number → 숫자, bool → true/false,
choice → 선택지 문자열). 한국어로, 짧게, 결론부터.

필요하면 아래 측정 기능을 직접 불러 확인한 뒤 답하세요. 전부 읽기만 하는 기능이라 형상을 바꾸지 않습니다.
파일 경로는 plan.json 의 산출물·작업 폴더에 있는 것을 쓰고, 지어내지 마세요. 측정 결과를 인용할 때는
어느 기능으로 무엇을 쟀는지 한 줄로 밝히세요."""

DEFAULT_KNOWLEDGE = [Path("var/docs/geometry_cleaning/GEOMETRY_CLEANING.md"), Path("docs/GEOMETRY_PIPELINE.md")]


def load_knowledge():
    """The reference documents, verbatim. No retrieval: the whole text goes in every call.

    On by default only where the prompt prefix is cached. Measured 2026-09-14 on OpenAI:
    the first turn billed 35,569 tokens at $0.072, the next 35,531 of 35,554 came back as
    cached input and cost $0.008. Prime publishes no cache price and charged the full
    $0.090 every turn, so there the documents stay off unless asked for."""
    if args.no_docs:
        return ""
    wanted = args.knowledge
    if not wanted and API["how"] == "OpenAI 직접" and not args.no_chat:
        wanted = ["docs"]
        print("  참고 문서: 기본으로 켬 (OpenAI 캐시가 있어 두 번째 턴부터 약 $0.008)")
    if not wanted:
        return ""
    root = HERE.parent
    paths = []
    for item in wanted:
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
    per_turn = {"gpt-5.6-terra": 2.0, "gpt-5.6-luna": 0.2, "gpt-5.6-sol": 4.0,
                "openai/gpt-5.6-terra": 2.5, "openai/gpt-5.6-terra-pro": 2.5,
                "openai/gpt-oss-120b": 0.35, "openai/gpt-oss-20b": 0.07}.get(args.model)
    # measured 2026-09-14: 68,050 characters of Korean documents came to 35,646 input tokens
    cached = " (첫 턴 기준, 캐시되면 1/9)" if API["how"] == "OpenAI 직접" else ""
    cost = f", {args.model} 기준 한 번에 약 ${len(body)/1.9*1e-6*per_turn:.3f}{cached}" if per_turn else ""
    print(f"  참고 문서 합계 {len(body):,}자 — 대화 한 번마다 다시 보냅니다{cost}")
    return ("\n\n아래는 이 파이프라인의 정본 문서입니다. 수치·규칙·이전 실측은 여기서 인용하고, "
            "여기에 없으면 모른다고 하세요. 문서와 plan.json 이 어긋나면 plan.json(이번 실행)이 우선입니다.\n" + body)


KNOWLEDGE = load_knowledge()


def digest():
    plan = json.loads(read_state_text(DIR / "plan.json")) if (DIR / "plan.json").exists() else {}
    keep = {k: plan.get(k) for k in ("status", "route", "diagnosis", "decisions", "checks", "questions", "assumptions", "deliverables", "warnings", "healing_result")}
    # terra takes a 1.05 M context, so the whole plan fits; the cap is only a guard
    return json.dumps(keep, ensure_ascii=False)[:120000]


def post(msgs, tools=None):
    # OpenAI's newer models reject max_tokens and want max_completion_tokens; Prime and a local
    # ollama still take the old name, so send whichever the endpoint accepts (measured 2026-09-14)
    limit_key = "max_tokens" if API["how"].startswith(("Prime", "로컬")) else "max_completion_tokens"
    body = {"model": args.model, "messages": msgs, limit_key: 1600, "temperature": 0.2}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
        # gpt-5.6 refuses function tools on /v1/chat/completions while it is reasoning:
        # "use /v1/responses or set reasoning_effort to 'none'" (measured 2026-09-14).
        # Turning reasoning off for the measuring turns is the small change; moving to the
        # responses API is the other way and would keep it.
        if API["how"] == "OpenAI 직접":
            body["reasoning_effort"] = "none"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "prime-cli/0.6.33"}
    if CFG["key"]:
        headers["Authorization"] = "Bearer " + CFG["key"]
    req = urllib.request.Request(CFG["url"] + "/chat/completions", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"모델 호출 실패 HTTP {exc.code}: {detail}") from None


def chat(history):
    """One turn. The model may call the read-only measurements first; every call is executed
    here, its result is fed back, and the loop ends when the model answers in words. Writing
    steps are not exposed: the rules in plan_geometry.py keep those."""
    if CFG is None:
        return "모델 연결이 없습니다(--no-chat 이거나 키가 없습니다).", None
    msgs = [{"role": "system", "content": SYSTEM + KNOWLEDGE + "\n\nplan.json:\n" + digest()}] + history[-12:]
    tools = None if args.no_tools else geometry_tools.schemas()
    used = []
    budget = max(1, args.max_tool_calls)   # measurements, not turns: one reply may ask for several
    while True:
        d = post(msgs, tools)
        msg = d["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        usage = d.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        print(f"  [대화] {args.model} 입력 {usage.get('prompt_tokens','?')} (캐시 {cached}) 출력 {usage.get('completion_tokens','?')}"
              + (f" · 도구 {[c['function']['name'] for c in calls]}" if calls else ""))
        if not calls:
            text = msg.get("content") or ""
            if used:
                text += "\n\n(직접 잰 것: " + ", ".join(used) + ")"
            return text, extract_answers(text)
        if budget <= 0:
            return ("측정 횟수 예산을 다 썼습니다(" + str(args.max_tool_calls) + "회). 질문을 좁혀 주세요."
                    + ("\n\n(잰 것: " + ", ".join(used) + ")" if used else "")), None
        calls = calls[:budget]
        budget -= len(calls)
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                arguments = json.loads(c["function"].get("arguments") or "{}")
            except Exception:
                arguments = {}
            result = geometry_tools.call(name, arguments)
            used.append(f"{name}({', '.join(f'{k}={v}' for k, v in arguments.items() if k != 'samples')})")
            print(f"        → {name} {arguments} : {str(result)[:120]}")
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": name,
                         "content": json.dumps(result, ensure_ascii=False)[:6000]})



def extract_answers(text):
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("answers")
    except Exception:
        return None




START_LOCK = threading.Lock()


def start_controller(answers=None, assume=False):
    with START_LOCK:
        if reconcile(DIR):
            return False
        return _start_controller(answers, assume)


def _start_controller(answers=None, assume=False):
    """Re-run the controller. It rewrites plan.json from scratch, so keep the previous record
    and never start on an empty answer set: a page with no open questions used to submit {},
    which wiped the answers of a finished run and restarted it (seen 2026-09-14)."""
    if STATE["proc"] is not None and STATE["proc"].poll() is None:
        return False
    if answers is not None and not answers:
        return False
    plan = DIR / "plan.json"
    if plan.exists():
        (DIR / "plan_prev.json").write_text(read_state_text(plan), encoding="utf-8")
    if answers:  # keep answers already given for questions the page no longer shows
        old_file = DIR / "answers.json"
        if old_file.exists():
            try:
                merged = json.loads(read_state_text(old_file))
            except Exception:
                merged = {}
            merged.update(answers)
            answers = merged
    cmd = [sys.executable, str(HERE / "plan_geometry.py"), "--in", str(args.src), "--out", str(DIR), "--no-render"]
    if answers is not None:
        (DIR / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
        cmd += ["--answers", str(DIR / "answers.json")]
    if assume:
        cmd.append("--assume-defaults")
    STATE["proc"] = subprocess.Popen(cmd, stdout=open(DIR / "ui_run.txt", "a"), stderr=subprocess.STDOUT)
    return True


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>형상 정리 통제기</title>
<style>body{font-family:"Noto Sans CJK KR",sans-serif;margin:0;display:grid;grid-template-columns:1fr 420px;height:100vh}
main{padding:18px 24px;overflow:auto}aside{border-left:1px solid #ddd;display:flex;flex-direction:column;height:100vh}
h1{font-size:20px;margin:0 0 6px}h2{font-size:15px;margin:18px 0 6px;color:#333}.lg{display:inline-block;background:#fff;border:1px solid #d9a9a3;color:#b23b2c;border-radius:10px;padding:1px 7px;margin:2px 3px;cursor:pointer;font-size:12px}
.q{border:1px solid #ccd;border-radius:6px;padding:10px 12px;margin:8px 0;background:#f8f9fb}
.q b{display:block;margin-bottom:4px}.q small{color:#666}.q input,.q select{margin-top:6px;padding:4px 6px;font-size:14px}
.ev img{max-width:100%;border:1px solid #ddd;margin-top:6px}.ok{color:#2a7}.bad{color:#c33}#log{font-family:monospace;font-size:12px;white-space:pre-wrap;background:#111;color:#ddd;padding:8px;max-height:220px;overflow:auto}
#chat{flex:1;overflow:auto;padding:12px}.m{margin:6px 0;padding:8px 10px;border-radius:6px;white-space:pre-wrap;font-size:14px}.u{background:#e8f0fe}.a{background:#f1f3f4}
#in{display:flex;border-top:1px solid #ddd}#in textarea{flex:1;border:0;padding:10px;font-size:14px;resize:none;height:70px}#in button{width:70px;border:0;background:#2b5;color:#fff}
button.run{padding:8px 14px;font-size:14px;margin-right:8px}.st{display:inline-block;padding:2px 8px;border-radius:10px;background:#eee;font-size:12px}</style></head>
<body><main><h1>형상 정리 통제기 <span class="st" id="status"></span></h1><div id="input"></div>
<style>body{grid-template-columns:minmax(0,1fr) 360px;background:#f7f9fc;color:#172033}main{min-width:0}aside{background:white}.q{background:white}button{cursor:pointer}button:disabled{opacity:.45;cursor:wait}#activity{position:sticky;top:0;z-index:5;background:#fff;border:1px solid #cbd5e1;border-radius:10px;padding:14px;margin:12px 0;box-shadow:0 3px 12px #0001}#activity strong{display:block;font-size:17px}#stage{overflow-wrap:anywhere;margin:6px 0}#elapsed,#activity-note{font-size:12px;color:#64748b}.loading{height:6px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin-top:10px}.loading span{display:block;width:30%;height:100%;background:#2563eb;animation:busy 1.6s ease-in-out infinite}@keyframes busy{from{transform:translateX(-100%)}to{transform:translateX(440%)}}#activity:not(.busy) .loading{display:none}#preview-state{font-size:12px;color:#64748b;margin:6px 0}details summary{cursor:pointer;padding:8px}#plan{overflow-wrap:anywhere}@media(max-width:900px){body{grid-template-columns:minmax(0,1fr)}aside{height:420px}main{overflow:visible}body{height:auto}}</style>
<section id="activity" role="status" aria-live="polite"><strong id="activity-title">상태 확인 중…</strong><div id="stage"></div><div id="elapsed"></div><div id="activity-note"></div><div class="loading" aria-label="계산 중, 전체 진행률 미제공"><span></span></div></section>
<div id="preview-state">미리보기 준비 상태를 확인 중입니다.</div>
<style>.preview-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0}.preview-tabs{display:inline-flex;padding:4px;gap:4px;border:1px solid #e2e8f0;background:#edf2f7;border-radius:11px}.preview-tabs button{border:0;border-radius:8px;padding:9px 18px;background:transparent;color:#64748b;font-family:inherit;font-size:12px;font-weight:600;transition:background .15s,color .15s}.preview-tabs button[aria-pressed="true"]{background:white;color:#2563eb;box-shadow:0 1px 4px #0f172a12}.preview-tabs button:disabled{opacity:1;color:#94a3b8;cursor:not-allowed}.preview-tabs button:not(:disabled):hover{color:#2563eb}.preview-toolbar small{font-size:11px;color:#94a3b8}#result-note{display:flex;align-items:flex-start;gap:10px;background:#fff;border:1px solid #e2e8f0;border-radius:11px;padding:12px 14px;margin:0 0 12px;color:#64748b;font-size:12px;line-height:1.7}#result-note:before{content:'i';display:grid;place-items:center;width:20px;height:20px;flex-shrink:0;border-radius:7px;background:#eff6ff;color:#2563eb;font-size:11px;font-weight:700}#result-note.warning{background:#fff7ed;border-color:#fed7aa;color:#9a3412}#result-note.warning:before{content:'!';background:#ffedd5;color:#c2410c}#result-note-text{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<div class="preview-toolbar"><div class="preview-tabs" role="group" aria-label="미리보기 형상 선택"><button id="view-original" aria-pressed="true" onclick="selectPreview('original')">원본</button><button id="view-result" aria-pressed="false" onclick="selectPreview('result')" disabled>결과</button></div><small>원본 · 생성 결과 비교</small></div>
<div id="result-note"><span id="result-note-text"></span></div>
<div id="viewer" style="width:100%;height:380px;border:1px solid #ccd;border-radius:6px;background:#f4f5f7;position:relative;margin:8px 0">
<div id="vhint" style="position:absolute;left:8px;top:6px;font-size:12px;color:#555;pointer-events:none">드래그로 돌리고 휠로 확대. 번호를 누르면 그 자리로, 표식을 누르면 해당 질문으로 갑니다.</div>
<button onclick="togglePins()" style="position:absolute;right:8px;top:6px;font-size:12px">번호 숨기기/보이기</button></div>
<div id="legend" style="font-size:12px;color:#444;margin:4px 0 8px 0;line-height:1.9"></div>
<style>#questions .question-card{padding:18px 20px;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 3px 10px #0f172a04;margin:12px 0}#questions .question-card b{font-size:15px;line-height:1.5;margin:0}.question-top{display:flex;align-items:center;justify-content:space-between;gap:12px}.question-badge{white-space:nowrap;border-radius:20px;background:#eff6ff;color:#2563eb;padding:4px 9px;font-size:11px}.question-description{font-size:12px;color:#64748b;line-height:1.7;margin:9px 0 12px}.question-field{display:flex;align-items:center;gap:10px;flex-wrap:wrap}#questions .question-field select,#questions .question-field input{box-sizing:border-box;min-width:130px;max-width:100%;font-family:inherit;font-size:13px;padding:10px 12px;margin:0;border:1px solid #cbd5e1;border-radius:9px;background:#f8fafc;color:#1e293b;outline:none}#questions .question-field input{width:180px}#questions .question-field select:focus,#questions .question-field input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px #dbeafe}.question-field button{padding:9px 12px;border:1px solid #cbd5e1;background:white;border-radius:8px;color:#475569;font-size:12px}.question-card details{font-size:12px;color:#64748b;margin-top:12px;border-top:1px solid #f1f5f9;padding-top:4px}.question-card details p{line-height:1.7;margin:8px}.question-actions{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 22px}button.run{margin:0;border:1px solid #cbd5e1;border-radius:10px;background:white;color:#475569;padding:11px 16px;font-family:inherit;font-size:13px}button.run:first-child{background:#2563eb;color:white;border-color:#2563eb}button.run:disabled{cursor:not-allowed;opacity:.45}button.run:not(:disabled):hover{filter:brightness(.96)}</style>
<h2>실행 전 확인</h2><p style="font-size:12px;color:#64748b;margin:6px 0">추천값을 확인하거나 변경한 뒤, 답을 저장해 다음 단계로 진행하세요.</p><div id="questions"></div>
<div class="question-actions"><button class="run" onclick="submitAnswers()">답 저장 후 이어서 실행 →</button><button class="run" onclick="runDefaults()">제안값으로 실행</button></div>
<style>.section-card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;margin:12px 0;box-shadow:0 3px 10px #0f172a04}.section-card h3{font-size:14px;margin:0 0 12px;font-weight:700;color:#334155}.section-caption{font-size:12px;line-height:1.7;color:#64748b;margin:0}.empty-card{display:flex;align-items:flex-start;gap:12px;padding:18px 20px;background:white;border:1px dashed #cbd5e1;border-radius:14px;color:#64748b;font-size:12px;line-height:1.7}.empty-icon{display:grid;place-items:center;width:30px;height:30px;flex-shrink:0;border-radius:10px;background:#eff6ff;color:#2563eb}.empty-card strong{display:block;color:#334155;font-size:13px;margin-bottom:4px}.route-row{display:flex;gap:12px;padding:11px 0;border-bottom:1px solid #f1f5f9}.route-row:last-child{border:0}.route-number{display:grid;place-items:center;flex-shrink:0;width:24px;height:24px;background:#eff6ff;color:#2563eb;border-radius:8px;font-size:11px;font-weight:700}.route-row strong{display:block;font-size:13px;font-weight:600}.route-row small{display:block;margin-top:4px;color:#64748b;line-height:1.6;font-size:11px}.check-row{padding:12px;border:1px solid #e2e8f0;border-radius:10px;margin:8px 0;display:flex;gap:10px;align-items:flex-start}.check-row.pass{background:#f0fdf4;border-color:#dcfce7}.check-row.fail{background:#fff7ed;border-color:#fed7aa}.check-tag{flex-shrink:0;font-size:10px;font-weight:700;border-radius:6px;padding:3px 7px;background:white;color:#166534}.fail .check-tag{color:#c2410c}.check-row strong{font-size:12px;display:block}.check-row small{display:block;font-size:11px;color:#64748b;margin-top:4px;line-height:1.6}.file-row{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 12px;margin:7px 0;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;font-size:12px}.file-row button{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:7px 10px;color:#2563eb;font-size:11px}.file-row code{font-family:inherit;font-weight:600;overflow-wrap:anywhere}.fold-panel{padding:0;background:white;border:1px solid #e2e8f0;border-radius:12px;margin:12px 0;overflow:hidden}.fold-panel>summary{padding:14px 18px;font-size:12px;font-weight:600;color:#475569}.fold-panel>div{padding:0 18px 16px}.fold-panel pre{white-space:pre-wrap;font-size:11px;line-height:1.7;overflow-wrap:anywhere}#log{background:#f8fafc;border-radius:9px;color:#475569;font-size:11px;padding:12px;line-height:1.7;max-height:250px}main>h2{font-size:15px;margin:24px 0 10px;letter-spacing:-.3px}main .question-actions{margin-bottom:24px}</style>
<h2>검사 결과와 처리 과정</h2><div id="plan"></div><details class="fold-panel"><summary>실행 로그</summary><div><div id="log"></div></div></details>
<details class="fold-panel"><summary>파일·CLI 상세 안내</summary><div style="font-size:12px;line-height:1.8;overflow-wrap:anywhere"><p>브라우저에서 답변한다면 아래 파일을 직접 편집할 필요는 없습니다.</p><div>질문 파일: <code id="cli-questions"></code></div><div>답변 파일: <code id="cli-answers"></code></div><p>파일로 답하려면 answers.json에 질문 ID와 답을 적으세요. 예: <code>{"units": "mm", "length_axis_now": "x"}</code><br>현재 질문의 ID·형식에 맞는 값만 사용하고, 제안값을 그대로 승인할지는 확인하세요.</p><p>계산이 답변 대기로 끝난 상태에서만 아래 명령을 실행하세요. 브라우저 실행 버튼과 동시에 실행하지 마세요.</p><pre id="cli-command" style="white-space:pre-wrap;background:#f1f5f9;padding:12px;border-radius:8px"></pre><small>위 명령은 답변 파일로 이어서 실행합니다. 단순 미리보기 명령이 아닙니다.</small></div></details></main>
<style>aside{min-width:0;border-left:1px solid #e2e8f0;background:#f8fafc}.chat-head{padding:22px 20px 16px;background:white;border-bottom:1px solid #e2e8f0}.chat-head h2{margin:0;font-size:18px;letter-spacing:-.4px}.chat-head p{margin:7px 0 0;font-size:12px;line-height:1.6;color:#64748b}.ai-label{font-size:10px;letter-spacing:1.2px;color:#2563eb;font-weight:700;margin-bottom:7px}#chat{min-height:0;padding:18px;scroll-behavior:smooth}.chat-empty{padding:24px 4px;color:#64748b;font-size:13px;line-height:1.8}.chat-empty strong{display:block;color:#334155;font-size:15px;margin-bottom:8px}.chat-example{display:block;text-align:left;width:100%;padding:11px 12px;background:#fff;border:1px solid #e2e8f0;border-radius:10px;margin:9px 0;color:#475569;font-size:12px}.chat-example:hover{border-color:#93c5fd;background:#eff6ff}.m{padding:13px 14px;margin:0 0 15px;border:1px solid #e2e8f0;border-radius:14px;font-size:13px;line-height:1.75;overflow-wrap:anywhere;box-shadow:0 2px 6px #0f172a04}.m.u{background:#eff6ff;border-color:#dbeafe;margin-left:20px}.m.a{background:white;margin-right:10px}.m-label{display:block;font-size:10px;font-weight:700;letter-spacing:.4px;margin-bottom:6px;color:#64748b}.m.u .m-label{color:#2563eb}.chat-wait{color:#64748b;display:flex;align-items:center;gap:9px}.chat-dot{width:12px;height:12px;border:2px solid #bfdbfe;border-top-color:#2563eb;border-radius:50%;animation:chat-spin .8s linear infinite}@keyframes chat-spin{to{transform:rotate(360deg)}}#in{display:block;margin:0;padding:14px 16px 12px;background:white;border-top:1px solid #e2e8f0}#in textarea{display:block;box-sizing:border-box;width:100%;height:86px;padding:12px;border:1px solid #cbd5e1;border-radius:12px;background:#f8fafc;font-family:inherit;font-size:13px;line-height:1.6;outline:none}#in textarea:focus{border-color:#3b82f6;box-shadow:0 0 0 3px #dbeafe}#in textarea::placeholder{color:#94a3b8}.chat-footer{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px}.chat-footer small{font-size:10px;color:#94a3b8}#in button{width:auto;padding:9px 18px;border-radius:9px;background:#2563eb;font-size:12px;font-weight:600;color:white}#in button:hover{background:#1d4ed8}#in button:disabled{opacity:.5}</style>
<aside aria-label="AI 형상 도우미"><header class="chat-head"><div class="ai-label">GEOMETRY ASSISTANT</div><h2>AI 형상 도우미</h2><p>현재 작업의 검사 결과와 처리 경로를 물어보세요.<br>계산 진행 상태는 왼쪽 상태 카드에 표시됩니다.</p></header><div id="chat" role="log" aria-live="polite"><div class="chat-empty" id="chat-empty"><strong>어떤 부분이 궁금한가요?</strong>검증 실패 이유나 다음 단계에 대해 질문해 보세요.<button class="chat-example" onclick="fillQuestion(this.textContent)">검증이 실패한 이유를 설명해줘</button><button class="chat-example" onclick="fillQuestion(this.textContent)">원본과 결과는 무엇이 달라졌어?</button><button class="chat-example" onclick="fillQuestion(this.textContent)">현재 처리 경로를 쉽게 설명해줘</button></div></div><div id="in"><textarea id="msg" aria-label="AI에게 질문" placeholder="형상 검사나 처리 과정에 대해 질문하세요…"></textarea><div class="chat-footer"><small>Ctrl+Enter로 보내기 · AI 답변은 검토가 필요해요</small><button id="chat-send" onclick="send()">보내기 ↗</button></div></div></aside>
<style>#log{background:#111827;color:#e2e8f0;border:1px solid #253247;border-radius:9px;padding:14px;line-height:1.8}#cli-command{color:#334155}</style>
<script>
let hist=[];let Q=[];let latestPlan={};let launching=false;let launchAt=0;let refreshing=false;let questionSignature='';
let previewMode='original';let previewFiles={};
function selectPreview(mode){previewMode=mode;syncPreview();}
function syncPreview(){const info=previewFiles[previewMode];const result=previewFiles.result;
 document.getElementById('view-result').disabled=!result;
 document.getElementById('view-original').disabled=!previewFiles.original;
 document.getElementById('view-original').setAttribute('aria-pressed',String(previewMode==='original'));
 document.getElementById('view-result').setAttribute('aria-pressed',String(previewMode==='result'));
 const failures=(latestPlan.checks||[]).filter(c=>!c.ok&&!c.resolved);
 const lines=previewMode==='result'&&result?['결과 파일: '+result.name+' ('+Math.round(result.bytes/1e6)+' MB)',...(failures.length?['검증 실패 — 정상 결과로 승인되지 않았습니다.',...failures.map(c=>c.name+': '+c.detail)]:['현재 기록에 미해결 검증 실패가 없습니다. 원본 보존·설계 의도는 별도 확인이 필요합니다.']),'결과 STL 전체를 불러오므로 큰 파일은 표시까지 시간이 걸립니다.']:[result?'원본 미리보기입니다. ‘결과’ 탭에서 생성된 STL을 확인하세요.':'수리 결과가 생성되면 ‘결과’ 탭이 활성화됩니다.'];document.getElementById('result-note-text').textContent=lines.join(String.fromCharCode(10));document.getElementById('result-note').classList.toggle('warning',previewMode==='result'&&!!result&&failures.length>0);
 window.previewSelection=info;if(window.ensurePreview)window.ensurePreview(info);
 if(window.updateMarkers)window.updateMarkers(previewMode==='result'?[]:Q);}
const statusNames={running:'계산 중',waiting_for_answers:'답변 대기',done:'완료',done_with_failed_checks:'완료 · 검증 실패 있음',failed:'실행 실패·중단',needs_customer:'고객 확인 필요'};
function activity(){const p=latestPlan;const busy=launching||p.process_active===true;const rt=p.runtime||{};
 document.getElementById('activity').classList.toggle('busy',busy);
 document.getElementById('activity-title').textContent=launching?'실행 시작 중…':busy?'계산 중 — 실행 중입니다':(statusNames[p.status]||'실행 대기');
 document.getElementById('stage').textContent=launching?'프로세스를 시작하고 있습니다.':busy?(rt.current_stage||'입력 읽기·초기 진단 중입니다.'):(p.status==='waiting_for_answers'?'아래 질문에 답하면 이어서 실행합니다.':rt.error||'');
 const start=launching?launchAt:rt.started_at*1000;const end=busy?Date.now():rt.ended_at*1000;
 const seconds=start&&end?Math.max(0,Math.floor((end-start)/1000)):null;
 document.getElementById('elapsed').textContent=(seconds==null?'':`경과 시간 ${Math.floor(seconds/60)}분 ${seconds%60}초`)+(busy&&rt.pid?` · PID ${rt.pid}`:'');
 document.getElementById('activity-note').textContent=busy?'로딩바는 실행 상태를 뜻합니다. 전체 완료율(%)은 계산 도구에서 제공하지 않습니다.':'';
 document.querySelectorAll('button.run').forEach(b=>b.disabled=busy||['done','done_with_failed_checks'].includes(p.status));}
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function healingPanel(h){if(!h)return '';const val=v=>v==null?'미보고':v===true?'True':v===false?'False':esc(v);
 const state=h.execution_status==='ok'?'실행 성공':h.execution_status==='failed'?'실행 실패':esc(h.execution_status);
 return '<div class="q"><b>힐링 STEP — '+esc(h.artifact)+'</b><div>실행 상태: heal: '+esc(h.execution_status)+' · '+state+'</div>'
 +[['closed','닫힘'],['valid','B-rep 유효성'],['floating_caps','떠 있는 캡'],['holes_found','발견한 구멍'],['holes_filled','메운 구멍'],['holes_left','남은 구멍'],['free_boundaries_measured','실측 잔여 자유경계']].map(([k,label])=>`<div class="${h[k]===false||(k==='floating_caps'&&h[k]>0)?'bad':''}">${label} (${k}): ${val(h[k])}</div>`).join('')
 +'<small>'+esc(h.note)+'</small></div>';}
async function refresh(){if(refreshing)return;refreshing=true;try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error('상태 조회 실패');const s=await r.json();const p=s.plan||{};
 latestPlan=p;if(p.process_active||Date.now()-launchAt>5000)launching=false;activity();
 document.getElementById('status').textContent=(statusNames[p.status]||p.status||'대기')+(p.route?' · '+p.route:'');
 document.getElementById('input').textContent=p.input||'';
 Q=s.questions||[];const qd=document.getElementById('questions');const signature=JSON.stringify([Q,p.status]);if(signature!==questionSignature){questionSignature=signature;
 if(!Q.length){qd.innerHTML='<div class="empty-card"><span class="empty-icon">✓</span><div><strong>지금은 답할 질문이 없습니다</strong>검사가 끝나고 확인할 항목이 생기면 여기에 표시됩니다.</div></div>'}else{qd.innerHTML=Q.map(q=>{let inp='';
  const cur=(q.answer!=null?q.answer:q.proposal);
  if(q.type==='bool')inp=`<select id="q_${q.id}"><option value="true" ${cur?'selected':''}>예</option><option value="false" ${!cur?'selected':''}>아니오</option></select>`;
  else if(q.type==='choice')inp=`<select id="q_${q.id}">${(q.choices||[]).map(c=>`<option ${c==cur?'selected':''}>${c}</option>`).join('')}</select>`;
  else inp=`<input id="q_${q.id}" type="number" step="any" value="${q.answer!=null?q.answer:q.proposal}"> ${q.unit||''}`;
  const ev=(q.evidence||[]).map(e=>e.match(/\\.png$/)?`<div class="ev"><img src="/files/${esc(e)}"></div>`:`<div><a href="/files/${esc(e)}" target="_blank">${esc(e)}</a></div>`).join('');
  const w=q.where||{};const hasWhere=(w.points&&w.points.length)||(w.plane_z!=null);
  const btn=hasWhere?` <button onclick="focusQ('${q.id}')">위치 보기</button>`:'';
  const live=(w.plane_z!=null&&q.type==='number')?` oninput="planeFromInput('${q.id}')"`:'';
  const done=q.answer!=null?` <span style="color:#197">· 답함 ${esc(q.answer)}</span>`:(q.assumed?' <span style="color:#a70">· 제안값으로 가정</span>':'');
  const titles={units:'파일 단위',length_axis_now:'차량 길이 방향',voxel_mm:'재표면화 복셀 크기',floor_z_mm:'평바닥 높이',half_model:'반쪽 모델 대칭 복원',close_rims:'바퀴 림 닫기',seal_below_mm:'자동 봉합 크기',keep_openings_mm:'유지할 개구부 크기'};
  const descriptions={units:'이 파일의 좌표 단위를 선택하세요. 단위에 따라 실제 차량 크기가 달라집니다.',length_axis_now:'현재 원본에서 차량 앞뒤 방향에 해당하는 축을 선택하세요. x축이 아니면 계산 전에 회전합니다.'};
  const title=titles[q.id]||q.question;const description=descriptions[q.id]||q.question;
  const card=`<div class="q question-card" id="card_${q.id}"><div class="question-top"><b>${esc(title)}</b><span class="question-badge">추천 ${esc(q.proposal)} ${esc(q.unit||'')}</span></div>${done}<p class="question-description">${esc(description)}</p><div class="question-field"><label for="q_${q.id}" style="font-size:12px;color:#475569">선택값</label>${inp.replace('<input ','<input '+live+' ')}${btn}</div><details><summary>질문·추천 근거 자세히 보기</summary><p>${esc(q.question)}</p><p>${esc(q.reason)}</p>${ev}</details></div>`;
  return q.answer!=null||q.assumed||['done','done_with_failed_checks'].includes(p.status)?`<details><summary>확인한 설정 · ${esc(q.id)}: ${esc(q.answer!=null?q.answer:q.proposal)}</summary>${card}</details>`:card}).join('');
 }
 window.Qcache=Q;if(window.updateMarkers)window.updateMarkers(Q);}
 const pd=document.getElementById('plan');const d=p.diagnosis||{};
 const openPlanPanels=new Set([...pd.querySelectorAll('details')].filter(el=>el.open).map(el=>el.querySelector('summary')?.textContent));
 const decisions=p.decisions||[];const seen=new Set();const unique=decisions.filter(x=>{const key=JSON.stringify([x.what,x.because]);if(seen.has(key))return false;seen.add(key);return true;});
 const route=p.route||'';let steps=[];
 if(route.includes('E-wrapcaps'))steps=['CAD 검사·봉합','작은 개구부 수리·STEP 저장','메쉬 생성','바닥 높이 확인','평바닥 생성·차량과 결합','랩 생성·검증','랩 기반 캡을 STEP에 반영','STEP 잔여 경계 검증'];
 else if(route.startsWith('C-flat-floor-wrap'))steps=['바닥 높이 확인','평바닥 생성','차량 메쉬와 결합','랩 생성','랩 수밀·채움률 검증'];
 else if(route==='D-resurface')steps=['복셀 크기 확인','메쉬 정점 병합·봉합','재표면화','수밀·체적 변화·경계상자 검증'];
 else if(route.startsWith('3-wrap-keep-openings'))steps=['유지할 개구부 크기 확인','메쉬 준비','랩 생성'+(route.endsWith('-coarse')?' (거친 알파)':''),'이음매 평활화','랩 형상 검증'];
 else if(route==='A-heal')steps=['CAD 검사·봉합','작은 개구부 수리·STEP 저장','메쉬 생성','남은 CAD 개구부 확인·후속 경로 결정'];
 const facts=[];const shown=v=>v===true?'True':v===false?'False':String(v);
 if(d.triangles!=null)facts.push(['입력 삼각형',Number(d.triangles).toLocaleString()]);
 if(d.bodies!=null)facts.push(['입력 몸체 수',d.bodies]);
 if(d.watertight!=null)facts.push(['입력 수밀 여부',shown(d.watertight)]);
 if(d.boundary_edge_share!=null)facts.push(['입력 경계 모서리 비율',(d.boundary_edge_share*100).toFixed(3)+' %']);
 if(d.underside_coverage!=null)facts.push(['바닥 덮임률',(d.underside_coverage*100).toFixed(1)+' %']);
 if(d.thickness_p5_mm!=null)facts.push(['두께 5% 분위',d.thickness_p5_mm+' mm']);
 const routeRows=list=>list.map((x,i)=>`<div class="route-row"><span class="route-number">${i+1}</span><div><strong>${esc(x.what)}</strong><small>${esc(x.because)}</small></div></div>`).join('');
 const checks=p.checks||[];const files=[...new Set(p.deliverables||[])];
 pd.innerHTML=healingPanel(p.healing_result)
 +'<section class="section-card"><h3>진단 결과</h3>'+(facts.length?facts.map(([name,value])=>`<div class="file-row"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join(''):'<p class="section-caption">초기 검사 결과가 준비되면 표시됩니다.</p>')+'<details class="fold-panel"><summary>전체 진단 데이터 보기</summary><div><pre>'+esc(JSON.stringify(d,null,2))+'</pre></div></details></section>'
 +'<section class="section-card"><h3>처리 경로</h3>'+(steps.length?'<p class="section-caption">선택된 작업 순서입니다. 각 단계의 완료를 뜻하지 않습니다.</p>'+routeRows(steps.map(what=>({what,because:''}))):'<p class="section-caption">아직 수리 경로가 확정되지 않았습니다.'+(route?' 현재 경로 코드: '+esc(route):'')+'</p>')+'</section>'
 +(unique.length?'<details class="fold-panel"><summary>진단 해석·경로 선택 근거</summary><div>'+routeRows(unique)+'</div></details>':'')
 +'<section class="section-card"><h3>검증 기록</h3>'+(checks.length?checks.map(c=>c.name==='STEP 되읽기·경계상자'?'<p class="section-caption">과거 힐링 실행 기록: '+(c.ok?'실행 성공':'실행 실패')+' · 실제 되읽기·경계상자 검사와 별개</p>':`<div class="check-row ${c.ok?'pass':'fail'}"><span class="check-tag">${c.resolved?'해결 기록':c.ok?'통과':'실패'}</span><div><strong>${esc(c.name)}</strong><small>${esc(c.detail)}${c.on_fail&&!c.ok?' · '+esc(c.on_fail):''}</small></div></div>`).join(''):'<p class="section-caption" style="margin-top:12px">아직 검증 결과가 없습니다.</p>')+'</section>'
 +(files.length?'<section class="section-card"><h3>생성된 파일</h3>'+files.map(f=>{const name=f.split(String.fromCharCode(92)).join('/').split('/').pop();return `<div class="file-row"><code>${esc(name)}</code><button onclick="selectPreview('result');document.getElementById('viewer').scrollIntoView({behavior:'smooth',block:'center'})">최종 STL 보기 ↗</button></div>`}).join('')+'</section>':'')
 +((p.assumptions||[]).length?'<section class="section-card"><h3>적용한 가정</h3><p class="section-caption">'+p.assumptions.map(a=>esc(a.id+' = '+a.value)).join(' · ')+'</p></section>':'')
 +(decisions.length>unique.length?'<details class="fold-panel"><summary>전체 결정 기록 · '+decisions.length+'건 (중복 포함)</summary><div>'+routeRows(decisions)+'</div></details>':'');
 pd.querySelectorAll('details').forEach(el=>{el.open=openPlanPanels.has(el.querySelector('summary')?.textContent);});
 document.getElementById('log').textContent=s.log||'';
 previewFiles=s.previews||{};syncPreview();
 const cli=s.cli_help||{};document.getElementById('cli-questions').textContent=cli.questions||'';document.getElementById('cli-answers').textContent=cli.answers||'';document.getElementById('cli-command').textContent=cli.command||'';
 }catch(e){document.getElementById('activity-title').textContent='서버 연결을 확인해 주세요';document.getElementById('activity-note').textContent='plan_ui.py 터미널이 켜져 있는지 확인하세요. 자동으로 다시 확인합니다.';}finally{refreshing=false;}}
function collect(){const a={};for(const q of Q){const el=document.getElementById('q_'+q.id);if(!el)continue;let v=el.value;
 if(q.type==='bool')v=(v==='true');else if(q.type==='number')v=parseFloat(v);a[q.id]=v}return a}
async function submitAnswers(){const a=collect();if(!Object.keys(a).length){alert('답할 질문이 없습니다. 다시 돌리려면 "기본값으로 실행"을 쓰세요.');return}
 await launch('/api/answers',{answers:a})}
async function launch(url,body){launching=true;launchAt=Date.now();activity();try{const r=await fetch(url,{method:'POST',body:JSON.stringify(body)});const s=await r.json();if(!s.started){launching=false;alert(s.why||'이미 실행 중이거나 시작하지 못했습니다.');}await refresh();}catch(e){launching=false;alert('서버에 연결하지 못했습니다.');}activity();}
async function runDefaults(){if(!confirm('미답변 질문에 제안값을 사용하고 이어서 실행합니다. 완료된 단계는 재사용합니다. 계속할까요?'))return;
 await launch('/api/run',{})}
let chatBusy=false;
function fillQuestion(text){const m=document.getElementById('msg');m.value=text;m.focus();}
function add(role,t){document.getElementById('chat-empty')?.remove();const c=document.getElementById('chat');const d=document.createElement('div');d.className='m '+(role==='user'?'u':'a');const label=document.createElement('span');label.className='m-label';label.textContent=role==='user'?'나':'AI 형상 도우미';const content=document.createElement('div');content.textContent=t;d.append(label,content);c.appendChild(d);c.scrollTop=c.scrollHeight;return content;}
async function send(){if(chatBusy)return;const m=document.getElementById('msg');const t=m.value.trim();if(!t)return;chatBusy=true;const button=document.getElementById('chat-send');button.disabled=true;button.textContent='답변 대기…';m.value='';add('user',t);hist.push({role:'user',content:t});
 const pending=add('assistant','');pending.innerHTML='<div class="chat-wait"><span class="chat-dot" aria-hidden="true"></span>작업 기록을 확인하고 있어요…</div>';
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history:hist})});if(!r.ok)throw new Error('HTTP '+r.status);const s=await r.json();if(typeof s.reply!=='string')throw new Error('응답 형식 오류');pending.textContent=s.reply;hist.push({role:'assistant',content:s.reply});
 if(s.suggested){for(const [k,v] of Object.entries(s.suggested)){const el=document.getElementById('q_'+k);if(el)el.value=(typeof v==='boolean')?String(v):v}}
 }catch(e){pending.textContent='답변을 받아오지 못했어요. UI 서버와 AI 연결 상태를 확인하고 다시 보내주세요.';pending.classList.add('bad');m.value=m.value||t;hist.pop();}
 finally{chatBusy=false;button.disabled=false;button.textContent='보내기 ↗';const c=document.getElementById('chat');c.scrollTop=c.scrollHeight;}}
document.getElementById('msg').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!e.isComposing){e.preventDefault();send();}});
refresh();setInterval(refresh,2000);setInterval(activity,1000);
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
let loadedRevision=null,loadingPreview=false;
window.ensurePreview=function(info){if(!info){document.getElementById('preview-state').textContent='미리보기 준비 중 — 생성되면 자동으로 표시합니다.';return;}const revision=info.url+'?v='+encodeURIComponent(info.revision);if(loadingPreview||revision===loadedRevision)return;loadingPreview=true;document.getElementById('preview-state').textContent='미리보기 불러오는 중…';
 loader.load(revision,g=>{if(meshObj){scene.remove(meshObj);meshObj.geometry.dispose();meshObj.material.dispose();}g.computeVertexNormals();
 meshObj=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x8a97a8,metalness:0.1,roughness:0.75,side:THREE.DoubleSide}));scene.add(meshObj);
 g.computeBoundingBox();bbox=g.boundingBox;fit(bbox.getCenter(new THREE.Vector3()),bbox.getSize(new THREE.Vector3()).length()*0.95);loadedRevision=revision;loadingPreview=false;document.getElementById('preview-state').textContent='표시 파일: '+info.name;if(window.updateMarkers)window.updateMarkers(previewMode==='result'?[]:Q);window.ensurePreview(window.previewSelection)},undefined,()=>{loadingPreview=false;document.getElementById('preview-state').textContent='미리보기 로드 실패 — 자동 재시도합니다.';});};
function fit(center,dist){controls.target.copy(center);camera.position.set(center.x-dist*0.9,center.y-dist*1.1,center.z+dist*0.7);camera.lookAt(center);controls.update();}
function pin(n,pos,r){const c=document.createElement('canvas');c.width=c.height=128;const x=c.getContext('2d');
 x.beginPath();x.arc(64,64,58,0,7);x.fillStyle='rgba(255,255,255,0.92)';x.fill();x.lineWidth=8;x.strokeStyle='#d23b2c';x.stroke();
 x.fillStyle='#c0392b';x.font='bold 68px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(String(n),64,68);
 const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,sizeAttenuation:true}));
 const d=bbox?bbox.getSize(new THREE.Vector3()).length():4000;const w=d*0.028;
 sp.position.copy(pos).add(new THREE.Vector3(0,0,Math.min(r,w)*0.6+w*0.6));sp.scale.set(w,w,1);sp.userData.pin=true;return sp;}
const qOf=new Map();
window.updateMarkers=function(Q){markers.clear();qOf.clear();let pz=null;let n=0;window.markerList=[];
 for(const q of Q){const w=q.where||{};const lines=w.lines||[];
  for(let i=0;i<(w.points||[]).length;i++){const p=w.points[i];n++;
  const d=bbox?bbox.getSize(new THREE.Vector3()).length():4000;
  const pos=new THREE.Vector3(p[0],p[1],p[2]);
  if(lines[i]&&lines[i].length>2){
   // the opening's own outline, drawn as a thin tube so it reads at any zoom
   const pts=lines[i].map(a=>new THREE.Vector3(a[0],a[1],a[2]));
   const curve=new THREE.CatmullRomCurve3(pts,true,'catmullrom',0.0);
   const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.min(pts.length*2,400),d*0.0022,6,true),
     new THREE.MeshBasicMaterial({color:0xe03327}));
   tube.userData.n=n;markers.add(tube);qOf.set(tube.uuid,q.id);
   const box=new THREE.Box3().setFromPoints(pts);pos.copy(box.getCenter(new THREE.Vector3()));
   if(window.showPins!==false)markers.add(pin(n,pos,0));
  }else{
   const r=Math.min(Math.max(p[3]||60,d*0.012),d*0.05);
   const m=new THREE.Mesh(new THREE.SphereGeometry(r,20,14),new THREE.MeshStandardMaterial({color:0xe04a3f,transparent:true,opacity:0.38,depthWrite:false}));
   m.position.copy(pos);m.userData.n=n;markers.add(m);qOf.set(m.uuid,q.id);
   if(window.showPins!==false)markers.add(pin(n,pos,r));}
  window.markerList.push({n:n,qid:q.id,text:p[4]||q.id,pos:[pos.x,pos.y,pos.z]});}
  if(w.plane_z!=null&&pz==null)pz=w.plane_z;}
 setPlane(pz);const leg=document.getElementById('legend');
 if(leg)leg.innerHTML=window.markerList.length?window.markerList.map(x=>`<span class="lg" onclick="focusN(${x.n})">${x.n}. ${x.text}</span>`).join(' '):'';}
window.focusN=function(n){const m=markers.children.find(o=>o.isMesh&&o.userData.n===n);if(!m)return;
 const b=new THREE.Box3().setFromObject(m);const c=b.getCenter(new THREE.Vector3());
 fit(c,Math.max(b.getSize(new THREE.Vector3()).length()*2.2,300));}
window.togglePins=function(){window.showPins=(window.showPins===false);if(window.Qcache)window.updateMarkers(window.Qcache);}
function setPlane(z){if(plane){scene.remove(plane);plane=null}if(z==null||!bbox)return;const s=bbox.getSize(new THREE.Vector3());
 plane=new THREE.Mesh(new THREE.PlaneGeometry(s.x*1.1,s.y*1.1),new THREE.MeshBasicMaterial({color:0x2b6cff,transparent:true,opacity:0.35,side:THREE.DoubleSide}));
 const c=bbox.getCenter(new THREE.Vector3());plane.position.set(c.x,c.y,z);scene.add(plane);}
window.planeFromInput=function(qid){const el=document.getElementById('q_'+qid);if(el)setPlane(parseFloat(el.value));};
window.focusQ=function(qid){const q=Q.find(x=>x.id===qid);if(!q)return;const w=q.where||{};const pts=w.points||[];
 if(pts.length){const c=new THREE.Vector3();for(const p of pts)c.add(new THREE.Vector3(p[0],p[1],p[2]));c.multiplyScalar(1/pts.length);const r=Math.max(...pts.map(p=>p[3]||60));fit(c,Math.max(r*6,600));}
 else if(w.plane_z!=null&&bbox){const c=bbox.getCenter(new THREE.Vector3());c.z=w.plane_z;fit(c,bbox.getSize(new THREE.Vector3()).length()*0.5);setPlane(w.plane_z);}
 const card=document.getElementById('card_'+qid);if(card){if(card.parentElement.tagName==='DETAILS')card.parentElement.open=true;card.scrollIntoView({behavior:'smooth',block:'center'});card.style.outline='2px solid #e04a3f';setTimeout(()=>card.style.outline='',2000);}};
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
renderer.domElement.addEventListener('click',e=>{const rc=renderer.domElement.getBoundingClientRect();mouse.x=((e.clientX-rc.left)/rc.width)*2-1;mouse.y=-((e.clientY-rc.top)/rc.height)*2+1;
 ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects(markers.children.filter(o=>o.isMesh));if(hit.length){const qid=qOf.get(hit[0].object.uuid);if(qid)window.focusQ(qid);}});
window.addEventListener('resize',()=>{camera.aspect=W()/Hh();camera.updateProjectionMatrix();renderer.setSize(W(),Hh());});
(function anim(){requestAnimationFrame(anim);controls.update();renderer.render(scene,camera);})();
window.ensurePreview(window.previewSelection);
</script></body></html>"""


def result_stl(plan):
    for value in reversed(plan.get("deliverables", [])):
        path = Path(value)
        candidates = [path] if path.is_absolute() else [HERE.parent / path, DIR / path]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.suffix.lower() == ".stl" and candidate.is_relative_to(DIR) and candidate.is_file():
                return candidate
    return None


def preview_info(path, kind):
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {"name": str(path.relative_to(DIR)), "bytes": stat.st_size,
            "revision": f"{stat.st_mtime_ns}-{stat.st_size}", "url": f"/preview/{kind}"}


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
        if self.path.split("?", 1)[0] in ("/preview/original", "/preview/result"):
            if self.path.split("?", 1)[0] == "/preview/original":
                path = DIR / "viewer.stl"
            else:
                plan_path = DIR / "plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
                path = result_stl(plan)
            if path is None or not path.is_file():
                return self._send(404, "STL not ready", "text/plain")
            return self._send(200, path.read_bytes(), "model/stl")
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/state":
            active = reconcile(DIR)
            plan = json.loads(read_state_text(DIR / "plan.json")) if (DIR / "plan.json").exists() else {}
            summary_path = DIR / "run" / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if "heal" in summary.get("stages", {}):
                        plan["healing_result"] = healing_result(summary)
                except (OSError, ValueError):
                    pass
            qs = json.loads(read_state_text(DIR / "questions.json")) if (DIR / "questions.json").exists() else []
            # the controller empties questions.json when it finishes; keep showing the ones it
            # asked (with the answers it used) so the markers and "위치 보기" stay on the page
            if not qs:
                qs = plan.get("questions", [])
            plan["process_active"] = active
            if STATE["proc"] is not None and STATE["proc"].poll() is None:
                plan["process_active"] = True
            viewer = DIR / "viewer.stl"
            viewer_revision = f"{viewer.stat().st_mtime_ns}-{viewer.stat().st_size}" if viewer.exists() else None
            log = read_state_text(DIR / "plan_log.txt")[-4000:] if (DIR / "plan_log.txt").exists() else ""
            previews = {"original": preview_info(viewer, "original"), "result": preview_info(result_stl(plan), "result")}
            def ps_quote(value):
                return "'" + str(value).replace("'", "''") + "'"
            cli_help = {"questions": str(DIR / "questions.json"), "answers": str(DIR / "answers.json"),
                        "command": "& " + ps_quote(sys.executable) + " " + ps_quote(HERE / "plan_geometry.py")
                        + " --in " + ps_quote(args.src.resolve()) + " --out " + ps_quote(DIR)
                        + " --answers " + ps_quote(DIR / "answers.json") + " --no-render"}
            return self._send(200, json.dumps({"plan": plan, "questions": qs, "log": log, "viewer_revision": viewer_revision, "previews": previews, "cli_help": cli_help}, ensure_ascii=False))
        if self.path.startswith("/files/"):
            rel = urllib.request.unquote(self.path[len("/files/"):].split("?", 1)[0])
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
            return self._send(200, json.dumps({"started": ok, "why": "" if ok else "답이 비었거나 이미 실행 중입니다"}))
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
