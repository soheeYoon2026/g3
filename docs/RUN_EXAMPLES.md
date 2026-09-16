# 실행 예시 — AI 통제기와 개별 명령

2026-09-16 기준. 여기 있는 명령은 전부 실제로 돌려 본 것이다. 옵션은 각 파일 `--help` 가 정본이다.

환경은 저장소 안 가상환경 하나다.

```bash
cd ~/2026/SU2_work/g3-version2
.venv/bin/python --version        # 3.9
```

---

## 1. AI 통제기로 한 번에 — 제일 흔한 길

통제기가 재고, 경로를 고르고, 도구를 돌리고, 검증하고, 사람이 정할 것에서 멈춘다.

### 1.1 질문 없이 끝까지 (제안값을 그대로 씀)

```bash
.venv/bin/python scripts/plan_geometry.py \
    --in ~/다운로드/CAS-A.stp \
    --out var/runs/plan-casa \
    --assume-defaults
```

끝나면 `var/runs/plan-casa/` 에 이렇게 남는다.

| 파일 | 내용 |
|---|---|
| `plan.json` | 진단·경로·결정·검증·가정·산출물 전부 |
| `questions.json` | 사람이 답해야 할 것 (없으면 `[]`) |
| `steps.json` | 끝난 단계와 입력 해시 (재실행 시 건너뛰기용) |
| `plan_log.txt` | 사람이 읽는 실행 기록 |
| `viewer.stl` | 화면용 경량 메쉬 |

### 1.2 멈춘 뒤 답을 주고 이어서

답이 없으면 종료코드 3 으로 멈춘다.

```bash
.venv/bin/python scripts/plan_geometry.py --in ~/다운로드/GTR35.stl --out var/runs/plan-gtr
# [정지] 고객 답이 필요한 질문 2개 → var/runs/plan-gtr/questions.json

cat > var/runs/plan-gtr/answers.json <<'JSON'
{ "units": "inch", "length_axis_now": "y", "keep_openings_mm": 13.0,
  "floor_z_mm": 201.5, "accept_coarse_29mm": "accept" }
JSON

.venv/bin/python scripts/plan_geometry.py --in ~/다운로드/GTR35.stl \
    --out var/runs/plan-gtr --answers var/runs/plan-gtr/answers.json
```

### 1.3 자주 쓰는 손잡이

```bash
--time-budget 1800     # 한 단계가 이 초를 넘으면 중단하고 거친 경로로 후퇴 (기본 1500)
--max-retries 2        # 검증 실패 시 정해진 손잡이로 재시도할 횟수
--no-render            # 근접 그림을 만들지 않음 (빠름)
--force                # 장부를 무시하고 모든 단계를 다시 실행
```

**종료코드**: 0 끝남 · 2 실행 실패 · 3 답 기다리는 중.

---

## 2. 질문 화면 + 대화

```bash
.venv/bin/python scripts/plan_ui.py --in ~/다운로드/CAS-A.stp --dir var/runs/plan-casa --port 8765
# http://127.0.0.1:8765
```

3D 뷰어에 구멍 테두리가 그려지고, 질문마다 제안값·이유·근거 그림이 붙는다. 오른쪽 대화창의 모델은 **읽기 전용 측정 기능을 직접 부를 수 있다**.

```bash
--api openai|prime|local|auto   # 기본 auto: LOCAL_LLM_URL → OPENAI_API_KEY → ~/.prime/config.json
--model gpt-5.6-terra           # OpenAI 직접일 때. Prime 이면 openai/… 접두사
--base-url http://127.0.0.1:11500/v1   # 로컬 ollama 등
--knowledge docs                # 기술문서를 대화에 싣기 (OpenAI 연결이면 기본 켜짐)
--no-docs                       # 문서 빼기
--no-tools                      # 모델의 측정 기능 호출 끄기
--max-tool-calls 6              # 한 메시지에 허용할 측정 횟수
--no-chat                       # 모델 없이 화면만
```

키는 대화에 붙여넣지 말고 파일로 둔다. `~/.config/openai/key` (권한 600). `OPENAI_API_KEY=...` 형식도 읽는다.

로컬 모델로 돌리려면:

```bash
var/llm/bin/ollama serve &                       # OLLAMA_MODELS, OLLAMA_HOST 지정
.venv/bin/python scripts/plan_ui.py --in X.stl --dir var/runs/x \
    --api local --base-url http://127.0.0.1:11500/v1 --model gpt-oss:20b
```

---

## 3. 측정만 하기 (읽기 전용, AI 가 부르는 것과 같은 기능)

```bash
.venv/bin/python scripts/geometry_tools.py --list        # 기능과 인자 규격

.venv/bin/python scripts/geometry_tools.py --call measure_mesh \
    --args '{"path": "var/runs/plan-gtr/sharp.stl"}'

.venv/bin/python scripts/geometry_tools.py --call list_openings \
    --args '{"path": "~/다운로드/GTR35.stl", "top": 10}'

.venv/bin/python scripts/geometry_tools.py --call compare_to_reference \
    --args '{"candidate": "var/runs/plan-gtr/sharp.stl",
             "reference": "var/runs/plan-gtr/input_mm.stl"}'

.venv/bin/python scripts/geometry_tools.py --call closed_openings \
    --args '{"candidate": "~/다운로드/car5_outer-wrapped-local-smoothed.stl",
             "reference": "~/2026/SU2_work/SU2/TestCases/optimization_rans/steady_oneram6/car5_outer.stl",
             "keep_above_mm": 13.0}'

.venv/bin/python scripts/geometry_tools.py --call section_profile \
    --args '{"reference": "var/runs/plan-gtr/input_mm.stl",
             "candidate": "var/runs/plan-gtr/sharp.stl", "axis": "x", "value_mm": -1367}'
```

`compare_to_reference` 는 좌표계·단위·단계가 다르면 `frame_warning` 을 함께 낸다. 경고가 비어 있어도 "확인됨" 이 아니라 "이 세 갈래로는 안 걸린다" 로 읽는다.

---

## 4. 개별 도구 (통제기가 부르는 것들)

### 4.1 닫지 않는 재표면화 — 구멍을 하나도 안 막고 수밀화

```bash
.venv/bin/python scripts/prepare_geometry.py --in X.stl --out var/runs/x/run \
    --no-mirror --resurface 2.0 --no-render
# 또는 단독으로
.venv/bin/python scripts/resurface_noclose.py --in X.stl --out out.stl --voxel 2.0
```

복셀은 가장 얇은 부위의 절반 이하. 닫힌 입력이면 오프셋 0, 열린 판이면 오프셋 = 복셀(껍질).

### 4.2 구멍을 지키는 랩

```bash
.venv/bin/python scripts/prepare_geometry.py --in car5_outer.stl --out var/runs/car5 \
    --wrap --keep-openings-above 13 --force-wrap --smooth-seams
```

`--keep-openings-above 13` 이 알파를 6.5 mm 로 정한다(지킬 구멍의 절반). 이미 수밀한 입력에는 `--force-wrap` 이 필요하다.

### 4.3 국소 재랩 — 지정한 틈 주변만 잘게

```bash
.venv/bin/python scripts/local_wrap.py --reference car5_outer.stl \
    --wrap var/runs/car5/wrap.stl --out local.stl --keep-above 13 --alpha 6.5
```

### 4.4 와인딩 수 등위면 — 알파 없는 위상 단계

```bash
.venv/bin/python scripts/winding_isosurface.py --in body_with_floor.stl \
    --out iso.stl --voxel 8 --report iso.json
```

큰 열림(바닥)은 먼저 막을 것. **방향 보정은 쓰지 말 것**(`--orient igl` 이 GT-R 바퀴를 지웠다).

### 4.5 거친 면을 선명하게 — 재메쉬 · 원본 투영 · 가는 랩

```bash
.venv/bin/python scripts/wrap_project_rewrap.py \
    --wrap var/runs/plan-gtr/assumed29/wrapped.stl \
    --reference var/runs/plan-gtr/input_mm.stl \
    --out sharp.stl --edge 10 --max-move 13 --fine-alpha 6.5 --report sharp.json
```

marching cubes 결과를 넣을 때는 `--edge 0` 으로 재메쉬를 건너뛴다.

### 4.6 평바닥 가정 + 랩

```bash
.venv/bin/python scripts/flat_floor_wrap.py --in X.stl --out var/runs/x/floor \
    --floor-z 150.2 --alpha-div 360
.venv/bin/python scripts/flat_floor_wrap.py --in X.stl --out var/runs/x/scan --no-wrap   # 높이만 훑기
```

### 4.7 STEP 에 랩 형상 캡 씌우기

```bash
.venv/bin/python scripts/close_with_wrap.py --step healed.stp --wrap wrapped.stl \
    --out healed_wrapcaps.stp --report wrapcaps.json --floor-z 150 --sew 0.5
.venv/bin/python scripts/close_with_wrap.py --step healed.stp --wrap wrapped.stl \
    --out /dev/null --list-only        # 어떤 고리가 있는지만 보기
```

---

## 5. 그림

```bash
# 덧댄 면(틈을 건너뛴 자리)을 빨갛게 칠한 네 방향 + 근접
.venv/bin/python scripts/render_gap_faces.py --reference input_mm.stl \
    --candidate sharp.stl --out figures/gaps.png --closeups 6

# 단면에서 붙은 구간(회색) 대 건너뛴 구간(빨강)
.venv/bin/python scripts/section_gaps.py --reference input_mm.stl --candidate sharp.stl \
    --out figures/sections.png --cols 3 \
    --cut "y=0,-2500,2500,100,1400;가운데 세로" \
    --cut "x=-1367,-1050,1050,100,900;뒤 휠하우스"

# 형상만 네 방향
.venv/bin/python scripts/render_geometry.py --step sharp.stl --out figures/body.png --no-holes

# 지난 작업의 질문에 구멍 테두리를 채워 넣기 (화면용)
.venv/bin/python scripts/mark_open_loops.py --dir var/runs/plan-gtr --mesh var/runs/plan-gtr/input_mm.stl --top 12
```

---

## 6. 시험

```bash
.venv/bin/python scripts/regression_check.py                    # 7케이스 22검사, 70초
.venv/bin/python scripts/regression_check.py --only gtr-sharpen
.venv/bin/python scripts/regression_check.py --update           # 기대값을 이번 측정으로 갱신
.venv/bin/python -m pytest tests/test_frame_guard.py -q         # 좌표계 검사 음성 대조군
```

파일이 없으면 **실패**로 끝난다(`--allow-missing` 으로 완화). 잰 것이 없어도 실패다.

---

## 7. 실제로 돌린 세 사례

| 형상 | 명령 | 결과 |
|---|---|---|
| car5 (포뮬러형, 구멍 보존) | `plan_geometry --assume-defaults` | 경로 D 재표면화, 복셀 1.9 mm, 73만 면, 수밀, 이탈 0.06 mm, 1.5분 |
| CAS-A (STEP, 반쪽) | `plan_geometry --assume-defaults` | 봉합 → 평바닥 z=150.2 → 랩 캡 STEP, 남은 고리 = 대칭면 + 바퀴 4, 4분 |
| GT-R (스캔 STL, 바닥 없음) | 답 5개 준 뒤 재개 | 15 mm 랩이 새서 평바닥 29 mm → 선명화 219만 면, 이탈 p50 0.22 mm |
| CAS-A 합본 25부품 (LES 의뢰) | `plan_geometry --assume-defaults` | 경로 D, 복셀 3 mm, 열린 고리 14 → 0, 몸체 5 유지, 체적 +0.01 %, **2분** |

---

## 8. 자주 걸리는 것

- **이미 수밀한 STL에 랩을 걸 때** `--force-wrap` 을 빼면 랩 층이 건너뛰고 `unchanged` 로 끝난다.
- **음수 인자**는 `--floor-z=-30` 처럼 `=` 로 붙여 쓴다.
- **통제기 출력 폴더를 다시 입력으로 주지 말 것**. 단위·축 변환이 두 번 걸린다(지금은 감지해서 건너뛴다).
- **화면에서 빈 답으로 실행**은 막혀 있다. 다시 돌리려면 "기본값으로 실행" 을 쓴다(확인을 묻는다).
- `pkill -f` 로 프로세스를 죽일 때 패턴이 자기 명령줄과 겹치면 **셸이 같이 죽는다**. pid 로 죽일 것.
