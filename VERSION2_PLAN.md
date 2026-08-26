# G3 version2 — 연결 맵과 개발 계획

작성: 2026-08-24
기준 문서: [docs/G3_HANDOVER_2026-08-24.md](docs/G3_HANDOVER_2026-08-24.md) (다운로드 폴더의 `G3_작업_전체_정리.md` 사본)

## 1. 이 폴더는 무엇인가

`../g3`(soheeYoon2026/g3, 브랜치 `fix/unify-surface-preprocessing`)를 히스토리째
로컬 클론한 뒤 `version2` 브랜치를 새로 판 개발 폴더다.

- 원본 g3에 커밋되지 않은 최신 작업 4개를 그대로 가져왔다:
  `README.md`, `scripts/evaluate_domino_v3.py` (수정분),
  `LIBRARIES.md`, `scripts/evaluate_pointnet_domino_split.py` (신규).
- origin은 동일하게 `git@github.com:soheeYoon2026/g3.git`를 가리킨다.
  push하면 원본 저장소의 `version2` 브랜치로 올라간다.
- `data/`, `var/`, `predictions/`는 gitignore 대상이라 빈 폴더로 시작한다.
  v1 산출물이 필요하면 `../g3/data`(441M), `../g3/var`(593M)에서 골라 온다.
  `models/`의 소형 체크포인트/평가 산출물(8.7M)은 복사해 두었다.

## 2. 연결 맵 — 2026-08-24 로컬 기준 확인 결과

### 2.1 서비스 추론 경로 (로컬 develop 브랜치에 이미 머지됨)

```
aox-next (develop)                      aox-django (develop)                    이 저장소
g3-preview-panel.tsx        ──POST──▶   app/g3_inference/views.py   ──POST──▶  aox_g3/service.py (FastAPI)
g3-three-viewer.tsx                     /api/v1/g3/inferences                  /v1/infer  (Bearer 토큰)
                                        · 팀 admin 권한 확인                    · infer_fields 실행
                                        · S3 _private/g3/inference-service.json· Cd/Cl + OOD + preview PNG 응답
                                          에서 GPU 서비스 url/token 조회
```

- **`integration/aox-django/src/app/g3_inference/`는 aox-django develop에 머지된
  코드와 바이트 단위로 동일함을 diff로 확인했다.** 이 저장소가 Django 연동부의
  원본(source of truth)이다.
- Django 설정 이름: `G3_INFERENCE_URL`, `G3_INFERENCE_TOKEN`,
  `G3_INFERENCE_TIMEOUT_SECONDS`, `G3_INFERENCE_CONFIG_BUCKET`,
  `G3_INFERENCE_CONFIG_KEY` (`integration/aox-django/src/config/settings/base.py`).
- 서비스 쪽 환경변수: `G3_MODEL_PATH`, `G3_COEFFICIENT_EXPERT`,
  `G3_API_TOKEN`(또는 `G3_API_TOKEN_FILE`), `G3_MAX_UPLOAD_BYTES`.

### 2.2 전체 Workbench G3 (temp/g3-workbench-transfer)

인수인계 문서의 4개 저장소 커밋 중 로컬에서 확인·확보한 것:

| 저장소 | 문서 커밋 | 로컬 상태 |
|---|---|---|
| aox-next | `21c24cc3` | **origin/temp/g3-workbench-transfer fetch 완료** (해시 일치) |
| aox-workbench | `8e3fda6` | **origin/temp/g3-workbench-transfer fetch 완료** (해시 일치) |
| aox-django-backend | `ce9215e` | 로컬 aox-django의 원격은 `ADRO-DEVEL/aox-django`이고 temp 브랜치가 없음. `aox-django-backend`는 **다른 저장소**로 보이며 미클론 — 확인 필요 |
| aox-next-admin | `003d40c` | 저장소 자체가 로컬에 없음 — 미클론 |

transfer 브랜치가 develop/canary 대비 추가하는 G3 파일:

- aox-next: `g3-surface-viewer.tsx`, `g3-profile-viewer.tsx`,
  `g3-surface-deform.ts(+spec)`, `g3-profile-deform.ts(+spec)`,
  `g3-profile-controls.tsx`, `g3-point-preview.worker.ts`,
  `g3-original-stl-export.ts` — 72 제어점 표면 편집·대칭·Undo·STL 내보내기 UI 전체.
- aox-workbench: `src/features/mesh/ui/g3/G3Workbench.tsx`,
  `G3SurfacePointEditor.tsx` (29파일, +1,467줄).

즉 **로컬 develop에는 "STL 업로드 → Cd/Cl 미리보기"까지만 머지돼 있고,
제어점 편집·ΔCd 비교·추천 UI는 fetch해 둔 transfer 브랜치에 있다.**

### 2.3 로컬에 없는 것 (작업 시 인지)

- 운영 체크포인트 `decoder-30epoch.pt`와 DoMINO 실험 산출물 — G3_TEST
  `/home/ubuntu/g3-v2/var/` (문서 §11). G3_TEST 미커밋 코드 정리가 남은 과제.
- 프로젝트/Celery 비동기 구조와 관리자 화면 백엔드 — `aox-django-backend`,
  `aox-next-admin` (미클론).
- 학습 원천 데이터 — S3 `aoxlabs-stage-static`, `aoxlabs-prod-static`,
  보존용 `s3://strain-bucket-adro`.

## 3. 학습 파이프라인 (이 저장소의 DoMINO v3 체인)

```
run_scheduled_collection.py      자정 수집 진입점 (수집 전용 안전 모드)
  └ collect_s3_training_data.py  AOX 이벤트 수집 → v2 NPZ 생성 → manifest 갱신
prepare_domino_su2_v3.py         SU2 표면 결과 → 유동 정렬 DoMINO 데이터 (Mach/AoA 보존)
build_domino_su2_v3.py           감사 기반 전체 v3 데이터셋 구축 + 물리 검증
build_quality_gated_manifest.py  Cd/Cl 라벨 품질 게이트 병합
split_domino_v3_groups.py        기하 그룹 결정론적 분할
create_group_holdout.py          그룹 격리 train/holdout manifest
finetune_domino_v3.py            SU2 Cd 검증 기반 DoMINO fine-tune (중복 제거 포함)
evaluate_domino_v3.py            사전학습/파인튠 체크포인트 평가  ← 수정 진행 중
evaluate_pointnet_domino_split.py PointNet을 동일 분할로 비교 평가 ← 신규 작성 중
```

전처리 v2 검증 절차는 [docs/G3_PREPROCESSING_V2_RUNBOOK.md](docs/G3_PREPROCESSING_V2_RUNBOOK.md),
야간 수집·카나리 운영은 [docs/G3_NIGHTLY_CANARY.md](docs/G3_NIGHTLY_CANARY.md) 참고.

## 4. 로컬 환경 (이 머신)

- `.venv`: Python 3.9 `--system-site-packages`. **torch 2.8.0+cu128, CUDA 사용
  가능(로컬 GPU)**. 원본 g3 venv에 없던 `fastapi`, `uvicorn`, `python-multipart`,
  `pytest`, `eval_type_backport`를 설치해 **추론 서비스를 로컬에서 띄울 수 있게
  했다.** (`eval_type_backport`는 `service.py`의 `str | None` 문법을 Python 3.9
  에서 평가하기 위해 필요 — G3_TEST처럼 3.10+ 환경이면 불필요)
- 테스트: `.venv/bin/python -m pytest -q` → 27개 통과 확인 (2026-08-24)
- 서비스 `/health` 기동 확인 완료 (2026-08-24, `g3_field_cdcl_v4.pt`로 model_ready)
- 서비스 로컬 기동 (스모크):

  ```bash
  cd /home/adro1234/2026/SU2_work/g3-version2
  G3_API_TOKEN=devtoken G3_MODEL_PATH=models/g3_field_cdcl_v4.pt \
    .venv/bin/uvicorn aox_g3.service:app --port 8005
  curl -s localhost:8005/health
  ```

  주의: `models/`의 로컬 체크포인트(cdcl_v4, g1_v2)는 전처리 v1 시절 산출물이라
  v2 추론 코드와 호환되지 않을 수 있다(런북 명시: v1 체크포인트는 v2 코드에서
  의도적으로 실패). 확실한 스모크는 런북 §3처럼 tiny 체크포인트를 새로 만들거나,
  G3_TEST/S3에서 현행 체크포인트를 받아서 한다.
- Django 연동 로컬 테스트: aox-django에 `G3_INFERENCE_URL=http://127.0.0.1:8005`,
  `G3_INFERENCE_TOKEN=devtoken` 환경변수를 주면 S3 설정 조회 없이 바로 붙는다.

## 5. version2 개발 계획 (인수인계 문서 §10 기반)

| # | 작업 | 어디서 하나 |
|---|---|---|
| 1 | NISMO 재검증 — AI 입력과 G2 원본의 STL 해시·단위·축·유동조건·G2 step 고정 | 이 폴더 (검증 스크립트) + G2 결과 대조 |
| 2 | 현행 체크포인트·전처리 코드 안전 커밋 | 이 폴더 `version2` 브랜치 + G3_TEST 코드 회수 |
| 3 | 같은 제어점의 안쪽·바깥쪽 변형, 여러 이동량 G2 계산 | transfer 브랜치 변형 로직(`g3-surface-deform.ts`) 재사용 + G2 파이프라인 |
| 4 | GT-R 외 차량 계열 원본·변형 쌍 추가 | 수집 파이프라인 (§3 체인) |
| 5 | 절대 Cd MAE + ΔCd MAE + 방향 정확도 통합 평가 | `evaluate_domino_v3.py` 확장 (진행 중인 수정 이어서) |
| 6 | 계열 완전 분리 Test에서 이긴 challenger만 서비스 적용 | `create_group_holdout.py` + 카나리 런북 |

바로 시작할 수 있는 것: #5 (평가 스크립트 수정이 이미 진행 중), #1 (로컬 SU2
결과 `optimization_rans/steady_oneram6` 계열 대조 가능), #2 중 로컬 몫.
G3_TEST 접속이 필요한 것: #2의 서버 코드 회수, 현행 decoder-30epoch.pt 확보.

## 6. 진행 로그

### 2026-08-24 — #5 평가 지표 통합, #1 재검증 도구 (로컬 몫 완료)

**#5: `aox_g3/eval_metrics.py` 신설.** 절대 Cd/Cl MAE + Spearman + **ΔCd/ΔCl
MAE·방향 정확도·순위상관**을 하나의 요약으로 계산한다. 쌍은 두 방식:

- 그룹 유도: split의 `group_id`가 같은 케이스끼리 전 조합 (기본값)
- 명시 쌍: `--pairs pairs.json`,
  형식 `{"pairs": [{"baseline": <run>, "variant": <run>}]}` — G3_TEST의
  변형 실험(27/39/57/84건)을 평가할 때는 이 방식을 쓴다

`evaluate_domino_v3.py`와 `evaluate_pointnet_domino_split.py` 둘 다 이 모듈을
쓰므로 DoMINO/PointNet 비교가 동일한 지표 정의로 나온다. `--direction-tolerance`
로 |ΔCd|가 작은 쌍을 방향 평가에서 제외할 수 있다.

주의: 로컬 `su2_labels_v3` split의 그룹 내 쌍(train 4, test 16개)은 준중복
지오메트리라 true ΔCd가 최대 0.0006 수준이다. 방향 정확도가 의미 있으려면
tolerance를 주거나, 변형 실험의 명시 쌍 manifest로 평가해야 한다.

**#1: `scripts/verify_case_identity.py` 신설.** AI 입력 STL과 G2 케이스의
동일성 리포트: STL sha256 + v3 geometry_digest(manifest와 동일 계산),
경계상자·축·단위 판정, +X 투영 전면적, cfg 유동조건 대조, **history 파일별
step 단위 CD/CL 추적과 특정 Cd 값이 나온 step 탐색**(`--find-cd`).

run_14 실검증 결과: 라벨 su2_cd는 `history.csv` 마지막 step(404)의 값이고
step 320부터 tolerance 안에서 수렴 구간이었다. 즉 현행 라벨 규약 =
"history.csv 마지막 행" (`prepare_domino_su2_v3.read_su2_coefficients`).
NISMO 재검증은 G3_TEST에서 다음처럼 실행한다:

```bash
python scripts/verify_case_identity.py \
  --stl <NISMO AI 입력.stl> --case-dir <NISMO G2 실행 디렉토리> \
  --find-cd 0.3207 --strict --out nismo_identity.json
```

테스트: `tests/test_eval_metrics.py`, `tests/test_verify_case_identity.py`
포함 40개 통과.

**남은 것:** #1의 NISMO 본 실행(G3_TEST 자료 필요), #2의 서버 코드 회수,
#3 이후 (변형 G2 계산, 타 차종 추가, challenger 게이트).

### 2026-08-25 — P1 최종 판정: G2↔LES ΔCd 교차검증 6쌍 완료

프로토콜 (7차례 시행착오 끝의 최종형, [docs/LES_PROTOCOL_NOTES.md](LES_PROTOCOL_NOTES.md)):
G2가 푼 수밀 표면(drivaer_N) · 3-level standard · 샘플 18k · 자유비행 ·
u_inf 0.10 · cs_l2 0.16 · **refine1/2 격자를 baseline에서 동결**(env
`LES_FIX_REFINE_JSON`, 배포 사본 amr_les.py 패치) · rear_wide만 쌍 단위
자체 격자(마이크로 삼각형 bbox핀이 이 형상만 불안정화 — 동료 경고 적중).
런당 cd_std 0.0003~0.0007, baseline 박스 간 4자리 재현.

| 쌍 | G2 ΔCd | LES ΔCd | 방향 |
|---|---:|---:|---|
| front_narrow_5mm | −0.0164 | −0.0097 | ✅ |
| roof_raise_5mm | −0.0046 | −0.0140 | ✅ |
| rear_narrow_10mm | −0.0019 | −0.0350 | ✅ |
| rear_narrow_25mm | +0.0083 | +0.0143 | ✅ |
| roof_lower_10mm | +0.0095 | −0.0262 | ❌ (3구성 부호 재현) |
| rear_wide_25mm | +0.0088 | −0.0029 | ❌ (\|LES\| 작음; v5 −0.0026 재현) |

**판정: 방향 일치 4/6 (67%), 크기 순위상관 0.03 (무상관).**

### 2026-08-26 — challenger-mixed-v1 학습·게이트 결과 (부분 개선, 승격 보류)

학습: SU2 54 + DrivAerML 71 혼합, decoder 모드 30에폭, per-epoch 검증
best-state, LR 1e-4. 1차 시도는 LR 2e-4에서 NaN 전멸(체크포인트 미저장) —
가드 추가 후 재학습에서 **run_1006 단독으로 비유한 loss 유발** 확인(스킵
처리; 다음 라운드 제외 대상). 체크포인트
`var/domino-automotive-runs/challenger-mixed-v1.pt` (G3_TEST).

| 게이트 | 현행 decoder-30epoch | challenger-mixed-v1 | 판정 |
|---|---|---|---|
| 미접촉 차형 63건 MAE / Spearman | 0.2516 / −0.03 | 0.2580 / **+0.11** | ❌ 절벽 미해결 |
| 학습노출 71건 MAE | 0.0334 | **0.0234** | ✅ 개선 |
| SU2 계열격리 test 8건 MAE | 0.0219 | 0.0256 | ≈ 동급 |
| gtr 6쌍 ΔCd 방향 (vs G2) | 4/6 | **5/6** | ✅ 개선 |
| gtr 6쌍 ΔCd MAE | 0.0068 | **0.0051** | ✅ 개선 |
| gtr 6쌍 ΔCd Spearman | 0.49 | **0.60** | ✅ 개선 |

해석: DrivAerML 71건 추가는 **ΔCd(추천 기능) 축을 뚜렷이 개선**했지만,
미접촉-임의형상 일반화(관문 2 절벽)는 못 메웠다 — DrivAer 모프는 "한 계열
추가"일 뿐, AOX 업로드의 임의 분포를 덮지 못함. 승격 게이트(미접촉 개선)
미통과로 **보류**. 참고: challenger는 G2 라벨로 학습된 만큼 분쟁 쌍에서
G2 쪽으로 정렬됨(vs LES 4/6→3/6).

다음 지렛대: ① WindsorML·AhmedML까지 계열 추가 ② 미접촉 63건에서 부품·
비정상 형상을 걸러낸 "완전체 차량" 공정 게이트 재구성 ③ encoder-tail
학습 모드 ④ 실패 시 제품 포지셔닝(OOD 게이트+계열 내 전용) 확정.

### 2026-08-26 — 공정 게이트 재판정: "순위능력 0"은 게이트 오염의 착시

manifest 유동조건으로 재분류하니 reaudit 265건 중 **완전체 자동차 조건
(30m/s·비압축·ref_area 1.2~4.5·AoA 0)은 182건**이고, 기존 "미접촉 63건"
게이트에는 **천음속 날개(run_1: Mach 0.84, 285m/s), 부품, 특수형상이
대량 혼입**돼 있었다. 미접촉 완전체 차 15건의 공정 게이트:

| 모델 | MAE | 상대오차 중앙 | Spearman |
|---|---:|---:|---:|
| pretrained | 0.3453 | 116% | −0.37 |
| **decoder-30epoch (현행)** | **0.1042** | **25%** | **0.86** |
| challenger-mixed-v1 | 0.1792 | 74% | 0.61 |

재판정 3줄:
1. **현행 모델은 처음 보는 완전체 차의 순위를 잘 매긴다(0.86)** — 절대값은
   ~25% 오차(NISMO와 일관). 관문 2의 "순위능력 0" 결론은 비자동차 혼입이
   만든 과잉 진단이었음.
2. **challenger는 공정 게이트에서 현행보다 후퇴** — DrivAerML 혼합이
   AOX-차 특화를 희석(ΔCd 축 개선과 맞바꿈). 승격 보류 유지.
3. 제품 시사점: (a) "신차 상대비교/순위" 기능은 현행 모델로도 유효 수준
   (b) 절대 Cd는 여전히 OOD 게이트 필요 (c) **부품·비차량 업로드를 입력단
   에서 분류·경고하는 게 모델 개선보다 싸고 급함** — 성능 문제의 절반은
   입력 분포 문제였다.

다음 라운드 설계 (혼합 희석 교훈 반영): DrivAerML을 같은 배치에 섞지 말고
**커리큘럼(DrivAerML 선학습 → SU2 전용 마무리)** 또는 도메인 가중 샘플링,
run_1006 제외. 게이트는 이 "완전체 15건"(+G2 신규 수집분)으로 동결.

### 2026-08-26 (2차) — WindsorML 신계열 프로브 + 업로드 게이트

**업로드 게이트 구현** (`aox_g3/upload_gate.py`, 커밋 58fda1a): 기하 규칙만으로
full_car/component/non_car 분류, 실데이터 16/16, `service.py` 응답에
`upload_gate` 경고 필드. YOLO(Ultralytics)는 AGPL이라 배제, 외부 비전 API는
고객 설계 유출 문제로 배제 — 애매 케이스 2차는 CLIP(MIT) 예정.

**WindsorML 프로브 (12건, 차 크기 ×4 스케일)** — 도중 발견·수정 2건:
윈저 필드는 무차원 계수(cpavg/cf*avg)로 저장(×0.5U² 변환 추가), 데시메이션
zero-area 셀이 DoMINO 면적 나눗셈에서 NaN 유발(제거 로직을 인제스트에 반영).
또한 DoMINO 래퍼의 좌표 정규화 박스가 DrivAer 고정 규격이라 모형 스케일
입력은 ×4 스케일업이 필수임을 확인.

| 모델 | Windsor MAE | Windsor Spearman |
|---|---:|---:|
| pretrained | 0.3245 | −0.17 |
| **decoder-30epoch** | **0.0629** | −0.10 |
| challenger-mixed | 0.1472 | −0.31 |

종합 판독: ① 완전 신계열에서도 현행 모델의 **절대 Cd는 0.06 수준으로 선방**
② 그러나 신계열 내부의 미세 변형 순위는 **세 모델 모두 무능**(Spearman ≈ 0)
— "계열 내 순위"는 그 계열 데이터로 파인튠해야만 생기는 능력(ΔCd 실험과
일관) ③ challenger는 신규 축 3연속 후퇴 → **보류 최종 확정**, 다음은
커리큘럼 방식.

1. **G2 ΔCd 라벨은 "방향 라벨"로는 조건부 유효** — 2/3 일치. 단 지붕·후미
   확폭처럼 박리 지배 변형에서 갈리며, 크기는 신뢰 불가.
2. **decoder-30epoch 재판정: G2 기준 4/6, LES 기준 4/6** — 모델의 오답이
   라벨 탓만은 아님(rear_narrow_10은 양쪽 다에게 틀림). 단 roof_lower에서
   모델은 LES 편(G2 라벨이 틀렸을 가능성), rear_wide에선 G2 편.
3. 실무 결론: ΔCd 학습 라벨로 G2 계속 사용 가능(방향 위주, 크기 가중 낮춤),
   **분쟁 쌍은 LES 중재** — 이 6쌍 중 2쌍처럼 솔버가 갈리는 케이스는
   학습에서 제외하거나 LES 값으로 대체. 추천 UX의 G2 검증 게이트 유지.
4. LES 캠페인 인프라(고정격자 A/B, 쌍 단위 격자, cs env)는 재사용 가능한
   형태로 박스에 배포돼 있음 (~/gtr-les/, 결과 runs_v5~v7 보존).

### 2026-08-24 (2차) — G3_TEST 접속: 관문 1 판정 + 서버 자료 회수

**관문 1 (NISMO 규약 검증) 결과 — 오차는 규약 문제가 아니라 진짜 모델 오차:**

- 인용값 "G2 기준 0.3207"의 원본 기록을 **어느 G2 산출물에서도 찾지 못함**.
  실측된 NISMO 계열 풀카 G2 값: run_60(원시 GT-R) su2_cd `0.3025`,
  gtr-smooth baseline 라벨 `0.30790`(history 최종=rbf 소스), 같은 형상 VTU
  적분 `0.31232`(+1.4%), 고항력 변형들의 적분값 `0.3202~0.3209`.
- 규약 차이 정량화: ref_area AI `3.1201`(bbox 전면적) vs G2 cfg `3.104241`
  (**0.5%**); history 라벨 vs VTU 적분 (**+1.4%**); G2 수렴 노이즈 tail std
  `1.5e-5` (무시 가능). 합쳐도 몇 % 수준 —
  **AI `0.4723` vs G2 `0.30~0.31`의 ~50% 오차는 분포 밖(OOD) 모델 오차.**
- 메시 동일성: AI 입력 `gtr-nismo-latest.stl`(419,428 tri)과 캠페인 baseline
  STL(511,548 tri)은 **다른 메시**. 향후 비교는 반드시 같은 메시로.
- G2 스케일 규약 발견: 잡의 `geometry_scale.json` — 소스 mm, 솔버 공간
  최대치수 5.000m 정규화(`solver_units_per_meter 0.99782`).

**보너스 — 실전 ΔCd 쌍 캠페인 발견:** `gtr-smooth` G2 캠페인 7케이스
(baseline + 변형 6종, ΔCd −0.0164~+0.0095, 노이즈 대비 100배 이상) →
[benchmarks/gtr_smooth_pairs.json](benchmarks/gtr_smooth_pairs.json)으로 동결.
`--pairs` 평가의 1호 실데이터이자 P1 캠페인의 기존 절반. 업로드만 되고
계산 안 된 변형 STL 7종이 S3에 대기 중 (P1 확장분).
LOO 파인튠 실험 기록: 5케이스 학습 → 홀드아웃 Cd MAE 0.0119→0.0082.

**#2 서버 자료 회수:** G3_TEST `~/g3-v2`의 미커밋 코드 47파일(서빙 스택,
ΔCd 빌더들, systemd 유닛) + 추적파일 수정 패치(626줄) + 소형 실험 기록을
`recovered/g3-test-20260824/`로 회수. 서버의 `evaluate_domino_v3.py` 수정본은
로컬 버전과 **별도 계보**이므로 병합 전 대조 필요
(`recovered/g3-test-20260824/g3v2-tracked-modifications.patch`).

**다음 실행 순서:** ① gtr-smooth 7케이스에 현행 decoder-30epoch를 돌려
`--pairs` ΔCd 방향 정확도 측정 (서버 GPU, 케이스당 ~1.5s) ② 미계산 변형
7종 G2 제출(P1 확장) ③ LES 파일럿 P0/P1.

### 2026-08-24 (3차) — ΔCd 방향 정확도 첫 실측 (① 완료)

G3_TEST GPU에서 통합 지표로 gtr-smooth 6쌍 평가
([benchmarks/results/gtr_smooth_decoder30_20260824.jsonl](benchmarks/results/gtr_smooth_decoder30_20260824.jsonl),
tolerance 0.001):

| 모델 | 절대 Cd MAE | Cd Spearman | ΔCd MAE | **방향 정확도** | ΔCd Spearman |
|---|---:|---:|---:|---:|---:|
| pretrained (NVIDIA) | 0.2374 | −0.14 | 0.0132 | 3/6 = 50% | −0.20 |
| decoder-30epoch | 0.0121 | 0.43 | 0.0068 | **4/6 = 67%** | 0.49 |

쌍별: 큰 변형일수록 정확 — front_narrow_5mm(trueΔ −0.0164)를 −0.0146으로
거의 재현, rear_wide/rear_narrow_25 방향 정답. 실패 2건: roof_lower_10mm
(true +0.0095 → pred −0.0021), rear_narrow_10mm(최소 변형 −0.0019 → +0.0101).
예측 Cd 범위(0.012)가 실제(0.026)의 절반 — 변화 과소평가 경향.

해석과 한계: ① 6쌍은 표본이 작아 67%는 ±지푸라기 수준 — 13쌍(미계산 7종
제출)으로 확장해야 유의미 ② 평가 변형들은 학습(8/13, reaudit-v1) 이후
생성(8/20)된 미학습 케이스지만, GT-R 계열 자체는 run_60으로 학습에 노출 —
계열 밖 일반화 검증은 아님 ③ 과거 "방향 50%" 보고 대비 개선으로 보이나
동일 세트 재현이 아니므로 직접 비교는 불가.

### 2026-08-26 (3차) — 커리큘럼 v1 판정: 승격 없음, 단 용도별 결론 도출

Stage A(DrivAerML 71, run_1006 복귀, 15ep) → Stage B(SU2 54 전용, 30ep,
base-checkpoint 이어받기). B단계 SU2 검증 0.0109 / 격리 test 0.0358.

| 모델 | 공정15 MAE/순위 | gtr ΔCd 방향/MAE | Windsor MAE/순위 |
|---|---|---|---|
| 현행 decoder-30ep | **0.104 / +0.86** | 4/6 / 0.0068 | **0.063** / −0.10 |
| 혼합 mixed-v1 | 0.179 / +0.61 | **5/6 / 0.0051** | 0.147 / −0.31 |
| 커리큘럼 v1 | 0.150 / **+0.83** | 4/6 / 0.0066 | 0.129 / −0.09 |

판정: ① 커리큘럼은 혼합의 특화 희석을 대부분 회복(순위 0.83)했지만 현행을
넘지 못함 — **절대 Cd/일반화 축 승격 없음, 현행 유지** ② 혼합 v1은 ΔCd
축(5/6, 0.0051)에서 유일하게 현행을 이김 — **추천 전용 expert 후보**로
보존 (v6 다중 expert 구조에 부합; 추천 파이프라인은 ΔCd만 사용) ③ 외부
계열 사전학습은 이 규모(SU2 54)에서 미접촉 개선을 사지 못함 — 공정 게이트의
진짜 지렛대는 **AOX 도메인 완전체 차 수집 확대** (야간 수집기 + 업로드
게이트 연동: full_car 판정 신규 G2 결과만 자동 편입).

### 2026-08-26 (4차) — 추천 전용 expert 배선 (서버 서비스)

`recovered/g3-test-20260824/scripts/domino_stl_service.py`(= G3_TEST 배포본)에
**작업별 모델 라우팅** 추가:

- `G3_DOMINO_RECOMMEND_CHECKPOINT` env로 추천 경로만 다른 체크포인트 사용
  (미설정 시 기존과 완전 동일 — 무변경 배포 확인 `recommend_expert_split:false`)
- 별도 resident engine(`_load_recommend_engine`)으로 두 모델 상주
- **재앵커링**(`_reanchor_to_absolute_model`): expert의 baseline Cd가 미리보기
  (/v1/infer, 서빙 모델)와 어긋나 **같은 차에 두 Cd가 표시되는 문제**를 발견 →
  절대값은 서빙 모델 앵커, ΔCd는 expert 값을 유지하고 후보 절대 Cd를
  `앵커 + ΔCd`로 재계산. 원본은 `expert_cd/expert_cl`로 보존.
- `/health`에 `checkpoint`·`recommend_checkpoint`·`recommend_expert_split`,
  추천 응답에 `absolute_model`·`delta_model` 노출.

검증 (NISMO, shadow 8011 vs 8012 동일 코드):
baseline 0.4723로 통일(expert 자체값 0.3608 보존), 후보 절대 Cd = baseline+ΔCd
일관성 OK, 상위 후보 2개는 두 모델 공통(point_00_out/point_03_out), 3위만 상이.

**프로덕션은 아직 미적용** — env 한 줄(`G3_DOMINO_RECOMMEND_CHECKPOINT=...
challenger-mixed-v1.pt`)을 systemd 유닛에 추가 후 재시작하면 전환되고,
그 줄을 지우면 즉시 롤백된다. 프론트가 `absolute_model`/`delta_model`을
표기하도록 하는 편이 사용자 혼선이 없다.

### 2026-08-26 (5차) — 프로덕션 전환 + 수집기 게이트 연동

**① 추천 expert 프로덕션 적용 완료.** systemd drop-in
`/etc/systemd/system/g3-domino-inference.service.d/recommend-expert.conf`
(env 한 줄) → 라이브 검증: `absolute_model: decoder-30epoch.pt`,
`delta_model: challenger-mixed-v1.pt`, baseline 0.4723(미리보기와 동일),
후보 Cd = baseline + ΔCd. **롤백 = 그 drop-in 파일 삭제 후 재시작.**

**② 수집기 ↔ 업로드 게이트 연동.** `aox_g3/upload_gate.py`에
`classify_case(conditions, geometry)` 추가(유동 레짐까지 판정:
속도 5~80 m/s, Mach ≤ 0.3, |AoA| ≤ 5°). 서버 수집기
`build_domino_s3_v4.py`가 채택 케이스마다 `shape_class`를 기록하고
manifest summary에 분포를 남긴다. `--require-car-case` 플래그를 주면
비차량 케이스를 아예 거부(기본은 태깅만 — 데이터는 남기고 분할에서 거른다).

기존 265건 소급 태깅 결과: **car_case 163 / component 50 /
non_car_shape 21 / unsure 29 / off_regime 2**.

**표준 게이트 동결** (`benchmarks/unseen_car_gate.json`, 13건 =
미접촉 ∩ car_case):

| 모델 | MAE | 상대중앙 | Spearman |
|---|---:|---:|---:|
| **현행 decoder-30ep** | **0.0881** | **24%** | **+0.87** |
| 혼합 mixed-v1 | 0.1639 | 75% | +0.60 |
| 커리큘럼 v1 | 0.1469 | 48% | +0.86 |

이제 challenger 승격 판정은 이 파일의 13건으로 재현 가능하다.

### 2026-08-26 (6차) — 프로덕션 추론에 게이트 노출 + 전면적 정의 수정

프로덕션 `/v1/infer`(domino_stl_service.py) 응답에 `upload_gate` 추가.
Django 프록시는 페이로드를 그대로 전달하므로 프론트는 이 필드만 읽으면
경고를 띄울 수 있다 (`verdict`, `reasons`, `features`).

**전면적 정의 버그 수정**: 초기 구현은 `Σ|n·A|/2`(모든 면의 투영합)이라
휠·언더바디·안쪽 패널이 겹쳐 NISMO를 6.18㎡(실제 3.12㎡)로 부풀렸고
`unsure` 오판을 냈다. **bbox 전면적(width × height)**으로 교체 —
NISMO 3.12㎡로 G2 REF_AREA와 일치, 판정도 `full_car`로 정상화.
실데이터 16/16 정확(차 11, 부품 5), 소급 태깅의 `unsure`가 29→2로 감소.

재태깅 결과 (265건): **car_case 190 / component 40 / non_car_shape 31 /
unsure 2 / off_regime 2**.

**표준 게이트 v2 동결** (`benchmarks/unseen_car_gate.json`, 15건):

| 모델 | MAE | 상대중앙 | Spearman |
|---|---:|---:|---:|
| **현행 decoder-30ep** | **0.0929** | **24%** | **+0.91** |
| 혼합 mixed-v1 | 0.1642 | 54% | +0.65 |
| 커리큘럼 v1 | 0.1362 | 45% | +0.90 |

프론트 작업(aox-next)은 `uploadGate.verdict !== 'full_car'`일 때
계수 옆에 주의 배지를 띄우는 것으로 충분하다 (DRF camelCase 변환 주의).

### 2026-08-26 (7차) — 신규 7쌍 LES 캠페인: 6/7 판정 불가, 방법론 교훈 확보

미계산 변형 7종을 LES로 라벨링해 13쌍으로 확장 시도. 형상은 G2를 안 돌린
Workbench 업로드라 수밀 표면이 없어 ① 정점 용접(열린에지 1,554 잔존)
② 합집합 bbox 고정 ③ standard/18k/free/cs 0.16으로 계산.

| 변형 | LES ΔCd | S/N | 판정 |
|---|---:|---:|---|
| roof_lower_20mm | **−0.0101** | **2.9** | ✅ 유효 |
| front_wide_10mm | −0.0039 | 1.1 | ❌ |
| front_narrow_10mm | −0.0025 | 0.7 | ❌ |
| front_wide_20mm | −0.0024 | 0.7 | ❌ |
| front_narrow_20mm | −0.0008 | 0.2 | ❌ |
| cabin_narrow_5mm | −0.0003 | 0.1 | ❌ |
| roof_lower_5mm | +0.0002 | 0.1 | ❌ |

**13쌍 중 방향 판정에 쓸 수 있는 것은 7쌍**(기존 6 + roof_lower_20mm).
신규 6쌍은 노이즈에 묻혔고, 앞부분 변형은 narrow/wide가 둘 다 항력 감소로
나오는 모순까지 보여 신호가 아님이 분명하다.

원인 두 겹: ① 열린 셸이라 plateau sd가 수밀 형상 대비 **4배**
(0.0035 vs 0.0005) ② bbox 고정이 격자는 통일해 주지만 **기준면적까지
고정**해 narrow/wide 변형의 전면적 감소분이 지워짐 — 지붕 변형만 전면적과
무관해 신호가 살아남았다(5mm 0 → 20mm −0.0101로 크기 단조성도 확인).

**부가 소득**: roof_lower가 세 번째 독립 LES 결과로 G2와 반대 방향을
지지했다(G2 +0.0095 vs LES −0.0262/−0.0101). G2의 지붕 변형 ΔCd 라벨은
학습에서 제외하거나 LES로 대체하는 쪽이 안전하다.

**다음 캠페인 설계 기준** (실측 근거):
1. ΔCd 라벨용 변형은 **노이즈의 3배 이상**(수밀 형상 기준 |ΔCd| ≥ 0.0015,
   열린 셸이면 ≥ 0.010) 크기로 설계한다.
2. 형상은 **G2를 먼저 돌려 수밀 표면(VTU)을 얻은 뒤 LES**에 넣는다 —
   순서를 바꾸면 노이즈가 4배가 된다.
3. 전면적이 변하는 변형은 기준면적 처리를 명시한다(고정 시 형상효과만 측정).

### 2026-08-26 (8차) — 표면장 접근: ΔCp 신호는 실재하고, 학습으로 두 배 개선됨

**동기**: ΔCd 하나로는 미세 변형이 CFD 노이즈에 묻히지만(v8/v9 실패), 표면장에는
신호가 남아 있는지 먼저 측정했다. gtr-smooth 6쌍에서 **변형 부위의 CFD ΔCp가
원거리의 1.27~2.33배**(6/6) — 성긴 G2 메시로도 국소 신호는 살아 있다.
쌍당 관측이 스칼라 1개가 아니라 **셀 4만 개**라 6쌍으로도 통계가 선다.

**신규 평가기** `scripts/evaluate_surface_delta.py` (커밋 29c84b2): 모델이 예측한
ΔCp 장을 CFD ΔCp 장과 비교(변형부 상관·신호비, 원거리는 대조군).

**신규 학습** `challenger-paired-v1`: 쌍 배치(baseline+변형 동시 통과) +
**ΔCp 차이 손실** + CFD ΔCp가 큰 셀에 최대 4배 가중. G2가 변형마다 재메싱해
셀 수가 달라 최근접 중심 매핑으로 대응(첫 시도는 이것 때문에 6쌍 전부 스킵).

| 지표 | 현행 decoder-30ep | **paired-v1** |
|---|---:|---:|
| 변형부 ΔCp 상관 (중앙, 학습쌍) | +0.30 | **+0.64** |
| 모델/CFD 신호비 (front_narrow) | 1.93 / 2.33 | **2.26** / 2.33 |
| 계열격리 test Cd MAE | 0.0219 | **0.0198** |
| 표준게이트15 MAE / 순위 | **0.0929 / +0.91** | 0.0993 / +0.82 |
| Windsor MAE | **0.0629** | 0.0643 |
| gtr ΔCd 방향 / MAE | **5/6 / 0.0051** | 4/6 / 0.0058 |

**판정**: 표면장 국소 신호는 확실히 개선(+0.30→+0.64, 신호비도 CFD에 근접)됐고
학습쌍 밖 지표(계열격리 test)도 좋아졌다. 그러나 **미접촉 게이트에서는 현행과
동급 또는 근소 열세**(0.0993/+0.82 vs 0.0929/+0.91)이고 적분 ΔCd 방향도 4/6로
내려갔다 — 즉 **국소 압력장은 잘 잡게 됐지만 그것이 적분 Cd 개선으로 이어지지
않았다.** 승격 보류, 그러나 방향성은 입증됐다.

**다음**: ① ΔCp 상관을 유지하면서 적분 Cd 손실을 함께 최적화(다중 목적) ②
학습쌍을 늘려 과적합 여지 축소 ③ 평가 지표를 "ΔCp 상관 + ΔCd 방향" 이원으로
동결해 이후 challenger를 판정.

### 2026-08-27 — 다중 목적(ΔCp + 적분 Cd) 학습: 승격 없음, 캠페인 1기 종결

`challenger-multi-v1` = paired-v1의 ΔCp 쌍 손실 + **미분 가능한 적분 Cd 손실**
(스텝당 4,096셀 표본을 N/K로 스케일한 불편추정, `sampled_cd()`), cd_weight 0.5.

| 모델 | 게이트15 MAE/순위 | Windsor MAE | ΔCd 방향/MAE | 변형부 ΔCp 상관 |
|---|---|---|---|---|
| **현행 decoder-30ep** | **0.0929 / +0.91** | **0.0629** | **5/6 / 0.0051** | +0.30 |
| 커리큘럼 v1 | 0.1362 / +0.90 | 0.1288 | 4/6 / 0.0066 | — |
| paired-v1 | 0.0993 / +0.82 | 0.0643 | 4/6 / 0.0058 | **+0.64** |
| multi-v1 | 0.1409 / +0.83 | 0.1598 | 5/6 / 0.0055 | +0.48 |

**판정**: multi-v1은 ΔCd 방향을 5/6으로 회복했지만 **미접촉 일반화가 크게
악화**(Windsor 0.0629→0.1598, 게이트 0.0929→0.1409)됐다. 학습 검증은 역대
최고(Cd MAE 0.0054)인데 계열격리 test는 0.0352 — 전형적 과적합이다.
적분 Cd 손실이 학습 계열의 절대값에 모델을 끌어당겼다.

**캠페인 1기 결론**: challenger 4종(혼합·커리큘럼·paired·multi) 모두
표준 게이트에서 현행을 넘지 못했다. **서비스 모델은 decoder-30epoch 유지**,
추천 경로만 mixed-v1 expert(프로덕션 적용 완료).

**얻은 것**: ① 표면장 ΔCp가 미세 변형의 실측 가능한 신호원임을 증명
(CFD 신호비 1.3~2.3배, 6/6) ② 그 신호를 학습으로 두 배 강화 가능
(+0.30→+0.64) ③ 그러나 **국소 정확도 → 적분 Cd → 일반화**로 이어지지
않으며, 적분 손실을 직접 넣으면 오히려 과적합된다는 것.

**다음 1순위**: 학습 데이터 자체를 늘리는 것 — 보유 형상 109대/78계열이
확인됐고(로컬 전용 스캔), G2 선행 → 수밀 표면 → LES 검증 파이프라인이
갖춰졌다. 손실 설계로는 더 짜낼 게 없다는 것이 오늘의 결론이다.
