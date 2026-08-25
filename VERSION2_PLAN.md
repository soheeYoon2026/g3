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
