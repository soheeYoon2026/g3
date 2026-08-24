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
