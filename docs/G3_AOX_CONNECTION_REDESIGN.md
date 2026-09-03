# G3 ↔ AOX 연결 재설계 — 작업 인수 문서

작성: 2026-09-03. 대상: aox-django / 인프라 / g3 서비스 담당.
목적: 지금의 "임시 터널 + S3 우편함 + 토큰 파일" 연결을 고정 주소·관리형 시크릿·헬스 알림 구조로 바꾸고, 형상 전처리용 잡 API를 열 준비.

---

## 1. 지금 구조 (확인된 사실)

```
soheeYoon2026/g3 (GPU 서버 /home/ubuntu/g3)
  aox_g3/service.py ─ uvicorn 127.0.0.1:8003
      엔드포인트: GET /health, POST /v1/infer   (형상 전처리 API 없음)
      인증: Authorization: Bearer <token>   (service.py:23-40)
      토큰 원본: /home/ubuntu/g3/.service-token
                 (deploy/g3-inference.service 의 G3_API_TOKEN_FILE; 대체 env G3_API_TOKEN)
        │
  deploy/g3-tunnel.service → scripts/run_service_tunnel.py
      cloudflared *quick tunnel* (trycloudflare.com, 재시작마다 URL 무작위)
      {url, token} 을 S3 에 기록:
        s3://adro-dev-us-east-1-static/_private/g3/inference-service.json
        │
aox-django  src/app/g3_inference/views.py:33-56  _service_config()
      1) env G3_INFERENCE_URL + G3_INFERENCE_TOKEN 있으면 사용
      2) 없으면 S3 위 파일에서 {url, token} 읽음 (버킷: G3_INFERENCE_CONFIG_BUCKET 또는 AWS_S3_BUCKET, 60초 캐시)
      3) 둘 다 없으면 ("", "") → 사용자에겐 추론 불가
      설정 이름: src/config/settings/base.py:21-25
        │
aox-next  src/app/_source/components/projects/g3-preview-panel.tsx
```

- aox-django 로컬 저장소에는 `.env` 파일이 없음 (`docker-compose.local.yml`만). 운영 환경변수는 배포 쪽에 있음.
- g3 저장소 `integration/aox-django`, `integration/aox-next` 사본은 실제 플랫폼 파일과 동일(5개 파일 diff 없음).
- **2026-09-03 시점 S3 설정 파일 404** → 터널이 떠 있지 않으면 플랫폼 추론 기능이 소리 없이 죽음. 알림 없음.

## 2. 왜 이렇게 됐나

- aox-django `5b59913` (2026-07-10) "proxy realtime GPU inference" — S3 폴백 도입
- g3 `afbbfe8` (2026-08-04) "establish G3 baseline" — 터널 스크립트 포함, 근거 기록 없음

제약: 켜고 끄는 EC2(IP 유동) · AOX 백엔드와 다른 리전/네트워크 · 공유 시크릿 저장소 없음.
→ 인프라를 안 건드리고 뚫는 가장 빠른 조합이었고, 프로토타입 전제("한 사람이 서버 켜고 터널 띄우고 프리뷰 본다")가 굳어짐.

## 3. 문제

| # | 문제 | 결과 |
|---|---|---|
| 1 | URL이 재시작마다 바뀌고 S3로만 전달 | 파일 없으면 조용히 죽음(오늘 404) |
| 2 | Bearer 토큰이 S3 평문 | 버킷 읽기 권한 = 토큰 노출, 회전 수동 |
| 3 | 요청 경로에 S3 조회 | S3 장애/지연 = 추론 장애 |
| 4 | 공개 인터넷 경유 | BAIC 형상(기밀) 전송 불가 |
| 5 | 동기 `/v1/infer`만 | 1~5분 걸리는 형상 전처리에 부적합 |
| 6 | 헬스/알림 없음 | 장애를 사용자가 먼저 발견 |

## 4. 목표 구조

```
aox-django ──(고정 주소, 사설 경로)──▶ g3 서비스
   env: G3_INFERENCE_URL=https://g3.<internal>   (배포 시크릿)
   token: SSM SecureString /aox/g3/inference-token  (기동 시 읽음)
   /api/g3/health → 서비스 /health 프록시 + 실패 알림
   /v1/prepare 잡: Celery 태스크가 제출·폴링, 결과는 사설 S3(SSE-KMS)
```

**주소 고정 — 둘 중 하나 (선행 확인 필요: aox-django 리전/VPC)**
- A. 둘 다 AWS: 같은 VPC 또는 피어링 + 내부 DNS(`g3.internal`) + 보안그룹으로 aox-django만 허용. 공개 인터넷 안 탐. **BAIC 형상 요건에 맞음 → 권장.**
- B. 다른 네트워크: cloudflared **named tunnel**(고정 호스트명) + Cloudflare Access 서비스 토큰. 계정 필요, 인터넷 경유(엣지 암호화).

**시크릿**: `.service-token` 파일 → SSM Parameter Store SecureString `/aox/g3/inference-token`. g3 `service.py`와 aox-django 양쪽이 기동 시 읽음. S3 평문 제거.

**폴백**: S3 발견 코드는 제거하거나, 남기면 실패 시 알림이 나가게.

## 5. 저장소별 작업

### g3 (GPU 서버 서비스)
- [ ] `aox_g3/service.py:23-27` 토큰 로드에 SSM 경로 추가(`G3_API_TOKEN_SSM=/aox/g3/inference-token`), 파일/ENV는 폴백
- [ ] `deploy/g3-inference.service` 환경 갱신
- [ ] `deploy/g3-tunnel.service` + `scripts/run_service_tunnel.py`: A안이면 제거, B안이면 named tunnel로 교체(URL 고정이면 S3 기록 불필요)
- [ ] `/health`에 모델·버전·커밋 해시 포함(배포 확인용)
- [ ] `/v1/prepare` 잡 API (아래 6절)
- [ ] GPU 서버에 version2 배포 + `pip install cgal` (형상 전처리 의존성)

### aox-django
- [ ] 배포 시크릿에 `G3_INFERENCE_URL` 고정값, `G3_INFERENCE_TOKEN`은 SSM에서 주입
- [ ] `g3_inference/views.py:37-56` S3 폴백 제거 또는 실패 알림
- [ ] `GET /api/g3/health` 추가(서비스 `/health` 프록시, 타임아웃 3초)
- [ ] `g3_prepare` 앱: 업로드 → Celery 태스크 → 서비스 `/v1/prepare` 제출/폴링 → 결과 저장·조회 뷰
- [ ] 업로드 파일은 사설 S3 접두어 + SSE-KMS, 보존 기간 정책

### 인프라
- [ ] aox-django 리전/VPC 확인 → A/B 결정
- [ ] A안: VPC 피어링/내부 DNS/보안그룹  ·  B안: Cloudflare 계정·named tunnel·Access
- [ ] SSM 파라미터 생성, 양쪽 IAM 역할에 `ssm:GetParameter` + KMS 복호화
- [ ] 헬스 실패 알림 채널(Slack 등)

## 6. `/v1/prepare` API 초안

```
POST /v1/prepare            multipart: file (STEP | STL), params (JSON, 선택)
  params: { seal_below?, close_near?: [[x,y,z,r]], mirror?: bool, auto?: bool, wrap?: bool }
  → 202 { job_id }

GET  /v1/prepare/{job_id}
  → { status: queued|running|done|failed, stage, progress, error? }

GET  /v1/prepare/{job_id}/result
  → { summary, intent: [ {size, centre, length, reason} ], params_used,
      files: { healed_step, mesh_stl, mesh_full_stl, render_step_png, render_mesh_png, wrap_stl? } }
      (파일은 사설 S3 presigned URL 또는 서비스 내부 경로)
```

서버 쪽 구현은 `scripts/prepare_geometry.py`를 그대로 잡 러너로 감싸면 됨(단계별 보고서·summary.json·intent.md 이미 생성).
`--auto`는 `aox_g3/autotune.py`가 파라미터를 제안하고 `params.json`에 근거를 남김.

## 7. 확인·결정 필요

1. **aox-django가 어느 리전/VPC에서 도는가** (A/B 결정의 유일한 선행 조건)
2. 형상 파일 보존 기간과 삭제 정책 (BAIC 기밀)
3. `/v1/prepare` 결과를 사용자에게 어떻게 보여줄지 — 특히 `intent` 항목(휠·언더바디 질문)이 UI에 나가야 의미가 있음
4. 기존 S3 폴백을 완전히 제거할지, 알림 붙여 남길지

## 8. 참고 위치

- 형상 파이프라인 사용법: `docs/GEOMETRY_PIPELINE.md`
- 설계 근거·측정 기록: `VERSION2_PLAN.md`
- 인수인계(8월): `docs/G3_HANDOVER_2026-08-24.md` 7·8절
- 플랫폼 G3 작업 브랜치: `temp/g3-workbench-transfer` (aox-django `ce9215e`)
