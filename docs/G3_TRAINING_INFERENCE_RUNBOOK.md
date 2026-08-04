# G3 학습·추론 운영 Runbook

마지막 갱신: 2026-08-03

기준 장비: 서울 GPU EC2 (`G4_TEST.pem`)

원격 작업 경로: `/home/ubuntu/g3`

## 1. 운영 원칙

- `G4_TEST.pem`은 EC2 접속 키이며 G4 solver/expert 이름과는 무관하다.
- 최종 차량 형상은 G3 결과만으로 확정하지 않고 G1/G2/G4 CFD로 재검증한다.
- 학습 산출물은 먼저 `challenger.pt`로 검증하고, gate를 통과하기 전에는 `models/registry/production.pt`를 변경하지 않는다.
- `ref_length`는 m, `ref_area`는 m² 기준으로 통일한다.
- XLB/Fidelity가 GPU를 크게 사용하는 동안에는 학습을 같이 실행하지 않는다.

## 2. 서버 접속과 상태 확인

```bash
ssh -i /home/adro1234/work/XLB/examples/car_shape_opt3/G4_TEST.pem \
  ubuntu@3.38.148.230

cd /home/ubuntu/g3
```

GPU, 메모리, 디스크를 확인한다.

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
free -h
df -h /
```

운영 서비스 상태를 확인한다.

```bash
systemctl status g3-inference g3-challenger g3-nightly.timer --no-pager
curl http://127.0.0.1:8003/health
```

## 3. 추론

### 3.1 직접 CLI 추론

Cd/Cl과 flow 산출물을 디스크에 모두 남길 때 사용한다.

```bash
/home/ubuntu/venv_g3/bin/python -m aox_g3.infer_fields \
  --stl /path/to/car.stl \
  --model models/registry/production.pt \
  --out-dir var/inference/car-test \
  --grid 96 64 48 \
  --u-x 30 \
  --density 1.225 \
  --viscosity 1.7894e-5 \
  --temperature 288.15 \
  --ref-length 5.0 \
  --ref-area 1.0 \
  --coefficient-expert g2_su2_clean \
  --device cuda \
  --png
```

Coefficient expert:

| Expert | 용도 | 현재 상태 |
|---|---|---|
| `g2_su2_clean` | 일반 G2/SU2 차량 | 기본 expert |
| `g2_su2_high_drag` | high-drag G2 영역 | experimental |
| `g1_openfoam` | G1/OpenFOAM 라벨 영역 | validated policy |
| `g4_lbm` | G4/XLB-LBM 라벨 영역 | experimental |

생성 파일:

```text
prediction.json
flow.vti
volume_field.vti
surface_pressure.vtp
streamlines.vtp
pressure_streamlines.png
vtp/manifest.json
```

```bash
jq . var/inference/car-test/prediction.json
```

G4 expert를 선택해도 flow 파일은 생성된다. 다만 현재 G4 데이터는 Cd 지도가 중심이며, volumetric flow는 G2 shared backbone에서 전이된 결과이므로 G4 CFD flow 정답으로 별도 검증해야 한다.

### 3.2 Challenger 직접 추론

```bash
/home/ubuntu/venv_g3/bin/python -m aox_g3.infer_fields \
  --stl /path/to/car.stl \
  --model /home/ubuntu/g3/var/retrain/20260803-corrected/challenger.pt \
  --out-dir /home/ubuntu/g3/var/retrain/20260803-corrected/test-inference \
  --grid 96 64 48 \
  --u-x 30 \
  --ref-length 5 \
  --ref-area 1 \
  --coefficient-expert g2_su2_clean \
  --device cuda \
  --png
```

### 3.3 운영 API smoke test

운영 API는 localhost `8003`, shadow challenger는 `8004`를 사용한다. 토큰을 출력하지 않도록 제공된 script를 사용한다.

```bash
/home/ubuntu/venv_g3/bin/python scripts/smoke_service_v6.py \
  --url http://127.0.0.1:8003 \
  --token-file /home/ubuntu/g3/.service-token \
  --stl /path/to/car.stl \
  --expected-model g3_field_g2_v6_final.pt
```

API 응답은 Cd/Cl, OOD, 압력·속도 범위와 선택적 PNG preview를 반환한다. API의 임시 작업 디렉터리는 요청 후 삭제되므로 `.vti`/`.vtp` 파일을 보관하려면 CLI 추론을 사용한다.

## 4. 학습

### 4.1 현재 권장 방식

직전 실험의 flow backbone은 고정 G2 비교에서 Cp/velocity 오차가 개선되었다. 따라서 backbone 250 epoch를 다시 학습하지 말고, 일단 해당 backbone을 재사용해 coefficient expert만 재학습한다.

재학습 전에 다음 manifest가 준비되어야 한다.

```text
g2-normal.json
g2-high-drag.json
g1-training.json
g1-lift-training.json
g4-training.json
```

필수 데이터 조건:

1. G1은 기존 lift manifest의 holdout 3건을 제외한 train 8건과 신규 lift 2건을 함께 사용한다.
2. G1 Cd와 Cl은 각각 validation 샘플이 반드시 존재하도록 split한다.
3. G2 coefficient expert는 한 optimization project의 여러 design state가 과도하게 가중되지 않도록 group 균형을 적용한다.
4. G4 STL은 m/mm를 통일한 뒤 `ref_length`를 m 단위로 재생성한다.
5. 운영 모델의 기존 validation과 별개의 golden holdout을 만든다.

### 4.2 Coefficient expert 재학습

아래 run directory에 수정된 manifest가 있다고 가정한다.

```bash
export G3_RUN_DIR=/home/ubuntu/g3/var/retrain/20260803-corrected
mkdir -p "$G3_RUN_DIR"

/home/ubuntu/venv_g3/bin/python -m aox_g3.train_coefficient_experts \
  --base /home/ubuntu/g3/var/manual/20260731-csv/backbone.pt \
  --out "$G3_RUN_DIR/experts.pt" \
  --expert g2_su2_clean "$G3_RUN_DIR/g2-normal.json" \
  --expert g2_su2_high_drag "$G3_RUN_DIR/g2-high-drag.json" \
  --expert g1_openfoam "$G3_RUN_DIR/g1-training.json" \
  --expert g1_openfoam "$G3_RUN_DIR/g1-lift-training.json" \
  --expert g4_lbm "$G3_RUN_DIR/g4-training.json" \
  --epochs 300 \
  --device cuda
```

```bash
/home/ubuntu/venv_g3/bin/python scripts/annotate_expert_policy.py \
  --checkpoint "$G3_RUN_DIR/experts.pt" \
  --out "$G3_RUN_DIR/challenger.pt"
```

예상 소요 자원:

| 항목 | expert-only 재학습 |
|---|---:|
| 시간 | 약 8~15분 |
| GPU VRAM | 약 1~3GB |
| RAM | 10~20GB 이내 |
| 추가 디스크 | 약 100~500MB |

### 4.3 Backbone 전체 학습

field backbone을 새로 만들어야 하는 경우에만 실행한다.

```bash
/home/ubuntu/venv_g3/bin/python -m aox_g3.train_fields \
  --manifest "$G3_RUN_DIR/g2-training.json" \
  --out "$G3_RUN_DIR/backbone.pt" \
  --epochs 250 \
  --group-balanced-sampling \
  --device cuda
```

이후 expert 학습의 `--base`를 새 `backbone.pt`로 변경한다. 직전 전체 pipeline 실측 시간은 약 20분, peak RAM은 약 18.4GB, run directory는 약 479MB였다.

## 5. 평가와 배포 전 확인

1. 독립 golden holdout에서 solver별 Cd/Cl MAE를 비교한다.
2. G2 holdout에서 Cp MAE와 velocity RMSE를 비교한다.
3. 대표 STL을 CLI로 추론해 flow 산출물과 OOD를 확인한다.
4. G4 flow는 G4 CFD 결과와 별도 교차 검증한다.
5. offline gate 통과 전에는 registry를 수정하지 않는다.

```bash
/home/ubuntu/venv_g3/bin/python scripts/evaluate_fixed_holdout.py --help
```

`data/validation/*`의 현재 fixed holdout은 운영 expert의 기존 validation case를 동결한 것이다. regression check에는 사용할 수 있지만, challenger 최종 승인은 새 독립 golden holdout을 기준으로 한다.

## 6. 자동 nightly pipeline 주의사항

`deploy/run_manual_csv_challenger.sh`과 `deploy/g3-nightly.json`의 현재 학습 구성에는 다음 수정이 필요하다.

- G1 기존 lift manifest merge
- Cd/Cl label-domain 별 validation 보장
- G2 coefficient group-balanced training
- G4 m/mm 단위 정규화
- 독립 golden holdout gate

위 수정 전에는 다음 파이프라인을 그대로 재실행하지 않는다.

```text
deploy/run_manual_csv_challenger.sh
deploy/g3-nightly.json
```

nightly 상태는 읽기 전용으로 확인한다.

```bash
systemctl list-timers g3-nightly.timer
journalctl -u g3-nightly.service -n 100 --no-pager
```

## 7. 관련 문서와 코드

- `docs/G3_V6_ARCHITECTURE.md`: v6 multi-expert 구조
- `docs/G3_V6_DATA_AUDIT.md`: 데이터 감사 결과
- `docs/G3_NIGHTLY_CANARY.md`: nightly, shadow, promotion 운영
- `aox_g3/infer_fields.py`: flow 추론 CLI
- `aox_g3/train_fields.py`: shared field backbone 학습
- `aox_g3/train_coefficient_experts.py`: solver/domain expert 학습
- `scripts/evaluate_fixed_holdout.py`: regression 평가
