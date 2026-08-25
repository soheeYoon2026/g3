# LES ΔCd 캠페인 프로토콜 노트 (Ahmed/SGS 세션 회신, 2026-08-25)

출처: "Ahmed body CFD 수렴성 진단 및 SGS 모델 검토" 세션의 실측 기반 회신.
gtr-smooth v6 캠페인 해석과 이후 모든 LES A/B 설계의 기준.

## 1. bbox 고정(마이크로 삼각형)은 정합 — 격자와 스케일을 동시에 고정

- L0 nx=400은 `_production3/4` 하드코딩. **blockage는 실루엣이 아니라 bbox
  기준**: `W,H = bbox` → `dx_stl = H / sqrt(blockage·ny·nz/(W/H))`.
  bbox를 고정하면 격자뿐 아니라 **형상→격자 스케일(dx_stl)이 고정**된다.
- 골프 캠페인의 fix_boxes는 refine1/2만 얼렸고 스케일은 통제 안 됐음 —
  bbox 고정이 그보다 강한 통제.
- **검증 2건 (런마다)**: ① `Solid cells: N/M` 값이 형상 간 동일 수준인지 +
  bbox 모서리에 고체 얼룩 없는지 (마이크로 삼각형은 SDF에선 사라지나 GWN
  부호 계산에는 면으로 들어감) ② area_ref도 bbox 유래라 함께 고정됨 —
  ΔCd에선 상쇄되지만 의도된 것 (v6에서 의도 맞음).

## 2. 격자 감도 실측 상한 (3-level, plateau 기준)

| 변화 | Cd 변화 |
|---|---:|
| 같은 형상 6반복 σ | **1.38%** |
| l2_wake 80→98 (+18%) | −0.8% |
| ny 144→176 (1.22×) | −5.0% |
| (같은 l2_wake 변화를 wake_survey로 재면) | −2.7% (3배 민감) |

## 3. 발산 처방: u_inf를 내리지 말고 cs를 올려라

- 원인: τ→0.5에서 SGS 소산 부족. `validate_cli.py`의
  `CS_LADDER=(0.08,0.12,0.16,0.22,0.30)` + `find_shared_cs(stls,...)` =
  "전 형상이 함께 견디는 최소 cs 하나"를 찾아 **전부 같은 값** 사용.
- A/B에서 형상마다 u_inf를 달리 주면 Re 차이가 ΔCd에 섞임 — 금지.
- 적용: rear_narrow_25가 cs_l2=0.12에서 발산하면 **7종 전부 cs_l2=0.16으로
  통일 재실행** (0.08 짝 계획 폐기).

## 4. 추정기: plateau가 최선, 후류 평면 금지, sd는 가중치로

Ahmed 35°(실험 0.26) 직접 비교: plateau +1.8% vs 후류평면 1.0~2.0L **−55.1%**
(누적 곡선이 정밀격자 경계마다 계단 손실: base 0.248 → L2출구 0.209 →
1.0L 0.120). 표면적분(cd_direct.L2_ibb)은 parked(15배)이고 GT-R↔NISMO 차이를
1/3로 압축해 읽음 — 금지.

**plateau_cd의 sd(창 평탄도) = 그 런의 불확도. sd>0.005면 자체 경고 —
그 런의 ΔCd는 가중을 낮춘다.**

## 5. 업로드 STL 용접 레시피 (nismo_weld 계보)

`xlb_archive/code/mesh_prep/make_opt_meshes.py` (기준판 `gtr_weld.py`):
부품 concatenate → `merge_vertices`를 허용오차 4단계(기본/1e-5/1e-4/1e-3)로
시도 → 열린에지 수 최소인 것 채택 → 매 단계 nondegenerate/unique/
remove_unreferenced. 같은 폴더에 weld_tol.py, pmf_repair.py, wheels_or_not.py.

덧: **GWN은 수밀이 아니라 "일관된 감김"만 요구** — 수프여도 감김이 일관되면
부호가 나온다. (우리 업로드 수프는 감김 일관 확인됨 — 4-level NaN의 원인은
부호 문제가 아닐 수 있음. 추후 업로드 직접 수용 경로에서 재검토.)
