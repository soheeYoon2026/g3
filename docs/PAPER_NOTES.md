# 참고 논문·데이터셋 라이선스 판정 (2026-08-25)

출처: `~/다운로드/vehicle-dynamics-papers.zip`, `wind-tunnel-papers.zip`.
판정 기준: AOX는 상업 SaaS — 학습 데이터·가중치는 상업 사용 가능해야 함.
근거는 각 논문 원문의 라이선스 명시 문구(pdftotext로 확인).

## 데이터셋 라이선스 판정

| 자산 | 라이선스 | AOX 상업 사용 | 근거 |
|---|---|---|---|
| **DrivAerML** (500 DrivAer 변형, HRLES) | CC-BY-SA 4.0 | ✅ 가능 | 논문 명시 "which permits commercial use" |
| **AhmedML** (Ahmed 변형) | CC-BY-SA 4.0 | ✅ 가능 | 논문 명시 permissive |
| **WindsorML** (355 Windsor 변형) | CC-BY-SA 4.0 | ✅ 가능 | 논문 명시 permissive |
| **DrivAerStar** (12,000 STAR-CCM+, FFD 20파라미터) | **CC BY-NC-SA 4.0** | ❌ **불가 (비상업)** | 논문 Section A: "released ... under CC BY-NC-SA 4.0" |
| DrivAerNet / DrivAerNet++ (4000/8000) | CC-BY-NC | ❌ 불가 (기존 금지 유지) | AhmedML 논문도 재확인 언급 |
| DoMINO 사전학습 가중치 | NVIDIA Open Model License | ✅ 가능 | 기존 확인 (README) |

주의 — **CC-BY-SA의 SA(동일조건변경허락)**: 데이터를 *재배포/파생 데이터셋 배포*하면
동일 라이선스 의무. 데이터로 *학습한 모델 가중치*에 SA가 전파되는지는 회색지대이나,
NVIDIA가 DrivAerML로 학습한 DoMINO를 자체 상업 라이선스로 배포한 선례가 있어
방어 가능. 파생 데이터셋을 외부 배포하지 않는 한 실무 리스크 낮음. 최종 상업 배포
전 법무 확인 권장.

결론: **DrivAerStar는 학습·서비스에 쓰지 않는다** (방법론·벤치마크 수치 참고는 논문
인용으로 가능). 다계열 확장은 DrivAerML + WindsorML + AhmedML (CC-BY-SA 3종)로.

## 활용 우선순위 (wind-tunnel 묶음)

1. **Tangsali et al. 2025, "A Benchmarking Framework for AI models in Automotive
   Aerodynamics"** (NVIDIA, arXiv 2507.10747) — 우리가 쓰는 `physicsnemo.cfd`
   평가 프레임워크의 원 논문. DoMINO/X-MeshGraphNet/FIGConvNet을 DrivAerML로
   비교. **G3-BENCH 지표·리포트 형식을 이 논문에 정렬할 것.** Table 1 = 모델·
   데이터셋 지형도.
2. **Ashton et al. AhmedML/DrivAerML/WindsorML 3부작** — 사전학습 원천이자
   **계열 밖 평가 프로브**(관문 2의 OOD 절벽 대응). WindsorML 소수 케이스부터
   평가에 투입 가능.
3. **DrivAerStar** — FFD 파라미터화 변형 + 산업급 라벨 + 풍동검증 1.04%.
   ΔCd 데이터 설계의 방법론 참고(비상업이라 데이터 자체는 사용 불가).
4. **Blockage 보정 2편** (2016 thesis; 2020 지면모사·blockage) — LES blockage
   0.05·자유류 외삽 프로토콜, ground-vs-free 비교조건의 문헌 근거.
5. **Ride-height/rake CFD-ML (2025), 지면효과 kriging (2023)** — ΔCd 추천
   도메인 선행연구 + BO/GP(kriging) 설계 참고.
6. **Zhang et al. 2006 ground-effect 리뷰** — 언더바디·지면효과 물리 기초.

## 비해당

vehicle-dynamics 묶음(레이스라인·랩타임·자율 레이싱 11편)은 G3 공력 모델과
무관. 주행 시뮬레이션 제품 기획 시 재검토.
