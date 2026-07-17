# ④Model (트윈 학습) 모델 분류 및 GUI 재설계 계획

작성: 2026-07-17 · 대상: `src/naviertwin/web/app.py` ④Model 패널 (웹 GUI)

---

## 1. 선행 연구 기준 — CFD 데이터 기반 예측 모델의 4대 계열

CFD 결과(스냅샷)로 "파라미터/시간 → 유동장"을 예측하는 디지털 트윈 모델은
문헌에서 크게 4계열로 수렴한다. (대표 문헌 병기)

### A. ROM — 차원 축소 + 계수 보간 (projection-based, non-intrusive)
- 원리: 스냅샷 행렬을 저차 기저로 축소(POD/SVD)한 뒤, 파라미터→축소계수를
  회귀(RBF/Kriging/GP)로 학습. 예측 시 계수를 역투영해 전체 장 복원.
- 대표 문헌: Berkooz et al. 1993 (POD), Benner et al. 2015 SIAM Review
  (파라메트릭 MOR), Hesthaven & Ubbiali 2018 (POD-NN, non-intrusive).
- 장점: 스냅샷 수십 장이면 충분. **메쉬 구조 무관**(1D/2D/3D 비정형 모두 —
  장을 벡터로 펴서 처리). 학습 수 초. 엔지니어링 트윈의 사실상 표준.
- 한계: 선형 기저라 이류 지배/불연속 문제(우리 필라멘트 데모!)에서 모드 수가
  급증(Kolmogorov n-width 한계). → 비선형 축소(AE)나 B/C 계열로 보완.
- 코드 재고: `dimensionality_reduction/linear/`(pod, randomized, incremental,
  shifted, weighted, space-time, mrPOD, DEIM, …), `nonlinear/`(autoencoder,
  cnn_ae, vae, quadratic manifold, OpInf, …), `surrogate/`(rbf, kriging,
  gp_scratch, ensemble, RFF, multi-fidelity).

### B. 직접 필드 회귀 — 좌표·파라미터 → 물리량 (Physics AI / neural field)
- 원리: (x, y, z, t/파라미터) → u 를 신경망으로 직접 회귀. 축소 단계 없음.
  물리 잔차(PDE loss)를 추가하면 PINN.
- 대표 문헌: Raissi et al. 2019 (PINN), NVIDIA PhysicsNeMo(구 Modulus) 백서.
- 장점: 메쉬 프리 — 임의 좌표에서 평가 가능, 데이터 희소해도 물리 제약으로
  보완 가능. 비정형/움직이는 형상에 자연스러움.
- 한계: 학습이 느리고(분 단위), 고주파/난류 장은 스펙트럴 바이어스로 어려움.
- 코드 재고: `physnemo/`(cfd_field_model=현 웹 탑재분, pina_wrapper PINN,
  deep_ritz, dd_pinn), PhysicsNeMo Module 체크포인트 래퍼.

### C. 신경 연산자 — 함수 → 함수 (operator learning)
- 원리: 초기장/경계조건 함수 전체를 입력으로 받아 해 함수 전체를 출력.
  해상도 불변(FNO), 임의 센서 위치(DeepONet).
- 대표 문헌: Li et al. 2021 (FNO), Lu et al. 2021 (DeepONet, Nature MI),
  PDEBench(NeurIPS 2022).
- 장점: 새로운 입력 조건에 ms 단위 추론. 다수 시나리오(수백~수천 샘플) 학습에
  최적.
- 한계: FNO/UNet은 **균일 격자 필수**(비정형 CFD 메쉬는 재보간 필요),
  데이터 요구량 큼. 단일 시계열 12스냅샷 같은 상황엔 부적합.
- 코드 재고: `operator_learning/`(fno/tfno/wno/lno, deeponet 5종, unet, kan,
  koopman/kno, latent_operator). 현재 웹 ⑧AI Bench에 FNO1D/2D만 노출.

### D. 동역학 모델 — 시간 전진 / 시스템 식별 (forecasting)
- 원리: 상태의 시간 전이 규칙 자체를 학습 — 선형(DMD/Koopman), 희소회귀
  (SINDy), 순환망(LSTM). 학습 구간 **너머로 외삽(forecast)** 가능한 유일 계열.
- 대표 문헌: Schmid 2010 (DMD), Brunton et al. 2016 (SINDy), Kutz et al.
  2016 (DMD book).
- 장점: 주기/준주기 유동 예보, 모드별 성장률·주파수 해석 부산물.
- 한계: 강한 비선형/천이 유동에서 장기 예보 불안정.
- 코드 재고: `flow_analysis/modal/`(dmd, dmd_advanced, spod, sindy, pykoopman),
  `operator_learning/koopman/`, LSTM(`neural/`).

### 차원(1D/2D/3D)별 통례 — 연구자들이 실제로 고르는 기준

| 데이터 형태 | 1순위 통례 | 대안 | 부적합 |
|---|---|---|---|
| 1D 신호/프로파일 (Burgers 등) | FNO1D, DeepONet | POD+RBF | — |
| 2D 균일 격자 (이미지형 장) | FNO2D, UNet | POD+Kriging, CNN-AE | — |
| 2D/3D **비정형 메쉬** (일반 CFD) | **POD+Kriging/RBF** (표준) | GNN(MeshGraphNet), DeepONet, PINN | FNO/UNet (재보간 없이는 불가) |
| 단일 시계열, 스냅샷 적음 (<50) | POD+보간, DMD | PINN | 연산자 학습 (데이터 부족) |
| 다수 케이스 스윕 (>100 샘플) | 연산자 학습 | POD+GP | — |
| 학습 구간 밖 시간 예보 | DMD/Koopman, LSTM | — | POD+보간 (내삽 전용) |

핵심: **일반 3D 비정형 CFD의 기본값은 여전히 A(POD+Kriging)** 이고, 균일 격자
다샘플이면 C, 물리 제약·희소 데이터면 B, 예보가 목적이면 D. 우리 앱의 현
기본값(A)은 문헌 통례와 일치한다.

---

## 2. 현 웹 GUI의 문제

1. **분류 축이 섞여 있음**: "Surrogate" 드롭다운에 rbf/kriging(=A의 계수
   회귀기)과 physicsnemo(=B, 아예 다른 계열)가 같은 층위로 나열 — 초심자에게
   "PhysicsNeMo도 Kriging 같은 보간법 중 하나"라는 오해를 유발.
2. **계열 C가 다른 패널(⑧AI Bench)에 고립**: 같은 "예측 모델 학습"인데 벤치
   전용처럼 보임. 로드한 CFD 데이터로는 연산자 학습을 못 함.
3. **계열 D 부재** (웹에는 없음).
4. **선택 가이드 부재**: 어떤 데이터에 어떤 모델이 맞는지 힌트가 없음 —
   사용자가 트윈 지식이 없으면 고를 수 없다.

---

## 3. 제안 — "방식 우선(method-first)" 2단 선택 + 자동 추천

### 3.1 정보 구조

```
④ Model (트윈 학습)
├─ [1단] 모델 방식 (4개 카드/라디오 — 한 줄 설명 포함)
│   Ⓐ ROM (축소+보간)            "적은 스냅샷·모든 메쉬. 엔지니어링 표준" [권장 배지]
│   Ⓑ Physics AI (직접 회귀)      "좌표→물리량 직접 학습. 메쉬 프리·희소 데이터"
│   Ⓒ 신경 연산자                 "다수 샘플·균일 격자. ms 추론" (→⑧Bench 연동)
│   Ⓓ 동역학 예보 (후속)           "학습 구간 밖 시간 외삽. DMD/Koopman"
├─ [2단] 방식별 세부 설정 (선택된 방식 것만 표시)
│   Ⓐ: Reducer(POD/rPOD/…) × 계수 회귀(RBF/Kriging/GP) × 모드 수
│   Ⓑ: 모델(PhysicsNeMo CFD Field/PINN) × epochs/hidden/samples
│   Ⓒ: 아키텍처(FNO/DeepONet/UNet) × epochs/modes/width
│   Ⓓ: 모델(DMD/Koopman) × rank/지연차원
├─ [자동 추천 캡션] 로드된 데이터 특성 기반
│   "현재 데이터: 12 타임스텝 · 2D 균일격자 · 단일 케이스
│    → 권장: Ⓐ ROM (스냅샷이 적어 연산자 학습에는 부족합니다)"
└─ [학습 버튼 1개] → 방식별 디스패치 (build_twin 확장)
```

### 3.2 자동 추천 규칙 (간단 휴리스틱, service 함수)

```
n_steps < 2                       → 학습 불가 안내
n_steps < 50 (단일 시계열)         → Ⓐ 권장
균일 격자(ImageData) & 샘플 많음    → Ⓒ 권장
비정형 메쉬                        → Ⓐ 권장, Ⓑ 차선 (Ⓒ는 "재보간 필요" 경고)
불연속 감지(인접셀 점프 큼)         → Ⓐ에 "모드 수를 늘리세요" 힌트 + Ⓑ 제안
```

### 3.3 용어 원칙
- 계열 이름은 한국어 + 괄호 영문 (예: "축소+보간 (ROM)").
- 각 선택지에 `subtitle` 한 줄: 무엇을 배우는지 + 언제 쓰는지.
- "Surrogate" 단독 표기 폐지 → Ⓐ 안에서 "계수 회귀기"로 명명.

---

## 4. 단계별 구현 계획

### Phase 1 — 정보 구조 재편 (신규 모델 0개, UI/상태만)
- `nt_model_method` 상태("rom"|"physics"|"operator") + 방식 선택 UI(카드 3개).
- 기존 rbf/kriging → Ⓐ 내부로, physicsnemo → Ⓑ로 재배치.
  (Surrogate 드롭다운에서 physicsnemo 옵션 제거 — 방식 선택으로 승격)
- Ⓒ 카드는 당분간 "⑧AI Bench에서 학습" 안내 + 패널 점프 링크.
- 자동 추천 캡션 (`service.recommend_method(dataset)` 신설).
- `build_twin()` 디스패치를 method 기준으로 변경. 기존 테스트 계약 유지
  (`nt_reducer`/`nt_surrogate` state 이름 보존).
- 검증: 기존 69 웹 테스트 + 추천 로직 단위 테스트 + 브라우저 확인.

### Phase 2 — Ⓐ 강화 (이미 core에 있는 것 노출)
- 계수 회귀기에 GP(불확실성 밴드) 추가 → ⑤Twin 예측에 ±σ 표시 여지.
- Reducer에 Incremental POD(대용량), Autoencoder(비선형, 불연속 대응) 추가.
- ⑦Compare가 확장된 조합을 자동 포함.

### Phase 3 — Ⓒ 통합 (로드한 CFD 데이터로 연산자 학습)
- 데모/균일격자 데이터 → FNO2D 직학습 경로 (`service.build_operator_twin`).
- 비정형 메쉬 → DeepONet(좌표 샘플링) 경로.
- ⑧Bench는 "표준 벤치마크 전용"으로 역할 명확화.

### Phase 4 — Ⓓ 동역학 예보
- DMD/Koopman 학습 → ⑤Twin 슬라이더가 학습 구간 **밖**까지 확장(외삽 구간은
  경고색 표시). SINDy 식 표시(설명가능성 보너스).

---

## 5. 리스크 / 결정 필요
- Phase 1에서 `nt_surrogate="physicsnemo"` 하위호환: 기존 상태값을 method로
  마이그레이션하는 shim 필요 (직전 커밋에서 도입한 값이라 부담 적음).
- Ⓒ의 비정형 메쉬 재보간은 품질 함정 — Phase 3에서 명시적 경고 UI 필수.
- GP 불확실성 밴드는 ⑤Twin 렌더 변경(±σ 필드) 수반 — Phase 2에서 범위 결정.
