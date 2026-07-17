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

---

# v2 — 딥리서치 재검증 및 패널 개편 계획 (2026-07-17)

웹서치 기반 딥리서치 2건(2022–2026 리뷰 논문 / 오픈소스 생태계) 반영.

## 6. 문헌 재검증 결과 — 4계열 골격은 유효, 축 2개와 계열 2개 보강

최신 리뷰들은 단일 평면 분류가 아니라 **직교 축 여러 개**로 분류한다:
- "무엇을 배우나"(mapping) 축 — 우리 Ⓐ~Ⓓ와 일치 (Azizzadenesheli et al.
  Nat. Rev. Phys. 2024 가 함수 근사(PINN) vs 연산자 학습을 명확히 구분 —
  Ⓑ/Ⓒ 분리 타당성 확인)
- **침습(intrusive) vs 비침습** 축 — ROM 커뮤니티 지배 축 (Kramer, Peherstorfer
  & Willcox, Annu. Rev. Fluid Mech. 2024). 우리 Ⓐ는 "비침습 투영 ROM".
  Operator Inference(OpInf)가 Ⓐ와 Ⓓ 사이의 명명된 선택지로 부상.
- **물리 주입 방식** 축 — 순수 데이터 / 물리 손실(PINN) / 하이브리드 클로저
- **결정론 vs 확률론** 축 — 디지털 트윈 문헌(Kapteyn & Willcox)에서는 UQ 가
  옵션이 아니라 정의 요소.

### 보강된 분류 (v2)
| 계열 | v1 대비 변경 |
|---|---|
| Ⓐ ROM (축소+보간) | EZyRB 식 Reduction(POD/AE) × Approximation(RBF/Kriging/GPR/ANN) 매트릭스로 세분. OpInf 를 "축소 동역학" 옵션으로 명명 (Ⓐ↔Ⓓ 교량) |
| Ⓑ 뉴럴 필드 (직접 회귀) | 학습 전략 태그 분리: supervised vs physics-informed (PINA 의 Problem/Model/Solver 구조 차용) |
| Ⓒ 신경 연산자 | **하위 분기 필수**: 균일 격자(FNO 계) vs **기하 인지(GNN/MeshGraphNet, GINO, Transolver)** — 후자가 산업 채택 1위 트렌드 (자동차 공력: 8천 형상 DrivAerNet++, GNN 600× 가속 벤치마크) |
| Ⓓ 동역학 예보 | DMD/Koopman/SINDy + OpInf-time |
| (신규, 문서만) Ⓔ 생성형 서로게이트 | diffusion 기반 (PDE-Refiner, GenCFD) — 연구 단계, 채택 최하위 |
| (신규, 문서만) Ⓕ PDE 파운데이션 모델 | Poseidon/DPOT/MPP 파인튜닝 — 소표본에서 from-scratch 연산자 대비 10× 표본 효율. 급부상 중이나 아직 연구 등급 |
| 교차 태그 | UQ(GP/앙상블/베이지안) · 물리주입(없음/loss/구조) · 다중충실도 |

### 데이터 규모 → 방식 결정표 (문헌 앵커 반영)
| 데이터 규모 | 고정 형상 (파라미터 스윕) | 가변 3D 형상 |
|---|---|---|
| 스냅샷 10–100 | **Ⓐ POD+GP/Kriging (UQ 포함)**, 시계열이면 Ⓓ | Ⓕ 파인튜닝 or Ⓐ |
| 100–1,000 | Ⓐ or OpInf; 메쉬프리 질의 필요시 Ⓑ | GNN/GINO 가시권, 다중충실도 GP |
| 1,000+ | Ⓒ (FNO/DeepONet) | Ⓒ 기하 인지 (GINO/Transolver) — 벤치마크 검증 영역 |

(임계값은 문헌 데이터포인트 기반 추론이며 상수 아님. GP 는 O(N³) 스케일 한계.)

## 7. 오픈소스 생태계 정렬 포인트

| 계열 | 사실상 표준 | 우리가 차용할 것 |
|---|---|---|
| ROM | pyMOR(346★), **EZyRB**(mathLab) | EZyRB 의 Reduction × Approximation 분해 = 우리 Ⓐ와 동일 → "EZyRB 호환 워크플로우" 표기 가능 |
| 서로게이트/UQ | SMT 2.x(897★), GPyTorch/BoTorch | SMT 식 능력 매트릭스 표(모델×정상/과도·메쉬·데이터규모) |
| 신경 연산자 | **neuraloperator**(3.8k★), **PhysicsNeMo**(3.1k★) | "Model Zoo" 명칭, 해상도 불변 태그. PhysicsNeMo v2 는 Solver/Constraint 추상화 통합 |
| PINN | DeepXDE(4.3k★), **PINA**(mathLab) | PINA 의 Problem/Model/Solver 축 — 아키텍처와 학습전략 분리 |
| 동역학 | PyDMD(1.2k★)/PySINDy(1.9k★)/PyKoopman | 이미 래핑함 — GUI 노출만 남음 |
| 벤치마크 | **The Well**(4.1k★, 15TB) > PDEBench(1.2k★, 인용 표준) > CFDBench | ⑧을 PDEBench 지표(nRMSE) + The Well 데이터셋 정렬 |
| 트윈 배포 | FMPy/PythonFMU, **ONNX→FMU**(DNV MLFMU 선례) | 웹 ⑥Export 에 ONNX/FMU 추가 (core 에 이미 구현됨) — 표준 트윈 배포 스토리 |

**경쟁 구도**: 오픈소스 다계열 CFD→트윈 GUI 는 사실상 공백 (CFDTwin 은
Fluent 종속 단일 기법). NavierTwin 의 포지셔닝 포인트.

## 8. 패널 존재 유무 판단 — ⑦Compare vs ⑧Bench 및 전체 구조

### 진단
| 패널 | 실체 | 판단 |
|---|---|---|
| ⑦ Compare | **로드한 내 데이터**에서 reducer×surrogate 전 조합 학습→RMSE/R²/지연 순위 | = "모델 선정(model selection)" 단계. ④Model 의 부속 기능이 맞음 |
| ⑧ AI Bench | 내 데이터와 무관한 **표준 벤치마크 문제**(Burgers/heat/cavity) 생성→FNO 학습 | = "연산자 실험실". 데이터 생성(→①의 일), 학습(→④의 일), 평가(→비교의 일)가 한 패널에 뭉침 |

**결론**: 둘은 목적이 달라 "그대로 합치면" 오히려 혼란. 올바른 통합은
해체·재배치다 — ⑦은 ④로 흡수, ⑧은 ①(데이터)+④(학습)로 분해 후 은퇴.

### 목표 패널 구조 (8 → 6)

```
① Import   (파일/데모 + 내장 벤치마크 데이터셋 로드)   ← ⑧의 데이터 생성 흡수
② Analyze  (와류/FFT — 후처리 진단. 그대로)
③ Reduce   (POD 진단: 에너지/모드 → ④의 모드 수 결정 지원. 캡션으로 역할 명시)
④ Model    (방식 선택 + 학습 + 자동 비교 리더보드)     ← ⑦ 흡수, ⑧의 학습 흡수
⑤ Twin     (예측 · UQ 밴드 · [향후] Ⓓ 예보 외삽)
⑥ Export   (+ ONNX / FMU — 표준 트윈 배포 경로)
```

### 단계 계획 (v2)
- **P2 (다음 단계, 소규모)** — 패널 정리 1차:
  - ⑦Compare 를 ④Model 하단 "자동 비교 (리더보드)" 섹션으로 이동
    (기존 콜백·다이얼로그·라이브 진행 재사용, 칩 스트립 7개로)
  - 비교 리더보드에 Physics AI(Ⓑ) 행 추가
  - ⑧ 명칭 "연산자 랩 (Benchmark Lab)" + nRMSE 용어 정렬 + "내 데이터와
    무관한 표준 문제 실험실" 캡션
  - ④ Ⓒ 카드에 "균일 격자(FNO) / 기하 인지(GNN — 예정)" 하위 표기
- **P3** — ROM 강화: Reduction×Approximation 매트릭스(AE·GPR 추가),
  GP 불확실성 → ⑤Twin ±σ 표시
- **P4** — 연산자 직학습: 균일격자→FNO2D, 내장 벤치마크 데이터를 ①Import
  소스로 이동, 리더보드에 연산자 포함, ⑧ 은퇴 (8→6 완성)
- **P5** — Ⓓ 동역학: DMD/Koopman 예보(외삽 구간 경고색), OpInf 옵션
- **P6** — 배포: 웹 ⑥Export 에 ONNX/FMU 버튼 (core 구현 재사용)

### 참고문헌 (v2 추가분)
Kramer/Peherstorfer/Willcox ARFM 2024 (OpInf) · Azizzadenesheli et al. Nat.
Rev. Phys. 2024 · Kovachki et al. JMLR 2023 · Vinuesa & Brunton Nat. Comput.
Sci. 2022 · Wang et al. arXiv:2408.12171 (ML-for-CFD 서베이) · Li et al. GINO
NeurIPS 2023 · Herde et al. Poseidon NeurIPS 2024 · Elrefaie et al.
DrivAerNet++ NeurIPS 2024 · Kapteyn & Willcox Nat. Comput. Sci. 2021 ·
arXiv:2504.06699 (산업 공력 CNN vs GNN 벤치마크)

---

# v3 — 핵심 파이프라인 vs 보조 진단 분리 (2026-07-17)

사용자 지적: "Analyze/Reduce 는 트윈 생성의 메인 방법이 아니지 않냐 — 부수적
분석이니 서브섹션으로 빼도 되지 않냐?" → 코드로 검증 후 확정.

## 9. 검증 — ②Analyze/③Reduce는 ④Model 학습과 완전히 비결합

`app.py::build_twin()` / `service.build_twin()` 모두 `nt_analysis_done`,
`nt_pod_done`, `self._pod_result` 를 전혀 참조하지 않는다. 학습 시 POD reducer
를 **내부에서 처음부터 새로 생성**한다 (`TwinEngine(reducer_type=..., n_modes=...)`
→ `.fit()`). ③Reduce 패널에서 실행한 POD 결과(에너지 차트, 모드 뷰어)는 학습에
전혀 재사용되지 않는 **진단 전용 사이드바**였다 — 공유되는 것은 `nt_n_modes`
슬라이더 값 하나뿐. ②Analyze(와류 식별/FFT)도 마찬가지로 완전히 독립.

사용자 판단이 정확했음을 코드가 확인시켜준다: 핵심 트윈 파이프라인은
**Import → Model → Twin → Export** 4단계뿐이고, Analyze/Reduce/연산자 랩은
전부 보조 도구다.

## 10. 조치 — 6패널로 정리 (1차: 병합만, 번호는 등장 순서 유지)

```
① Import
② 부가 분석 (선택)         ← 구 ②Analyze + ③Reduce 통합, 캡션으로 "학습에
                              필요 없다" 명시. 내부는 "와류 식별" / "POD 진단"
                              두 서브섹션(구분선+소제목)으로 구성.
③ Model (트윈 학습)         ← 핵심. 자동 비교 리더보드 포함(v2 P2).
④ Twin (예측)               ← 핵심.
⑤ Export (저장)             ← 핵심.
⑥ 연산자 랩 (Benchmark Lab) ← 보조. 표준 문제 실험실.
```

파이프라인 칩도 6개로: "② 분석" 칩은 `nt_analysis_done || nt_pod_done` 중
하나만 완료돼도 점등(두 진단 중 무엇을 했든 "분석 좀 해봤다"는 신호로 충분).

검증: 브라우저에서 6패널/6칩 렌더, POD→Model→연산자 랩 점프 링크(버그 발견해
수정: 점프 대상이 ⑤Export 를 가리키던 것을 ⑥연산자 랩 index[5]로 정정) 확인,
웹 테스트 91개 통과.

## 11. 2차 — 핵심/보조 물리적 재배치 + 그룹 헤더 (2026-07-17, 후속)

§10 은 병합만 했을 뿐 번호가 등장 순서(② 부가분석, ⑥ 연산자 랩)라 핵심 4단계와
보조 2단계가 여전히 시각적으로 뒤섞여 있었다. 사용자 요청으로 완전 분리:

```
[핵심 파이프라인]                    [보조 도구 (선택 — 트윈 학습에는 필요 없음)]
① Import                            ⑤ 부가 분석 (선택)
② Model (트윈 학습)                  ⑥ 연산자 랩 (Benchmark Lab)
③ Twin (시간→필드 예측)
④ Export (저장)
```

- 부가 분석 패널을 Import 직후에서 **Export 뒤**로 물리적으로 이동, 연산자
  랩은 그대로 마지막 유지 — 핵심 4단계(①~④)가 서로 연속하도록 재번호.
  번호는 그대로 두되(비고유 numbering 제거는 하지 않음 — 목록 내 읽기
  순서를 유지하는 편이 상태 인덱스·회귀 위험 대비 이득이 큼) 두 그룹 사이에
  `text-overline` 라벨 두 개("핵심 파이프라인" / "보조 도구 (선택 — 트윈
  학습에는 필요 없음)")를 드로어에 직접 삽입해 시각적으로 블록을 나눴다.
  Vuetify 의 `v-expansion-panels` 그룹 인덱싱은 `useGroupItem` 등록 순서
  기반이라(DOM 인접 위치가 아니라 컴포넌트 마운트 순서) 사이에 일반 `html.Div`
  를 끼워 넣어도 `nt_open_panels` 인덱스가 깨지지 않음을 확인.
- 파이프라인 칩도 같은 두 블록으로 나눠 두 줄로 렌더(`_chip_row` 헬퍼) —
  1줄: ①~④ 핵심, "보조" 캡션, 2줄: ⑤~⑥ 보조.
- 인덱스 재계산: Import=0, Model=1, Twin=2, Export=3, 부가분석=4, 연산자랩=5.
  전체 코드베이스에서 이전 번호를 참조하던 상태/에러 메시지·독스트링·주석을
  전수 검색(`①②③④⑤⑥` grep)해 전부 정합화.
- 검증: 브라우저에서 두 overline 라벨 렌더 확인, POD(⑤, index 4) → Model
  (②, index 1) → Twin 예측(③, index 2) → 연산자 랩 점프(⑥, index 5) 전체
  흐름을 실제 클릭/상태조작으로 실행해 인덱스 정합 확인. 웹 테스트 91개 통과
  (재배치는 패널 배치/텍스트만 바꿔 콜백 계약 불변이라 신규 테스트 불필요).

---

# v4 — 일반화된 입출력 데이터 모델 로드맵 (2026-07-17)

사용자 제기: (a) 다중 출력 필드 동시 학습, (b) steady vs unsteady 분류,
(c) 입력/출력 격자 분리, (d) 형상이 다른 케이스(자동차 공력 스타일) 지원,
(e) 일반 N-입력/M-출력. 순서대로 판단과 로드맵.

## 12. 현 데이터 모델의 근본 제약 진단

웹 앱의 암묵적 가정: **"데이터셋 1개 = 케이스 1개 = 고정 메쉬 + 시계열"**.
그래서 입력이 항상 시간(t) 하나였다. (b)~(e)는 전부 이 가정을 깨는 요구이며,
모델 계열 문제가 아니라 **문제 정의(Problem) 계층**의 문제다 — PINA 의
Problem/Model/Solver 분리(§7)에서 Problem 에 해당.

코어가 이미 가진 것 (재사용 가능):
- `PhysicsNeMoCFDFieldModel.from_datasets(datasets, field_names, params,
  parameter_names)` — **다중 케이스 + 파라미터 테이블 + 다중 출력** 지원.
  데스크톱 Model 패널은 이미 케이스 목록 + 파라미터 CSV 로 이 경로를 쓴다.
- `TwinEngine.fit(snapshots, params)` — params 는 (n_samples, n_params)
  임의 차원. ROM 도 다중 입력 파라미터를 이미 지원 (웹이 t 만 넣을 뿐).
- 데스크톱 `twin_panel` 의 N-파라미터 스핀박스 (`_param_spins`) — ③Twin 의
  다중 입력 슬라이더 UI 선례.
- 제약: `cfd_field_model` 은 **모든 케이스가 동일 메쉬**를 요구
  (`all CFD cases must share the same mesh coordinates/topology` 검사).
  → 형상 가변은 이 경로로는 불가. GNN/GINO(§6 v2) 또는 SDF 인코딩 필요.

## 13. 판단 — steady/unsteady 는 "문제 유형" 축이 맞다

동의. 단, 모델 방식(ROM/Physics/연산자) 카드와 섞지 말고 **직교하는 상위
축**으로: 문제 유형이 입력 공간을 정의하고, 모델 방식은 매핑 함수를 정의.

| 문제 유형 | 입력 | 데이터 형태 | 현황 |
|---|---|---|---|
| A. 비정상 단일 케이스 | t | 1 케이스 × T 스텝 | ✅ 현재 유일 지원 |
| B. 정상 파라미터 스윕 | μ = (μ₁..μₖ) | N 케이스 × 1 스텝 + 파라미터 표 | 코어 ✅ / 웹 ❌ |
| C. 비정상 + 파라미터 | (μ, t) | N 케이스 × T 스텝 | 코어 부분 / 후순위 |
| D. 형상 가변 (정상) | 형상 파라미터 or 기하 자체 | N 형상 × 1 스텝 | GNN/GINO or SDF 필요 |

자동 판별 규칙: 케이스 1 + T>1 → A. 케이스 N + 파라미터 표 → B.
①Import 에 "케이스 세트 로드"(여러 파일 + 파라미터 CSV)가 생기면 B 가 열린다.

## 14. 입력/출력 격자 분리 판단

- ROM/Physics 의 "입력"은 격자가 아니라 파라미터 벡터 → 입력 격자 개념은
  연산자 학습(함수 입력)에서만 유효. 혼동 방지를 위해 UI 용어로 "입력
  격자"는 연산자 방식에서만 노출할 것.
- **출력 격자 분리는 Physics AI 가 공짜로 가능**: 신경장은 (x,y,z,μ)→u 라
  임의 좌표에서 평가 가능. 현재 `predict()` 가 학습 좌표(self._coords)에
  고정돼 있을 뿐 — `predict_at(coords, params)` 하나 추가하면 업로드한
  다른 메쉬/해상도 재샘플 출력이 열린다 (M3).
- ROM 출력은 학습 메쉬에 묶임(모드가 메쉬 벡터) — 분리하려면 보간 후처리.

## 15. 형상 가변(자동차 스타일) 판단

PhysicsNeMo 데모(DrivAerNet 등)가 하는 것: 형상별 정상 유동 → 기하 인지
연산자(MeshGraphNet/DoMINO) 또는 SDF 조건화. 우리 로드맵:
- 1단계 (현실적 중간 단계): **형상 파라미터화** — 사용자가 형상당 파라미터
  벡터(CSV)를 주면 문제 유형 B 로 처리. 단, 동일 메쉬 제약 때문에 공통
  배경 격자(바운딩 박스 균일 격자)에 재샘플 + SDF(x) 입력 피처 추가가 필요
  — cfd_field_model 확장 (M4a).
- 2단계 (정공법): GNN/MeshGraphNet·GINO 노출 (코어에 구현 존재, v2 §6 의
  산업 채택 1위 트렌드) — 연산자 랩에서 검증 후 ②Model 로 승격 (M4b).

## 16. 로드맵 (M-시리즈)

- **M1 (완료)**: Physics AI 다중 출력 — `build_physics_ai_twin(fields=[...])`,
  UI 복수 선택(physics 방식일 때), 예측을 `split_multi_prediction` 으로
  필드별 `twin_<name>` 분해 표시. per-field 검증 지표 반환.
- **M2 (완료)**: 문제 유형 B — 정상 파라미터 스윕. 아래 §17 참조.
- **M3 (완료)**: 출력 격자 자유화 — 아래 §18 참조.
- **M4a (완료)**: 형상 가변 — 공통 격자 재샘플 + SDF. 아래 §19 참조.
- **M4b**: 기하 인지 연산자 (MeshGraphNet/GINO) — 연산자 랩 검증 → 승격.
- **M5 (완료)**: Ⓓ 동역학 예보 (PyDMD). 아래 §20 참조. OpInf 는 미착수.
- 교차: ②Model 입력·출력 섹션이 문제 유형을 표시/전환하는 허브가 된다.

## 17. M2 구현 기록 — 정상 파라미터 스윕 (2026-07-17)

문제 유형 B 를 웹에 열었다. 앱의 "데이터셋 1개 = 케이스 1개" 가정을 깨는 첫
작업이라, 상태 모델을 A/B 두 갈래로 명시 분기했다.

### 데이터 계약
- **폴더 1개 = 케이스 세트**. 폴더 안 CFD 파일 1개 = 운전조건 1개(파일명 정렬
  순서 = CSV 행 순서). 폴더 안 CSV 1개 = 파라미터 표(행=케이스, 열=파라미터).
  CSV 가 없으면 `case_index` 폴백(경고 로그 + UI 에 출처 표시).
- 각 케이스는 **마지막 타임스텝**(정상해의 수렴 해)만 `snapshot_dataset()` 으로
  단일 스냅샷화 → 케이스당 정확히 1 스냅샷이라 params 행 수와 1:1 로 맞는다.
  이것이 ROM/Physics 양쪽 경로를 같은 계약으로 통일하는 핵심.
- OpenFOAM 처럼 **디렉토리가 케이스**인 형태는 범위 밖 (파일 확장자 기반) —
  후속 과제.

### 계층별 추가
- service: `load_case_set()`, `build_twin_from_cases()`(ROM),
  `build_physics_ai_twin_from_cases()`(Physics, 다중 출력 겸용).
  `predict_twin()` 이 스칼라 t 뿐 아니라 k 차원 벡터를 받도록 일반화(하위호환).
  엔진 metadata 에 `problem_type="steady_sweep"`, `param_names/mins/maxs` 기록.
- app: `nt_case_mode` 를 축으로 A/B 분기. `case_datasets/case_params/
  case_param_names` 인스턴스 필드. `_set_case_set()` 은 `_set_dataset()`(전체
  리셋)을 **먼저** 부른 뒤 케이스 상태를 채운다 — 순서 의존적이라 주석 명시.
  `_restore_sweep_engine()` 로 스윕 엔진 복원(예측 전용 — 프로젝트엔 케이스
  1개만 담겨 재학습 불가).
- UI: ①Import "케이스 세트 로드" + 파일 브라우저 `nt_fb_mode`("single"|
  "caseset") 로 폴더 액션 전환. ②Model 은 입력 캡션/버튼 라벨/학습 캡션이
  문제 유형에 따라 바뀜. ③Twin 은 파라미터별 슬라이더 k 개(v-for).

### 배운 것 (재발 방지)
- **VBtn 의 첫 위치 인자는 텍스트 child 라 바인딩이 안 된다** — 표현식 튜플을
  넣으면 문자열이 날것으로 렌더된다(브라우저 검증에서 발견). 동적 라벨은
  `with VBtn(...): html.Span("{{ expr }}")` 로. 페이지 전역에서 미평가 표현식
  잔재를 grep 하는 JS 스니펫으로 다른 곳은 없음을 확인.
- **배열 원소 v-model 은 trame dirty 감지가 안 된다** — `nt_twin_params[i]` 에
  `@update:model-value="flushState('nt_twin_params')"` 를 붙여야 서버로
  push 된다. 브라우저에서 슬라이더를 실제로 움직여(20→20.6) 서버 status 가
  `inlet_velocity=20.6` 로 찍히는 것까지 확인.

### 제약 (알려진 것)
- 자동 비교(리더보드)는 시계열 경로 전용 — 케이스 세트에선 명확한 안내와 함께
  버튼 비활성. 후속 과제.
- 모든 케이스가 **동일 메쉬**여야 함(ROM 은 스냅샷 행렬, Physics 는 core 의
  좌표 일치 검사). 형상 가변은 M4a/M4b.

검증: 브라우저에서 실제 5케이스 × 2파라미터(inlet_velocity, angle_of_attack)
스윕 폴더를 만들어 로드→학습→예측 전 과정 확인. 웹 테스트 105개 통과
(신규 10개: CSV/폴백/행수불일치/스윕 ROM 정확도 복원/Physics 스윕/앱 E2E).

## 18. M3 구현 기록 — 출력 격자 자유화 (2026-07-17)

§14 의 판단("출력 격자 분리는 Physics AI 가 사실상 공짜 — predict() 가 학습
좌표에 고정돼 있을 뿐")을 실현했다.

### 계층별 추가
- core `PhysicsNeMoCFDFieldModel.predict_at(coords, params)` — 임의 좌표에서
  평가. 기존 `predict()` 는 `predict_at(self._coords, params)` 로 위임하도록
  리팩터(동작 불변, 기존 core 테스트 13개 통과로 확인).
- service `predict_to_mesh(engine, params, target)` — 대상 메쉬 좌표에서
  평가해 `twin_<field>` 를 붙인 **새** 데이터셋 반환(원본 불변). 다중 출력은
  field-major 벡터를 `reshape(n_fields, n_locations)` 로 분해 — 저장된
  output_fields spec 의 start/end 는 학습 격자 기준이라 쓰지 않는다(중요).
  ROM 엔진이면 명확한 RuntimeError("POD 모드가 학습 메쉬에 묶여 있습니다").
- app `predict_on_mesh()` / `restore_training_mesh()` + `_swap_view_dataset()`
  — `_set_dataset()` 과 달리 **모델/케이스/POD 상태를 리셋하지 않고 뷰어
  대상만 교체**한다. `_origin_dataset` 에 원본을 보관해 복귀.
  파일 브라우저에 `nt_fb_mode="predict_mesh"` 추가(제목/힌트/액션 전환).
- 가드: 시계열(문제 유형 A)에서 예측 격자로 전환한 상태면 `self.dataset` 이
  학습 데이터가 아니므로 재학습을 막는다("학습 격자로 복귀 후"). 케이스
  세트는 `case_datasets` 로 학습하므로 예측 격자 상태에서도 재학습 가능.

### 검증
- 단위: 5×5 학습 → **9×9 평가**에서 참값 대비 상대오차 < 15% (해상도 비종속
  주장을 실제로 검증). 원본 불변, ROM 거부, 좌표/파라미터 차원 검증.
- 브라우저 E2E: 24×24(576점) 케이스 세트로 Physics AI 학습 → **60×60(3600점)
  격자에 예측**(6배 촘촘) → 뷰어 전환 → 학습 상태(케이스 5개/엔진) 보존 확인
  → 복귀 시 576점으로 정확히 원복.
- 웹 테스트 112개 통과 (신규 7개).

### 부수 발견 (M2 검증이 실제로 잡은 것)
대상 메쉬 파일을 케이스 폴더 안에 두면 6번째 케이스로 세어져 CSV 행수
검증이 `params CSV row count mismatch: 5 vs CFD snapshots 6` 으로 정확히
막았다 — 계약("폴더 안 모든 CFD 파일 = 케이스")이 의도대로 작동함을 우연히
확인. 대상 메쉬는 별도 폴더에 둘 것.

### 남은 제약
- ROM 은 여전히 학습 메쉬 고정 (원리상 — 분리하려면 보간 후처리 필요).
- **입력** 격자 개념은 연산자 학습에서만 유효하므로 여기서 다루지 않았다
  (§14) — ROM/Physics 의 입력은 파라미터 벡터다.

## 19. M4a 구현 기록 — 형상 가변 케이스 (2026-07-17)

§15 의 1단계("형상 파라미터화 + 공통 배경 격자 재샘플")를 구현했다. 이로써
"모든 케이스가 동일 메쉬" 제약이 풀려 **자동차 형상 스윕류 시나리오가 1차
근사로 동작**한다.

### 접근
케이스마다 메쉬가 다르면 스냅샷 행렬(ROM)도 공유 좌표(Physics)도 성립하지
않는다 → **모든 케이스를 같은 균일 격자에 올려 문제 유형 B 로 환원**한다.
이는 형상 가변 ROM 의 표준 전처리다.

- `meshes_are_identical(datasets)` — 좌표 일치 판정.
- `resample_cases_to_common_grid(datasets, resolution)` — 합집합 바운딩 박스
  (2% 여유) → 최장 축 기준 등방 격자(두께 0 축은 1칸, 즉 2D 자동 처리) →
  케이스별 `grid.sample(case.mesh)`.
- `load_case_set(..., resample="auto")` — 메쉬가 다를 때만 재샘플(같으면 불필요
  한 보간 오차를 피하려 건너뜀). `resample=False` 면 명확한 이유로 거부.

### SDF — VTK 표면 대신 거리변환
형상 경계를 모델에 알려주려면 부호거리(SDF)가 필요한데, `compute_implicit_
distance` 류의 표면 기반 방식은 2D/열린 형상에서 취약하다. 대신 `sample()` 이
주는 **`vtkValidPointMask` 의 거리변환(scipy EDT)** 으로 계산했다:
`sdf = EDT(mask) - EDT(1-mask)`, 축별 물리 간격으로 스케일. 2D/3D·열린/닫힌
형상 어디서나 동작하고 VTK 표면 추출이 불필요하다.
- 부호 규약: **+ = 유체, − = 고체/도메인 밖**. 패딩된 바깥 테두리도 정확히
  음수로 나온다(비유체라는 뜻이므로 의도대로).
- `sdf` 는 `DERIVED_EXTRA` 에 넣어 **뷰어로는 보이되 학습 대상(출력) 선택지
  에서는 빠지도록** 했다 — 물리량이 아니라 기하 정보다.
- 함정: EDT `sampling` 은 배열 rank(3)와 길이가 같아야 한다. 처음에 두께 0 축을
  빼고 2개만 넘겨 `RuntimeError` — 테스트가 잡았다.

### 검증
- 단위 5개: 재샘플 거부(resample=False), auto 재샘플 후 전 케이스 메쉬 일치,
  sdf 부호(중심=음수) 및 **반지름↑ → 고체 영역↑**, 형상 가변 ROM E2E 학습·예측,
  메쉬 동일 시 재샘플 건너뜀.
- 브라우저 E2E: 반지름이 다른 원기둥 5케이스(점 수 2496→2396 로 실제 상이)를
  로드 → 자동 재샘플(33×22, 726점) 안내 표시 → `sdf` 는 필드 목록엔 있고 학습
  선택지엔 없음 확인 → radius 로 ROM 학습 → **학습에 없던 r=0.105 에서 보간
  예측** 성공.
- 서버측 몽타주로 sdf 0-등고선이 원기둥을 따라가고 반지름에 따라 커지는 것,
  압력장이 형상에 따라 변하는 것을 육안 확인. 웹 테스트 121개 통과.

### 남은 제약 / 다음
- **SDF 가 아직 입력 피처가 아니다** — 필드로 계산·표시만 한다. 신경장 입력에
  넣으려면 core 에 per-(케이스, 점) 특징 지원이 필요(현재 입력은 좌표 + 케이스별
  파라미터뿐). 이것이 **M4a-2**. ROM 경로는 SDF 없이도 성립하므로(공통 격자 POD
  + 형상 파라미터 보간 = 교과서적 방법) M4a-1 만으로 형상 가변이 실제 동작한다.
- 고체 내부 격자점의 물리량은 0 으로 채워진다(`sample` 기본) — 케이스 간
  일관되므로 ROM 은 성립하지만, 정밀도가 필요하면 마스킹 학습이 낫다.
- 정공법인 기하 인지 연산자(GNN/GINO)는 **M4b** 로 남았다 — 재샘플 없이 원본
  메쉬를 그대로 다루는 방식이며 산업 채택 1위 트렌드(v2 §6). 쓸 수 있는 OSS
  재고는 §21 참조.

## 20. M5 구현 기록 — Ⓓ 동역학 예보 (PyDMD) (2026-07-17)

"오픈소스를 최대한 사용" 지침에 따라 **직접 구현 대신 PyDMD 를 쓴다**. 코어에
이미 `flow_analysis/modal/dmd.py`(PyDMD 래퍼)가 있었고 웹에만 없었다 — 마지막
빈 계열이자, 학습 구간 **밖을 외삽**할 수 있는 유일한 계열이다.

### 추가
- core `digital_twin/dmd_engine.py::DMDTwinEngine` — `DMDAnalyzer.reconstruct(t)`
  를 `predict(params)` 계약으로 감싸 `TwinEngine`/`PhysicsAITwinEngine` 과 같은
  덕타이핑을 갖게 했다 → 웹의 예측·저장 경로를 수정 없이 재사용.
- service `build_dmd_twin(dataset, field, method)` — 학습 범위(param_min/max)와
  **외삽 상한(forecast_max = 학습 구간 × 1.5)**, 모드 주파수/성장률, 그리고
  **재구성 오차**를 반환.
- app: "동역학 예보 (DMD)" 방식 카드, DMD 변형 선택, ③Twin 슬라이더 상한을
  forecast_max 로 확장 + 학습 상한(`nt_twin_train_max`) 초과 시 슬라이더 색과
  경고문. 케이스 세트는 시간축이 없어 명확히 거부.

### PyDMD 를 실제로 검증하다 발견한 것 (중요)
래퍼를 그냥 노출하지 않고 정확도를 측정한 결과 **코어의 기존 버그 2건**이 드러났다.
기존 테스트는 `fit()` 만 하고 재구성 품질을 보지 않아 전혀 못 잡고 있었다.
- **기본 method `fbdmd` 가 발산한다** — 이상적인 진행파 데이터에서도 rel=1.4,
  실제 데모에선 rel=6792. PyDMD 의 FbDMD 자체 `reconstructed_data` 도 동일하게
  나빠 PyDMD 쪽 문제로 보인다. → 코어 기본값을 **`dmd`** 로 변경(데스크톱
  reduce_panel 콤보 순서도 검증된 것 우선으로 수정). 웹은 검증된
  `dmd`/`spdmd` 만 노출.
- **`hodmd` 는 예외를 던진다** — 지연 임베딩으로 모드 차원이 (80,24) 로 바뀌어
  `reconstruct` 의 (n_features, n_modes) 가정과 안 맞는다. 웹에서 제외하고
  코어 docstring 에 명시(수정은 범위 밖).
- **우리 `reconstruct()` 자체는 정확하다** — PyDMD 가 fit 한 객체에 그대로
  적용하면 rel=2e-14 로 PyDMD 의 `reconstructed_data` 와 일치. 초기 실패는 전부
  데이터/랭크 문제였다.
- **실수 진동 데이터는 물리 모드당 랭크 2 가 필요**(켤레쌍) — `n_modes` 를 물리
  모드 수로 주면 과소적합. `svd_rank=0`(자동)이 올바르게 4를 고른다 → UI 는
  자동에 맡기고 그 이유를 캡션에 적었다.

### 적합도를 UI 에 강제 노출한 이유
DMD 는 데이터가 안 맞아도 **조용히** 크게 틀린다(경고 없음). 이 앱의 기본
데모(필라멘트)는 불연속·강이류라 DMD 부적합이며 실제로 **재구성 오차 66%** 가
나온다. 그래서 `reconstruction_error` 를 반환해 신호등(<10% 초록 / <30% 주황 /
그 이상 빨강 + "예보를 믿지 마세요")으로 띄운다 — 사용자가 적합 여부를 판단할
유일한 근거다.

### 검증
- 회귀 테스트 신설(`tests/test_dmd_basic.py::TestDMDAccuracy`): 기본 method 로
  진행파 재구성 rel<1e-6, 주파수 복원(0.207/0.430 참값 일치), **절반 학습 →
  나머지 절반 외삽 rel<1e-6**. 옛 기본값 `fbdmd` 로는 rel=1.41 이라 이 테스트가
  실제로 그 버그를 잡는다(확인함).
- service 테스트: 적합 데이터 재구성/주파수/외삽, **부적합 데이터에서 오차가
  크게 보고되는지**, 미검증 method 차단, 타임스텝 부족 거부.
- 브라우저 E2E: 동역학 카드 → 학습(오차 66.3% 빨강 배지 + 경고) → 슬라이더
  상한 3 > 학습 상한 2 → t=1.0(구간 내)엔 경고 숨김, t=2.8(외삽)엔 경고 표시 →
  외삽 예측 성공. 마크다운 `**` 이 HTML 에 날것으로 노출되던 것도 발견해 수정.
- 웹+DMD 테스트 132개 통과.

## 21. 오픈소스 재고 — 실측 (2026-07-17)

"오픈소스를 최대한 사용" 지침에 따라 환경을 실측했다. **설치돼 있는데 안 쓰는
것**이 핵심 기회다.

| 패키지 | 버전 | 현황 |
|---|---|---|
| **neuraloperator** | 2.0.0 | **설치됨, 미사용** — 우리 FNO 는 직접 구현. FNO/TFNO/**GINO** 레퍼런스(3.8k★) |
| pydmd | 2025.8.1 | 코어가 래핑, **M5 로 웹 노출 완료** |
| pymor | 2025.2.0 | `linear/pymor_pod.py` 에서 래핑 |
| smt | 2.13.0 | `rbf_surrogate`/`kriging_surrogate` 백엔드로 이미 사용 중 |
| torch_geometric | 2.7.0 | `gnn/meshgraphnets.py` 등에서 래핑 (웹 미노출) |
| gpytorch / botorch | 1.15 / 0.17 | `bayesian_opt_botorch` 에서 사용 (웹 미노출) |
| nvidia-physicsnemo | 2.0.0 | Module 체크포인트 래퍼로 사용 (④Export) |
| pina | 0.1.1 | 설치됨, 미사용 |
| pysindy / pykoopman / ezyrb / deepxde | — | 미설치 (코어에 래퍼만 존재) |

### 우선순위 제안
1. **M4b 를 neuraloperator 의 GINO 로** — 기하 인지 연산자를 직접 구현하지 말 것.
   torch_geometric 기반 `gnn/meshgraphnets.py` 도 이미 있다.
2. ~~**⑥연산자 랩의 FNO 를 neuraloperator 로 교체/병기**~~ → **완료, §22 참조.**
3. **GP 불확실성(UQ)** — gpytorch 로 ③Twin 에 ±σ (v2 P3, 디지털 트윈 문헌에서
   UQ 는 정의 요소).

## 22. 구현 기록 — neuraloperator 백엔드 병기 (2026-07-17)

§21 의 2번. "레퍼런스가 설치돼 있는데 자체 구현을 쓰는" 상태를 해소했다.
**교체가 아니라 병기** — 같은 계약으로 꽂아 같은 벤치에서 비교 가능하게 했다.

M4b(GINO)를 먼저 하지 않은 이유: GINO 는 비정형 형상 + 수백~수천 샘플이 전제인데
우리 벤치는 균일 격자(Burgers/heat/cavity)이고 형상 케이스는 5개뿐이라, 지금
붙이면 동작하지 않는 장난감이 된다. 반면 이 작업은 지금 데이터로 즉시 검증된다.

### 추가
- core `operator_learning/fno/neuralop_fno.py::NeuralOpFNO` — neuralop 의 `FNO`
  를 `BaseOperator` 의 fit/predict 계약으로 감싼다. 두 가지를 흡수한다:
  - **레이아웃**: neuralop 은 채널 우선 `(B, C, ...)`, 우리 데이터셋은 채널
    마지막 `(B, ..., C)` → 래퍼가 왕복 transpose.
  - **학습 루프**: neuralop 의 FNO 는 순수 `nn.Module`(fit 없음) → 자체 FNO 와
    동일한 Adam+MSE 루프와 `epoch_callback`(라이브 진행) 을 래퍼가 제공.
  - 1D/2D 를 `n_dim` 하나로 처리 (neuralop 은 `n_modes` 튜플 길이로 차원 결정).
- `bench.train_operator(backend="builtin"|"neuralop")`, 결과에 `backend` 기록.
- ⑥연산자 랩에 "FNO 구현" 선택 (기본 **neuralop** — OSS 우선 지침).

### 함정 (테스트가 잡음)
`bench.evaluate_sample` 은 단일 샘플을 **배치 차원 없이** `predict({"x": X[idx]})`
로 넘긴다. 자체 FNO 는 이를 관용하지만 레퍼런스는 거부한다("inputs must have
same number of dimensions"). → 래퍼가 배치 유무를 모두 받아 입력과 같은 형태로
돌려주도록 맞췄다(기존 앱 계약 보존).

### 실측 비교 (Burgers 64샘플 × N=64, 동일 하이퍼파라미터·시드)
| epochs | backend | test nRMSE | 학습 | 추론 |
|---|---|---|---|---|
| 60 (기본) | builtin | 0.0765 | 4.6s | 2.45ms |
| 60 (기본) | **neuralop** | **0.0660** | 5.5s | 4.04ms |
| 20 | **builtin** | **0.111** | **0.5s** | 2.04ms |
| 20 | neuralop | 0.137 | 6.6s | 3.82ms |

**정직한 결론**: 레퍼런스가 항상 우월한 게 아니다. 기본 epoch(60)에선 neuralop
이 정확하지만(−14% nRMSE), 짧은 학습에선 자체 구현이 더 빠르고 정확하다 —
neuralop 은 채널 MLP·lifting/projection 등 기계가 더 많아 수렴이 느리고 무겁다.
그래서 **교체가 아니라 선택지**로 두고 기본만 레퍼런스로 잡았다.

검증: 백엔드별 계약 일치(1D/2D), 레이아웃 왕복, 배치 유무 관용, 잘못된 backend
거부. 브라우저에서 두 백엔드로 학습해 요약에 `[neuralop]`/`[builtin]` 이 정확히
찍히는 것 확인. 테스트 145개 통과.
