# NavierTwin 기술 명세서 (SPEC)

> 시스템이 **무엇**인지 정의하는 문서. 기술 스택, 지원 포맷, 디렉토리 구조, 전체 기법 목록, 설계 원칙, 참고 문헌을 포함한다.
> 구현 우선순위·버전 범위는 `PLAN.md`, Phase별 진행 현황은 `ROADMAP.md` 참조.

---

## 1. 개요

CFD 후처리 결과 데이터 → AI/ROM/Operator Learning → 디지털 트윈 변환 Windows 데스크톱 툴.
비상업용(오픈소스). PySide6 GUI + PyVista 3D + 로컬 GPU 학습. 타겟: 엔지니어 일반 사용자.

---

## 2. 기술 스택

| 레이어 | 기술 |
|--------|------|
| GUI | PySide6 (Qt6), QSS 다크테마, i18n(한/영) |
| 3D 시각화 | PyVista + pyvistaqt, pvpython(ParaView 배치 렌더링) |
| CFD I/O | meshio, foamlib(OpenFOAM 현대적 래퍼), ofpp(MIT, 경량 OpenFOAM 파서), fluidfoam, pyCGNS, h5py, SU2 Python Wrapper |
| OpenFOAM 자동화 | fluidsimfoam (케이스 생성·실행·후처리 자동화) |
| 내부 포맷 | HDF5 (.ntwin) — 메쉬+필드+메타+모델가중치 |
| POD/ROM | modred(BSD-2, MPI 병렬 POD/BPOD), pyMOR(BSD-2, 종합 ROM 프레임워크) |
| 차원축소(선형) | NumPy/SciPy |
| 차원축소(비선형) | PyTorch |
| ROM/모달 | PyDMD, PySPOD(SPOD 병렬), NumPy |
| 데이터 기반 방정식 발견 | PySINDy (SINDy, PDE 발견) |
| Koopman 분석 | PyKoopman, pykoop (scikit-learn 호환 Koopman 연산자) |
| Surrogate | SMT, scikit-learn, PyTorch |
| Operator Learning | neuraloperator(FNO/TFNO/GINO/UQNO, v2.0+), deepxde(DeepONet), PyTorch(U-Net) |
| 잠재공간 연산자 | PyTorch (L-DeepONet, PI-Latent-NO) |
| Koopman Neural Operator | PyTorch (KNO, IKNO, FlowDMD) |
| State Space Model | mamba-ssm, PyTorch (MNO, DeepOMamba) |
| 생성 모델 | PyTorch (Diffusion Model, Score-based) |
| GNN | PyTorch Geometric (GNN surrogate, MeshGraphNets, EGNO) |
| 시계열 | PyTorch (LSTM, Transformer, Neural ODE — torchdiffeq, Mamba) |
| Equivariant NN | e3nn, escnn(Steerable CNN, SO(3)/SE(3)), PyTorch |
| PINN | NVIDIA PhysicsNEMO, PINA (PyTorch+Lightning 경량 대안) |
| PINN 검증 참조 솔버 | FEniCSx/DOLFINx (고정밀 FEM) |
| 미분가능 CFD | JAX-Fluids (JAX 기반, end-to-end 미분 가능) |
| 데이터동화 | DAPPER(EnKF/파티클필터 벤치마크), filterpy, NumPy |
| 불확실성 정량화(UQ) | UQpy(PyTorch 통합), OpenTURNS(PCE/Sobol), SALib(Sobol/Morris/FAST) |
| 최적화 | OpenMDAO(MDO, NASA), DL4TO/PyTopo3D(위상최적화) |
| 심볼릭 회귀 | PySR (Julia 백엔드, 방정식 자동 발견) |
| 설명가능성 | SHAP, captum |
| 모델 내보내기 | ONNX, TorchScript |
| API 서버 | FastAPI (선택) |
| 패키징 | PyInstaller + pyinstaller-hooks-contrib(Apache 2.0, torch/PySide6 훅) + Inno Setup |
| 벤치마크 데이터셋 | PDEBench, AirfRANS, CFDBench, FlowBench |

---

## 3. 지원 CFD 포맷

- **비상업:** OpenFOAM (polyMesh+field), SU2 (.su2+.vtk), Code_Saturne/Nektar++ (XDMF/HDF5)
- **상업:** Fluent (.cas/.dat, .cas.h5/.dat.h5), CFX (.res→EnSight), STAR-CCM+ (VTK/CGNS/EnSight), Tecplot (.dat/.plt/.szplt)
- **범용:** VTK/VTU, CGNS, EnSight Gold, HDF5/XDMF

**전략:** 전체 → VTK UnstructuredGrid → 내부 HDF5 정규화

---

## 4. 디렉토리 구조

```
NavierTwin/
├── src/naviertwin/
│   ├── core/
│   │   ├── cfd_reader/                # 포맷별 Reader + 팩토리
│   │   │
│   │   ├── dimensionality_reduction/
│   │   │   ├── linear/                # POD, Randomized SVD, BPOD, ICA
│   │   │   └── nonlinear/             # AE, VAE, CNN-AE, GNN-AE, Diffusion Maps, CPOD, Tucker
│   │   │
│   │   ├── flow_analysis/
│   │   │   ├── modal/                 # DMD, SPOD, PGD
│   │   │   ├── vortex/                # Q-criterion, λ₂, LCS
│   │   │   ├── statistics/            # FFT, PSD, Wavelet, 두점상관
│   │   │   ├── boundary_layer/        # y+, Cf, δ/θ/H
│   │   │   └── thermofluids/          # Nu, Re, Pr, 엔트로피생성
│   │   │
│   │   ├── surrogate/                 # RBF, Kriging, NN, Ensemble, MoE
│   │   │
│   │   ├── operator_learning/
│   │   │   ├── fno/                   # FNO, TFNO, Adaptive FNO, WNO, LNO
│   │   │   ├── deeponet/              # DeepONet, PI-DeepONet, MIONet
│   │   │   ├── latent_operator/       # L-DeepONet, PI-Latent-NO
│   │   │   ├── koopman/               # KNO, IKNO, FlowDMD
│   │   │   ├── kan/                   # KANO (Kolmogorov-Arnold Neural Operator)
│   │   │   └── unet/                  # U-Net
│   │   │
│   │   ├── gnn/
│   │   │   ├── gnn_surrogate/         # GNN surrogate (비정형 메쉬)
│   │   │   ├── meshgraphnets/         # MeshGraphNets (시간발전 포함)
│   │   │   ├── egno/                  # E(n)-Equivariant GNN Operator
│   │   │   └── graph_transformer/     # HAMLET (Graph Transformer Neural Operator)
│   │   │
│   │   ├── state_space/
│   │   │   ├── mamba_neural_op/       # MNO (Mamba Neural Operator)
│   │   │   └── deepomamba/            # DeepOMamba (DeepONet + Mamba)
│   │   │
│   │   ├── generative/
│   │   │   ├── diffusion_pde/         # Score-based Diffusion PDE solver
│   │   │   ├── wavelet_diffusion/     # Wavelet Diffusion Neural Operator
│   │   │   └── conditional_gen/       # 조건부 유동장 생성
│   │   │
│   │   ├── time_series/
│   │   │   ├── lstm/                  # LSTM
│   │   │   ├── transformer/           # Temporal Transformer
│   │   │   ├── temporal_no/           # TNO (Temporal Neural Operator)
│   │   │   ├── neural_ode/            # Neural ODE (torchdiffeq)
│   │   │   └── latent_dynamics/       # 잠재공간 동역학 (AE + Neural ODE)
│   │   │
│   │   ├── equivariant/
│   │   │   ├── group_equiv_fno/       # Group Equivariant FNO
│   │   │   └── physics_embedded/      # Physics-embedded E(n)-equivariant GNN
│   │   │
│   │   ├── physnemo/                  # PhysicsNEMO PINN
│   │   │
│   │   ├── multi_fidelity/            # 멀티피델리티, 전이학습
│   │   ├── data_augmentation/         # 물리기반 증강, 합성 스냅샷
│   │   ├── physics_correction/        # 보존법칙 사후보정, Hybrid ROM
│   │   ├── online_learning/           # Active learning, 온라인 업데이트
│   │   │
│   │   ├── data_assimilation/         # EnKF, 4D-Var, Particle Filter
│   │   ├── optimization/              # Bayesian Opt, 역문제, 위상최적화, UQ
│   │   ├── sensitivity/               # Sobol indices, 인과분석
│   │   │
│   │   ├── digital_twin/             # 실시간 예측 엔진
│   │   ├── validation/               # RMSE/R²/L2, 해석해비교, 모델간비교
│   │   ├── explainability/           # SHAP, Attention viz, KANO symbolic
│   │   ├── export/                    # ONNX, TorchScript
│   │   └── report/                    # Jinja2→HTML/PDF 자동보고서
│   │
│   ├── gui/
│   │   ├── main_window.py
│   │   ├── panels/                    # 6단계 워크플로우 패널
│   │   ├── widgets/                   # VTK뷰어, 슬라이더, 비교대시보드
│   │   ├── wizard/                    # 튜토리얼 위자드
│   │   └── styles/                    # QSS, i18n
│   └── utils/                         # 설정, 로거, Undo/Redo
│
├── tests/
├── resources/                         # 아이콘, 샘플데이터
├── installer/                         # Inno Setup
├── CLAUDE.md
├── SPEC.md                            # 기술 명세 (이 파일)
├── PLAN.md                            # 구현 계획 및 의사결정
├── ROADMAP.md                         # Phase별 진행 현황
├── pyproject.toml
└── main.py
```

---

## 5. GUI 워크플로우

```
[1.Import] → [2.Analyze] → [3.Reduce] → [4.Model] → [5.Twin] → [6.Export]
```

| 패널 | 기능 |
|------|------|
| 1. Import | 파일/폴더 선택, 포맷 자동감지, 타임스텝/변수 선택, HDF5 변환 |
| 2. Analyze | 유동분석 (Q-crit, DMD, SPOD, FFT, y+, Nu), 민감도(Sobol), 시각화 |
| 3. Reduce | 차원축소 (POD/AE/GNN-AE), 에너지 누적 그래프, 모드 선택 |
| 4. Model | 모델 선택 트리(§5.1), 하이퍼파라미터, GPU, 학습모니터링, 멀티피델리티, 증강, 물리보정 |
| 5. Twin | 파라미터→실시간예측→3D시각화, 데이터동화, 시계열예측, 온라인학습, 오차지표, SHAP |
| 6. Export | ONNX/TorchScript, 자동보고서(PDF), .ntwin 저장, 모델비교대시보드 |

**레이아웃:** 좌측 설정패널 + 우측 3D뷰어 + 하단 로그/진행률 + 모델비교 대시보드

### 5.1 Model 패널 선택 트리

```
모델 유형 선택
├── Classical Surrogate
│   └── RBF, Kriging, NN, Ensemble, MoE
├── Operator Learning
│   ├── Fourier 계열: FNO, TFNO, Adaptive FNO, WNO, LNO, Spectral-Refiner
│   ├── DeepONet 계열: DeepONet, PI-DeepONet, MIONet, Sequential DeepONet, NFNO-DeepONet
│   ├── 잠재공간 연산자: L-DeepONet, PI-Latent-NO
│   ├── Koopman 계열: KNO, IKNO, FlowDMD
│   ├── KAN 계열: KANO (해석가능 연산자)
│   └── U-Net
├── GNN 계열
│   └── GNN Surrogate, MeshGraphNets, EGNO, HAMLET
├── State Space Model
│   └── MNO (Mamba Neural Operator), DeepOMamba
├── 생성 모델
│   └── Diffusion PDE, Wavelet Diffusion NO, 조건부 유동장 생성
├── 시계열/동역학
│   └── LSTM, Transformer, TNO, Neural ODE, Latent Space Dynamics
├── Equivariant
│   └── Group Equivariant FNO, Physics-embedded E(n)-GNN, Lie Algebra Canonicalization
└── PINN
    └── PhysicsNEMO, Domain Decomposition PINN
```

---

## 6. 전체 기법 목록

### 6.1 차원축소

| 기법 | 분류 | 설명 |
|------|------|------|
| POD/PCA | 선형 | SVD 기반 에너지 최대 보존 모드 |
| Randomized SVD | 선형 | 대용량 스냅샷 고속 POD 근사 |
| BPOD | 선형 | 관측/제어 가능성 기반, 제어계 연동 |
| ICA | 선형 | 통계적 독립 성분 분리 |
| Autoencoder | 비선형 | NN 기반 비선형 압축 |
| VAE | 비선형 | 잠재공간 샘플링 → 새 유동장 생성 |
| CNN-AE | 비선형 | 정형격자 이미지 기반 |
| GNN-AE | 비선형 | 비정형 메쉬 그래프 기반 |
| Diffusion Maps | 비선형 | 데이터 기하구조 보존 |
| Tucker Decomposition | 텐서 | 3D 텐서 직접 분해, 메쉬 구조 보존 |
| CPOD | 하이브리드 | 물리 보존법칙 제약 POD |

### 6.2 Surrogate

| 기법 | 설명 |
|------|------|
| RBF | 방사기저함수 보간 |
| Kriging/GP | 가우시안 프로세스, 불확실성 정량화 포함 |
| NN Surrogate | 다층 퍼셉트론 기반 |
| Model Ensemble | 다중 모델 평균 → 정확도↑ 불확실성↓ |
| Mixture of Experts (MoE) | 영역별 전문 모델 자동 선택 |

### 6.3 Operator Learning — Fourier 계열

| 기법 | 설명 |
|------|------|
| FNO | 주파수 도메인 PDE 연산자, 해상도 독립적 |
| TFNO | Tucker-factorized FNO — 파라미터 90% 절감 |
| Adaptive FNO | 적응적 주파수 모드 선택 |
| WNO (Wavelet Neural Operator) | 웨이블릿 기반, 공간 국소 신호 포착에 강점 |
| LNO (Laplace Neural Operator) | 라플라스 변환 기반, 과도 응답 정확 근사 |
| Spectral-Refiner | FNO 파인튜닝으로 난류 정확도 향상 (ICLR 2025) |

### 6.4 Operator Learning — DeepONet 계열

| 기법 | 설명 |
|------|------|
| DeepONet | 함수→함수 연산자 학습, 비균일 그리드 지원 |
| PI-DeepONet | 물리법칙 내장 DeepONet, 학습 데이터 5-10배 절약 |
| MIONet | 다중 입력 연산자 (복수 파라미터 함수 동시 입력) |
| Sequential DeepONet | 시간 의존 하중의 순차 예측 |
| NFNO-DeepONet | 비균일 Fourier 변환 + DeepONet, 불규칙 격자 직접 처리 |

### 6.5 Operator Learning — 잠재공간 (Latent)

| 기법 | 설명 |
|------|------|
| L-DeepONet | AE 잠재공간에서 DeepONet 학습 → 고차원 실시간 예측 |
| PI-Latent-NO | 물리 제약 잠재 연산자 — 라벨 데이터 불필요, 학습시간 15-67% 단축 (CMAME 2026) |

### 6.6 Operator Learning — Koopman 계열

| 기법 | 설명 |
|------|------|
| KNO (Koopman Neural Operator) | 비선형 PDE를 선형 예측으로 변환, 장기 예측에 강점 |
| IKNO (Invertible KNO) | 가역 신경망 기반 Koopman, 비데카르트 도메인 확장 가능 (2025) |
| FlowDMD | Coupling Flow INN 기반 Koopman 임베딩 학습 |

### 6.7 Operator Learning — KAN 계열

| 기법 | 설명 |
|------|------|
| KANO | Kolmogorov-Arnold Neural Operator — 학습된 연산자의 해석적(symbolic) 복원 가능, 해석성 극대화 |

### 6.8 GNN 기반

| 기법 | 설명 |
|------|------|
| GNN Surrogate | 메쉬→그래프 직접 입력, 메쉬 변경에도 재학습 불필요 |
| MeshGraphNets | 시간발전 포함 GNN 시뮬레이터 (DeepMind) |
| EGNO | E(n)-Equivariant GNN Operator — 3D 동역학 대칭성 보존 (ICML 2024) |
| HAMLET | Graph Transformer Neural Operator — Transformer + GNN 결합 |
| Graph Neural PDE Solver | 보존법칙/유사성-등변 GNN 솔버 |

### 6.9 State Space Model (SSM) 기반

| 기법 | 설명 |
|------|------|
| MNO (Mamba Neural Operator) | Mamba SSM + 양방향 스캔, Transformer 대비 최대 90% 오차 감소, 선형 복잡도 (JCP 2025) |
| DeepOMamba | DeepONet + Mamba 최적 조합 — 고차원 장기 적분, FNO 대비 10배 빠름 (JCP 2025) |

### 6.10 생성 모델 기반

| 기법 | 설명 |
|------|------|
| Score-based Diffusion PDE | 확산 모델로 PDE 해 생성, 불확실성 자연 포함 |
| Wavelet Diffusion NO | 웨이블릿 확산 신경 연산자 (ICLR 2025) |
| 조건부 유동장 생성 | 파라미터 조건부로 다양한 유동장 샘플 생성 |

### 6.11 시계열 / 동역학

| 기법 | 설명 |
|------|------|
| DMD | 시계열→주파수·성장률 모드 분리, 시간예측 |
| SPOD | 주파수 도메인 POD, 비정상 유동 분석 |
| LSTM | 순환 신경망 기반 시계열 예측 |
| Transformer | 어텐션 기반 장거리 시계열 |
| TNO (Temporal Neural Operator) | FNO 인코더 + 시간 브랜치 → 시간 외삽 오차 누적 거의 없음 (Nature Sci. Rep. 2025) |
| Neural ODE | 연속 시간 동역학, 불균일 타임스텝에 강건 |
| Latent Space Dynamics | AE 잠재공간에서 Neural ODE로 시간적분 (Gonzalez & Balajewicz, Lee & Carlberg) |

### 6.12 Equivariant (대칭성 보존)

| 기법 | 설명 |
|------|------|
| Group Equivariant FNO | 회전/반사 대칭 보존 FNO (ICML 2023) |
| Physics-embedded E(n)-GNN | 물리 보존법칙 + E(n)-등변 GNN PDE 솔버 (NeurIPS 2022) |
| Lie Algebra Canonicalization | 임의 리 군 하의 등변 신경 연산자 (ICLR 2025) |

### 6.13 PINN

| 기법 | 설명 |
|------|------|
| PhysicsNEMO | NVIDIA PINN 프레임워크, 로컬 GPU 학습 |
| PINA | PyTorch+Lightning 기반 경량 PINN, 복잡한 경계조건 지원 |
| Domain Decomposition PINN | 도메인 분할 순차 학습 → 비선형 전달방정식 수렴 안정화 |

### 6.14 데이터 기반 방정식 발견 / Koopman

| 기법 | 설명 |
|------|------|
| SINDy (PySINDy) | 희소 회귀로 지배 방정식 자동 발견, PDE-FIND 포함 |
| Koopman 연산자 (PyKoopman/pykoop) | 비선형 시스템의 선형 Koopman 좌표 추정 |
| PySR | Julia 백엔드 심볼릭 회귀 — AI ROM을 수식으로 복원 |

### 6.15 디지털 트윈 핵심

| 기법 | 설명 |
|------|------|
| Data Assimilation (EnKF, DAPPER) | 실측 센서 + 모델 실시간 융합 |
| 4D-Var | 시공간 최적화 기반 데이터 동화 |
| Particle Filter | 비선형/비가우시안 시스템 |
| Multi-fidelity modeling | coarse mesh→fine mesh 전이학습, CFD 고해상도 데이터 비용 절감 |
| Transfer learning | 유사 형상/조건의 기존 모델을 새 케이스에 적응 |
| Active learning | 모델 불확실성 높은 영역에서 추가 CFD 시뮬레이션 자동 요청 |
| Online learning | 새 데이터 유입 시 모델 점진적 업데이트 (재학습 없이) |
| Physics correction | surrogate/ROM 출력이 보존법칙(질량, 운동량, 에너지) 만족하도록 사후 보정 |
| Hybrid ROM | POD-Galerkin + NN 잔차 보정 결합 |
| Data augmentation | 물리적 대칭성/갈릴레이 불변성 기반 증강, 합성 스냅샷 생성 |

### 6.16 유동 분석 (후처리)

| 기법 | 설명 |
|------|------|
| Q-criterion / λ₂ | 와류 코어 자동 식별 |
| LCS (Lagrangian Coherent Structures) | 라그랑지안 입자 궤적 기반 유동 분리선 |
| FFT / PSD | 주파수 분석, 압력 맥동/진동 |
| Wavelet Transform | 시간-주파수 동시 분석, 과도 현상 |
| 두점 상관 (Two-point correlation) | 공간 상관 길이 스케일, 난류 정량화 |
| y+ / Cf / δ·θ·H | 벽면 분석 (y+ 분포, 마찰계수, 경계층 두께) |
| Nu / Re / Pr | 무차원수 자동 계산 및 분포 시각화 |
| 엔트로피 생성 (Entropy generation) | 비가역 손실 공간 분포, 열역학 최적화 |
| PGD (Proper Generalized Decomposition) | 다파라미터 공간 분리 텐서 분해 |

### 6.17 최적화 / 불확실성

| 기법 | 설명 |
|------|------|
| Bayesian Optimization | 최소 샘플로 최적 파라미터 탐색 |
| 역문제 (Inverse Problem) | 출력(압력분포 등)→입력(형상, 경계조건) 역추정 |
| 위상 최적화 (Topology Optimization) | 유동 경로 최적 형상 도출 (DL4TO, PyTopo3D) |
| UQ - Monte Carlo | 입력 불확실성 전파 분석 |
| UQ - PCE (Polynomial Chaos Expansion) | 다항식 카오스 전개 (OpenTURNS) |
| Sobol indices | 전역 민감도 분석, 중요 파라미터 순위 (SALib) |
| Causal inference | 입출력 간 인과관계 자동 추출 및 정량화 |
| MDO (OpenMDAO) | 다분야 최적화, CFD+구조+AI 연동 (NASA) |

### 6.18 설명가능성

| 기법 | 설명 |
|------|------|
| SHAP | 모델 입출력 기여도 해석 |
| Attention visualization | Transformer/HAMLET 집중 영역 시각화 |
| Feature importance | surrogate 변수 중요도 순위 |
| KANO symbolic recovery | 학습된 연산자의 수식 자동 복원 (해석적 표현) |
| PySR symbolic recovery | 심볼릭 회귀로 AI 모델을 수식으로 변환 |

---

## 7. 설계 원칙

- **팩토리 패턴 통일:** Reader, 차원축소, Surrogate, Operator 모두 통일 인터페이스 (`fit/predict` 또는 `fit/encode/decode`)
- **core↔gui 분리:** core 모듈은 Qt 의존 금지. GUI는 시그널/슬롯으로 core와 통신
- **프로젝트 파일 (.ntwin):** HDF5 기반 세션 저장/복원, 모델 가중치+설정 이력 포함
- **GPU 폴백:** NVIDIA GPU 미탑재 시 CPU 모드 동작 (PINN/GNN/SSM 등 일부 제외), CUDA 버전 자동 체크
- **Undo/Redo 스택:** 전 패널 공통 명령 스택
- **튜토리얼 위자드:** 첫 사용자가 샘플 데이터로 전체 파이프라인을 따라가는 가이드 모드
- **i18n:** 한국어/영어 최소 지원
- **벤치마크 프레임워크:** 모델 간 정확도/속도/파라미터 수 자동 비교 대시보드

---

## 8. 참고 문헌

| 기법 | 핵심 논문/출처 |
|------|---------------|
| FNO | Li et al., ICLR 2021; neuraloperator library (v2.0+, 2025) |
| DeepONet | Lu et al., Nature Machine Intelligence 2021 |
| L-DeepONet | Kontolati et al., Nature Communications 2024 |
| PI-Latent-NO | Karumuri et al., CMAME 2026 |
| KNO/IKNO | Xiong et al., JCP 2024; Jin et al., arXiv 2025 |
| KANO | arXiv:2509.16825 |
| MNO | Mamba Neural Operator, JCP 2025 |
| DeepOMamba | Hu et al., JCP 2025 |
| TNO | Temporal Neural Operator, Nature Sci. Rep. 2025 |
| MeshGraphNets | Pfaff et al., ICML 2021 |
| EGNO | Xu et al., ICML 2024 |
| HAMLET | arXiv 2024 |
| Wavelet Diffusion NO | ICLR 2025 |
| Group Equiv FNO | ICML 2023 |
| Lie Algebra Equivariant | ICLR 2025 |
| Latent Space Dynamics | Gonzalez & Balajewicz 2018; Lee & Carlberg 2020 |
| PhysicsNEMO | NVIDIA docs.nvidia.com/physicsnemo |
| Spectral-Refiner | ICLR 2025 |
| foamlib | Gerlero et al., JOSS 2025 |
| PySPOD | Mengaldo et al., Computer Physics Communications 2024 |
| PySINDy | de Silva et al., JOSS 2020; Kaptanoglu et al., JOSS 2022 |
| PyKoopman | Hirsh et al., arXiv 2023 |
| pykoop | Dahdah & Forbes, JOSS 2025 |
| PINA | Coscia et al., GitHub mathLab/PINA |
| FEniCSx | Baratta et al., FEniCS 2023; DOLFINx v0.10, 2025 |
| JAX-Fluids | Bezgin et al., Computer Physics Communications 2025 |
| DAPPER | Raanes et al., JOSS 2024 |
| UQpy | Olivier et al., ScienceDirect 2025 (v4.2) |
| OpenTURNS | Baudin et al., Springer 2017; AI-UQ app 2025 |
| SALib | Herman & Usher, JOSS 2017; Iwanaga et al., SESMO 2022 |
| OpenMDAO | Gray et al., AIAA 2019; NASA v3.37+ 2024 |
| PySR | Cranmer, Genetic Programming & Evolvable Machines 2024 |
| PDEBench | Takamoto et al., NeurIPS 2022 |
| AirfRANS | Bonnet et al., NeurIPS 2022 |
| CFDBench | Luo et al., arXiv 2023 |
| FlowBench | arXiv:2409.18032, 2024 |
| escnn | Cesa et al., QUVA-Lab/escnn |
| ofpp | xu-xianghua/ofpp, GitHub (MIT) |
| modred | Belson et al., GitHub belson17/modred (BSD-2) |
| pyMOR | Milk et al., SIAM J. Sci. Comput. 2016; pymor/pymor (BSD-2) |
| VTKHDF 포맷 | Kitware Blog, "How to write time-dependent data in VTKHDF files", 2023 |
| pyinstaller-hooks-contrib | pyinstaller/pyinstaller-hooks-contrib (Apache 2.0) |

---

## 9. 모듈 구현 전략

핵심 모듈의 구현 방법과 라이브러리 선택 근거를 정리한다.

### 9.1 CFD Reader — OpenFOAM 폴백 체인

```
pv.POpenFOAMReader (VTK 내장, 가장 안정)
  └── 실패 시 → ofpp (MIT, ASCII/binary 양쪽, numpy 직반환)
                └── 실패 시 → foamlib (GPL-3.0, 격리 필요)
```

- `pv.POpenFOAMReader`: `.foam` 더미 파일이 케이스 루트에 있어야 함. 병렬 분해(decomposedCase) 지원. `mesh.point_data["U"]`로 직접 필드 접근.
- `ofpp.FoamMesh`: MIT 라이선스로 직접 포함 가능. `points`, `faces`, `owner`, `neighbour`를 읽어 PyVista `UnstructuredGrid`로 재조립.
- `foamlib`: GPL-3.0이므로 서브프로세스 격리 또는 오픈소스 전용 사용.

### 9.2 CFD Reader — Fluent / CGNS

| 포맷 | 리더 | 비고 |
|------|------|------|
| Fluent `.cas.h5` | `pv.FLUENTCFFReader` | PyVista 내장 (MIT) |
| Fluent `.cas/.dat` | `pv.FluentReader` | PyVista 내장 (MIT) |
| CGNS | `pv.CGNSReader` | VTK 9.0+ 내장 |
| CGNS (변환) | `meshio` (`src/meshio/cgns/`) | MIT, CGNS → VTK 브릿지 |

### 9.3 POD/SVD 구현

```python
# pod.py 핵심 — sklearn randomized SVD (대용량 스냅샷)
from sklearn.utils.extmath import randomized_svd
import numpy as np

def compute_pod(snapshots: np.ndarray, n_modes: int):
    """snapshots: (n_features, n_snapshots)"""
    U, s, Vt = randomized_svd(snapshots, n_components=n_modes)
    energy = np.cumsum(s**2) / np.sum(s**2)  # 에너지 누적 기여율
    return U, s, Vt, energy
```

- **modred** (`modred/pod.py`): MPI 병렬화 필요 시 사용. `InnerProductArray`로 임의 데이터 타입 지원.
- **pyMOR** (`src/pymor/algorithms/svd_ei.py`): Gram-Schmidt POD 기반, `singular_values` 직접 노출. 아키텍처 참고.
- 대용량 단순 케이스: `torch.linalg.svd` (GPU 가속 가능).

### 9.4 Q-criterion / λ₂ 구현

```python
# q_criterion.py
def compute_q_criterion(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """PyVista VTK GradientFilter 기반 — cell-centered velocity 주의"""
    if "U" in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()
    return mesh.compute_derivative(scalars="U", qcriterion=True, vorticity=True)

# lambda2.py — numpy 직접 구현
def compute_lambda2(grad_u: np.ndarray) -> np.ndarray:
    """grad_u: (N_cells, 3, 3) 속도 구배 텐서"""
    S = 0.5 * (grad_u + grad_u.transpose(0, 2, 1))     # 변형률 텐서
    O = 0.5 * (grad_u - grad_u.transpose(0, 2, 1))     # 회전률 텐서
    M = S @ S + O @ O
    eigvals = np.linalg.eigvalsh(M)                     # (N, 3) — 오름차순
    return eigvals[:, 1]                                # λ₂ = 두 번째 고유값
```

### 9.5 y+ 벽면 분석 구현

```python
# yplus.py
import numpy as np
from numpy.typing import NDArray

def compute_yplus(
    wall_shear_stress: NDArray,  # shape (N_wall_cells, 3) [Pa]
    rho: float,                  # 밀도 [kg/m³]
    nu: float,                   # 동점성계수 [m²/s]
    y_wall: NDArray,             # 첫 번째 셀 중심까지 거리 [m]
) -> NDArray:
    tau_w = np.linalg.norm(wall_shear_stress, axis=-1)
    u_tau = np.sqrt(tau_w / rho)   # friction velocity
    return u_tau * y_wall / nu

def estimate_first_cell_height(
    y_plus_target: float, Re: float, L: float,
    nu: float, rho: float, U_inf: float
) -> float:
    """Schlichting 경계층 상관식 기반 첫 번째 셀 높이 추정"""
    Cf = 0.026 * Re**(-1/7)
    tau_w = 0.5 * Cf * rho * U_inf**2
    u_tau = np.sqrt(tau_w / rho)
    return y_plus_target * nu / u_tau
```

- OpenFOAM 케이스의 경우 `postProcess -func wallShearStress` 결과를 ofpp/foamlib로 직접 읽어 사용.
- 결과 없으면 속도 구배 텐서의 벽면 법선 성분에서 Python으로 직접 계산.

### 9.6 .ntwin 내부 포맷 (VTKHDF 기반)

ParaView 직접 호환을 위해 VTKHDF 표준을 기반으로 확장:

```
project.ntwin  (HDF5)
├── VTKHDF/                      ← ParaView VTKHDF 표준 그룹
│   ├── attrs: Version=(2,0), Type="UnstructuredGrid"
│   ├── Points                   shape (N_total, 3), resizable
│   ├── Connectivity
│   ├── Offsets
│   ├── Types                    VTK cell type codes
│   ├── NumberOfPoints           per-timestep
│   ├── NumberOfCells
│   ├── PointData/
│   │   ├── U                   velocity, shape (N_total, 3)
│   │   └── p                   pressure, shape (N_total,)
│   └── Steps/
│       ├── Values               time values array
│       ├── PointOffsets         per-timestep pointer into Points
│       └── PointDataOffsets/
│           ├── U
│           └── p
└── NavierTwin/                  ← 확장 그룹 (NavierTwin 전용)
    ├── Metadata/
    │   ├── project_info         JSON string (이름, 작성자, 날짜)
    │   └── cfd_params           JSON string (Re, 경계조건 등)
    ├── Models/
    │   ├── POD/
    │   │   ├── modes            shape (n_features, n_modes)
    │   │   ├── singular_values
    │   │   └── energy
    │   └── FNO/
    │       └── weights          TorchScript 직렬화 bytes
    └── Sessions/
        └── last_state           JSON string (GUI 상태)
```

**구현 원칙:**
- 메쉬 토폴로지(Points, Connectivity)는 첫 타임스텝에만 저장, 이후 `PointOffsets`으로 포인터만 이동
- field는 `dset.resize()` + `dset[n:]` append 패턴으로 스트리밍 저장
- ParaView에서 `VTKHDF/` 그룹만 읽으면 타임스텝 시각화 즉시 가능

### 9.7 PySide6 + pyvistaqt Qt 뷰어 패턴

```python
# widgets/vtk_viewer.py
import os
os.environ["QT_API"] = "pyside6"
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider, QComboBox
from PySide6.QtCore import Qt, Signal

class VtkViewer(QWidget):
    timestep_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter, stretch=9)

        # 타임스텝 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.timestep_changed)
        layout.addWidget(self.slider, stretch=1)

        # 컬러맵 선택기
        self.cmap_box = QComboBox()
        self.cmap_box.addItems(["coolwarm", "viridis", "jet", "rainbow"])
        self.cmap_box.currentTextChanged.connect(self._rerender)
        layout.addWidget(self.cmap_box)

    def show_mesh(self, mesh, scalars: str):
        self.plotter.clear()
        self.plotter.add_mesh(mesh, scalars=scalars, cmap=self.cmap_box.currentText())
        self.plotter.reset_camera()

    def _rerender(self, _):
        self.plotter.render()
```

- `QtInteractor`를 `QVBoxLayout`에 직접 임베드 (독립 창 모드는 `BackgroundPlotter` 사용)
- VTK 내장 위젯(`add_slider_widget`)과 Qt 위젯을 혼용하지 말 것 — UI 일관성 저하

### 9.8 PyInstaller + PyTorch/CUDA 패키징 전략

```python
# installer/naviertwin.spec  (핵심 부분)
from PyInstaller.utils.hooks import collect_all

torch_datas, torch_bins, torch_hidden   = collect_all("torch")
pyside6_datas, pyside6_bins, pyside6_h  = collect_all("PySide6")
pyvista_datas, _, pyvista_h             = collect_all("pyvista")

a = Analysis(
    ["main.py"],
    datas=[*torch_datas, *pyside6_datas, *pyvista_datas,
           ("src/naviertwin/gui/styles/*.qss", "styles")],
    binaries=[*torch_bins, *pyside6_bins],
    hiddenimports=[
        *torch_hidden, *pyside6_h, *pyvista_h,
        "vtkmodules.all",          # PyVista 동적 VTK 로드
        "torch._C._jit",
    ],
    excludes=["tkinter", "matplotlib", "torch.distributions"],
)
```

**필수 규칙:**
- `--onedir` 사용 (`--onefile`은 CUDA DLL 경로 문제로 실패)
- `vtkmodules.all` hiddenimport 필수
- `--copy-metadata torch` 추가 (일부 PyTorch 기능이 패키지 메타데이터를 읽음)
- Windows Defender 오탐지 방지를 위해 Inno Setup + 코드 서명 권장
