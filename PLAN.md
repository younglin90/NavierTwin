# NavierTwin 구현 계획서 (PLAN)

> 시스템을 **어떻게** 만들지 정의하는 문서. 프로젝트 목표, 버전별 범위, 우선순위 기준, 기술 선택 근거를 포함한다.
> 기술 명세(스택·기법 목록)는 `SPEC.md`, Phase별 진행 현황은 `ROADMAP.md` 참조.

---

## 1. 프로젝트 목표

| 구분 | 내용 |
|------|------|
| 핵심 목표 | CFD 엔지니어가 시뮬레이션 결과를 AI/ROM으로 즉시 디지털 트윈화할 수 있는 로컬 GUI 툴 제공 |
| 타겟 사용자 | CFD 코드를 직접 짜기 어려운 엔지니어 일반 사용자 (연구자·산업체) |
| 핵심 가치 | 로컬 실행(데이터 외부 유출 없음), 오픈소스, 설치 즉시 사용 가능 |
| 비목표 | 클라우드 SaaS, CFD 솔버 자체 개발, 상업 라이선스 |

---

## 2. 버전별 범위 및 근거

### v1.0 (MVP) — 핵심 파이프라인 검증

**목표:** 가장 흔한 CFD 포맷(OpenFOAM, Fluent, VTK)을 읽고, 기본 ROM+Surrogate로 디지털 트윈을 완성하는 end-to-end 파이프라인 동작 확인.

| 영역 | 항목 | 근거 |
|------|------|------|
| I/O | OpenFOAM, Fluent, VTK Reader | 산업·학계 가장 널리 쓰이는 3종 |
| 차원축소 | POD, Randomized SVD | 선형 방법이 가장 안정적이고 디버깅 쉬움 |
| 모달 | DMD | 시계열 ROM의 사실상 표준 |
| 유동분석 | Q-criterion, FFT/PSD, y+ | 와류·스펙트럼·벽면 — 가장 많이 요청되는 3종 |
| Surrogate | RBF, Kriging | 구현 단순, 소량 데이터에서도 동작 |
| PINN | PhysicsNEMO | NVIDIA 공식 지원으로 CUDA 통합 용이 |
| 디지털 트윈 | predict(params)→field 엔진 | MVP 핵심 기능 |
| 검증 | RMSE, R², L2 norm | 모든 모델에 공통 적용 가능한 최소 지표 |
| 내보내기 | .ntwin 프로젝트 저장/복원 | 세션 재현성 확보 |
| GUI | 기본 6패널 + 3D VTK 뷰어 | 워크플로우 전체 흐름 확인 |
| 패키징 | PyInstaller | Windows 배포 최소 요건 |

### v1.5 — 신경 연산자 + DA + 보고서

**목표:** 연구자가 실제로 논문에 쓸 수 있는 수준의 모델(FNO, DeepONet)과 데이터 동화, 자동 보고서 추가.

| 영역 | 항목 | 근거 |
|------|------|------|
| I/O | CGNS, STAR-CCM+ | 항공우주·자동차 산업 대응 |
| 차원축소 | Autoencoder, GNN-AE | 비정형 메쉬 비선형 압축 |
| 모달 | SPOD (PySPOD) | 난류 비정상 유동 분석 표준 |
| 유동분석 | Wavelet, 두점상관 | 과도현상·난류 스케일 분석 |
| Surrogate | NN Surrogate, Ensemble | 고차원 입출력 대응 |
| FNO 계열 | FNO, TFNO, WNO | neuraloperator v2.0 기반, 해상도 독립 |
| DeepONet 계열 | DeepONet, PI-DeepONet, MIONet | 비균일 격자 직접 처리 |
| 잠재공간 | L-DeepONet | 고차원 실시간 예측 |
| Koopman | KNO | 장기 예측 ROM |
| GNN | GNN Surrogate | 비정형 메쉬 직접 입력 |
| 시계열 | LSTM | 기본 순환 시계열 예측 |
| 데이터동화 | EnKF (DAPPER) | 실시간 센서 융합 |
| 보정 | Physics correction | 보존법칙 만족 보장 |
| 증강 | 물리기반 Data augmentation | 소량 데이터 문제 완화 |
| 최적화 | Bayesian Opt, Sobol indices (SALib) | 파라미터 탐색 + 민감도 분석 |
| 설명 | SHAP | 모델 해석 기본 요건 |
| 검증 | 해석해 자동비교 (Couette, Poiseuille) | 구현 정확도 자동 검증 |
| 내보내기 | ONNX export | 타 플랫폼 배포 |
| 보고서 | Jinja2→PDF 자동보고서 | 결과 공유 자동화 |
| GUI | 튜토리얼 위자드, i18n(한/영) | 진입 장벽 낮춤 |
| 패키징 | Inno Setup 설치파일 | Windows 정식 설치 |
| PINN | Domain Decomposition PINN | 복잡 도메인 수렴 안정화 |
| 방정식 발견 | PySINDy | 데이터→방정식 자동 추출 |
| UQ | UQpy, OpenTURNS | AI 예측 불확실성 정량화 |

### v2.0 — 고급 모델 전체 통합

**목표:** GNN/SSM/생성모델/등변 신경망 등 최신 연구 기법을 모두 포함하여 연구 플랫폼으로 완성.

| 영역 | 항목 |
|------|------|
| I/O | 전체 포맷 커버 |
| 차원축소 | VAE, Tucker, CPOD, Diffusion Maps |
| 모달 | PGD |
| 유동분석 | LCS, 엔트로피생성 |
| Surrogate | Mixture of Experts (MoE) |
| FNO 계열 | Adaptive FNO, LNO, Spectral-Refiner |
| DeepONet 계열 | Sequential DeepONet, NFNO-DeepONet |
| 잠재공간 | PI-Latent-NO |
| Koopman | IKNO, FlowDMD |
| KAN | KANO |
| GNN | MeshGraphNets, EGNO, HAMLET |
| SSM | MNO (Mamba Neural Operator), DeepOMamba |
| 생성 모델 | Diffusion PDE, Wavelet Diffusion NO, 조건부 유동장 생성 |
| 시계열 | Transformer, TNO, Neural ODE, Latent Space Dynamics |
| Equivariant | Group Equivariant FNO, E(n)-GNN, Lie Algebra Canonicalization |
| 피델리티 | Transfer learning |
| 데이터동화 | 4D-Var, Particle Filter |
| 보정 | Hybrid ROM (POD-Galerkin + NN) |
| 학습 | Active learning, Online learning |
| 최적화 | 역문제, 위상최적화(DL4TO), UQ(MC, PCE), MDO(OpenMDAO), Causal inference |
| 설명 | Attention visualization, KANO symbolic recovery, PySR symbolic recovery |
| 검증 | 모델간 비교 대시보드 |
| 내보내기 | FastAPI 서버 모드 |
| GUI | Undo/Redo, 모델비교 대시보드 |
| 패키징 | 자동 업데이트 |
| 미분가능 CFD | JAX-Fluids 통합 (end-to-end 미분) |

---

## 3. 우선순위 기준

1. **사용 빈도** — 가장 많은 CFD 사용자가 필요로 하는 기능 먼저
2. **구현 안정성** — 성숙한 라이브러리 기반 모듈 먼저, 최신 연구 코드는 후순위
3. **의존성 최소화** — v1.0에서 무거운 선택적 의존성(e.g. JAX, Julia/PySR)은 설치 옵션으로
4. **검증 가능성** — 해석해가 있는 기법(Couette, Poiseuille)으로 자동 검증 가능한 것 우선
5. **GPU 없이도 동작** — CPU 폴백이 가능한 기법을 우선 구현

---

## 4. 기술 선택 근거

| 선택 | 이유 | 대안 검토 |
|------|------|-----------|
| PySide6 (Qt6) | 크로스플랫폼, Python 바인딩 성숙도, 상업 무료 | wxPython (생태계 작음), tkinter (3D 통합 어려움) |
| PyVista | VTK Python 래퍼 최고 성숙도, pyvistaqt로 Qt 통합 용이 | Mayavi (유지보수 약화), vedo |
| HDF5 (.ntwin) | 메쉬+필드+가중치를 단일 파일에, 계층 구조, 대용량 지원 | NetCDF (CFD 표준 아님), SQLite (바이너리 배열 비효율) |
| foamlib | 2025 JOSS, 비동기 지원, 타입힌트 완비 | fluidfoam (바이너리 미지원), PyFoam (구식 API) |
| neuraloperator v2.0+ | FNO 원저자 라이브러리, GINO/UQNO 포함, 활발한 개발 | 자체 구현 (유지보수 부담) |
| DAPPER | DA 알고리즘 벤치마크 표준, 2024 JOSS | filterpy (기본 EnKF만 지원) |
| SALib | Sobol/Morris/FAST 통합, 2022 SESMO | 자체 구현 (검증 비용 큼) |
| PySR | Julia 백엔드로 속도 탁월, 2024 학술지 게재 | gplearn (성능 낮음) |

---

## 5. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| PhysicsNEMO 설치 복잡 (CUDA 버전 종속) | PINA를 경량 대안으로 병행 지원 |
| mamba-ssm Windows 지원 불안정 | WSL 또는 Linux 환경 권장 문서화, CPU 폴백 |
| PySR Julia 런타임 번들링 어려움 | PyInstaller 배포 시 선택적 플러그인으로 분리 |
| neuraloperator 대형 모델 GPU 메모리 | 배치 크기 자동 조정, gradient checkpointing 옵션 |
| 상업 포맷(Fluent .cas.h5) 스펙 비공개 | meshio + h5py 조합으로 역공학, 공개 스펙 최대 활용 |
