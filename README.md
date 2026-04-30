# NavierTwin

CFD 후처리 결과를 AI/ROM/Operator Learning 기반 디지털 트윈으로 변환하는 로컬 GUI 도구입니다.

## 현재 상태

- 개발 단계: ROADMAP 기준 릴리스 하드닝 진행 중
- 패키지 버전 확인: `naviertwin --version`
- 고객용 release smoke와 코어/회귀 스위트 통과, ruff 린트 깨끗
- 로컬 실행 · MIT 오픈소스 · 데이터 외부 유출 없음

상세 범위는 다음 문서 참고.

- 기술 명세: [`SPEC.md`](SPEC.md)
- 구현 계획: [`PLAN.md`](PLAN.md)
- 진행 현황: [`ROADMAP.md`](ROADMAP.md)

## 구현된 핵심 기능

### CFD I/O
- OpenFOAM (pv/ofpp/foamlib 폴백), VTK, Fluent (.cas/.dat ASCII), CGNS, Gmsh (.msh), SU2 (.su2)
- 내부 포맷: `.ntwin` (VTKHDF 확장, HDF5)

### 전처리
- **메쉬 생성**: 파라미터 채널/원통/NACA 익형 (Gmsh OCC)
- **메쉬 후처리**: PyMeshLab 단순화/스무딩 + PyVista 품질 보고서

### 차원축소
- 선형: Snapshot POD, Randomized SVD, Incremental POD, MRPOD, Constrained POD (제약 null-space 투영)
- 비선형: Autoencoder, β-VAE (샘플링 생성), GNN-AE (torch_geometric)
- 텐서: Tucker 분해 (HOSVD + HOOI)
- 기하학적: Diffusion Maps (Coifman-Lafon α-정규화)

### 모달/통계
- DMD, SPOD (Welch-block + PySPOD 옵션), PGD (3D greedy)
- FFT/PSD, CWT 웨이블릿, 두점 상관 + 적분 길이
- LCS FTLE (RK4 flow-map + Cauchy-Green)

### 유동 분석
- Q-criterion / λ₂, y+ / Cf / δ₉₉ / δ* / θ / H
- 무차원수: Re, Pr, Nu, Pe, Gr, Ra
- 엔트로피 생성율 (Bejan, thermal + viscous)
- Couette / Poiseuille 2D / Pipe 해석해 + 수치해 자동 비교

### Surrogate
- RBF, Kriging (SMT), Bayesian Optimization (GP + EI), Co-Kriging 멀티피델리티

### 신경 연산자
- **FNO1D/2D**, TFNO2D (Tucker-factorized), WNO1D (웨이블릿)
- **DeepONet**, PI-DeepONet (물리 잔차), MIONet (multi-input)
- U-Net 2D, KANO1D (KAN + spectral)
- C4 Equivariant FNO (회전 평균)

### GNN
- GNN Surrogate (GCN N-layer), MeshGraphNets (Encode-Process-Decode)

### 시계열 / Koopman
- LSTM, Temporal Transformer, Neural ODE (torchdiffeq + RK4 폴백)
- KNO (Koopman Neural Operator), Latent Dynamics (AE + Neural ODE)

### 생성 모델
- Score-based Diffusion PDE (DDPM 형태)

### PINN / 물리 보정
- PINNSolver (PDE 잔차 + BC), Hybrid ROM (POD + NN 잔차)
- 선형 제약 투영, 질량 보존 스케일링
- SINDy (STLSQ + PySINDy 옵션)

### 데이터 동화 / UQ / 최적화
- EnKF (inflation), Particle Filter (SIR + systematic resample), 4D-Var (선형), UKF
- Sobol 민감도 (Saltelli + SALib 옵션), MC 전파, KernelSHAP, Attention viz
- NSGA-II 다목적, SIMP 위상 최적화 (2D), Bayesian Optimization

### 멀티피델리티 / Transfer / Active
- Additive Co-Kriging, freeze/finetune 전이학습
- Variance-based active learning + loop

### 디지털 트윈 / GUI
- `NavierTwinPipeline`: 6 단계 end-to-end 오케스트레이터
- 데이터셋→트윈 자동 빌드: CFD reader 또는 CSV snapshot 시퀀스에서 metrics/report/checkpoint/engine 생성, manifest에 artifact SHA256/bytes 기록
- 자동 모델 sweep: ROM/surrogate 후보군을 같은 데이터에서 평가하고 RMSE 기준 랭킹
- GUI: Import / Analyze / Reduce / Model / Twin / Export / Compare / Simulation / Explain / Post-Tools 10 탭
- 분석 패널: Q-criterion / λ₂ / FFT / y+ / 해석해 비교, SPOD, SINDy, Wavelet/STFT, BL, nondim, FTLE, PGD, entropy quick checks
- 모델 패널: Kriging/RBF + 신경 연산자(FNO/TFNO/DeepONet/UNet/WNO) + Active Learning 후보 추천/Online Update
- Twin 패널: 파라미터 예측, 저장/로드, Surrogate/Bayesian inverse-design, NSGA-II/SIMP 설계 quick-check, 4D-Var/PF/UKF 동화 quick-check
- Export 패널: `.ntwin`, VTK/CSV, 고객 보고서, ONNX/TorchScript/FMI-FMU 모델 아티팩트
- Explain 탭: Kernel SHAP feature attribution, symbolic expression fit, Attention weight matrix/top-k token viz
- Post-Tools: 29개 facade 연산(PSD/Reynolds/flux/integrals/EOF/derivatives/topology)을 GUI/CLI/API 공통 이름으로 실행
- Tools 메뉴: benchmark smoke, pipeline-demo 산출물 생성, CSV 스냅샷→트윈 생성, 저장된 트윈 예측/검증, 트윈 산출물 ZIP 패키징/검증, FastAPI 서버 start/stop
- 모델 비교 대시보드, loss curve 위젯, 튜토리얼 위자드, i18n/테마 전환

### 설명가능성
- Kernel SHAP (MC Shapley), Granger causality, 상관행렬

### Export / 배포
- ONNX, TorchScript (trace/script), FMI 2.0 FMU handoff archive
- Jinja2 HTML 보고서 + weasyprint PDF
- FastAPI REST 서버 (`/health`, `/reduce`, `/reduce/pod`, `/analytic/*`, `/simulate/lbm_cavity`, `/optimize/bayesian`)
- PyInstaller spec + Inno Setup (`installer/naviertwin.iss`)

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[core,dev]"

# 고객용 smoke 검증
python scripts/release_smoke.py

# wheel artifact 검증
python scripts/wheel_smoke.py --install-smoke

# sdist artifact 검증
python scripts/sdist_smoke.py --install-smoke

# 로컬 릴리스 메타데이터 기반 업데이트 확인
naviertwin --version
naviertwin update-check --metadata examples/release-metadata.example.json

# 최소 quickstart smoke (복붙 가능한 설치 확인)
naviertwin --version
naviertwin preflight tests/fixtures/tiny_square.su2 --json

# 기본 benchmark / 서버 / 자동 고도화 dry-run
naviertwin benchmark --kind burgers
naviertwin server --host 0.0.0.0 --port 8000
naviertwin autorefine --iterations 1 --dry-run

# 설치/런타임 환경 진단
naviertwin doctor --json
naviertwin doctor --json --output /tmp/naviertwin-doctor.json

# CFD 입력 데이터 readiness 점검
naviertwin preflight tests/fixtures/tiny_square.su2 --json
naviertwin preflight tests/fixtures/tiny_square.su2 --json --output /tmp/naviertwin-preflight.json

# 고객 지원용 진단 번들 생성
naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2
naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2 --zip

# 라이선스/의존성 실사 리포트
python scripts/license_report.py --json --output /tmp/naviertwin-license-report.json

# 합성 파이프라인 데모 산출물 생성
naviertwin pipeline-demo --outdir /tmp/naviertwin-pipeline-demo

# ROM/surrogate 후보 자동 비교
naviertwin model-sweep --reducers pod --n-modes 2,3,5 --surrogates rbf,kriging --json

# CFD/CSV 데이터셋에서 트윈 산출물 생성
naviertwin build-twin --csv-snapshots "case/snapshots/*.csv" --field-column U --outdir /tmp/naviertwin-twin --json
naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json
naviertwin predict-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --output /tmp/naviertwin-prediction.csv --json
naviertwin benchmark-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --warmup 2 --repeat 20 --output /tmp/naviertwin-latency.json --json
naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json
naviertwin validate-twin --artifacts-dir /tmp/naviertwin-deploy --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json
naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --json
naviertwin inspect-twin-package --package /tmp/naviertwin-twin.zip --json
naviertwin verify-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-deploy --json

# 전체 core 회귀 수집
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mpl pytest --collect-only -q

# GUI 실행
naviertwin --gui
```

선택 의존성 (Gmsh / PyMeshLab / Dedalus / FastAPI / PyTorch Geometric):

```bash
pip install -e ".[full]"
```

## API 사용 예시

```python
from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 30))  # (n_features, n_snapshots)

pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=5, surrogate_kind="kriging")
pipe.load_snapshots(X, field_name="U")
pipe.reduce()
params = np.linspace(0, 1, 30).reshape(-1, 1)
pipe.fit_surrogate(params)
metrics = pipe.validate(params[-8:], pipe.state.coeffs[-8:])
pipe.export_report("report.html", project="Demo")
```

## REST 서버

```bash
naviertwin server --host 0.0.0.0 --port 8000
```

주요 엔드포인트:

- `GET /health`
- `POST /analytic/couette`
- `POST /analytic/poiseuille_2d`
- `POST /reduce` — `pod`, `incremental_pod`, `mrpod`
- `POST /reduce/pod` — 하위 호환 POD 전용
- `POST /simulate/lbm_cavity`
- `POST /optimize/bayesian`

## 프로젝트 구조

```text
src/naviertwin/
├── core/
│   ├── cfd_reader/           # OpenFOAM/VTK/Fluent/CGNS/Gmsh/SU2
│   ├── dimensionality_reduction/  # POD/rSVD/cPOD/AE/VAE/GNN-AE/Tucker/DiffMaps
│   ├── flow_analysis/        # Q-crit/DMD/SPOD/FFT/CWT/2pc/BL/nondim/LCS/PGD/엔트로피
│   ├── surrogate/            # RBF/Kriging
│   ├── operator_learning/    # FNO/TFNO/WNO/DeepONet/PI-DeepONet/MIONet/UNet/KANO/KNO
│   ├── gnn/                  # GNN Surrogate / MeshGraphNets
│   ├── time_series/          # LSTM/Transformer/NeuralODE/Latent Dynamics
│   ├── generative/           # Diffusion PDE
│   ├── equivariant/          # C4 Equivariant FNO
│   ├── physnemo/             # PINNSolver
│   ├── physics_correction/   # 제약 투영 + Hybrid ROM
│   ├── data_assimilation/    # EnKF/PF/4D-Var
│   ├── optimization/         # BO/NSGA-II/SIMP/MC
│   ├── sensitivity/          # Sobol/Granger/correlation
│   ├── multi_fidelity/       # Co-Kriging/Transfer
│   ├── online_learning/      # Active learning, OnlineKriging/OnlineNN updates
│   ├── explainability/       # KernelSHAP, Attention viz, symbolic regression
│   ├── digital_twin/         # TwinEngine / Pipeline
│   ├── validation/           # metrics / analytic solutions
│   ├── export/               # .ntwin / ONNX / TorchScript
│   ├── report/               # Jinja2 HTML/PDF
│   └── tools/                # Gmsh 메쉬 생성 / PyMeshLab 후처리
├── gui/
│   ├── panels/               # 10 탭 패널 + Post-Tools facade
│   ├── widgets/              # VTK viewer / loss curve / compare / analytic compare
│   ├── wizard/               # 튜토리얼
│   └── styles/               # QSS + i18n (ko/en)
├── api/                      # FastAPI REST 서버
└── utils/                    # config / logger / undo_redo / i18n
```

## 테스트 전략

- 고객용 smoke: 패키징 메타데이터, CLI, GUI offscreen, Post-Tools 패널을 우선 검증한다.
- 권장 smoke 명령: `python scripts/release_smoke.py`
- wheel artifact 검증: `python scripts/wheel_smoke.py --install-smoke`
- sdist artifact 검증: `python scripts/sdist_smoke.py --install-smoke`
- 업데이트 메타데이터 검증: `naviertwin update-check --metadata examples/release-metadata.example.json`
- benchmark smoke: `naviertwin benchmark --kind burgers`
- REST 서버 실행: `naviertwin server --host 0.0.0.0 --port 8000`
- 자동 고도화 dry-run: `naviertwin autorefine --iterations 1 --dry-run`
- 설치/런타임 환경 진단: `naviertwin doctor --json`
- CFD 입력 데이터 readiness 점검: `naviertwin preflight tests/fixtures/tiny_square.su2 --json --output /tmp/naviertwin-preflight.json`
- 고객 지원 번들 생성: `naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2`
- 고객 지원 번들 생성(ZIP 포함): `naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2 --zip`
- 라이선스/의존성 실사 리포트: `python scripts/license_report.py --json --output /tmp/naviertwin-license-report.json`
- 합성 파이프라인 데모: `naviertwin pipeline-demo --outdir /tmp/naviertwin-pipeline-demo`
- ROM/surrogate 후보 자동 비교: `naviertwin model-sweep --reducers pod --n-modes 2,3,5 --surrogates rbf,kriging --json`
- CFD/CSV 데이터셋에서 트윈 산출물 생성: `naviertwin build-twin --csv-snapshots "case/snapshots/*.csv" --field-column U --outdir /tmp/naviertwin-twin --json`
- 저장된 트윈 예측 실행: `naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json`
- 배포 트윈 디렉토리 예측 실행: `naviertwin predict-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --output /tmp/naviertwin-prediction.csv --json`
- 배포 트윈 지연시간 측정: `naviertwin benchmark-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --warmup 2 --repeat 20 --output /tmp/naviertwin-latency.json --json`
- 저장된 트윈 검증 실행: `naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json`
- 배포 트윈 디렉토리 검증 실행: `naviertwin validate-twin --artifacts-dir /tmp/naviertwin-deploy --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json`
- 트윈 산출물 ZIP 패키징(README.txt/delivery.json 포함): `naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --json`
- 트윈 전달 ZIP 구성 조회: `naviertwin inspect-twin-package --package /tmp/naviertwin-twin.zip --json`
- 트윈 전달 ZIP 검증/안전 추출: `naviertwin verify-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-deploy --json`
- 전체 core 회귀: `QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mpl pytest -q`
- 전체 collection 안전성: `QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mpl pytest --collect-only -q`
- optional 의존성이 필요한 모듈은 `pytest.mark.optional`로 기본 core 실행에서 제외한다.
- pyMOR/MPI 계열 optional 테스트는 `NAVIER_TWIN_RUN_PYMOR=1`을 명시한 환경에서 별도로 실행한다.

## 라이선스

MIT License (상업적 이용 가능). 자세한 내용은 `LICENSE` 참조.
