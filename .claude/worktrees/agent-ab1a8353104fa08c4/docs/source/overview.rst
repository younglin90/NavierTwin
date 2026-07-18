개요 (Overview)
================

NavierTwin 은 CFD 시뮬레이션 결과를 AI/ROM (Reduced Order Model) 기반 **디지털
트윈** 으로 변환하는 올인원 로컬 GUI/CLI 도구입니다.

주요 기능
---------

- **CFD I/O**: OpenFOAM / VTK / Fluent / CGNS / Gmsh / SU2 지원
- **차원축소**: POD / Randomized SVD / BPOD / cPOD / AE / VAE / GNN-AE / Tucker / Diffusion Maps
- **신경 연산자**: FNO / TFNO / WNO / DeepONet / PI-DeepONet / MIONet / U-Net / KANO / HAMLET / GNN / MeshGraphNets
- **시계열 / Koopman**: LSTM / Transformer / Neural ODE / KNO / IKNO / FlowDMD / ESN / TNO
- **PINN / 물리 보정**: PINNSolver / Deep Ritz / Domain Decomposition / HybridROM + HybridROMAdv
- **생성 모델**: Score-based Diffusion PDE / Wavelet Diffusion / Conditional VAE / Langevin sampler
- **데이터 동화 / UQ**: EnKF / Particle Filter / 4D-Var / Sobol (Saltelli/FAST/PAWN/Delta) / PCE / MC
- **최적화**: Bayesian Optimization (BoTorch qEI/UCB) / NSGA-II / SIMP / NLopt (14 algos) / SurrogateOpt
- **Surrogate**: RBF / Kriging (SMT) / KPLS / GEKPLS / IDW / QP / Ensemble / MoE
- **실시간 Digital Twin**: StreamingDigitalTwin + EnKF 동화
- **외부 솔버**: foamlib (OpenFOAM) / LBM D2Q9 / Burgers / Heat / SPH / FVM + Lettuce / flowtorch / JAX-Fluids 래퍼
- **Explainability**: KernelSHAP / Granger / Attention viz / Symbolic regression

디자인 원칙
-----------

1. **로컬 실행** — 데이터 외부 유출 없음
2. **Optional dependencies** — 핵심 기능은 자체 구현, 고급 기능은 [full] extras
3. **GPL-3.0 오픈소스**
4. **재현 가능성** — Profile JSON 으로 seed + 의존성 버전 고정

아키텍처
---------

- ``core/`` — 순수 Python 연산 로직 (Qt 독립)
- ``gui/`` — PySide6 기반 10 탭 인터페이스
- ``api/`` — FastAPI REST 서버 (optional)
- ``utils/`` — config, logger, i18n, profile, callbacks
