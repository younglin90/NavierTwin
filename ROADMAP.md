# NavierTwin 로드맵

> Phase별 세부 태스크 체크리스트. 버전별 목표·범위·근거는 `PLAN.md` 참조.

## 현재 단계: v5.0~v5.2 진행 중 — 일반 CFD 트윈 플랫폼 전환

상세 설계·갭 분석·전략 카탈로그: `.omc/plans/twin-platform-roadmap.md`

- [x] v5.0-M1: 능력 기반 전략 레지스트리(`core/digital_twin/strategies.py`) —
      ②Model 카드/데스크톱 어드바이저가 로드 시점에 가능/불가+이유 표시
- [x] v5.0-M1: 비정상×다케이스 시간축 보존 — 케이스 세트가 (μ, t) 로 학습,
      ③Twin 에 t 슬라이더 자동 생성 (`sweep_unsteady` 데모, PVD 우선 규칙)
- [x] v5.0-M2: 벡터 성분 보존 — U 가 U_x/U_y/U_z 채널로 학습(방향 유지),
      U_mag 는 파생
- [x] v5.2: ParametricDMD — 비정상 스윕의 (μ, t) 예보 (partitioned + ezyrb 보간)
- [x] GUI 패리티: 데스크톱 전략 어드바이저 + 데모 데이터 메뉴 + 웹 엔진
      param_names 호환
- [ ] v5.2: EZyRB 서로게이트 (POD-GPR·UQ / POD-NN) — 진행 중
- [ ] v5.2: FNO+SDF 채널 (형상=SDF 입력 채널, DeepCFD/Thuerey 방식) — 진행 중
- [x] v5.2: EZyRB 서로게이트 — `ezyrb_gpr`(예측 σ UQ)/`ezyrb_ann` 키, 리더보드 포함
- [x] v5.2: FNO+SDF 채널 core — `GeometryFNO2D` + `cases_to_grid_tensors`
      (형상=SDF 채널 · 조건=브로드캐스트 채널, DeepCFD/Thuerey 방식)
- [x] v5.2: GeometryFNO service/app 배선 — operator 전략이 정상 형상가변 케이스
      세트에서 활성화(공통 격자+SDF 채널), 예측은 공통 격자 뷰어 전환
- [x] v5.2: ParametricDMD(비정상 스윕 μ,t 예보), EZyRB(GPR·UQ/NN) 배선
- [x] v5.4: 셀별 오차장(twin_error) + 실제/트윈 요약 지표 + 정직한 외삽 인지
- [x] v5.6 P0: 학습 디바이스 배지(GPU/CPU), AMP·미니배치(A3, ~2.2배), OOD 3단계
      지지집합 상태(IN/NEAR/OUT), GeometryFNO 마스크 손실(0-채움 셀 loss 제외)
- [x] 검토 반영: 데이터 계약 우선 재정렬(로드맵 §6½), signed SDF 폐곡면 한정 강제
- [x] v5.1: 경계조건 UI — ①Import 에 벽 선택 모드(trame server 픽킹) + wall-distance
      계산 버튼. region growing/seed 확장은 후속(현재는 단일 픽 누적)
- [x] v5.4: 좌(실제)/우(트윈) 분할 뷰어 — 독립 Plotter 2개, 공통 컬러 범위(clim) 강제,
      카메라는 수동 동기화 버튼(실시간 드래그 연동은 아님, 문서화됨)
- [x] v5.6 P1: 케이스 로드/재샘플 병렬화(thread_map, ~4.6배 실측)
- [ ] v5.1 후속: seed+region growing 확장, BC 값 입력 폼(속도/압력/온도), CGNS
      ZoneBC 자동 wall 인식
- [ ] v5.6 P1+: 리더보드 조합 병렬, remap 오차 바닥 분리, 그룹 스플릿+train-only
      정규화, MPI 배치 CLI(클러스터)

## 직전 단계: v4.2.0 + 17 rounds — 연구 플랫폼 + 전영역 성숙화 ✅

---

## v0.x — 기반 구축

### v0.1.0 — 프로젝트 스캐폴딩 ✅
- [x] `pyproject.toml` 작성 (setuptools, `[core]` / `[full]` / `[dev]` optional extras)
- [x] `src/naviertwin/` 전체 디렉토리 구조 생성 (SPEC.md §4 기준)
- [x] 각 모듈 `__init__.py` + `base.py` 추상 클래스 작성
- [x] `utils/config.py` — JSON 기반 설정 관리
- [x] `utils/logger.py` — 공통 로거
- [x] `main.py` — CLI 진입점
- [x] `pyproject.toml` 테스트/린터 설정

### v0.2.0 — CFD I/O 기초 + .ntwin 포맷 ✅
- [x] `core/cfd_reader/base.py` — `BaseReader` ABC, `CFDDataset` 데이터클래스
- [x] `core/cfd_reader/reader_factory.py` — 확장자 기반 자동 감지
- [x] `core/cfd_reader/openfoam_reader.py` — `pv.POpenFOAMReader` 우선, `ofpp` 폴백
- [x] `core/cfd_reader/vtk_reader.py` — VTK/VTU/STL (PyVista)
- [x] `core/export/ntwin_format.py` — `.ntwin` HDF5 저장/로드 (VTKHDF 구조)
- [x] 타임스텝 append 스트리밍 저장 구현
- [x] `tests/test_cfd_reader.py`

### v0.3.0 — 기초 유동 분석 ✅
- [x] `core/flow_analysis/vortex/q_criterion.py` — Q-criterion + λ₂
- [x] `core/flow_analysis/statistics/fft_psd.py` — FFT, PSD, 주파수 피크
- [x] `core/flow_analysis/boundary_layer/yplus.py` — y+, u_tau, Cf, 첫 번째 셀 높이 추정
- [x] `tests/test_flow_analysis.py`

---

## v1.x — MVP & 핵심 ROM

### v1.0.0 — MVP 릴리스 ✅
- [x] `core/dimensionality_reduction/linear/pod.py` — Snapshot POD, 에너지 누적 곡선 (modred)
- [x] `core/dimensionality_reduction/linear/randomized_svd.py`
- [x] `core/flow_analysis/modal/dmd.py` — DMD, FbDMD (PyDMD)
- [x] `core/surrogate/rbf_surrogate.py` (SMT)
- [x] `core/surrogate/kriging_surrogate.py` (SMT)
- [x] `core/digital_twin/twin_engine.py` — `predict(params) → field` 파이프라인
- [x] `core/validation/metrics.py` — RMSE, R², L2 norm
- [x] `gui/main_window.py` — 6패널 탭 호스트
- [x] `gui/panels/import_panel.py`
- [x] `gui/panels/analyze_panel.py`
- [x] `gui/panels/reduce_panel.py`
- [x] `gui/panels/model_panel.py`
- [x] `gui/panels/twin_panel.py`
- [x] `gui/panels/export_panel.py`
- [x] `gui/widgets/vtk_viewer.py` — AutoTessell식 `QtInteractor` viewer + headless 정적 fallback
- [x] `gui/styles/dark_theme.qss`
- [x] `.ntwin` 프로젝트 저장/복원 연동
- [x] `installer/naviertwin.spec` — PyInstaller `--onedir` 설정
- [x] `tests/test_reduction.py`, `tests/test_surrogate.py`, `tests/test_twin.py`

### v1.1.0 — CFD I/O 확장 (3 리더) ✅
- [x] `core/cfd_reader/fluent_reader.py` — `.cas/.dat` ASCII (pv.FluentReader → meshio → FluentASCIIParser, sibling .dat 자동 감지)
- [x] `core/cfd_reader/cgns_reader.py` — CGNS (`pv.CGNSReader` → pyCGNS → h5py → meshio)
- [x] `core/cfd_reader/gmsh_reader.py` — `.msh` v2.2/v4.1 (gmsh probe → meshio)
- [x] `core/cfd_reader/_mesh_utils.py` — 공통 메쉬 변환 헬퍼
- [x] `tests/test_cfd_io_expansion.py` — 26 테스트 (25 passed, 1 skipped/optional)

### v1.1.1 — SU2 + 메쉬 툴 + 해석해 검증 ✅
- [x] `core/cfd_reader/su2_reader.py` — SU2 `.su2` (meshio → SU2ASCIIParser, sibling `.csv` 자동 병합)
- [x] `core/tools/mesh_generator.py` — 채널/원통/NACA 익형 파라미터 메쉬 (Gmsh OCC)
- [x] `core/tools/mesh_processor.py` — simplify/smooth (PyMeshLab) + quality_report (PyVista 폴백)
- [x] `core/validation/analytic_solutions.py` — Couette / Poiseuille 2D / Poiseuille Pipe + Dedalus optional
- [x] `core/validation/analytic_solutions.compare_against_analytic` + metrics 연동
- [x] `gui/widgets/analytic_compare_widget.py` — Matplotlib 임베드 비교 시각화
- [x] `gui/panels/analyze_panel.py` — "해석해 비교" 5번째 분석 탭
- [x] `tests/test_su2_reader.py` (8 passed), `tests/test_mesh_tools.py` (6 passed), `tests/test_analytic.py` (7 passed, 1 optional skip)

### v1.2.0 — 비선형 차원축소 + SPOD + 고급 유동분석 ✅
- [x] `core/dimensionality_reduction/nonlinear/autoencoder.py` — PyTorch AE (MSE + Adam)
- [x] `core/dimensionality_reduction/nonlinear/vae.py` — β-VAE (reparameterization + ELBO + sample())
- [x] `core/dimensionality_reduction/nonlinear/gnn_ae.py` — GNN-AE (torch_geometric GCNConv, optional)
- [x] `core/flow_analysis/modal/spod.py` — Welch-block SPOD + PySPOD 백엔드 옵션
- [x] `core/flow_analysis/statistics/wavelet.py` — CWT (PyWavelets) + STFT 폴백
- [x] `core/flow_analysis/statistics/two_point_corr.py` — R(r) + 적분 길이 스케일
- [x] `core/flow_analysis/boundary_layer/boundary_layer.py` — δ99, δ*, θ, H + Cf
- [x] `core/flow_analysis/thermofluids/nondim.py` — Re / Pr / Nu / Pe / Gr / Ra
- [x] `core/data_augmentation/augmentation.py` — 갈릴레이 shift, reflect, rotate_2d, scale, symmetric
- [x] `tests/test_nonlinear_reduction.py` — 11 테스트 (AE/VAE/GNN-AE/SPOD/2pc/BL/nondim/aug) 전부 통과

---

## v2.x — 신경 연산자

### v2.0.0 — FNO / DeepONet / U-Net MVP ✅
- [x] `core/operator_learning/fno/fno.py` — SpectralConv1d/2d + FNO1D/FNO2D (PyTorch 직접 구현)
- [x] `core/operator_learning/deeponet/deeponet.py` — branch/trunk MLP + bias (trunk 좌표 캐시)
- [x] `core/operator_learning/unet/unet.py` — 2-level U-Net (encoder/decoder + skip)
- [x] `tests/test_operator_learning.py` — 8 테스트 통과 (FNO1D/FNO2D/DeepONet/UNet2D + not-fitted 가드)
- [x] `fno/tfno.py` — Tucker-factorized FNO2D (파라미터 90% 절감)
- [x] `fno/wno.py` — Wavelet Neural Operator 1D (pywt optional)
- [x] `deeponet/pi_deeponet.py` — Physics-Informed DeepONet (물리 잔차 λ_phys)
- [x] `deeponet/mionet.py` — Multiple-Input Operator Network (product/concat merge)
- [x] Model 패널 GUI — FNO/TFNO/DeepONet/UNet/WNO 데모 학습 버튼 통합
- [x] `tests/test_operator_learning_ext.py` — 7 pass + 1 skip
- [x] `fno/adaptive_fno.py` / `fno/lno.py` / `fno/spectral_refiner.py` (v2.0.x)
- [x] `deeponet/sequential_deeponet.py` / `nfno_deeponet.py` (v2.0.x)
- [x] `latent_operator/l_deeponet.py` / `latent_operator/pi_latent_no.py` (v2.0.x)
- [ ] 학습 loss curve 실시간 플롯 (v2.0.x)

### v2.1.0 — GNN 계열 ✅ (부분 완료 — GCN surrogate + MeshGraphNets MVP)
- [x] `core/gnn/gnn_surrogate/gnn_surrogate.py` — GCNConv 기반 node-level surrogate
- [x] `core/gnn/meshgraphnets/meshgraphnets.py` — Encode-Process-Decode + rollout
- [x] `tests/test_gnn.py` — 6 tests pass
- [ ] `core/gnn/egno/egno.py` — E(n)-Equivariant GNN (e3nn) (v2.1.x)
- [x] `core/gnn/graph_transformer/hamlet.py` (v2.1.x)

### v2.2.0 — 시계열 + Koopman 신경 연산자 ✅ (부분 완료)
- [x] `core/time_series/lstm/lstm.py` — LSTM autoregressive (lookback 윈도우 + rollout)
- [x] `core/time_series/transformer/transformer_ts.py` — causal mask Transformer encoder
- [x] `core/time_series/neural_ode/neural_ode.py` — torchdiffeq + RK4 폴백
- [x] `core/operator_learning/koopman/kno.py` — encoder/decoder + 선형 Koopman K
- [x] `tests/test_time_series.py` — 6 tests pass
- [x] `core/time_series/latent_dynamics/latent_dynamics.py` — AE + Neural ODE (v4.0.0 에서 구현)
- [x] `core/operator_learning/koopman/ikno.py` — Real-NVP 가역 Koopman (round10)
- [x] `core/time_series/temporal_no/tno.py` (v2.2.x)
- [x] `core/operator_learning/koopman/flowdmd.py` (v2.2.x)
- [x] `core/flow_analysis/modal/pykoopman_wrapper.py` (v2.2.x)

---

## v3.x — 디지털 트윈 완성

### v3.0.0 — 데이터 동화 + UQ + 기본 최적화 ✅ (부분 완료)
- [x] `core/data_assimilation/enkf.py` — Stochastic EnKF + inflation
- [x] `core/data_assimilation/particle_filter.py` — Bootstrap SIR + systematic resample
- [x] `core/sensitivity/sobol_analysis.py` — Saltelli 샘플링 + Sobol S1/ST (SALib 옵션 wrapper)
- [x] `core/optimization/mc_propagation.py` — 평균/표준편차/백분위수 MC 전파
- [x] `core/optimization/bayesian_opt.py` — GP + EI 최소화 (scikit-learn)
- [x] `tests/test_da_uq.py` — 7 tests pass
- [x] `core/data_assimilation/four_dvar.py` — 선형 4D-Var (round6)
- [x] `core/optimization/uq_surrogate.py` — PCE + Sobol (round9)
- [x] `core/optimization/surrogate_opt.py` — RBF + L-BFGS-B (round9)
- [x] `core/sensitivity/causal_analysis.py` — Pearson + Granger (round6)
- [ ] `core/data_assimilation/enkf_hpc.py` (v3.0.x, pyPDAF)

### v3.1.0 — PINN + 물리 보정 + 방정식 발견 ✅ (부분 완료)
- [x] `core/physnemo/pina_wrapper.py` — PINNSolver (PINA-style, PyTorch 직접)
- [x] `core/physics_correction/physics_correction.py` — 선형 제약 투영 + 질량 보존
- [x] `core/physics_correction/hybrid_rom.py` — POD + NN 잔차 보정
- [x] `core/flow_analysis/modal/sindy_wrapper.py` — STLSQ 자체 구현 + PySINDy 백엔드
- [x] `tests/test_pinn_correction.py` — 7 tests (1D Poisson 수렴 포함)
- [x] `core/explainability/symbolic_regression.py` — PySR + poly fallback (round13)
- [x] `core/multi_fidelity/multi_fidelity.py` — Additive Co-Kriging (v5.1.0)
- [x] `core/physnemo/physnemo_wrapper.py` (NVIDIA PhysicsNEMO, v3.1.x)
- [x] `core/physnemo/dd_pinn.py` — Domain Decomposition PINN (v3.1.x)

### v3.2.0 — GUI 완성 + 배포 ✅ (핵심 MVP 완료)
- [x] `gui/styles/i18n/ko.json`, `en.json` + `utils/i18n.py` Translator
- [x] `utils/undo_redo.py` — Command 스택 (max_size cap 포함)
- [x] `core/report/generator.py` — Jinja2 HTML + weasyprint PDF
- [x] `core/export/onnx_export.py` — opset / dynamic_axes / legacy 경로 fallback
- [x] `core/export/torchscript_export.py` — trace / script 지원
- [x] `tests/test_export_report.py` — 11 tests pass
- [x] `gui/wizard/tutorial_wizard.py` — 5 페이지 QWizard (round5)
- [x] 모델 비교 대시보드 GUI — `ModelCompareWidget` + MainWindow 탭 통합 (round5, round17)
- [x] `gui/widgets/loss_curve_widget.py` — 실시간 loss (round7)
- [x] `installer/naviertwin.iss` — Inno Setup 스크립트 (round8)
- [x] `core/digital_twin/pipeline.py` — end-to-end 오케스트레이터 (round5)
- [x] `api/server.py` — FastAPI REST 엔드포인트 (round4)
- [x] `core/explainability/shap_explainer.py` — KernelSHAP (v5.0.0)
- [x] `core/explainability/attention_viz.py` — MultiheadAttention 시각화 (round13)

---

## v4.x — 최첨단 모델

### v4.0.0 — 잠재 동역학 + 생성 모델 MVP ✅
- [x] `core/time_series/latent_dynamics/latent_dynamics.py` — AE + Neural ODE (RK4) 잠재 적분
- [x] `core/generative/diffusion_pde/diffusion_pde.py` — DDPM-style 유동장 생성
- [x] `tests/test_latent_generative.py` — 6 tests pass
- [ ] `core/state_space/mamba_neural_op/mno.py` (v4.0.x — mamba-ssm WSL 필요)
- [ ] `core/state_space/deepomamba/deepomamba.py` (v4.0.x)
- [ ] flowtorch 파이프라인 연동 (v4.0.x, GPL)

### v4.1.0 — 생성 모델 + KAN ✅
- [x] `core/generative/diffusion_pde/diffusion_pde.py` — DDPM-style (v4.0.0)
- [x] `core/generative/wavelet_diffusion/wavelet_diffusion_no.py` — DWT+Diffusion (round14)
- [x] `core/generative/conditional_gen/conditional_gen.py` — cVAE (round10)
- [x] `core/operator_learning/kan/kano.py` — KAN + spectral (round1)
- [x] `core/dimensionality_reduction/nonlinear/tucker_decomp.py` — HOSVD+HOOI (round1)

### v4.2.0 — Equivariant NN + 고급 분해 ✅ (부분 완료)
- [x] `core/equivariant/group_equiv_fno/group_equiv_fno.py` — C4 회전 평균 FNO2D
- [x] `core/dimensionality_reduction/linear/cpod.py` — null-space 투영 POD
- [x] `tests/test_equivariant_cpod.py` — 5 tests pass
- [x] `core/equivariant/physics_embedded/physics_embedded_gnn.py` — EGNN translation/rotation equivariant (round14)
- [x] `core/dimensionality_reduction/nonlinear/diffusion_maps.py` — Coifman-Lafon (round6)
- [x] `core/flow_analysis/modal/pgd.py` — greedy rank-1 (round4)
- [x] `core/flow_analysis/vortex/lcs.py` — FTLE via RK4 flow-map (round4)
- [x] `core/flow_analysis/thermofluids/entropy_gen.py` — Bejan (round4)
- [x] `core/equivariant/physics_embedded/lie_algebra_no.py` (v4.2.x)

---

## v5.x — 연구 플랫폼

### v5.0.0 — 고급 최적화 + 인증 ROM
- [x] `core/optimization/moo_optimizer.py` (pygmo2, GPL)
- [x] `core/optimization/topology_opt.py` (DL4TO, PyTopo3D)
- [ ] `core/optimization/mdo_pipeline.py` (OpenMDAO)
- [ ] `core/optimization/inverse_problem.py` (Firedrake+pyadjoint, LGPL)
- [ ] `core/optimization/su2_adjoint.py` (SU2, LGPL)
- [ ] `core/dimensionality_reduction/linear/certified_rb.py` (RBniCSx, LGPL)
- [ ] `core/dimensionality_reduction/linear/neural_rb.py` (dlrbnicsx, LGPL)

### v5.1.0 — 멀티피델리티 + Active/Online Learning
- [x] `core/multi_fidelity/transfer_learning.py`
- [x] `core/online_learning/active_learning.py`
- [x] `core/online_learning/online_learning.py`
- [x] `core/physics_correction/hybrid_rom_adv.py` 고도화
- [ ] `core/digital_twin/jax_fluids_wrapper.py` (JAX-Fluids, optional)

### v5.2.0 — 설명가능성 + FastAPI + 완전 배포
- [x] `core/explainability/shap_explainer.py` (SHAP)
- [x] `core/explainability/attention_viz.py` (captum)
- [ ] `core/explainability/kano_symbolic.py`
- [ ] `core/explainability/pysr_recovery.py` (PySR, optional)
- [ ] 모델 비교 대시보드 완성
- [x] `api/server.py` — FastAPI REST 서버
- [ ] 자동 업데이트 (GitHub Releases 기반)
- [ ] Sphinx + MkDocs API 문서 자동 생성

---

## 완료된 항목
(완료 시 위에서 여기로 이동)

---

## 라운드 기반 고도화 (v4.2.0 이후)

### 연산자 학습 확장
- [x] `operator_learning/fno/tfno.py` — Tucker-factorized FNO (v2.0.1)
- [x] `operator_learning/fno/wno.py` — WNO (v2.0.1)
- [x] `operator_learning/fno/adaptive_fno.py` — rFFT 기반 modes 자동 선택 (round11)
- [x] `operator_learning/fno/spectral_refiner.py` — low→high 2단계 학습 (round11)
- [x] `operator_learning/fno/lno.py` — Laplace 복소 pole/residue (round15)
- [x] `operator_learning/deeponet/pi_deeponet.py` — 물리 잔차 (v2.0.1)
- [x] `operator_learning/deeponet/mionet.py` — 복수 branch (v2.0.1)
- [x] `operator_learning/deeponet/sequential_deeponet.py` — GRU branch (round11)
- [x] `operator_learning/latent_operator/l_deeponet.py` — 잠재 DeepONet (round12)
- [x] `operator_learning/latent_operator/pi_latent_no.py` — PI-Latent-NO (round12)
- [x] `operator_learning/koopman/ikno.py` — Real-NVP invertible (round10)
- [x] `gnn/graph_transformer/hamlet.py` — Dense self-attention + position emb (round15)

### 5.0/5.1 연구 플랫폼
- [x] `optimization/moo_optimizer.py` — NSGA-II (v5.0.0)
- [x] `optimization/topology_opt.py` — SIMP 2D (v5.0.0)
- [x] `optimization/uq_surrogate.py` — PCE + Sobol (round9)
- [x] `optimization/surrogate_opt.py` — RBF + L-BFGS-B (round9)
- [x] `multi_fidelity/transfer_learning.py` — freeze + finetune (v5.1.0)
- [x] `online_learning/active_learning.py` — variance-based selection (v5.1.0)
- [x] `data_assimilation/four_dvar.py` — 선형 해석해 (round6)
- [x] `sensitivity/causal_analysis.py` — Pearson + Granger (round6)
- [x] `explainability/symbolic_regression.py` — PySR + polynomial fallback (round13)
- [x] `explainability/attention_viz.py` — MultiheadAttention (round13)
- [x] `surrogate/ensemble.py` — Ensemble + MoE k-means gating (round13)

### 생성/잠재 모델
- [x] `generative/conditional_gen/conditional_gen.py` — cVAE (round10)
- [x] `generative/wavelet_diffusion/wavelet_diffusion_no.py` — DWT+DDPM (round14)

### 대칭성 보존
- [x] `equivariant/group_equiv_fno/group_equiv_fno.py` — C4 회전 평균 (v4.2.0)
- [x] `equivariant/physics_embedded/physics_embedded_gnn.py` — EGNN (round14)

### GUI & 배포
- [x] `gui/wizard/tutorial_wizard.py` — 5 단계 QWizard (round5)
- [x] `gui/widgets/model_compare_widget.py` — RMSE/R² 바 차트 (round5)
- [x] `gui/widgets/loss_curve_widget.py` — 학습 손실 실시간 (round7)
- [x] `gui/widgets/analytic_compare_widget.py` — 해석해 ↔ 수치 (v1.1.1)
- [x] MainWindow i18n + 7번째 Compare 탭 (round17)
- [x] `installer/naviertwin.iss` — Windows Inno Setup (round8)
- [x] `utils/i18n.py` + ko/en 번역 JSON (v3.2.0)
- [x] `utils/undo_redo.py` — Command 스택 (v3.2.0)

### API / 내보내기
- [x] `api/server.py` — FastAPI (/health, /reduce/pod, /analytic/*, /optimize/bayesian) (round4)
- [x] `core/export/onnx_export.py` + `torchscript_export.py` (v3.2.0)
- [x] `core/report/generator.py` — Jinja2 + weasyprint (v3.2.0)
- [x] `core/digital_twin/pipeline.py` — 6 단계 오케스트레이터 (round5)

### 실전 예제
- [x] `examples/cavity_benchmark.py` — POD/AE/FNO 재구성 비교 (round16)

---

## 총 진행 상황

- **307+ 테스트 통과 / 4 skipped** (optional: pywt / pymeshlab / dedalus / onnxscript)
- Ruff 린트 통과 전체 모듈
- v1.1.0 → v4.2.0 + 17 rounds 고도화 완료

---

## 최종 통계 (20+ rounds 완료)

### 양적 지표
- **334 passed / 4 skipped** (optional: pywt / pymeshlab / dedalus / onnxscript)
- **48 개 feature/docs commits** (v1.1.1 → rounds 1-23)
- **195 소스 파일** (src/naviertwin/)
- **41 테스트 파일** (tests/)

### 라운드별 산출물 (round 11-23 추가분)
- **Round 11** — SequentialDeepONet (GRU branch), AdaptiveFNO1D (energy-based mode selection), SpectralRefiner (low→high res 2단계)
- **Round 12** — L-DeepONet (AE latent + operator), PI-Latent-NO (물리 residual 추가)
- **Round 13** — SymbolicRegressor (PySR + poly fallback), EnsembleSurrogate, MixtureOfExperts (k-means gating), Attention viz
- **Round 14** — EGNN (translation + rotation equivariance 검증), WaveletDiffusionNO (DWT + DDPM)
- **Round 15** — HAMLET (dense self-attention + position), LNO1D (Laplace pole/residue)
- **Round 16** — Cavity benchmark 예제 (POD/AE/FNO 비교)
- **Round 17** — MainWindow i18n + 7번째 Compare 탭
- **Round 18** — 문서 전면 동기화, version 4.2.17
- **Round 19** — TNO (Temporal NO), FlowDMD (INN + DMD), KoopmanAnalysis (pykoopman + DMD)
- **Round 20** — SO2Canonicalizer (Lie equivariance), NFNODeepONet (비균일 격자), HybridROMAdv (제약 투영)
- **Round 21** — OnlineKriging/OnlineNN, DomainDecompPINN, PhysicsNEMOWrapper
- **Round 22** — LBMD2Q9 (D2Q9 LBGK 자체 구현), Lettuce/flowtorch/JAX-Fluids 래퍼
- **Round 23** — LBM → POD → Kriging 완전 파이프라인 데모 (R²=1.0 달성)

### 구현 완성도
거의 모든 SPEC.md §6 기법 (차원축소/모달/Surrogate/Operator Learning/GNN/SSM 제외/생성모델/시계열/Equivariant/PINN/방정식 발견/DA/UQ/최적화/설명가능성) 의 **MVP 또는 완성본** 이 구현되었으며, 실제 파이프라인 통합 데모로 유효성 확인.

### 남은 여정 (선택)
- mamba-ssm (MNO, DeepOMamba) — Windows 지원 불안정으로 후순위
- Certified RB (RBniCSx) — LGPL, Firedrake 의존성 무거움
- pyPDAF 대규모 DA — Fortran 컴파일 필요

---

## 🎯 Round 40 Milestone (v4.2.40)

### 최종 통계
- **383 passed / 4 skipped** tests
- **63 commits**, **215 소스 파일**, **53 테스트 파일**
- **4 실전 예제** (cavity / LBM / Burgers+FNO / Streaming Burgers)

### 추가 도메인 (rounds 33-39)
- **StreamingDigitalTwin + Burgers** (round 33) — 실시간 EnKF (4.9% → 3.0%)
- **CLI 서브커맨드** (round 34) — benchmark/server/pipeline
- **RL flow control** (round 35) — GaussianPolicy + REINFORCE
- **Turbulence** (round 36) — k-ε closure + E(k) 스펙트럼 + Kolmogorov 기울기
- **Helmholtz + 압축성** (round 37) — 주기 2D 분해 + Mach/isentropic
- **QMC samplers** (round 38) — Halton / LHS / Sobol + 스케일링
- **음향 모드** (round 39) — 1D duct + Strouhal + Womersley

### 사용 가능 도메인 total
1. CFD I/O (7 formats)
2. 메쉬 생성/후처리 (2)
3. 차원축소 (선형 5 + 비선형 6)
4. 모달/통계 (9)
5. 유동 분석 (10+)
6. 신경 연산자 (FNO 6 + DeepONet 5 + Latent 2 + KNO 3 + UNet 2 + KANO + HAMLET)
7. GNN (2)
8. 시계열 (5 + ESN + TNO)
9. Equivariant (3)
10. 생성 모델 (3)
11. PINN (3)
12. 물리 보정 (3)
13. DA (3)
14. UQ + 최적화 (7+)
15. 민감도 + 설명 (5)
16. Surrogate (4) + 멀티피델리티 (2) + Online (3)
17. Digital Twin (3 엔진 + Streaming)
18. **Turbulence** (k-ε + E(k))
19. **압축성** (Mach/isentropic)
20. **음향** (duct + Strouhal)
21. **RL** (policy gradient)
22. **QMC** (Halton/LHS/Sobol)
23. External 솔버 래퍼 (LBM + Lettuce + flowtorch + JAX-Fluids)
24. PDE 솔버 (Burgers + Heat)
25. Export (ntwin + ONNX + TorchScript)
26. Report (Jinja2 + weasyprint)
27. API (FastAPI 5 엔드포인트)
28. GUI (6 패널 + 6 위젯 + 위자드 + i18n + Compare 탭)

---

## 🎯 Round 47 Milestone (v4.2.47)

- **409 passed / 4 skipped** tests (46 rounds 완료)
- **70 commits, 223 src files, 59 test files, 4 examples**

### 라운드 41-46 추가 산출물
- **Round 41** — POD-Galerkin linear ROM with input matrix
- **Round 42** — Langevin score-based sampler + Euler-Maruyama SDE
- **Round 43** — Wasserstein 1D + MMD Gaussian + KL divergence
- **Round 44** — Chebyshev spectral (Gauss-Lobatto + Trefethen D 행렬) + Lagrange
- **Round 45** — Benchmark dataset catalog (Burgers/Heat/Cavity)
- **Round 46** — SPH M4 cubic kernel (1/2/3D) + density/gradient

### 추가 도메인
29. **POD-Galerkin reduced dynamical system** (with input channel)
30. **Langevin sampling + SDE integration**
31. **Statistical distances** (W₁, MMD, KL)
32. **Spectral methods** (Chebyshev + Lagrange)
33. **Benchmark registries** (파라미터 가변 PDE 데이터셋)
34. **SPH** (kernel, density, gradient)

---

## 🏆 Round 50 Milestone (v4.2.50)

- **416 passed / 4 skipped** tests
- **74 commits / 225 src files / 61 test files / 4 examples**
- **50 rounds 완료** (초기 목표 11 버전 + 추가 39 rounds 고도화)

### 라운드 48-49 추가
- **Round 48** — Deep Ritz solver (변분 에너지 최소화 PINN)
- **Round 49** — FVM upwind + MUSCL-Hancock + minmod limiter + 질량 보존

### 최종 도메인 총합
35+ 독립 기술 도메인 + 4 엔드-투-엔드 예제 + GUI + REST API + CLI.
초기 v1.1.0 단일 리더에서 시작해 **연구 플랫폼 + 실전 배포** 수준 도달.

이 milestone 이후 기능 추가는 희귀 영역 (Mamba SSM / Firedrake 기반 Certified RB /
Fortran pyPDAF) 이며, 일반 사용자에게는 현재 범위가 충분합니다.

---

## 🔧 Round 51-58: 라이브러리 활용도 개선 집중

**전략 변경**: 자체 구현 일변도 → 검증된 라이브러리 래핑.

| Round | 라이브러리 | 이전 활용도 | 개선 후 |
|------|------|------|------|
| 51 | botorch + gpytorch | 0% | qEI/UCB 배치 BO ✅ |
| 52 | SALib | 20% (Sobol) | 100% (Morris/FAST/PAWN/Delta/Sobol) |
| 53 | nlopt | 0% | 14 알고리즘 |
| 54 | pydmd | 20% (DMD/FbDMD) | 60% (+HODMD/MrDMD/OptDMD/HAVOK/DMDc) |
| 55 | smt | 30% (RBF/Kriging) | 80% (+KPLS/GEKPLS/IDW/QP + LHS/FullFact) |
| 56 | pymor | 0% | POD/DEIM/GramSchmidt |
| 57 | foamlib | 0% | case 파라미터 스윕 + dict 조작 |
| 58 | pymeshlab | 10% (simplify/smooth) | 60% (+Taubin/remesh/curvature/hole) |

**458 passed / 83 commits / v4.2.58**

### 라이브러리 활용도 점검 답변
> **"모든 라이브러리 제대로 잘 이용하고 있는거 맞지?"**

- Round 50 시점: **부분적** (설치된 라이브러리 중 절반만 활용, 자체 구현 과다)
- Round 58 시점: **핵심 라이브러리 8종 전체 활용** (botorch/SALib/nlopt/pydmd/smt/pymor/foamlib/pymeshlab)
- 남은 것: `physicsnemo` (가용성 체크만), `pyCGNS` (h5py 폴백에 의존)
- 미설치: pyspod/pysindy/modred/flowtorch/pysr/shap/torchdiffeq/e3nn/escnn/pygmo/dedalus/jaxfluids/lettuce — 자체 구현 완비로 보완

---

## 🏆 Round 630 Milestone — Commercial Post-Processor Parity

**프로젝트 정체성 확립**: NavierTwin은 CFD **솔버**가 아닌, CFD 결과를 받아
AI/ROM/차원축소로 **의미 있는 데이터를 추출**하는 후처리 도구이다.

### R591–600 (커버리지 강화)
| Round | 영역 | Δ |
|-------|------|---|
| 591–594 | safe_yaml, main CLI, dim_reduction lazy, OpenFOAMReader | 33–47% → 75%+ |
| 595–596 | WNO1D 에러 경로, WaveletDiffusionNO 패킹 | |
| 597–599 | onnx_wrap, device utils, CGNSReader 폴백 체인 | |
| 600 | **커버리지 84% → 85%, 약점 모듈 27 → 16** | gate ratchet |

### R601–605 (신규 ROM 인프라)
- **R601** sparse_sensor — QR-pivot 최적 센서 배치 + 그리디 폴백 + reconstruct
- **R602** SensorDMDPipeline — POD 기저 + 센서 + 재구성 워크플로
- **R603** spectral_energy — 1D/2D 에너지 스펙트럼, Kolmogorov -5/3 적합, 적분 길이
- **R604** MRPOD — 다중 해상도 POD (Gaussian-pyramid + per-scale SVD)
- **R605** IncrementalPOD — Brand 2006 rank-1 SVD 갱신, 망각 인수 지원

### R606–629 (상용 툴 동급 후처리 모듈)
| Round | 모듈 | 상용 툴 대응 |
|-------|------|--------------|
| 606 | reynolds_stats | Tecplot 360 Time-Average / CFD-Post Statistics |
| 607 | psd | MATLAB Signal Processing Toolbox |
| 608 | surface_integrals | Tecplot 360 / CFD-Post Force/Moment |
| 609 | quadrant_pdf | Quadrant analysis (Wallace 1972) + KDE |
| 610 | two_point | Tecplot Two-Point Stats + Taylor microscale |
| 611 | stat_convergence | Fluent Sample Convergence + Geweke |
| 612 | plane_flux | CFD-Post Surface Integral / EnSight Flux |
| 613 | time_interp | Tecplot Time-Aware Sliding |
| 614 | coord_transform | EnSight Cylindrical/Spherical |
| 615 | slice_extract | Tecplot Slice + CFD-Post Line |
| 616 | expression_eval | CFD-Post Custom Expressions (AST sandbox) |
| 617 | phase_lock | Fluent Phase-Locked Sampling |
| 618 | running_moments | EnSight Variable Statistics (Welford+Pébay) |
| 619 | denoise | MATLAB sgolayfilt/hampel + wavelet shrinkage |
| 620 | quantile_stats | Tecplot box-and-whisker + ECDF |
| 621 | eof_analysis | NCL/CDO eofunc + Lumley + North test + Varimax |
| 622 | goodness_of_fit | MATLAB kstest/adtest/chi2gof + Shapiro-Wilk |
| 623 | conditional_sampling | Trigger averaging + 사분면 마스크 |
| 624 | grid_derivatives | Tecplot Calculate Variables (2차/4차 정확) |
| 625 | critical_points | Vector Field Topology (Helman 1991) |
| 626 | anisotropy | Pope §11.5 Lumley triangle + barycentric |
| 627 | morphology | SciPy ndimage 등가 (4-/8-conn) |
| 628 | cell_volume | Tecplot Compute Cell Volume + Volume Integral |
| 629 | truncation_criteria | Eckart-Young + scree + AIC/BIC |

### 상태 (R630 시점)
- **2535 테스트 수집** / **677 commits** / **39 라운드 후처리 패리티 추가**
- ruff 0 errors / coverage 85%+ gate
- 후처리 능력: **상용 툴 (Tecplot 360 / CFD-Post / EnSight) 핵심 기능 동등 수준 도달**
- 차세대 차별화 포인트: AI/ROM 통합 (이미 보유) + 검증 인프라 (R561–590)
