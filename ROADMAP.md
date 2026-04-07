# NavierTwin 로드맵

> Phase별 세부 태스크 체크리스트. 버전별 목표·범위·근거는 `PLAN.md` 참조.

## 현재 단계: v0.1.0 — 프로젝트 스캐폴딩

---

## v0.x — 기반 구축

### v0.1.0 — 프로젝트 스캐폴딩
- [ ] `pyproject.toml` 작성 (setuptools, `[core]` / `[full]` / `[dev]` optional extras)
- [ ] `src/naviertwin/` 전체 디렉토리 구조 생성 (SPEC.md §4 기준)
- [ ] 각 모듈 `__init__.py` + `base.py` 추상 클래스 작성
- [ ] `utils/config.py` — JSON 기반 설정 관리
- [ ] `utils/logger.py` — 공통 로거
- [ ] `main.py` — CLI 진입점
- [ ] `pytest.ini` / `pyproject.toml` 테스트 설정
- [ ] `ruff.toml` + `.isort.cfg` 린터 설정

### v0.2.0 — CFD I/O 기초 + .ntwin 포맷
- [ ] `core/cfd_reader/base_reader.py` — `BaseReader` ABC, `CFDDataset` 데이터클래스
- [ ] `core/cfd_reader/reader_factory.py` — 확장자 기반 자동 감지
- [ ] `core/cfd_reader/openfoam_reader.py` — `pv.POpenFOAMReader` 우선, `ofpp` 폴백
- [ ] `core/cfd_reader/vtk_reader.py` — VTK/VTU/STL (meshio, PyVista)
- [ ] `core/export/ntwin_format.py` — `.ntwin` HDF5 저장/로드 (VTKHDF 구조)
- [ ] 타임스텝 append 스트리밍 저장 구현
- [ ] `tests/test_cfd_reader.py`

### v0.3.0 — 기초 유동 분석
- [ ] `core/flow_analysis/vortex/q_criterion.py` — Q-criterion (`pv.compute_derivative`)
- [ ] `core/flow_analysis/vortex/lambda2.py` — λ₂ (numpy eigvalsh 직접 구현)
- [ ] `core/flow_analysis/statistics/fft_psd.py` — FFT, PSD, 주파수 피크
- [ ] `core/flow_analysis/boundary_layer/yplus.py` — y+, u_tau, Cf, 첫 번째 셀 높이 추정
- [ ] `tests/test_flow_analysis.py`

---

## v1.x — MVP & 핵심 ROM

### v1.0.0 — MVP 릴리스
- [ ] `core/dimensionality_reduction/linear/pod.py` — Snapshot POD, 에너지 누적 곡선 (modred)
- [ ] `core/dimensionality_reduction/linear/randomized_svd.py`
- [ ] `core/flow_analysis/modal/dmd.py` — DMD, FbDMD (PyDMD)
- [ ] `core/surrogate/rbf_surrogate.py` (SMT)
- [ ] `core/surrogate/kriging_surrogate.py` (SMT)
- [ ] `core/digital_twin/twin_engine.py` — `predict(params) → field` 파이프라인
- [ ] `core/validation/metrics.py` — RMSE, R², L2 norm
- [ ] `gui/main_window.py` — 6패널 탭 호스트
- [ ] `gui/panels/import_panel.py`
- [ ] `gui/panels/analyze_panel.py`
- [ ] `gui/panels/reduce_panel.py`
- [ ] `gui/panels/model_panel.py`
- [ ] `gui/panels/twin_panel.py`
- [ ] `gui/panels/export_panel.py`
- [ ] `gui/widgets/vtk_viewer.py` — `QtInteractor` 임베드, 타임스텝 슬라이더, 컬러맵 선택기
- [ ] `gui/styles/dark_theme.qss`
- [ ] `.ntwin` 프로젝트 저장/복원 연동
- [ ] `installer/naviertwin.spec` — PyInstaller `--onedir` 설정
- [ ] `tests/test_reduction.py`, `tests/test_surrogate.py`, `tests/test_twin.py`

### v1.1.0 — CFD I/O 확장 + 검증 강화
- [ ] `core/cfd_reader/fluent_reader.py` — `.cas/.dat`, `.cas.h5/.dat.h5`
- [ ] `core/cfd_reader/cgns_reader.py` — CGNS (`pv.CGNSReader`, pyCGNS)
- [ ] `core/cfd_reader/su2_reader.py` — SU2 Python Wrapper
- [ ] `core/tools/mesh_generator.py` — 채널/실린더/익형 파라미터 메쉬 (Gmsh)
- [ ] `core/tools/mesh_processor.py` — 단순화, 스무딩, 품질 검사 (PyMeshLab)
- [ ] `core/validation/analytic_solutions.py` — Couette, Poiseuille (Dedalus)
- [ ] 해석해 vs 수치해 자동 비교 GUI 패널 연동
- [ ] `tests/test_analytic.py`

### v1.2.0 — 비선형 차원축소 + SPOD + 고급 유동분석
- [ ] `core/dimensionality_reduction/nonlinear/autoencoder.py` — AE (PyTorch)
- [ ] `core/dimensionality_reduction/nonlinear/vae.py` — VAE (PyTorch)
- [ ] `core/dimensionality_reduction/nonlinear/gnn_ae.py` — GNN-AE (PyTorch Geometric)
- [ ] `core/flow_analysis/modal/spod.py` — SPOD (PySPOD)
- [ ] `core/flow_analysis/statistics/wavelet.py` — 웨이블릿 분석
- [ ] `core/flow_analysis/statistics/two_point_corr.py` — 난류 공간 상관
- [ ] `core/flow_analysis/boundary_layer/boundary_layer.py` — δ, θ, H, Cf
- [ ] `core/flow_analysis/thermofluids/nondim.py` — Nu, Re, Pr
- [ ] `core/data_augmentation/augmentation.py` — 갈릴레이 불변성, 대칭 변환
- [ ] `tests/test_nonlinear_reduction.py`

---

## v2.x — 신경 연산자

### v2.0.0 — FNO / DeepONet 계열
- [ ] `core/operator_learning/fno/fno.py` — FNO (neuraloperator)
- [ ] `core/operator_learning/fno/tfno.py` — TFNO
- [ ] `core/operator_learning/fno/wno.py` — WNO
- [ ] `core/operator_learning/fno/adaptive_fno.py` — Adaptive FNO
- [ ] `core/operator_learning/fno/lno.py` — LNO
- [ ] `core/operator_learning/fno/spectral_refiner.py` — Spectral-Refiner
- [ ] `core/operator_learning/deeponet/deeponet.py` — DeepONet (deepxde)
- [ ] `core/operator_learning/deeponet/pi_deeponet.py` — PI-DeepONet
- [ ] `core/operator_learning/deeponet/mionet.py` — MIONet
- [ ] `core/operator_learning/deeponet/sequential_deeponet.py`
- [ ] `core/operator_learning/deeponet/nfno_deeponet.py`
- [ ] `core/operator_learning/latent_operator/l_deeponet.py`
- [ ] `core/operator_learning/latent_operator/pi_latent_no.py`
- [ ] `core/operator_learning/unet/unet.py`
- [ ] 학습 모니터링 (loss curve, 검증 오차) GUI 연동
- [ ] `tests/test_operator_learning.py`

### v2.1.0 — GNN 계열
- [ ] `core/gnn/gnn_surrogate/gnn_surrogate.py` (PyTorch Geometric)
- [ ] `core/gnn/meshgraphnets/meshgraphnets.py`
- [ ] `core/gnn/egno/egno.py` — E(n)-Equivariant GNN (e3nn)
- [ ] `core/gnn/graph_transformer/hamlet.py`
- [ ] `tests/test_gnn.py`

### v2.2.0 — 시계열 + Koopman 신경 연산자
- [ ] `core/time_series/lstm/lstm.py`
- [ ] `core/time_series/transformer/transformer_ts.py`
- [ ] `core/time_series/temporal_no/tno.py`
- [ ] `core/time_series/neural_ode/neural_ode.py` (torchdiffeq)
- [ ] `core/time_series/latent_dynamics/latent_dynamics.py`
- [ ] `core/operator_learning/koopman/kno.py`
- [ ] `core/operator_learning/koopman/ikno.py`
- [ ] `core/operator_learning/koopman/flowdmd.py`
- [ ] `core/flow_analysis/modal/pykoopman_wrapper.py` (PyKoopman, pykoop)
- [ ] `tests/test_time_series.py`

---

## v3.x — 디지털 트윈 완성

### v3.0.0 — 데이터 동화 + UQ + 기본 최적화
- [ ] `core/data_assimilation/enkf.py` (DAPPER)
- [ ] `core/data_assimilation/enkf_hpc.py` (pyPDAF)
- [ ] `core/data_assimilation/4dvar.py`
- [ ] `core/data_assimilation/particle_filter.py` (filterpy)
- [ ] `core/sensitivity/sobol_analysis.py` (SALib)
- [ ] `core/optimization/uq_surrogate.py` (UQpy, OpenTURNS)
- [ ] `core/optimization/mc_propagation.py`
- [ ] `core/optimization/bayesian_opt.py` (scikit-optimize + NLopt)
- [ ] `core/optimization/surrogate_opt.py` (NLopt)
- [ ] `core/sensitivity/causal_analysis.py`
- [ ] `tests/test_da.py`, `tests/test_uq.py`

### v3.1.0 — PINN + 물리 보정 + 방정식 발견
- [ ] `core/physnemo/physnemo_wrapper.py` (NVIDIA PhysicsNEMO)
- [ ] `core/physnemo/pina_wrapper.py` (PINA)
- [ ] `core/physnemo/dd_pinn.py` — Domain Decomposition PINN
- [ ] `core/physics_correction/physics_correction.py`
- [ ] `core/physics_correction/hybrid_rom.py` — POD-Galerkin + NN
- [ ] `core/flow_analysis/modal/sindy_wrapper.py` (PySINDy)
- [ ] `core/explainability/symbolic_regression.py` (PySR, optional)
- [ ] `core/multi_fidelity/multi_fidelity.py`
- [ ] `tests/test_pinn.py`, `tests/test_correction.py`

### v3.2.0 — GUI 완성 + 배포
- [ ] `gui/wizard/tutorial_wizard.py`
- [ ] `gui/styles/i18n/ko.json`, `en.json`
- [ ] `utils/undo_redo.py` — Command 스택
- [ ] 모델 비교 대시보드 GUI
- [ ] `core/report/generator.py` (Jinja2 → PDF)
- [ ] `core/export/onnx_export.py`
- [ ] `core/export/torchscript_export.py`
- [ ] `installer/naviertwin.iss` — Inno Setup
- [ ] optional extras 분리 빌드 (`[core]` vs `[full]`)
- [ ] `tests/test_export.py`, `tests/test_report.py`

---

## v4.x — 최첨단 모델

### v4.0.0 — State Space Model + Neural ODE 고도화
- [ ] `core/state_space/mamba_neural_op/mno.py` (mamba-ssm)
- [ ] `core/state_space/deepomamba/deepomamba.py`
- [ ] `core/time_series/latent_dynamics/latent_dynamics_adv.py` 고도화
- [ ] flowtorch 파이프라인 연동 (GPL)
- [ ] Lettuce LBM 스냅샷 생성 연동

### v4.1.0 — 생성 모델 + KAN
- [ ] `core/generative/diffusion_pde/diffusion_pde.py`
- [ ] `core/generative/wavelet_diffusion/wavelet_diffusion_no.py`
- [ ] `core/generative/conditional_gen/conditional_gen.py`
- [ ] `core/operator_learning/kan/kano.py`
- [ ] `core/dimensionality_reduction/nonlinear/tucker_decomp.py`

### v4.2.0 — Equivariant NN + 고급 분해
- [ ] `core/equivariant/group_equiv_fno/group_equiv_fno.py` (e3nn)
- [ ] `core/equivariant/physics_embedded/physics_embedded_gnn.py` (escnn)
- [ ] `core/equivariant/physics_embedded/lie_algebra_no.py`
- [ ] `core/dimensionality_reduction/linear/cpod.py`
- [ ] `core/dimensionality_reduction/nonlinear/diffusion_maps.py`
- [ ] `core/flow_analysis/modal/pgd.py`
- [ ] `core/flow_analysis/vortex/lcs.py`
- [ ] `core/flow_analysis/thermofluids/entropy_gen.py`

---

## v5.x — 연구 플랫폼

### v5.0.0 — 고급 최적화 + 인증 ROM
- [ ] `core/optimization/moo_optimizer.py` (pygmo2, GPL)
- [ ] `core/optimization/topology_opt.py` (DL4TO, PyTopo3D)
- [ ] `core/optimization/mdo_pipeline.py` (OpenMDAO)
- [ ] `core/optimization/inverse_problem.py` (Firedrake+pyadjoint, LGPL)
- [ ] `core/optimization/su2_adjoint.py` (SU2, LGPL)
- [ ] `core/dimensionality_reduction/linear/certified_rb.py` (RBniCSx, LGPL)
- [ ] `core/dimensionality_reduction/linear/neural_rb.py` (dlrbnicsx, LGPL)

### v5.1.0 — 멀티피델리티 + Active/Online Learning
- [ ] `core/multi_fidelity/transfer_learning.py`
- [ ] `core/online_learning/active_learning.py`
- [ ] `core/online_learning/online_learning.py`
- [ ] `core/physics_correction/hybrid_rom_adv.py` 고도화
- [ ] `core/digital_twin/jax_fluids_wrapper.py` (JAX-Fluids, optional)

### v5.2.0 — 설명가능성 + FastAPI + 완전 배포
- [ ] `core/explainability/shap_explainer.py` (SHAP)
- [ ] `core/explainability/attention_viz.py` (captum)
- [ ] `core/explainability/kano_symbolic.py`
- [ ] `core/explainability/pysr_recovery.py` (PySR, optional)
- [ ] 모델 비교 대시보드 완성
- [ ] `api/server.py` — FastAPI REST 서버
- [ ] 자동 업데이트 (GitHub Releases 기반)
- [ ] Sphinx + MkDocs API 문서 자동 생성

---

## 완료된 항목
(완료 시 위에서 여기로 이동)
