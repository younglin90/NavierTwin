# NavierTwin 구현 계획서 (PLAN)

> 시스템을 **어떻게** 만들지 정의하는 문서. 프로젝트 목표, 버전별 범위, 우선순위 기준, 기술 선택 근거를 포함한다.
> 기술 명세(스택·기법 목록)는 `SPEC.md`, Phase별 진행 현황은 `ROADMAP.md` 참조.

---

## 1. 프로젝트 목표

| 구분 | 내용 |
|------|------|
| 핵심 목표 | CFD 엔지니어가 시뮬레이션 결과를 AI/ROM으로 즉시 디지털 트윈화할 수 있는 로컬 GUI 툴 제공 |
| 타겟 사용자 | CFD 코드를 직접 짜기 어려운 엔지니어 일반 사용자 (연구자·산업체) |
| 핵심 가치 | 로컬 실행(데이터 외부 유출 없음), GPL-3.0 오픈소스, 설치 즉시 사용 가능 |
| 비목표 | 클라우드 SaaS, CFD 솔버 자체 개발, 상업 라이선스 |

---

## 2. 버전 로드맵 전체 개요

```
v0.x  기반 구축       ██████
v1.x  MVP & 핵심 ROM  ████████████
v2.x  신경 연산자     ████████
v3.x  디지털 트윈     ████████
v4.x  최첨단 모델     ████████
v5.x  연구 플랫폼     ████████
```

| 버전 | 단계 | 핵심 목표 |
|------|------|-----------|
| v0.1 | 기반 | 프로젝트 스캐폴딩 |
| v0.2 | 기반 | CFD I/O 기초 (OpenFOAM, VTK) + .ntwin 포맷 |
| v0.3 | 기반 | 기초 유동 분석 (Q-criterion, FFT, y+) |
| v1.0 | MVP | 선형 ROM + 기본 Surrogate + 디지털 트윈 엔진 + GUI + 패키징 |
| v1.1 | MVP | CFD I/O 확장 (Fluent, CGNS, Gmsh) + 검증 (Dedalus) |
| v1.2 | MVP | 비선형 차원축소 (AE, VAE) + SPOD + 고급 유동분석 |
| v2.0 | 신경연산자 | FNO/TFNO/WNO + DeepONet 계열 |
| v2.1 | 신경연산자 | GNN 계열 (MeshGraphNets, EGNO, HAMLET) |
| v2.2 | 신경연산자 | 시계열 + Koopman 신경 연산자 |
| v3.0 | 디지털트윈 | 데이터 동화 + UQ + 최적화 기본 |
| v3.1 | 디지털트윈 | PINN + 물리 보정 + PySINDy |
| v3.2 | 디지털트윈 | GUI 완성 + 보고서 + ONNX + i18n + 위자드 |
| v4.0 | 최첨단모델 | SSM (Mamba) + Neural ODE + 잠재 동역학 |
| v4.1 | 최첨단모델 | 생성 모델 (Diffusion PDE) + KAN(KANO) |
| v4.2 | 최첨단모델 | Equivariant NN + 대칭성 보존 모델 |
| v5.0 | 연구플랫폼 | 고급 최적화 (다목적, 위상, MDO) + 인증 ROM |
| v5.1 | 연구플랫폼 | 멀티피델리티 + Active/Online learning + Hybrid ROM |
| v5.2 | 연구플랫폼 | 설명가능성 + 모델 비교 대시보드 + FastAPI |

---

## 3. 버전별 상세 범위

---

### v0.x — 기반 구축

---

#### v0.1.0 — 프로젝트 스캐폴딩

**목표:** 전체 디렉토리 구조, 공통 인터페이스, 개발 환경 구축.

| 영역 | 항목 | 비고 |
|------|------|------|
| 패키지 | pyproject.toml 작성 (setuptools, optional extras 구조) | `[core]`, `[full]`, `[dev]` extra 분리 |
| 구조 | `src/naviertwin/` 전체 디렉토리 생성 | SPEC.md §4 기준 |
| 인터페이스 | 각 모듈 `__init__.py` + `base.py` 추상 클래스 | `fit/predict`, `fit/encode/decode` |
| 유틸 | `utils/config.py` (JSON 설정), `utils/logger.py` | 전 모듈 공통 사용 |
| 엔트리포인트 | `main.py` — CLI 실행 진입점 | |
| 테스트 | pytest 설정, `tests/` 구조 | |
| CI | ruff lint, isort 설정 | |

---

#### v0.2.0 — CFD I/O 기초 + .ntwin 포맷

**목표:** OpenFOAM/VTK 읽기 → PyVista UnstructuredGrid → .ntwin HDF5 저장/복원.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| Reader | `BaseReader` ABC, `CFDDataset` 데이터클래스 | — |
| Reader | `reader_factory.py` — 확장자 기반 자동 감지 | — |
| Reader | `openfoam_reader.py` — `pv.POpenFOAMReader` 우선, `ofpp` 폴백 | PyVista, ofpp |
| Reader | `vtk_reader.py` — VTK/VTU/STL | meshio, PyVista |
| 내부 포맷 | `.ntwin` HDF5 저장/로드 — VTKHDF 구조 기반 | h5py |
| 내부 포맷 | 타임스텝 append 스트리밍 저장 | h5py |
| 테스트 | `tests/test_cfd_reader.py` | pytest |

---

#### v0.3.0 — 기초 유동 분석

**목표:** 가장 많이 쓰이는 후처리 3종 구현 및 단위 테스트.

| 영역 | 항목 | 구현 방법 |
|------|------|---------|
| 와류 | `q_criterion.py` — Q-criterion, λ₂ | `pv.compute_derivative` + numpy eigvalsh |
| 스펙트럼 | `fft_psd.py` — FFT, PSD, 주파수 피크 | NumPy/SciPy |
| 벽면 | `yplus.py` — y+, u_tau, Cf, 첫 번째 셀 높이 추정 | 해석 공식 직접 구현 |
| 테스트 | `tests/test_flow_analysis.py` | pytest |

---

### v1.x — MVP & 핵심 ROM

---

#### v1.0.0 — MVP 릴리스

**목표:** end-to-end 파이프라인 완성. CFD 읽기 → POD/DMD → Surrogate → 디지털 트윈 예측 → GUI → Windows 배포.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 차원축소 | `pod.py` — Snapshot POD, 에너지 누적 곡선 | modred, sklearn.randomized_svd |
| 차원축소 | `randomized_svd.py` — 대용량 고속 근사 | sklearn |
| 모달 | `dmd.py` — DMD, FbDMD (노이즈 강건) | PyDMD |
| Surrogate | `rbf_surrogate.py` | SMT |
| Surrogate | `kriging_surrogate.py` | SMT |
| 디지털 트윈 | `twin_engine.py` — `predict(params) → field` 파이프라인 | — |
| 검증 | `metrics.py` — RMSE, R², L2 norm | NumPy/SciPy |
| GUI | `main_window.py` — 6패널 탭 호스트 | PySide6 |
| GUI | `panels/import_panel.py` | PySide6 |
| GUI | `panels/analyze_panel.py` | PySide6 |
| GUI | `panels/reduce_panel.py` | PySide6 |
| GUI | `panels/model_panel.py` | PySide6 |
| GUI | `panels/twin_panel.py` | PySide6 |
| GUI | `panels/export_panel.py` | PySide6 |
| GUI | `widgets/vtk_viewer.py` — `QtInteractor` 임베드, 타임스텝 슬라이더 | pyvistaqt |
| GUI | `styles/dark_theme.qss` | — |
| 내보내기 | `.ntwin` 프로젝트 저장/복원 | h5py |
| 패키징 | PyInstaller `--onedir` + `naviertwin.spec` | pyinstaller-hooks-contrib |
| 테스트 | `tests/test_reduction.py`, `tests/test_surrogate.py` | pytest |

---

#### v1.1.0 — CFD I/O 확장 + 검증 강화

**목표:** 산업 포맷 지원 확장, 메쉬 생성·처리, 해석해 기반 자동 검증.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| Reader | `fluent_reader.py` — `.cas/.dat`, `.cas.h5/.dat.h5` | `pv.FLUENTCFFReader`, meshio |
| Reader | `cgns_reader.py` — CGNS | `pv.CGNSReader`, pyCGNS |
| Reader | `su2_reader.py` — `.su2` + `.csv` 결과 | SU2 Python Wrapper |
| 메쉬 생성 | `mesh_generator.py` — 채널/실린더/익형 파라미터 메쉬 | Gmsh Python API (GPL) |
| 메쉬 처리 | `mesh_processor.py` — 단순화, 스무딩, 품질 검사 | PyMeshLab (GPL) |
| 검증 | `analytic_solutions.py` — Couette, Poiseuille 해석해 | Dedalus (GPL) |
| 검증 | 해석해 vs 수치해 자동 비교 GUI | — |
| 테스트 | `tests/test_analytic.py` | pytest |

---

#### v1.2.0 — 비선형 차원축소 + SPOD + 고급 유동분석

**목표:** 비정형 메쉬 비선형 압축, 주파수 공간 POD, 난류·경계층 분석 완성.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 차원축소 | `autoencoder.py` — AE, VAE | PyTorch |
| 차원축소 | `gnn_ae.py` — GNN 기반 비정형 메쉬 AE | PyTorch Geometric |
| 모달 | `spod.py` — SPOD, 주파수별 모드 | PySPOD |
| 유동분석 | `wavelet.py` — 웨이블릿 시간-주파수 분석 | PyWavelets |
| 유동분석 | `two_point_corr.py` — 난류 공간 상관 길이 | NumPy |
| 유동분석 | `boundary_layer.py` — δ, θ, H, Cf 경계층 두께 | NumPy |
| 유동분석 | `nondim.py` — Nu, Re, Pr 무차원수 분포 | NumPy |
| 데이터 증강 | `augmentation.py` — 갈릴레이 불변성, 대칭 변환 | NumPy/PyTorch |
| 테스트 | `tests/test_nonlinear_reduction.py` | pytest |

---

### v2.x — 신경 연산자

---

#### v2.0.0 — Fourier / DeepONet 계열

**목표:** 해상도 독립 PDE 연산자 학습. FNO 계열과 DeepONet 계열 통합.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| FNO | FNO, TFNO, WNO | neuraloperator v2.0+ |
| FNO | Adaptive FNO, LNO | neuraloperator |
| FNO | Spectral-Refiner (난류 정확도 향상) | neuraloperator |
| DeepONet | DeepONet, PI-DeepONet, MIONet | deepxde |
| DeepONet | Sequential DeepONet, NFNO-DeepONet | deepxde / PyTorch |
| 잠재공간 | L-DeepONet — AE 잠재공간 + DeepONet | PyTorch |
| 잠재공간 | PI-Latent-NO — 물리 제약 잠재 연산자 | PyTorch |
| U-Net | U-Net 필드 예측 | PyTorch |
| 학습 지원 | GPU 학습 모니터링 (loss curve, 검증 오차) | — |
| 학습 지원 | 물리기반 데이터 증강 연동 | — |
| 테스트 | `tests/test_operator_learning.py` | pytest |

---

#### v2.1.0 — GNN 계열

**목표:** 비정형 메쉬를 그래프로 직접 다루는 GNN 모델 통합.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| GNN | `gnn_surrogate.py` — 메쉬→그래프, 재학습 없이 다른 메쉬 대응 | PyTorch Geometric |
| GNN | `meshgraphnets.py` — 시간발전 포함 GNN 시뮬레이터 | PyTorch Geometric |
| GNN | `egno.py` — E(n)-Equivariant GNN Operator (ICML 2024) | PyTorch Geometric, e3nn |
| GNN | `hamlet.py` — Graph Transformer Neural Operator | PyTorch Geometric |
| GNN-AE | 비정형 메쉬 Graph AE 개선 (v1.2 연동) | PyTorch Geometric |
| 테스트 | `tests/test_gnn.py` | pytest |

---

#### v2.2.0 — 시계열 + Koopman 신경 연산자

**목표:** 비정상 유동의 시간 발전 예측. LSTM부터 Neural ODE까지 통합.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 시계열 | `lstm.py` — LSTM 시계열 예측 | PyTorch |
| 시계열 | `transformer_ts.py` — 어텐션 기반 장거리 시계열 | PyTorch |
| 시계열 | `tno.py` — Temporal Neural Operator (Nature Sci. Rep. 2025) | PyTorch |
| 시계열 | `neural_ode.py` — 연속 시간 동역학 | torchdiffeq |
| 잠재 동역학 | `latent_dynamics.py` — AE + Neural ODE 잠재 시간 적분 | PyTorch, torchdiffeq |
| Koopman NO | `kno.py` — KNO 비선형→선형 변환 | PyTorch |
| Koopman NO | `ikno.py` — Invertible KNO (2025) | PyTorch |
| Koopman NO | `flowdmd.py` — Coupling Flow INN 기반 Koopman | PyTorch |
| Koopman 분석 | `pykoopman_wrapper.py` — 데이터 기반 Koopman 연산자 | PyKoopman, pykoop |
| 테스트 | `tests/test_time_series.py` | pytest |

---

### v3.x — 디지털 트윈 완성

---

#### v3.0.0 — 데이터 동화 + UQ + 기본 최적화

**목표:** 실측 센서와 모델을 실시간 융합. 예측 불확실성 정량화. Surrogate 기반 최적화.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 데이터동화 | `enkf.py` — Ensemble Kalman Filter | DAPPER |
| 데이터동화 | `enkf_hpc.py` — 대규모 앙상블 DA | pyPDAF (LGPL) |
| 데이터동화 | `4dvar.py` — 시공간 최적화 기반 DA | NumPy/SciPy |
| 데이터동화 | `particle_filter.py` — 비선형/비가우시안 DA | filterpy |
| UQ | `sobol_analysis.py` — 전역 민감도 (Sobol indices) | SALib |
| UQ | `uq_surrogate.py` — PCE, Kriging 기반 UQ | UQpy, OpenTURNS |
| UQ | `mc_propagation.py` — Monte Carlo 불확실성 전파 | UQpy |
| 최적화 | `bayesian_opt.py` — BO (acquisition function + NLopt) | scikit-optimize, NLopt |
| 최적화 | `surrogate_opt.py` — Surrogate-based Optimization (RBF/Kriging + NLopt) | NLopt |
| 최적화 | `sensitivity.py` — Causal inference, 변수 중요도 | SALib, pingouin |
| 테스트 | `tests/test_da.py`, `tests/test_uq.py` | pytest |

---

#### v3.1.0 — PINN + 물리 보정 + 방정식 발견

**목표:** 물리 법칙을 직접 내장한 학습 방법과 데이터에서 방정식을 자동 발견하는 기능 추가.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| PINN | `physnemo_wrapper.py` — NS/에너지 방정식 템플릿 | NVIDIA PhysicsNEMO |
| PINN | `pina_wrapper.py` — 경량 PINN (복잡 경계조건) | PINA |
| PINN | `dd_pinn.py` — Domain Decomposition PINN | PyTorch |
| 물리 보정 | `physics_correction.py` — 보존법칙 사후 보정 (질량·운동량·에너지) | PyTorch |
| 물리 보정 | `hybrid_rom.py` — POD-Galerkin + NN 잔차 보정 | PyTorch |
| 방정식 발견 | `sindy_wrapper.py` — SINDy, PDE-FIND | PySINDy |
| 방정식 발견 | `symbolic_regression.py` — AI 모델 → 수식 자동 복원 | PySR (optional) |
| 멀티피델리티 | `multi_fidelity.py` — coarse→fine mesh 전이학습 | PyTorch |
| 테스트 | `tests/test_pinn.py`, `tests/test_correction.py` | pytest |

---

#### v3.2.0 — GUI 완성 + 배포

**목표:** 사용자 경험 완성. 한/영 지원, 튜토리얼, 보고서 자동 생성, Windows 정식 설치.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| GUI | `wizard/tutorial_wizard.py` — 샘플 데이터로 전체 파이프라인 가이드 | PySide6 |
| GUI | i18n 한/영 전환 (`styles/i18n/ko.json`, `en.json`) | PySide6 |
| GUI | `utils/undo_redo.py` — 전 패널 공통 Command 스택 | — |
| GUI | 모델 비교 대시보드 (정확도/속도/파라미터 수) | PySide6, pyqtgraph |
| 보고서 | `report/generator.py` — Jinja2 → HTML → PDF | Jinja2, weasyprint |
| 보고서 | 자동 그림·표 삽입 (유동장 스냅샷, 오차 곡선) | Matplotlib |
| 내보내기 | `export/onnx_export.py` — ONNX 변환 | ONNX |
| 내보내기 | `export/torchscript_export.py` | TorchScript |
| 패키징 | Inno Setup `.iss` 스크립트 — Windows 설치 파일 | Inno Setup |
| 패키징 | optional extras 분리 빌드 (`[core]` vs `[full]`) | — |
| 테스트 | `tests/test_export.py`, `tests/test_report.py` | pytest |

---

### v4.x — 최첨단 모델

---

#### v4.0.0 — State Space Model + Neural ODE 고도화

**목표:** Mamba 기반 선형 복잡도 연산자와 연속 시간 동역학 모델 통합.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| SSM | `mno.py` — Mamba Neural Operator (JCP 2025) | mamba-ssm, PyTorch |
| SSM | `deepomamba.py` — DeepONet + Mamba (JCP 2025) | mamba-ssm, PyTorch |
| 시계열 | `latent_dynamics_adv.py` — AE + Neural ODE 고도화 | torchdiffeq |
| Koopman | IKNO, FlowDMD 개선 | PyTorch |
| LBM 스냅샷 | `lbm_snapshot.py` — Lettuce GPU 기반 빠른 스냅샷 생성 | Lettuce (MIT) |
| flowtorch 연동 | PyTorch 텐서 직접 파이프라인 | flowtorch (GPL) |

---

#### v4.1.0 — 생성 모델 + KAN

**목표:** 확산 모델로 유동장 생성, KAN 기반 해석 가능 연산자.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 생성 모델 | `diffusion_pde.py` — Score-based Diffusion PDE 솔버 | PyTorch |
| 생성 모델 | `wavelet_diffusion_no.py` — Wavelet Diffusion NO (ICLR 2025) | PyTorch |
| 생성 모델 | `conditional_gen.py` — 파라미터 조건부 유동장 생성 | PyTorch |
| KAN | `kano.py` — KANO, 학습된 연산자의 해석적 수식 복원 | PyTorch |
| VAE | VAE 고도화 — 잠재 공간 샘플링으로 새 유동장 생성 | PyTorch |
| Tucker | `tucker_decomp.py` — Tucker 텐서 분해 | PyTorch |

---

#### v4.2.0 — Equivariant NN + 고급 텐서 분해

**목표:** 물리 대칭성 보존 모델 통합. 회전/반사 불변 학습.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| Equivariant | `group_equiv_fno.py` — Group Equivariant FNO (ICML 2023) | e3nn, PyTorch |
| Equivariant | `physics_embedded_gnn.py` — E(n)-GNN PDE 솔버 (NeurIPS 2022) | e3nn, escnn |
| Equivariant | `lie_algebra_no.py` — Lie Algebra Canonicalization (ICLR 2025) | escnn |
| 차원축소 | `cpod.py` — 물리 보존법칙 제약 POD | NumPy/SciPy |
| 차원축소 | `diffusion_maps.py` — 데이터 기하구조 보존 | scikit-learn |
| 모달 | `pgd.py` — Proper Generalized Decomposition | NumPy/SciPy |
| 유동분석 | `lcs.py` — LCS (라그랑지안 입자 궤적) | NumPy |
| 유동분석 | `entropy_gen.py` — 엔트로피 생성 공간 분포 | NumPy |

---

### v5.x — 연구 플랫폼

---

#### v5.0.0 — 고급 최적화 + 인증 ROM

**목표:** 항공우주·에너지 수준의 다목적 최적화와 오차 보장 ROM.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 다목적 최적화 | `moo_optimizer.py` — NSGA-II, MOEA/D, 파레토 프런트 | pygmo2 (GPL) |
| 위상 최적화 | `topology_opt.py` — SIMP 3D 위상 최적화 | DL4TO, PyTopo3D |
| MDO | `mdo_pipeline.py` — 다분야 최적화 (CFD+구조+AI) | OpenMDAO |
| 역문제 | `inverse_problem.py` — 압력분포→형상/경계조건 역추정 | Firedrake+pyadjoint (LGPL) |
| SU2 민감도 | `su2_adjoint.py` — Discrete Adjoint 형상 민감도 | SU2 (LGPL) |
| 인증 ROM | `certified_rb.py` — Certified Reduced Basis (오차 상한) | RBniCSx (LGPL) |
| 인증 ROM | `neural_rb.py` — DL + Certified RB | dlrbnicsx (LGPL) |

---

#### v5.1.0 — 멀티피델리티 + Active/Online Learning

**목표:** 데이터 효율 극대화. coarse→fine 전이, 불확실 영역 자동 샘플링.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 멀티피델리티 | `multi_fidelity_adv.py` — 멀티 레벨 전이학습 고도화 | PyTorch |
| 전이학습 | `transfer_learning.py` — 유사 형상/조건 모델 재사용 | PyTorch |
| Active learning | `active_learning.py` — 불확실성 기반 CFD 시뮬레이션 자동 요청 | scikit-learn, UQpy |
| Online learning | `online_learning.py` — 새 데이터 유입 시 점진적 업데이트 | PyTorch |
| Hybrid ROM | `hybrid_rom_adv.py` — POD-Galerkin + NN 잔차 보정 고도화 | PyTorch |
| 미분가능 CFD | `jax_fluids_wrapper.py` — JAX-Fluids end-to-end 미분 | JAX-Fluids (optional) |

---

#### v5.2.0 — 설명가능성 + FastAPI + 완전 배포

**목표:** 연구 플랫폼 완성. 모델 해석, API 서버, 자동 업데이트.

| 영역 | 항목 | 라이브러리 |
|------|------|-----------|
| 설명가능성 | `shap_explainer.py` — 입출력 기여도 분석 | SHAP |
| 설명가능성 | `attention_viz.py` — Transformer/HAMLET 집중 영역 | captum |
| 설명가능성 | `kano_symbolic.py` — KANO 수식 자동 복원 | PyTorch |
| 설명가능성 | `pysr_recovery.py` — PySR 심볼릭 회귀 | PySR (Julia, optional) |
| 비교 대시보드 | 모델간 정확도/속도/파라미터 수 자동 비교 UI | PySide6, pyqtgraph |
| API | `api/server.py` — FastAPI REST 서버 (선택 모드) | FastAPI, uvicorn |
| 패키징 | 자동 업데이트 (GitHub Releases 기반) | — |
| 문서 | Sphinx + MkDocs API 문서 자동 생성 | Sphinx |

---

## 4. 우선순위 기준

1. **사용 빈도** — 가장 많은 CFD 사용자가 필요로 하는 기능 먼저
2. **구현 안정성** — 성숙한 라이브러리 기반 모듈 먼저, 최신 연구 코드는 후순위
3. **의존성 최소화** — `[core]` 배포판은 MIT/BSD 위주. GPL/LGPL·무거운 의존성(JAX, Julia, Fortran)은 `[full]` optional extra
4. **검증 가능성** — 해석해가 있는 기법(Couette, Poiseuille)으로 자동 검증 가능한 것 우선
5. **GPU 없이도 동작** — CPU 폴백이 가능한 기법을 우선 구현
6. **증분 배포** — 각 버전은 이전 버전에서 독립적으로 기능 추가 가능해야 함

---

## 5. 기술 선택 근거

| 선택 | 이유 | 대안 검토 |
|------|------|-----------|
| PySide6 (Qt6) | 크로스플랫폼, Python 바인딩 성숙도, 상업 무료 | wxPython (생태계 작음), tkinter (3D 통합 어려움) |
| PyVista | VTK Python 래퍼 최고 성숙도, pyvistaqt로 Qt 통합 용이 | Mayavi (유지보수 약화), vedo |
| HDF5 (.ntwin) | 메쉬+필드+가중치를 단일 파일에, VTKHDF 기반 ParaView 호환 | NetCDF (CFD 표준 아님), SQLite (바이너리 배열 비효율) |
| foamlib | 2025 JOSS, 비동기 지원, 타입힌트 완비 | fluidfoam (바이너리 미지원), PyFoam (구식 API) |
| neuraloperator v2.0+ | FNO 원저자 라이브러리, GINO/UQNO 포함, 활발한 개발 | 자체 구현 (유지보수 부담) |
| DAPPER | DA 알고리즘 벤치마크 표준, 2024 JOSS | filterpy (기본 EnKF만 지원) |
| pyPDAF | Fortran 수준 DA 효율, LGPL-3.0, 2025 GMD 게재 | DAPPER (프로토타이핑용, 대규모 부적합) |
| SALib | Sobol/Morris/FAST 통합, 2022 SESMO | 자체 구현 (검증 비용 큼) |
| PySR | Julia 백엔드로 속도 탁월, 2024 학술지 게재 | gplearn (성능 낮음) |
| Dedalus (검증) | 스펙트럼법 고정밀, 설치 간단, GPL-3.0 호환 | FEniCSx (더 무거움), 자체 구현 (검증 비용) |
| Gmsh | OCC 기반 최고 품질 비정형 메쉬, Python API 완비 | SALOME (GUI 의존), netgen (제한적) |
| NLopt | 30+ 알고리즘 단일 인터페이스, LGPL-2.1 | scipy.optimize (알고리즘 수 적음) |
| pygmo2 | ESA 주도 다목적 최적화 표준, NSGA-II 포함 | pymoo (MIT이지만 기능 적음) |

---

## 6. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| PhysicsNEMO 설치 복잡 (CUDA 버전 종속) | PINA를 경량 대안으로 병행 지원 |
| mamba-ssm Windows 지원 불안정 | WSL 또는 Linux 환경 권장 문서화, CPU 폴백 |
| PySR Julia 런타임 번들링 어려움 | optional extra 분리, 설치 안내 자동화 |
| neuraloperator 대형 모델 GPU 메모리 | 배치 크기 자동 조정, gradient checkpointing 옵션 |
| 상업 포맷(Fluent .cas.h5) 스펙 비공개 | meshio + h5py 조합으로 역공학, 공개 스펙 최대 활용 |
| GPL 의존성 Windows 배포 복잡도 | `[core]` 배포판은 MIT/BSD만 포함. `[full]` extra로 분리 |
| Gmsh/SU2/FEniCSx 설치 복잡 (Windows) | conda-forge 경로 우선, 실패 시 기능 비활성화 + 안내 메시지 |
| pyPDAF Fortran 컴파일 필요 | conda 설치 경로 권장, 실패 시 DAPPER로 자동 폴백 |
| Firedrake Linux 전용 | Docker/WSL 환경에서만 지원, Windows 직접 설치 불가 안내 |
