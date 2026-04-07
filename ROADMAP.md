# NavierTwin 로드맵

## 현재 단계: v1.0 MVP

### Phase 0: 프로젝트 스캐폴딩
- [ ] pyproject.toml 작성 (의존성: PySide6, pyvista, meshio, foamlib, pyCGNS, numpy, scipy, torch, smt, pysindy, pyspod, pykoopman, pykoop, pina, uqpy, salib, openmdao, pysr, dapper 등)
- [ ] src/naviertwin/ 전체 디렉토리 구조 생성 (PLAN.md §4 기준)
- [ ] 각 모듈 __init__.py + base.py 추상 클래스 작성
- [ ] main.py 엔트리포인트
- [ ] utils/ (config, logger) 기본 구현

### Phase 1: CFD Reader
- [ ] base_reader.py: BaseReader ABC, CFDDataset 데이터클래스
- [ ] reader_factory.py: 경로/확장자 기반 자동 감지
- [ ] openfoam_reader.py: fluidfoam 기반 polyMesh+field 파싱
- [ ] vtk_reader.py: meshio 기반 VTK/VTU 읽기
- [ ] fluent_reader.py: .cas/.dat + .cas.h5/.dat.h5
- [ ] 내부 HDF5 (.ntwin) 변환/저장/로드
- [ ] tests/test_cfd_reader.py

### Phase 2: 차원축소 (선형)
- [ ] base.py: BaseReducer ABC (fit/encode/decode/reconstruct)
- [ ] pod.py: Snapshot POD (SVD)
- [ ] randomized_svd.py
- [ ] 에너지 누적 그래프 유틸
- [ ] tests/test_reduction.py

### Phase 3: DMD
- [ ] dmd.py: PyDMD 래퍼 또는 자체 구현
- [ ] tests/test_dmd.py

### Phase 4: Surrogate (기본)
- [ ] base_surrogate.py: BaseSurrogate ABC (fit/predict)
- [ ] rbf_surrogate.py
- [ ] kriging_surrogate.py (SMT)
- [ ] tests/test_surrogate.py

### Phase 5: PINN
- [ ] physnemo_wrapper.py: PhysicsNEMO 래퍼
- [ ] 사전정의 NS/에너지 방정식 템플릿

### Phase 6: 유동분석 (기본)
- [ ] q_criterion.py
- [ ] fft_psd.py
- [ ] yplus.py
- [ ] tests/test_flow_analysis.py

### Phase 7: 디지털 트윈 엔진
- [ ] twin_engine.py: predict(params) → field 파이프라인
- [ ] Surrogate 또는 PINN 선택 → PyVista 메쉬 복원

### Phase 8: 검증
- [ ] metrics.py: RMSE, R², L2 norm
- [ ] 해석해 비교 유틸 (Couette, Poiseuille)

### Phase 9: GUI
- [ ] main_window.py: 탭/패널 호스트
- [ ] panels/import_panel.py
- [ ] panels/analyze_panel.py
- [ ] panels/reduce_panel.py
- [ ] panels/model_panel.py
- [ ] panels/twin_panel.py
- [ ] panels/export_panel.py
- [ ] widgets/vtk_viewer.py: PyVista Qt 뷰어
- [ ] styles/dark_theme.qss

### Phase 10: 내보내기 / 패키징
- [ ] export/onnx_export.py
- [ ] .ntwin 프로젝트 저장/복원
- [ ] PyInstaller 설정

---

## 완료된 항목
(완료 시 위에서 여기로 이동)
