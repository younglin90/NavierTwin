---
name: NavierTwin Roadmap Status
description: 현재 v1.1.0 CFD I/O 확장 완료 (2026-04-07 기준)
type: project
---

현재 단계: **v1.1.0 진행 중** — 3 리더 구현 완료, 나머지 v1.1.x로 이연

**Why:** v1.0 MVP 완료 후 CFD I/O 확장(Fluent/CGNS/Gmsh)을 우선 구현.
su2_reader / mesh_generator / analytic_solutions 는 v1.1.x로 이연.

**How to apply:** 다음 작업은 ROADMAP.md v1.1.0 미완료 항목(su2_reader 등) 또는 v1.2.0(비선형 차원축소) 중 우선순위 확인 후 진행.

---

## 완료된 버전

### v0.1.0 ✅ — 프로젝트 스캐폴딩
pyproject.toml, src/naviertwin/ 전체 구조, base.py들, main.py, utils/

### v0.2.0 ✅ — CFD I/O 기초 + .ntwin 포맷
BaseReader, ReaderFactory, OpenFOAMReader, VTKReader, ntwin_format.py (HDF5)

### v0.3.0 ✅ — 기초 유동분석
Q-criterion/λ₂, FFT/PSD, y+, Cf

### v1.0.0 ✅ — MVP 릴리스
POD, Randomized SVD, DMD, RBF/Kriging surrogate, TwinEngine, validation metrics,
GUI 6패널(Import/Analyze/Reduce/Model/Twin/Export), VtkViewer, dark_theme.qss,
PyInstaller spec. 테스트 106 passed.

### v1.1.0 (부분 완료) — CFD I/O 확장

**완료 (2026-04-07):**
- `fluent_reader.py`: pv.FluentReader → meshio → FluentASCIIParser, sibling .dat 자동 감지, 바이너리 → ValueError
- `cgns_reader.py`: pv.CGNSReader → CGNS.MAP(pyCGNS 6.3.5 확인) → h5py → meshio
- `gmsh_reader.py`: gmsh probe → meshio (.msh v2.2/v4.1)
- `_mesh_utils.py`: meshio/pyvista → CFDDataset 공통 헬퍼
- `tests/test_cfd_io_expansion.py`: 26 테스트 (25 passed, 1 skipped)
- 전체 테스트: 131 passed, 1 skipped

**이연 (v1.1.x):**
- su2_reader.py, mesh_generator.py (Gmsh), mesh_processor.py (PyMeshLab)
- analytic_solutions.py (Dedalus), 해석해 vs 수치해 GUI 패널

---

## 다음 단계 후보

- v1.1.x: su2_reader, mesh_generator, analytic_solutions
- v1.2.0: AE/VAE 비선형 차원축소, SPOD, 웨이블릿 분석
