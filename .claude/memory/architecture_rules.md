---
name: NavierTwin Architecture Rules
description: 코딩 규칙, 아키텍처 원칙, 디렉토리 구조 — CLAUDE.md + PLAN.md §7 기반
type: project
---

**코딩 규칙:**
- 언어: Python 3.10+, 타입 힌트 필수
- docstring: Google style, 한국어 가능
- import 정렬: isort 호환

**아키텍처 원칙:**
- 팩토리 패턴 통일: `fit()/predict()` 또는 `fit()/encode()/decode()` 인터페이스
- 각 모듈 디렉토리에 `__init__.py` + `base.py` (추상 클래스) 필수
- core↔gui 분리: core 모듈은 Qt 의존 금지, GUI는 시그널/슬롯으로 통신
- 내부 데이터: HDF5 (.ntwin), 메쉬는 PyVista UnstructuredGrid 통일
- 설정: JSON 기반 (`utils/config.py`)
- GPU 폴백: NVIDIA GPU 미탑재 시 CPU 모드 (일부 제외), CUDA 버전 자동 체크
- Undo/Redo 스택: 전 패널 공통 명령 스택
- i18n: 한국어/영어 최소 지원

**디렉토리 구조 (핵심):**
```
NavierTwin/
├── src/naviertwin/
│   ├── core/
│   │   ├── cfd_reader/
│   │   ├── dimensionality_reduction/ (linear/, nonlinear/)
│   │   ├── flow_analysis/ (modal/, vortex/, statistics/, boundary_layer/, thermofluids/)
│   │   ├── surrogate/
│   │   ├── operator_learning/ (fno/, deeponet/, latent_operator/, koopman/, kan/, unet/)
│   │   ├── gnn/
│   │   ├── state_space/
│   │   ├── generative/
│   │   ├── time_series/
│   │   ├── equivariant/
│   │   ├── physnemo/
│   │   ├── digital_twin/
│   │   ├── validation/
│   │   ├── explainability/
│   │   ├── export/
│   │   └── report/
│   ├── gui/ (main_window.py, panels/, widgets/, wizard/, styles/)
│   └── utils/ (config, logger, Undo/Redo)
├── tests/
├── resources/
├── installer/
├── CLAUDE.md, PLAN.md, ROADMAP.md
├── pyproject.toml
└── main.py
```
