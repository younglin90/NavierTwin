# NavierTwin — Claude Code 지시서

## 프로젝트 개요
CFD 후처리 결과 → AI/ROM/Operator Learning → 디지털 트윈 변환 Windows 데스크톱 툴.
비상업용(오픈소스). Python 3.10+.

## 반드시 읽을 문서
- `PLAN.md`: 전체 아키텍처, 기술 스택, 디렉토리 구조, 전체 기법 목록
- `ROADMAP.md`: 현재 진행 단계, 완료/미완료 태스크

## 코딩 규칙
- 언어: Python 3.10+, 타입 힌트 필수
- GUI: PySide6 (Qt6)
- 3D: PyVista + pyvistaqt
- AI/ML: PyTorch (CUDA 지원)
- 테스트: pytest, tests/ 디렉토리
- 린터: ruff
- 패키지 관리: pyproject.toml (setuptools)
- docstring: Google style, 한국어 가능
- import 정렬: isort 호환

## 아키텍처 규칙
- 모든 모듈은 팩토리 패턴: `fit()/predict()` 또는 `fit()/encode()/decode()` 인터페이스
- 각 모듈 디렉토리에 `__init__.py` + `base.py` (추상 클래스) 필수
- 내부 데이터 포맷: HDF5 (.ntwin), 메쉬는 PyVista UnstructuredGrid 통일
- GUI 시그널/슬롯으로 core↔gui 분리 (core는 Qt 의존 금지)
- 설정: JSON 기반 (`utils/config.py`)

## 디렉토리 구조
`PLAN.md`의 §4 참조. 이 구조를 반드시 따를 것.

## 현재 단계
`ROADMAP.md` 참조. 현재 v1.0 MVP 단계.
