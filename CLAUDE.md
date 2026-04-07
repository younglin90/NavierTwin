# NavierTwin — Claude Code 지시서

## 프로젝트 개요
CFD 후처리 결과 → AI/ROM/Operator Learning → 디지털 트윈 변환 Windows 데스크톱 툴.
비상업용(오픈소스). Python 3.10+.

## 반드시 읽을 문서
- `SPEC.md`: 기술 명세 — 기술 스택, 지원 포맷, 디렉토리 구조, 전체 기법 목록, 설계 원칙, 참고 문헌
- `PLAN.md`: 구현 계획 — 버전별 범위 정의, 우선순위 기준, 기술 선택 근거, 리스크
- `ROADMAP.md`: 진행 현황 — Phase별 체크리스트, 완료/미완료 태스크

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

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
