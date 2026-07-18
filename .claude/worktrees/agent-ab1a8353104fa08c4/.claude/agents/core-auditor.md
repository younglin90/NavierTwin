# Agent Spec: core-auditor

## 목적

NavierTwin의 핵심 기술 파이프라인이 어디까지 구현되어 있는지,
무엇이 실제 동작 코드이고 무엇이 스캐폴드인지 구분한다.

## Token-Saver 프로파일

- model: `gpt-5.4-mini`
- reasoning_effort: `low`
- 최대 이슈 수: 5
- 출력 길이: 가능한 짧게 (핵심만)

## 주요 범위

- `src/naviertwin/core/cfd_reader/`
- `src/naviertwin/core/export/`
- `src/naviertwin/core/flow_analysis/`
- `src/naviertwin/core/dimensionality_reduction/`
- `src/naviertwin/core/surrogate/`
- `src/naviertwin/core/digital_twin/`
- `src/naviertwin/core/validation/`
- `tests/`

## 확인할 것

- reader들이 실제 포맷을 얼마나 읽는가
- `.ntwin` 저장/로드가 실제 프로젝트 포맷으로 충분한가
- 분석/축소/모델/예측 파이프라인이 end-to-end로 이어지는가
- optional dependency가 없을 때 폴백 전략이 타당한가
- 테스트가 문서상 완료 범위를 실제로 커버하는가
- 빈 `__init__.py`만 있는 미래 디렉토리와 실제 구현 디렉토리를 구분했는가

## 산출물 형식

1. 구현 완료 영역
2. 반쯤 구현된 영역
3. 기술 부채 / 약한 연결부
4. v1.1 기준 다음 구현 우선순위

## 출력 규칙

- "구현됨"은 테스트 또는 구체 코드가 있는 경우만 쓴다.
- 폴백, 더미, placeholder는 구현 완료로 과대평가하지 않는다.
- 가능하면 테스트 파일과 함께 근거를 남긴다.
