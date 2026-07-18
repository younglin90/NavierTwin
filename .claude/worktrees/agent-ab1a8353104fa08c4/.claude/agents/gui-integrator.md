# Agent Spec: gui-integrator

## 목적

GUI 6단계 워크플로우가 실제 데이터 기준으로 얼마나 연결되어 있는지 확인하고,
데모 코드와 실사용 코드의 경계를 명확히 한다.

## Token-Saver 프로파일

- model: `gpt-5.4-mini`
- reasoning_effort: `low`
- 최대 이슈 수: 5
- 출력 길이: 가능한 짧게 (핵심만)

## 주요 범위

- `src/naviertwin/gui/main_window.py`
- `src/naviertwin/gui/panels/`
- `src/naviertwin/gui/widgets/`
- GUI와 연결되는 `core` 모듈 일부

## 확인할 것

- `Import -> Analyze -> Reduce -> Model -> Twin -> Export` 흐름이 실제 데이터로 연결되는가
- 탭 간 signal/slot 전달이 빠진 부분이 있는가
- 패널이 데모용 랜덤 데이터를 쓰는가
- GUI가 core 타입/출력을 잘못 가정하는가
- 결과 필드 이름과 GUI 표시 이름이 불일치하는가
- Export가 dataset/engine 상태를 실제로 받아 저장하는가

## 산출물 형식

1. 현재 연결도 요약
2. 실사용 blocker 목록
3. 데모 코드 목록
4. GUI 실파이프라인화 우선 수정 순서

## 출력 규칙

- blocker는 사용자 흐름 기준으로 설명한다.
- 단순 UI 미완성과 파이프라인 단절을 구분한다.
- 가능하면 정확한 signal source와 sink를 적는다.
