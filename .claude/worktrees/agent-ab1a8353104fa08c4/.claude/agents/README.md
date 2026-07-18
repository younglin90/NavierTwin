# NavierTwin Sub-Agent Specs

이 디렉토리는 NavierTwin 저장소를 병렬로 조사하거나 구현할 때 사용할
역할별 서브에이전트 스펙을 저장한다.

## 기본 원칙

- 각 에이전트는 맡은 범위만 본다.
- 문서 조사와 구현 작업을 섞지 않는다.
- 먼저 현재 상태를 요약하고, 그 다음 blocker와 다음 액션을 정리한다.
- 가능하면 파일 경로와 라인 근거를 함께 남긴다.
- 다른 에이전트의 책임 범위를 침범하지 않는다.

## 권장 역할

- `docs-auditor.md`
  문서, 버전 문자열, 로드맵, 패키지 메타데이터 정합성 점검
- `core-auditor.md`
  core 파이프라인 구현 상태와 기술 부채 점검
- `gui-integrator.md`
  GUI 단계별 연결 상태와 실사용 blocker 점검
- `test-runner.md`
  테스트/실행환경/패키징 관점 문제 점검

## Token-Saver 기본값

토큰 절약이 목표일 때 아래 프로파일을 기본으로 사용한다.

- model: `gpt-5.4-mini`
- reasoning_effort: `low`
- 출력 원칙:
  - 불필요한 서론 금지
  - 핵심 이슈 최대 5개
  - 파일 근거는 꼭 필요한 라인만
  - 결론은 5~10줄 내

## 권장 사용 순서

1. `docs-auditor`
2. `core-auditor`
3. `gui-integrator`
4. `test-runner`

## 구현 단계 확장

조사 단계가 끝나면 같은 이름으로 `*-worker.md`를 추가해
수정 작업 전용 에이전트 세트를 만들 수 있다.
