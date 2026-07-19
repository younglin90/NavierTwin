# NavierTwin 로드맵

> Phase별 세부 태스크 체크리스트. 버전별 목표·범위·근거는 `PLAN.md` 참조.

## 현재 단계: v5.0~v5.8 — 일반 CFD 트윈 플랫폼 전환 ✅ (2026-07-19 완료)

Route 1(공통 격자 재샘플: ROM/DMD/EZyRB/GeometryFNO)과 Route 2(메쉬 네이티브,
재샘플 없음: mesh_gnn/GINO/MeshGraphNets) 양쪽 다 배선 완료. 4계층 저장
아키텍처(원본 불변·canonical VTKHDF 캐시·Zarr ML 캐시·MLflow 실험 관리) 완료.
대형 3D 자원 계획, Electron 배포, 운영 센서 연결, 확장 CGNS, Transolver,
API 보안·관측성까지 확장했다. 산업 데이터·클러스터 실측은 아래 외부 검증에서
별도 관리한다.

상세 설계·갭 분석·전략 카탈로그: `.omc/plans/twin-platform-roadmap.md`

- [x] v5.0-M1: 능력 기반 전략 레지스트리(`core/digital_twin/strategies.py`) —
      ②Model 카드/데스크톱 어드바이저가 로드 시점에 가능/불가+이유 표시
- [x] v5.0-M1: 비정상×다케이스 시간축 보존 — 케이스 세트가 (μ, t) 로 학습,
      ③Twin 에 t 슬라이더 자동 생성 (`sweep_unsteady` 데모, PVD 우선 규칙)
- [x] v5.0-M2: 벡터 성분 보존 — U 가 U_x/U_y/U_z 채널로 학습(방향 유지),
      U_mag 는 파생
- [x] v5.2: ParametricDMD — 비정상 스윕의 (μ, t) 예보 (partitioned + ezyrb 보간)
- [x] GUI 패리티: 데스크톱 전략 어드바이저 + 데모 데이터 메뉴 + 웹 엔진
      param_names 호환
- [x] v5.2: EZyRB 서로게이트 — `ezyrb_gpr`(예측 σ UQ)/`ezyrb_ann` 키, 리더보드 포함
- [x] v5.2: FNO+SDF 채널 core — `GeometryFNO2D` + `cases_to_grid_tensors`
      (형상=SDF 채널 · 조건=브로드캐스트 채널, DeepCFD/Thuerey 방식)
- [x] v5.2: GeometryFNO service/app 배선 — operator 전략이 정상 형상가변 케이스
      세트에서 활성화(공통 격자+SDF 채널), 예측은 공통 격자 뷰어 전환
- [x] v5.2: ParametricDMD(비정상 스윕 μ,t 예보), EZyRB(GPR·UQ/NN) 배선
- [x] v5.4: 셀별 오차장(twin_error) + 실제/트윈 요약 지표 + 정직한 외삽 인지
- [x] v5.6 P0: 학습 디바이스 배지(GPU/CPU), AMP·미니배치(A3, ~2.2배), OOD 3단계
      지지집합 상태(IN/NEAR/OUT), GeometryFNO 마스크 손실(0-채움 셀 loss 제외)
- [x] 검토 반영: 데이터 계약 우선 재정렬(로드맵 §6½), signed SDF 폐곡면 한정 강제
- [x] v5.1: 경계조건 UI — ①Import 에 벽 선택 모드(trame server 픽킹) + wall-distance
      계산 버튼. region growing/seed 확장은 후속(현재는 단일 픽 누적)
- [x] v5.4: 좌(실제)/우(트윈) 분할 뷰어 — 독립 Plotter 2개, 공통 컬러 범위(clim) 강제,
      카메라는 수동 동기화 버튼(실시간 드래그 연동은 아님, 문서화됨)
- [x] v5.6 P1: 케이스 로드/재샘플 병렬화(thread_map, ~4.6배 실측)
- [x] v5.1 후속: BC 값 입력 폼(속도/압력/온도/사용자 정의) — 기존 wall-picking
      카드 확장, OpenFOAM patch 메타와 병합된 통합 목록(service.list_boundary_patches)
- [x] v5.6 P1+: 리더보드 조합 병렬(ThreadPoolExecutor, 완료순 진행률 보고로 갱신)
- [x] 검토 §6½ #2: 그룹(trajectory/case) 스플릿 + train-only 정규화 primitive
      (`core/preprocessing/group_split.py`) — 아직 학습 경로 미배선, 척추만 확보
- [x] v5.6 P1+: remap 오차 바닥 분리(reconstruction test) — `estimate_remap_floor()`,
      공통 격자 왕복 오차를 GeometryFNO 학습 결과에 표기(모델 오차와 분리)
- [x] v5.6 P1+: 그룹 스플릿을 실제 학습 경로에 배선 — `build_geometry_fno_twin(
      group_split=True)`, held-out 케이스만 평가(4-way 일반화 라벨), 기본값은
      이전과 동일(하위 호환)
- [x] v5.1 후속: seed+region growing 확장 — `grow_wall_selection()` BFS
      (edge-connected 이웃만, 모서리 넘어 새는 것 방지) + "선택 확장" 버튼
- [x] v5.1 후속: CGNS ZoneBC 자동 wall 인식 — 폴백 리더(pyCGNS/h5py)에
      Elements_t 셀 연결성 파싱 추가(점 구름 탈피), BC_t → boundary_patches
      (OpenFOAM 과 동일 계약) + auto_wall_patches. MIXED/NGON/NFACE/멀티존도
      v5.8에서 연결성·필드·zone-qualified BC를 보존하도록 확장
- [x] 검토 §6½ #5: 보존량(rho/mass_flux 등) 점 보간 재샘플 경고 —
      `flag_conserved_fields()`, coarsen/케이스 재샘플 UI 에 warning 표시
- [x] GeometryFNO held-out 검증 토글(웹 UI) — 시그니처 geometry_id 자동 부여,
      공통 격자 재샘플로 시그니처가 뭉치면 케이스=그룹 폴백(문서화됨)
- [x] v5.6 P1+: MPI 배치 CLI(클러스터) — `naviertwin batch-train --config jobs.json`,
      `jobs[rank::size]` 라운드로빈 분배, mpi4py 없거나 단독 실행 시 rank 0/size 1
      순차 폴백, MPI 초기화는 헤드리스 CLI 전용(GUI 이벤트 루프 금지)
- [x] 검토 §6½ #8+#10: 모델 등급제(production/domain/experimental tier —
      ②Model 카드에 "실험적" 뱃지) + DataProfile topological/embedding 차원 분리
- [x] 검토 canonical data model 1단계: `core/data_model/signature.py` —
      topology/coordinate sha256 해시로 동일 격자 O(1) 판정,
      `assign_geometry_ids()` → 그룹 스플릿 `group_ids` 자동 연결
- [x] 검토 §6½ #6: Zarr 텐서 캐시(`core/storage/tensor_cache.py`) — 케이스
      세트 텐서화(cases_to_grid_tensors) 결과를 시그니처 해시 키로 캐시,
      재학습 시 재계산 생략(`build_geometry_fno_twin(use_tensor_cache=True)`)
- [x] Route 2 스코핑 연구(`.omc/research/route2-mesh-native-wiring.md`) —
      torch_geometric 2.7.0 실증, physicsnemo 는 torch 2.11 비호환으로 차단,
      mesh_gnn 을 첫 타깃으로 추천
- [x] **Route 2 첫 배선: `mesh_gnn`** — 메쉬 네이티브(재샘플 없음) 케이스 세트
      트윈, `varying_mesh` 덕타이핑으로 원본 케이스 메쉬(진짜 구멍) 위에
      그대로 표시(앱 예측 경로 수정 0). tier="experimental". karman_shapes
      헤드리스 플로우로 검증
- [x] **Route 2 두 번째 배선: `gino`** — neuraloperator GINO, 점군 기반이라
      고정 그래프 없이 예측 좌표가 학습 케이스와 달라도 그대로 동작
      (mesh_gnn 의 kNN 폴백 불필요). tier="experimental"
- [x] 검토 §6½ #6: MLflow 실험 추적(`core/experiment/tracking.py`) — 학습
      실행마다 strategy/params/metrics 기록, mlflow 미설치/실패는 조용한
      no-op(학습 흐름을 절대 막지 않음)
- [x] canonical data model 마무리: `strategies._same_mesh_points()` 를
      DatasetSignature 해시 기반으로 배선(해시 불일치 시 기존 allclose
      허용오차 로직으로 폴백 — 판정 결과 100% 하위 호환, 빠른 경로만 추가)
- [x] MeshGraphNets `predict()` stale edge_features 버그 수정 — 그래프
      override 시 edge_features 를 재계산하지 않고 재사용하던 결함
      (Route 2 연구가 발견, 재현 테스트로 고정)
- [x] 검토 §6½ #5 후속: 보존량 volume-weighted 재샘플 옵션
      (`conservative_resample_to_grid`) — 체커보드 밀도장 기준 점 보간
      오차 100% → 부피가중 오차 0.002%로 실측 검증. 기본값(미지정)은
      기존 동작과 bit-동일
- [x] GUI 테스트 스위트 행(hang) 수정 — `QApplication.setStyleSheet()` 가
      매 `MainWindow` 생성마다 프로세스 내 살아있는 모든 위젯을 재폴리시해
      테스트가 누적될수록 선형으로 느려지던 근본 원인 수정 (36개 테스트
      무한대기 → 43초 완주)
- [x] zarr_reader.py 를 zarr 3.x 호환(`create_array`, `create_dataset` 폴백)으로 수정
- [x] 기존(세션 무관) 테스트 실패 10건 정리 — 검증 리포트 계약 누락 키,
      emit 스크립트 UTF-8 인코딩 누락, postproc 스모크 테스트의 연산
      순서 의존성(`load_rom` 이 `save_rom` 산출물에 암묵 의존) 등
- [x] **Route 2 세 번째 배선: `mesh_gnn_mp`(MeshGraphNets)** — 메시지
      패싱 edge_features 를 실제로 쓰는 케이스 세트 파라메트릭 트윈.
      `fit()` 대신 `_model` 직접 구동(케이스별 다른 edge_index 대응),
      한 스텝 델타 예측으로 정상장 회귀를 프레이밍
- [x] 검토 §6½ #9 잔여분: 분할 뷰어 시간(timestep) 동기 — 좌측 슬라이더
      변경 시 우측 트윈 예측도 같은 시간으로 자동 재계산(정상 트윈은
      조용한 no-op). 슬라이스 동기는 웹 GUI에 슬라이스 기능 자체가 아직
      없어 보류(문서화됨)
- [x] canonical VTKHDF 저장 계층(4계층 중 tier 2) — 기존 `.ntwin`
      포맷을 재사용한 `CanonicalCache`(mtime+size 키), 원본 파일 재파싱
      없이 캐시 히트. `use_canonical_cache=False`(기본)는 기존 동작과
      100% 동일

## 트윈 전략 확장 + GUI 비정상 지원 통일 (2026-07-19)

사용자 요청: "트윈 전략부의 기법들이 모두 비정상을 지원할 수 있다면 그렇게
하도록 GUI 셋팅구조를 바꿔라. 비정상/시계열 관련해서 트윈 예측 버튼에
연결 안 된 것들을 해결해달라. 대안 라이브러리에서 트윈 학습과 관련된
것들 모두를 사용할 수 있게 해달라. 우리는 결과만으로 트윈학습 시키는
앱이다"에 대한 응답.

- [x] **GUI 비정상 지원 표시 통일** — 전략 카드마다 "정상 + 비정상 모두
      지원" 캡션을 `strategy_report()`가 실어 나르는 `supports_time_in_sweep`
      필드로 표시(하드코딩 아님 — 새 전략이 미지원을 선언하면 자동으로 안
      뜬다). 케이스 세트가 비정상이면 카드 목록 아래 요약 배지도 추가.
      mesh_gnn/gino/mesh_gnn_mp spec의 "정상 스윕 전용"이라는 오해 소지
      주석을 "케이스 세트 전용(단일 케이스 미지원) — 정상/비정상 모두
      지원"으로 정정.
- [x] **dynamics 전략에 대안 예보 백엔드 4종** — 기존 `core/time_series/`
      아래 완성돼 있었지만 미배선이던 LSTM/Koopman 신경 연산자(KNO)/
      Latent-ODE/Neural-ODE 를 POD 계수 시계열 위에서 동작하는 DMD의
      대안 선택지로 연결. 케이스 세트(비정상 포함) μ 조건화는 "새 μ = 새
      초기조건"으로 처리(RBF로 초기 윈도우 보간).
- [x] **Route 2 네 번째 배선: `deeponet`** — operator 전략의 3번째
      backend(builtin/neuralop GeometryFNO 다음), branch=운전조건 μ,
      trunk=케이스 쿼리 좌표 순수 데이터 지도학습(PDE 잔차 없음). 공통
      격자 재샘플이 아예 없고 임의 좌표에서 예측 가능.
- [x] **새 전략: `mesh_gnn_rollout`** — 원본 MeshGraphNets의 진짜 자기회귀
      시간 롤아웃(`u_{t+1}=u_t+MGN(u_t,edge)`)을 단일 케이스 시계열
      전용으로 배선. `mesh_gnn_mp`(정상 케이스 세트를 "1스텝 가짜
      트레젝토리"로 재해석한 파라미터 회귀)와 반대 방향 — 다른 9개 전략과
      달리 케이스 세트가 아니라 **단일 케이스만** 지원. 롤아웃 캐싱으로
      이미 계산한 범위는 모델 재호출 없이 즉시 반환.
      **전략 총 10개**(rom/physics/dynamics/operator/mesh_gnn/gino/
      mesh_gnn_mp/transolver/deeponet/mesh_gnn_rollout)로 확장.
- [x] **범위에서 명시적으로 제외**(이유 포함) — PINN/PI-DeepONet(PDE 잔차
      콜백을 직접 작성해야 해 "결과만으로" 원칙과 불일치), Diffusion
      모델(비조건부 생성모델이라 μ→field 결정론적 예측 계약과 불일치),
      EnKF/UKF/4D-Var(이미 `/twin/stream/*` 온라인 세션으로 트윈 예측과
      개념적으로 분리돼 있어 억지로 합치면 오히려 혼란), DeepXDE/Warp/
      JAX-CFD(솔버 라이브러리지 결과 기반 트윈학습 라이브러리가 아님).

## 확장 기반 M13-M15 ✅ (2026-07-18)

- [x] **M13 대형 3D/HPC 자원 계획** — Route 1/2별 총 데이터·host/device peak
      추정, point chunk/case batch/microbatch/gradient accumulation/MPI rank 계획,
      `naviertwin plan-scale` JSON CLI, 웹 Operation 패널. 수백만 점 workload를
      bounded-memory 계약으로 자동 검증한다.
- [x] **M14 Windows-native Electron 실행** — Linux 직접 실행과 Windows
      `wsl.exe` 실행 명령 분리, `\\wsl.localhost`/`\\wsl$` 경로 자동 변환,
      shell-free 인자 전달, 환경 설정 조기 진단, 자식 프로세스 scoped 종료.
- [x] **M15 운영 센서 기반** — thread-safe bounded buffer, out-of-order/duplicate
      처리, 품질 상태, nearest/linear timestamp 정렬, interpolation gap/stale/missing
      차단, 원자적 assimilation interval gate. REST `/twin/stream/sensor`와
      `/twin/stream/align`, 웹 Operation 상태 점검을 기존 EnKF 세션에 연결했다.

## 제품화 기반 M16-M20 ✅ (2026-07-19)

- [x] **M16 Electron 배포** — Electron 43, AppImage/NSIS `electron-builder`,
      Linux/Windows CI matrix, 패키지 계약 테스트, npm audit 0건, Linux unpacked
      실제 패키징 검증
- [x] **M17 운영 연결기·복구** — MQTT/OPC UA/Kafka lazy optional adapter,
      공통 JSON/CSV payload 계약, SQLite WAL 영속 저장, sequence-safe upsert,
      재시작 복구, 수집·정렬·동화 metrics, REST close/metrics
- [x] **M18 CGNS 확장** — MIXED fixed elements, NGON polygon,
      NGON+NFACE VTK polyhedron, pyCGNS/h5py multizone, zone-qualified BC와
      합성 fixture 검증
- [x] **M19 Transolver** — learned physics-slice attention 기반 가변 점 수
      mesh-native operator, case-set train-only 정규화, 저장/복원, service/engine,
      전략 레지스트리와 Electron 웹 UI 배선
- [x] **M20 API 운영 기반 (100%)** — plaintext/SHA-256 API-key constant-time 인증,
      in-process sliding-window + SQLite multi-worker 공유 rate limit, 모든 business
      route의 `/api/v1` alias, health/readiness, request body cap, request ID/API
      version/server timing, 보안 헤더, OpenAPI security scheme, JSON/Prometheus,
      Uvicorn TLS/multi-worker/proxy trust CLI. 무설정 로컬 동작은 하위 호환

### 로컬 수용 범위 완성도

| 영역 | 완성도 | 수용 기준 |
|---|---:|---|
| CFD 데이터 로드·canonical 저장 | 100% | VTK/CGNS/OpenFOAM/Gmsh/SU2/Fluent ASCII + 캐시 계약 |
| 전처리·경계·재샘플 | 100% | BC picking, SDF, 해상도 축소, 보존량 옵션, train-only split |
| 트윈 학습 Route 1/2 | 100% | ROM/DMD(+LSTM/Koopman/Latent-ODE/Neural-ODE 대안)/FNO/DeepONet + GCN/GINO/MGN/Transolver/MGN-롤아웃 10전략 배선 |
| 실제·트윈 비교 | 100% | 분할 뷰, 공통 범위, 오차장/지표, extrapolation 상태 |
| 운영 센서·동화 | 100% | buffer/align/connectors/persistence/recovery/metrics 계약 |
| Electron 배포 기반 | 100% | AppImage/NSIS 설정, 패키지 테스트, Linux pack 검증 |
| API 보안·관측성 기반 | 100% | 인증/rate limit/version/metrics 계약 |
| 로컬 자동 검증 | 100% | ruff, Python regression, Electron Node tests/audit |

100%는 저장소에서 재현 가능한 **로컬 코드 수용 범위**다. 실제 설비 프로토콜,
산업 데이터 정확도, 클러스터 scaling, OS 코드 서명은 외부 환경 수용시험이며
코드 완성도에 섞지 않는다.

### 독립 검증 (2026-07-19)

위 M0~M20(Canonical Twin Architecture Foundation 포함)은 다른 AI 도구가
작성한 뒤 커밋되지 않은 채로 남아있던 것을, 별도 세션이 **자체 서술을 그대로
믿지 않고** 독립적으로 감사·보완했다:

- 전체 테스트 스위트 **4055 passed, 0 failed**(16 skipped, 11 deselected) 실측 확인
- M9(lineage 왕복) — 실제 재현 시 왕복 오차 정확히 `2.65e-8` 로 서술과 일치
- M10(36-scenario 벤치마크) — `--execute` 재실행으로 `36/36`, 오차 `0.0` 확인
- M11(비정상 트윈 temporal leakage 방지) — 4개 엔진(GeometryFNO/mesh_gnn/GINO/MGN)
  전부에서 source-case 그룹이 스플릿 경계를 넘지 않음을 실측 재현
- M12(DDP best-fit 스케줄러) — 진짜 best-fit-decreasing 알고리즘임을 코드로 확인
  (first-fit 을 과장한 게 아님)
- API 보안 — API-key 비교는 이미 `secrets.compare_digest`(상수시간) 사용 중,
  rate limiter 는 100-스레드/8-프로세스 동시성 테스트로 안전성 실증.
  **CSP 헤더 누락은 실제 발견해 추가**
- Transolver — 테스트 2개→15개로 확대, **단일 케이스 거절 로직 누락 버그를
  실제 발견해 수정**(다른 3개 Route 2 전략과 동등하지 않던 부분)
- Electron 패키징 — `electron-builder --dir` 실제 빌드 + 헤드리스 실행까지 검증,
  **패키징된 빌드에서만 나타나는 asar 경로 버그 2건을 실제 발견해 수정**
  (로그 파일 쓰기 실패로 크래시, Python 서버 루트를 못 찾는 문제) — 스크린샷으로
  GUI 정상 렌더링까지 확인

즉 "100%"는 자기 주장이 아니라 **재현·검증된 100%**다. 감사 과정에서 실제
버그(Transolver 단일 케이스 거절 누락, Electron 패키징 경로 2건, CSP 헤더
누락) 4건을 발견해 전부 수정했다.

## 외부 환경 수용시험·선택 연구

아래는 로컬에서 정직하게 완료 판정할 수 없는 외부 환경 의존 작업이다:

- **산업 3D/HPC 실측** — M13 계획을 실제 수백만 셀 CFD 데이터와 Slurm/PBS
  클러스터에 적용해 I/O throughput, peak RSS/VRAM, MPI scaling efficiency를
  측정하고 site별 job template를 확정해야 한다.
- **운영 설비 수용시험** — 구현된 MQTT/OPC UA/Kafka·SQLite 복구·API 인증을
  실제 broker/PLC/tag namespace/인증서/방화벽 정책에 연결해 site SLO를 승인한다.
- **모델 등급제의 Domain-Specific/Experimental 확장 카탈로그**
  (UPT/DoMINO/AB-UPT/Transolver++) — Transolver 기본 배선 완료. 나머지 우선순위는 실사용 데이터 특성에
  따라 갈린다(Production Core 4종은 이미 확보: ROM/DMD/GeometryFNO+
  mesh_gnn/GINO/MGN 이 Route 1/2 커버).

## 직전 단계: v4.2.0 + 17 rounds — 연구 플랫폼 + 전영역 성숙화 ✅

---

## v0.x — 기반 구축

### v0.1.0 — 프로젝트 스캐폴딩 ✅
- [x] `pyproject.toml` 작성 (setuptools, `[core]` / `[full]` / `[dev]` optional extras)
- [x] `src/naviertwin/` 전체 디렉토리 구조 생성 (SPEC.md §4 기준)
- [x] 각 모듈 `__init__.py` + `base.py` 추상 클래스 작성
- [x] `utils/config.py` — JSON 기반 설정 관리
- [x] `utils/logger.py` — 공통 로거
- [x] `main.py` — CLI 진입점
- [x] `pyproject.toml` 테스트/린터 설정

### v0.2.0 — CFD I/O 기초 + .ntwin 포맷 ✅
- [x] `core/cfd_reader/base.py` — `BaseReader` ABC, `CFDDataset` 데이터클래스
- [x] `core/cfd_reader/reader_factory.py` — 확장자 기반 자동 감지
- [x] `core/cfd_reader/openfoam_reader.py` — `pv.POpenFOAMReader` 우선, `ofpp` 폴백
- [x] `core/cfd_reader/vtk_reader.py` — VTK/VTU/STL (PyVista)
- [x] `core/export/ntwin_format.py` — `.ntwin` HDF5 저장/로드 (VTKHDF 구조)
- [x] 타임스텝 append 스트리밍 저장 구현
- [x] `tests/test_cfd_reader.py`

### v0.3.0 — 기초 유동 분석 ✅
- [x] `core/flow_analysis/vortex/q_criterion.py` — Q-criterion + λ₂
- [x] `core/flow_analysis/statistics/fft_psd.py` — FFT, PSD, 주파수 피크
- [x] `core/flow_analysis/boundary_layer/yplus.py` — y+, u_tau, Cf, 첫 번째 셀 높이 추정
- [x] `tests/test_flow_analysis.py`

---

## v1.x — MVP & 핵심 ROM

### v1.0.0 — MVP 릴리스 ✅
- [x] `core/dimensionality_reduction/linear/pod.py` — Snapshot POD, 에너지 누적 곡선 (modred)
- [x] `core/dimensionality_reduction/linear/randomized_svd.py`
- [x] `core/flow_analysis/modal/dmd.py` — DMD, FbDMD (PyDMD)
- [x] `core/surrogate/rbf_surrogate.py` (SMT)
- [x] `core/surrogate/kriging_surrogate.py` (SMT)
- [x] `core/digital_twin/twin_engine.py` — `predict(params) → field` 파이프라인
- [x] `core/validation/metrics.py` — RMSE, R², L2 norm
- [x] `gui/main_window.py` — 6패널 탭 호스트
- [x] `gui/panels/import_panel.py`
- [x] `gui/panels/analyze_panel.py`
- [x] `gui/panels/reduce_panel.py`
- [x] `gui/panels/model_panel.py`
- [x] `gui/panels/twin_panel.py`
- [x] `gui/panels/export_panel.py`
- [x] `gui/widgets/vtk_viewer.py` — AutoTessell식 `QtInteractor` viewer + headless 정적 fallback
- [x] `gui/styles/dark_theme.qss`
- [x] `.ntwin` 프로젝트 저장/복원 연동
- [x] `installer/naviertwin.spec` — PyInstaller `--onedir` 설정
- [x] `tests/test_reduction.py`, `tests/test_surrogate.py`, `tests/test_twin.py`

### v1.1.0 — CFD I/O 확장 (3 리더) ✅
- [x] `core/cfd_reader/fluent_reader.py` — `.cas/.dat` ASCII (pv.FluentReader → meshio → FluentASCIIParser, sibling .dat 자동 감지)
- [x] `core/cfd_reader/cgns_reader.py` — CGNS (`pv.CGNSReader` → pyCGNS → h5py → meshio)
- [x] `core/cfd_reader/gmsh_reader.py` — `.msh` v2.2/v4.1 (gmsh probe → meshio)
- [x] `core/cfd_reader/_mesh_utils.py` — 공통 메쉬 변환 헬퍼
- [x] `tests/test_cfd_io_expansion.py` — 26 테스트 (25 passed, 1 skipped/optional)

### v1.1.1 — SU2 + 메쉬 툴 + 해석해 검증 ✅
- [x] `core/cfd_reader/su2_reader.py` — SU2 `.su2` (meshio → SU2ASCIIParser, sibling `.csv` 자동 병합)
- [x] `core/tools/mesh_generator.py` — 채널/원통/NACA 익형 파라미터 메쉬 (Gmsh OCC)
- [x] `core/tools/mesh_processor.py` — simplify/smooth (PyMeshLab) + quality_report (PyVista 폴백)
- [x] `core/validation/analytic_solutions.py` — Couette / Poiseuille 2D / Poiseuille Pipe + Dedalus optional
- [x] `core/validation/analytic_solutions.compare_against_analytic` + metrics 연동
- [x] `gui/widgets/analytic_compare_widget.py` — Matplotlib 임베드 비교 시각화
- [x] `gui/panels/analyze_panel.py` — "해석해 비교" 5번째 분석 탭
- [x] `tests/test_su2_reader.py` (8 passed), `tests/test_mesh_tools.py` (6 passed), `tests/test_analytic.py` (7 passed, 1 optional skip)

### v1.2.0 — 비선형 차원축소 + SPOD + 고급 유동분석 ✅
- [x] `core/dimensionality_reduction/nonlinear/autoencoder.py` — PyTorch AE (MSE + Adam)
- [x] `core/dimensionality_reduction/nonlinear/vae.py` — β-VAE (reparameterization + ELBO + sample())
- [x] `core/dimensionality_reduction/nonlinear/gnn_ae.py` — GNN-AE (torch_geometric GCNConv, optional)
- [x] `core/flow_analysis/modal/spod.py` — Welch-block SPOD + PySPOD 백엔드 옵션
- [x] `core/flow_analysis/statistics/wavelet.py` — CWT (PyWavelets) + STFT 폴백
- [x] `core/flow_analysis/statistics/two_point_corr.py` — R(r) + 적분 길이 스케일
- [x] `core/flow_analysis/boundary_layer/boundary_layer.py` — δ99, δ*, θ, H + Cf
- [x] `core/flow_analysis/thermofluids/nondim.py` — Re / Pr / Nu / Pe / Gr / Ra
- [x] `core/data_augmentation/augmentation.py` — 갈릴레이 shift, reflect, rotate_2d, scale, symmetric
- [x] `tests/test_nonlinear_reduction.py` — 11 테스트 (AE/VAE/GNN-AE/SPOD/2pc/BL/nondim/aug) 전부 통과

---

## v2.x — 신경 연산자

### v2.0.0 — FNO / DeepONet / U-Net MVP ✅
- [x] `core/operator_learning/fno/fno.py` — SpectralConv1d/2d + FNO1D/FNO2D (PyTorch 직접 구현)
- [x] `core/operator_learning/deeponet/deeponet.py` — branch/trunk MLP + bias (trunk 좌표 캐시)
- [x] `core/operator_learning/unet/unet.py` — 2-level U-Net (encoder/decoder + skip)
- [x] `tests/test_operator_learning.py` — 8 테스트 통과 (FNO1D/FNO2D/DeepONet/UNet2D + not-fitted 가드)
- [x] `fno/tfno.py` — Tucker-factorized FNO2D (파라미터 90% 절감)
- [x] `fno/wno.py` — Wavelet Neural Operator 1D (pywt optional)
- [x] `deeponet/pi_deeponet.py` — Physics-Informed DeepONet (물리 잔차 λ_phys)
- [x] `deeponet/mionet.py` — Multiple-Input Operator Network (product/concat merge)
- [x] Model 패널 GUI — FNO/TFNO/DeepONet/UNet/WNO 데모 학습 버튼 통합
- [x] `tests/test_operator_learning_ext.py` — 7 pass + 1 skip
- [x] `fno/adaptive_fno.py` / `fno/lno.py` / `fno/spectral_refiner.py` (v2.0.x)
- [x] `deeponet/sequential_deeponet.py` / `nfno_deeponet.py` (v2.0.x)
- [x] `latent_operator/l_deeponet.py` / `latent_operator/pi_latent_no.py` (v2.0.x)
- [x] 학습 loss curve 실시간 플롯 (`LossCurveWidget`, ModelPanel 연동)

### v2.1.0 — GNN 계열 ✅ (부분 완료 — GCN surrogate + MeshGraphNets MVP)
- [x] `core/gnn/gnn_surrogate/gnn_surrogate.py` — GCNConv 기반 node-level surrogate
- [x] `core/gnn/meshgraphnets/meshgraphnets.py` — Encode-Process-Decode + rollout
- [x] `tests/test_gnn.py` — 6 tests pass
- [ ] `core/gnn/egno/egno.py` — E(n)-Equivariant GNN (e3nn) (v2.1.x)
- [x] `core/gnn/graph_transformer/hamlet.py` (v2.1.x)

### v2.2.0 — 시계열 + Koopman 신경 연산자 ✅ (부분 완료)
- [x] `core/time_series/lstm/lstm.py` — LSTM autoregressive (lookback 윈도우 + rollout)
- [x] `core/time_series/transformer/transformer_ts.py` — causal mask Transformer encoder
- [x] `core/time_series/neural_ode/neural_ode.py` — torchdiffeq + RK4 폴백
- [x] `core/operator_learning/koopman/kno.py` — encoder/decoder + 선형 Koopman K
- [x] `tests/test_time_series.py` — 6 tests pass
- [x] `core/time_series/latent_dynamics/latent_dynamics.py` — AE + Neural ODE (v4.0.0 에서 구현)
- [x] `core/operator_learning/koopman/ikno.py` — Real-NVP 가역 Koopman (round10)
- [x] `core/time_series/temporal_no/tno.py` (v2.2.x)
- [x] `core/operator_learning/koopman/flowdmd.py` (v2.2.x)
- [x] `core/flow_analysis/modal/pykoopman_wrapper.py` (v2.2.x)

---

## v3.x — 디지털 트윈 완성

### v3.0.0 — 데이터 동화 + UQ + 기본 최적화 ✅ (부분 완료)
- [x] `core/data_assimilation/enkf.py` — Stochastic EnKF + inflation
- [x] `core/data_assimilation/particle_filter.py` — Bootstrap SIR + systematic resample
- [x] `core/sensitivity/sobol_analysis.py` — Saltelli 샘플링 + Sobol S1/ST (SALib 옵션 wrapper)
- [x] `core/optimization/mc_propagation.py` — 평균/표준편차/백분위수 MC 전파
- [x] `core/optimization/bayesian_opt.py` — GP + EI 최소화 (scikit-learn)
- [x] `tests/test_da_uq.py` — 7 tests pass
- [x] `core/data_assimilation/four_dvar.py` — 선형 4D-Var (round6)
- [x] `core/optimization/uq_surrogate.py` — PCE + Sobol (round9)
- [x] `core/optimization/surrogate_opt.py` — RBF + L-BFGS-B (round9)
- [x] `core/sensitivity/causal_analysis.py` — Pearson + Granger (round6)
- [ ] `core/data_assimilation/enkf_hpc.py` (v3.0.x, pyPDAF)

### v3.1.0 — PINN + 물리 보정 + 방정식 발견 ✅ (부분 완료)
- [x] `core/physnemo/pina_wrapper.py` — PINNSolver (PINA-style, PyTorch 직접)
- [x] `core/physics_correction/physics_correction.py` — 선형 제약 투영 + 질량 보존
- [x] `core/physics_correction/hybrid_rom.py` — POD + NN 잔차 보정
- [x] `core/flow_analysis/modal/sindy_wrapper.py` — STLSQ 자체 구현 + PySINDy 백엔드
- [x] `tests/test_pinn_correction.py` — 7 tests (1D Poisson 수렴 포함)
- [x] `core/explainability/symbolic_regression.py` — PySR + poly fallback (round13)
- [x] `core/multi_fidelity/multi_fidelity.py` — Additive Co-Kriging (v5.1.0)
- [x] `core/physnemo/physnemo_wrapper.py` (NVIDIA PhysicsNEMO, v3.1.x)
- [x] `core/physnemo/dd_pinn.py` — Domain Decomposition PINN (v3.1.x)

### v3.2.0 — GUI 완성 + 배포 ✅ (핵심 MVP 완료)
- [x] `gui/styles/i18n/ko.json`, `en.json` + `utils/i18n.py` Translator
- [x] `utils/undo_redo.py` — Command 스택 (max_size cap 포함)
- [x] `core/report/generator.py` — Jinja2 HTML + weasyprint PDF
- [x] `core/export/onnx_export.py` — opset / dynamic_axes / legacy 경로 fallback
- [x] `core/export/torchscript_export.py` — trace / script 지원
- [x] `tests/test_export_report.py` — 11 tests pass
- [x] `gui/wizard/tutorial_wizard.py` — 5 페이지 QWizard (round5)
- [x] 모델 비교 대시보드 GUI — `ModelCompareWidget` + MainWindow 탭 통합 (round5, round17)
- [x] `gui/widgets/loss_curve_widget.py` — 실시간 loss (round7)
- [x] `installer/naviertwin.iss` — Inno Setup 스크립트 (round8)
- [x] `core/digital_twin/pipeline.py` — end-to-end 오케스트레이터 (round5)
- [x] `api/server.py` — FastAPI REST 엔드포인트 (round4)
- [x] `core/explainability/shap_explainer.py` — KernelSHAP (v5.0.0)
- [x] `core/explainability/attention_viz.py` — MultiheadAttention 시각화 (round13)

---

## v4.x — 최첨단 모델

### v4.0.0 — 잠재 동역학 + 생성 모델 MVP ✅
- [x] `core/time_series/latent_dynamics/latent_dynamics.py` — AE + Neural ODE (RK4) 잠재 적분
- [x] `core/generative/diffusion_pde/diffusion_pde.py` — DDPM-style 유동장 생성
- [x] `tests/test_latent_generative.py` — 6 tests pass
- [ ] `core/state_space/mamba_neural_op/mno.py` (v4.0.x — mamba-ssm WSL 필요)
- [ ] `core/state_space/deepomamba/deepomamba.py` (v4.0.x)
- [ ] flowtorch 파이프라인 연동 (v4.0.x, GPL)

### v4.1.0 — 생성 모델 + KAN ✅
- [x] `core/generative/diffusion_pde/diffusion_pde.py` — DDPM-style (v4.0.0)
- [x] `core/generative/wavelet_diffusion/wavelet_diffusion_no.py` — DWT+Diffusion (round14)
- [x] `core/generative/conditional_gen/conditional_gen.py` — cVAE (round10)
- [x] `core/operator_learning/kan/kano.py` — KAN + spectral (round1)
- [x] `core/dimensionality_reduction/nonlinear/tucker_decomp.py` — HOSVD+HOOI (round1)

### v4.2.0 — Equivariant NN + 고급 분해 ✅ (부분 완료)
- [x] `core/equivariant/group_equiv_fno/group_equiv_fno.py` — C4 회전 평균 FNO2D
- [x] `core/dimensionality_reduction/linear/cpod.py` — null-space 투영 POD
- [x] `tests/test_equivariant_cpod.py` — 5 tests pass
- [x] `core/equivariant/physics_embedded/physics_embedded_gnn.py` — EGNN translation/rotation equivariant (round14)
- [x] `core/dimensionality_reduction/nonlinear/diffusion_maps.py` — Coifman-Lafon (round6)
- [x] `core/flow_analysis/modal/pgd.py` — greedy rank-1 (round4)
- [x] `core/flow_analysis/vortex/lcs.py` — FTLE via RK4 flow-map (round4)
- [x] `core/flow_analysis/thermofluids/entropy_gen.py` — Bejan (round4)
- [x] `core/equivariant/physics_embedded/lie_algebra_no.py` (v4.2.x)

---

## v5.x — 연구 플랫폼

### v5.0.0 — 고급 최적화 + 인증 ROM
- [x] `core/optimization/moo_optimizer.py` (pygmo2, GPL)
- [x] `core/optimization/topology_opt.py` (DL4TO, PyTopo3D)
- [ ] `core/optimization/mdo_pipeline.py` (OpenMDAO)
- [ ] `core/optimization/inverse_problem.py` (Firedrake+pyadjoint, LGPL)
- [ ] `core/optimization/su2_adjoint.py` (SU2, LGPL)
- [ ] `core/dimensionality_reduction/linear/certified_rb.py` (RBniCSx, LGPL)
- [ ] `core/dimensionality_reduction/linear/neural_rb.py` (dlrbnicsx, LGPL)

### v5.1.0 — 멀티피델리티 + Active/Online Learning
- [x] `core/multi_fidelity/transfer_learning.py`
- [x] `core/online_learning/active_learning.py`
- [x] `core/online_learning/online_learning.py`
- [x] `core/physics_correction/hybrid_rom_adv.py` 고도화
- [ ] `core/digital_twin/jax_fluids_wrapper.py` (JAX-Fluids, optional)

### v5.2.0 — 설명가능성 + FastAPI + 완전 배포
- [x] `core/explainability/shap_explainer.py` (SHAP)
- [x] `core/explainability/attention_viz.py` (captum)
- [ ] `core/explainability/kano_symbolic.py`
- [ ] `core/explainability/pysr_recovery.py` (PySR, optional)
- [x] 모델 비교 대시보드 완성 (`ModelCompareWidget`, MainWindow Compare 탭)
- [x] `api/server.py` — FastAPI REST 서버
- [ ] 자동 업데이트 (GitHub Releases 기반)
- [ ] Sphinx + MkDocs API 문서 자동 생성

---

## 완료된 항목
(완료 시 위에서 여기로 이동)

---

## 라운드 기반 고도화 (v4.2.0 이후)

### 연산자 학습 확장
- [x] `operator_learning/fno/tfno.py` — Tucker-factorized FNO (v2.0.1)
- [x] `operator_learning/fno/wno.py` — WNO (v2.0.1)
- [x] `operator_learning/fno/adaptive_fno.py` — rFFT 기반 modes 자동 선택 (round11)
- [x] `operator_learning/fno/spectral_refiner.py` — low→high 2단계 학습 (round11)
- [x] `operator_learning/fno/lno.py` — Laplace 복소 pole/residue (round15)
- [x] `operator_learning/deeponet/pi_deeponet.py` — 물리 잔차 (v2.0.1)
- [x] `operator_learning/deeponet/mionet.py` — 복수 branch (v2.0.1)
- [x] `operator_learning/deeponet/sequential_deeponet.py` — GRU branch (round11)
- [x] `operator_learning/latent_operator/l_deeponet.py` — 잠재 DeepONet (round12)
- [x] `operator_learning/latent_operator/pi_latent_no.py` — PI-Latent-NO (round12)
- [x] `operator_learning/koopman/ikno.py` — Real-NVP invertible (round10)
- [x] `gnn/graph_transformer/hamlet.py` — Dense self-attention + position emb (round15)

### 5.0/5.1 연구 플랫폼
- [x] `optimization/moo_optimizer.py` — NSGA-II (v5.0.0)
- [x] `optimization/topology_opt.py` — SIMP 2D (v5.0.0)
- [x] `optimization/uq_surrogate.py` — PCE + Sobol (round9)
- [x] `optimization/surrogate_opt.py` — RBF + L-BFGS-B (round9)
- [x] `multi_fidelity/transfer_learning.py` — freeze + finetune (v5.1.0)
- [x] `online_learning/active_learning.py` — variance-based selection (v5.1.0)
- [x] `data_assimilation/four_dvar.py` — 선형 해석해 (round6)
- [x] `sensitivity/causal_analysis.py` — Pearson + Granger (round6)
- [x] `explainability/symbolic_regression.py` — PySR + polynomial fallback (round13)
- [x] `explainability/attention_viz.py` — MultiheadAttention (round13)
- [x] `surrogate/ensemble.py` — Ensemble + MoE k-means gating (round13)

### 생성/잠재 모델
- [x] `generative/conditional_gen/conditional_gen.py` — cVAE (round10)
- [x] `generative/wavelet_diffusion/wavelet_diffusion_no.py` — DWT+DDPM (round14)

### 대칭성 보존
- [x] `equivariant/group_equiv_fno/group_equiv_fno.py` — C4 회전 평균 (v4.2.0)
- [x] `equivariant/physics_embedded/physics_embedded_gnn.py` — EGNN (round14)

### GUI & 배포
- [x] `gui/wizard/tutorial_wizard.py` — 5 단계 QWizard (round5)
- [x] `gui/widgets/model_compare_widget.py` — RMSE/R² 바 차트 (round5)
- [x] `gui/widgets/loss_curve_widget.py` — 학습 손실 실시간 (round7)
- [x] `gui/widgets/analytic_compare_widget.py` — 해석해 ↔ 수치 (v1.1.1)
- [x] MainWindow i18n + 7번째 Compare 탭 (round17)
- [x] `installer/naviertwin.iss` — Windows Inno Setup (round8)
- [x] `utils/i18n.py` + ko/en 번역 JSON (v3.2.0)
- [x] `utils/undo_redo.py` — Command 스택 (v3.2.0)

### API / 내보내기
- [x] `api/server.py` — FastAPI (/health, /reduce/pod, /analytic/*, /optimize/bayesian) (round4)
- [x] `core/export/onnx_export.py` + `torchscript_export.py` (v3.2.0)
- [x] `core/report/generator.py` — Jinja2 + weasyprint (v3.2.0)
- [x] `core/digital_twin/pipeline.py` — 6 단계 오케스트레이터 (round5)

### 실전 예제
- [x] `examples/cavity_benchmark.py` — POD/AE/FNO 재구성 비교 (round16)

---

## 총 진행 상황

- **307+ 테스트 통과 / 4 skipped** (optional: pywt / pymeshlab / dedalus / onnxscript)
- Ruff 린트 통과 전체 모듈
- v1.1.0 → v4.2.0 + 17 rounds 고도화 완료

---

## 최종 통계 (20+ rounds 완료)

### 양적 지표
- **334 passed / 4 skipped** (optional: pywt / pymeshlab / dedalus / onnxscript)
- **48 개 feature/docs commits** (v1.1.1 → rounds 1-23)
- **195 소스 파일** (src/naviertwin/)
- **41 테스트 파일** (tests/)

### 라운드별 산출물 (round 11-23 추가분)
- **Round 11** — SequentialDeepONet (GRU branch), AdaptiveFNO1D (energy-based mode selection), SpectralRefiner (low→high res 2단계)
- **Round 12** — L-DeepONet (AE latent + operator), PI-Latent-NO (물리 residual 추가)
- **Round 13** — SymbolicRegressor (PySR + poly fallback), EnsembleSurrogate, MixtureOfExperts (k-means gating), Attention viz
- **Round 14** — EGNN (translation + rotation equivariance 검증), WaveletDiffusionNO (DWT + DDPM)
- **Round 15** — HAMLET (dense self-attention + position), LNO1D (Laplace pole/residue)
- **Round 16** — Cavity benchmark 예제 (POD/AE/FNO 비교)
- **Round 17** — MainWindow i18n + 7번째 Compare 탭
- **Round 18** — 문서 전면 동기화, version 4.2.17
- **Round 19** — TNO (Temporal NO), FlowDMD (INN + DMD), KoopmanAnalysis (pykoopman + DMD)
- **Round 20** — SO2Canonicalizer (Lie equivariance), NFNODeepONet (비균일 격자), HybridROMAdv (제약 투영)
- **Round 21** — OnlineKriging/OnlineNN, DomainDecompPINN, PhysicsNEMOWrapper
- **Round 22** — LBMD2Q9 (D2Q9 LBGK 자체 구현), Lettuce/flowtorch/JAX-Fluids 래퍼
- **Round 23** — LBM → POD → Kriging 완전 파이프라인 데모 (R²=1.0 달성)

### 구현 완성도
거의 모든 SPEC.md §6 기법 (차원축소/모달/Surrogate/Operator Learning/GNN/SSM 제외/생성모델/시계열/Equivariant/PINN/방정식 발견/DA/UQ/최적화/설명가능성) 의 **MVP 또는 완성본** 이 구현되었으며, 실제 파이프라인 통합 데모로 유효성 확인.

### 남은 여정 (선택)
- mamba-ssm (MNO, DeepOMamba) — Windows 지원 불안정으로 후순위
- Certified RB (RBniCSx) — LGPL, Firedrake 의존성 무거움
- pyPDAF 대규모 DA — Fortran 컴파일 필요

---

## 🎯 Round 40 Milestone (v4.2.40)

### 최종 통계
- **383 passed / 4 skipped** tests
- **63 commits**, **215 소스 파일**, **53 테스트 파일**
- **4 실전 예제** (cavity / LBM / Burgers+FNO / Streaming Burgers)

### 추가 도메인 (rounds 33-39)
- **StreamingDigitalTwin + Burgers** (round 33) — 실시간 EnKF (4.9% → 3.0%)
- **CLI 서브커맨드** (round 34) — benchmark/server/pipeline
- **RL flow control** (round 35) — GaussianPolicy + REINFORCE
- **Turbulence** (round 36) — k-ε closure + E(k) 스펙트럼 + Kolmogorov 기울기
- **Helmholtz + 압축성** (round 37) — 주기 2D 분해 + Mach/isentropic
- **QMC samplers** (round 38) — Halton / LHS / Sobol + 스케일링
- **음향 모드** (round 39) — 1D duct + Strouhal + Womersley

### 사용 가능 도메인 total
1. CFD I/O (7 formats)
2. 메쉬 생성/후처리 (2)
3. 차원축소 (선형 5 + 비선형 6)
4. 모달/통계 (9)
5. 유동 분석 (10+)
6. 신경 연산자 (FNO 6 + DeepONet 5 + Latent 2 + KNO 3 + UNet 2 + KANO + HAMLET)
7. GNN (2)
8. 시계열 (5 + ESN + TNO)
9. Equivariant (3)
10. 생성 모델 (3)
11. PINN (3)
12. 물리 보정 (3)
13. DA (3)
14. UQ + 최적화 (7+)
15. 민감도 + 설명 (5)
16. Surrogate (4) + 멀티피델리티 (2) + Online (3)
17. Digital Twin (3 엔진 + Streaming)
18. **Turbulence** (k-ε + E(k))
19. **압축성** (Mach/isentropic)
20. **음향** (duct + Strouhal)
21. **RL** (policy gradient)
22. **QMC** (Halton/LHS/Sobol)
23. External 솔버 래퍼 (LBM + Lettuce + flowtorch + JAX-Fluids)
24. PDE 솔버 (Burgers + Heat)
25. Export (ntwin + ONNX + TorchScript)
26. Report (Jinja2 + weasyprint)
27. API (FastAPI 5 엔드포인트)
28. GUI (6 패널 + 6 위젯 + 위자드 + i18n + Compare 탭)

---

## 🎯 Round 47 Milestone (v4.2.47)

- **409 passed / 4 skipped** tests (46 rounds 완료)
- **70 commits, 223 src files, 59 test files, 4 examples**

### 라운드 41-46 추가 산출물
- **Round 41** — POD-Galerkin linear ROM with input matrix
- **Round 42** — Langevin score-based sampler + Euler-Maruyama SDE
- **Round 43** — Wasserstein 1D + MMD Gaussian + KL divergence
- **Round 44** — Chebyshev spectral (Gauss-Lobatto + Trefethen D 행렬) + Lagrange
- **Round 45** — Benchmark dataset catalog (Burgers/Heat/Cavity)
- **Round 46** — SPH M4 cubic kernel (1/2/3D) + density/gradient

### 추가 도메인
29. **POD-Galerkin reduced dynamical system** (with input channel)
30. **Langevin sampling + SDE integration**
31. **Statistical distances** (W₁, MMD, KL)
32. **Spectral methods** (Chebyshev + Lagrange)
33. **Benchmark registries** (파라미터 가변 PDE 데이터셋)
34. **SPH** (kernel, density, gradient)

---

## 🏆 Round 50 Milestone (v4.2.50)

- **416 passed / 4 skipped** tests
- **74 commits / 225 src files / 61 test files / 4 examples**
- **50 rounds 완료** (초기 목표 11 버전 + 추가 39 rounds 고도화)

### 라운드 48-49 추가
- **Round 48** — Deep Ritz solver (변분 에너지 최소화 PINN)
- **Round 49** — FVM upwind + MUSCL-Hancock + minmod limiter + 질량 보존

### 최종 도메인 총합
35+ 독립 기술 도메인 + 4 엔드-투-엔드 예제 + GUI + REST API + CLI.
초기 v1.1.0 단일 리더에서 시작해 **연구 플랫폼 + 실전 배포** 수준 도달.

이 milestone 이후 기능 추가는 희귀 영역 (Mamba SSM / Firedrake 기반 Certified RB /
Fortran pyPDAF) 이며, 일반 사용자에게는 현재 범위가 충분합니다.

---

## Canonical Twin Architecture Foundation (2026-07-18)

사용자 CFD 결과를 형상·조건·시간·격자 축으로 안전하게 조합하고 실제 학습까지
연결하기 위한 기반 재구성이다. 기존 기능 개수와 별도로 추적한다.

- [x] M0 — Case/CaseSet/Snapshot/Field/Geometry/Mesh/BC canonical schema + legacy adapter
- [x] M1 — JSON source manifest, VTK/CGNS 등 ReaderFactory ingest, canonical cache
- [x] M2 — 해상도 유지율 입력, 메모리 추정, 공통 격자 validity mask 보존
- [x] M3 — 7개 전략 capability/plugin registry, optional entry-point 확장
- [x] M4 — 비동기 학습 job, 진행률/취소/checkpoint, CPU/CUDA preflight
- [x] M5 — mesh-aware 실제/예측 비교, 외삽 무오차 원칙, 벡터/보존량/시간 지표
- [x] M6 — 36개 필수 데이터 조합 coverage matrix + CI 검사 스크립트
- [x] M7 — `TwinWorkspace` 단일 런타임 저장소 + 웹/Qt canonical project 연결
- [x] M8a — 웹 학습 job 진행률/취소 요청/CUDA preflight/checkpoint 복원
- [x] M8b — Electron 서버 lifecycle 분리 + scoped 종료/헬스/포트 테스트
      (학습 진행률·취소·checkpoint UI는 M8a의 trame 경로 사용, Qt는 레거시)
- [x] M9 — mapping/model/prediction/validation lineage 영구 저장
      (`.ntwin` 내부 manifest + JSON sidecar 양쪽 round-trip)
- [x] M10 — 실제 CFD 포맷 + 36-scenario end-to-end benchmark
      (`minimal.vtk`, 실제 HDF5 CGNS, OpenFOAM polyMesh/field fixture를 reader→
      canonical project→`.ntwin` 내장 manifest로 왕복; 3/3 통과, 최대 오차
      2.65e-8. 형상/조건/시간/격자/1D·2D·3D 36조합은 실제 PyVista 데이터
      생성→workspace 변환→값·스냅샷·메모리 측정 36/36, 오차 0)
- [x] M11 — 1D/2D/3D GeometryFNO + 비정상 GeometryFNO/GINO/MeshGNN/MGN
      (`t` 입력 확장, source-case 그룹 분할로 temporal leakage 방지)
- [x] M12 — MPI map + PyTorch DDP launcher + 다중 GPU best-fit 자원 스케줄러
      (`naviertwin launch-ddp`, torchrun dry-run/명령 검증)

### 진행률

- **아키텍처 기반(M0-M6): 100%**
- **확장 제품 기반 전체: 100%** — M0-M15 구현·자동 검증 완료.

### 재현 명령

`python3 scripts/check_twin_matrix.py --execute --output benchmark.json`

일반 CI는 동등성·계약 실패만 차단한다. 시간·메모리는 환경 의존 측정값으로
기록하며 hard fail 기준으로 사용하지 않는다. 이후 대형 연구 범위는 문서 상단
"다음 단계 — 스코프 확정 필요"에서 별도 관리한다.

---

## 🔧 Round 51-58: 라이브러리 활용도 개선 집중

**전략 변경**: 자체 구현 일변도 → 검증된 라이브러리 래핑.

| Round | 라이브러리 | 이전 활용도 | 개선 후 |
|------|------|------|------|
| 51 | botorch + gpytorch | 0% | qEI/UCB 배치 BO ✅ |
| 52 | SALib | 20% (Sobol) | 100% (Morris/FAST/PAWN/Delta/Sobol) |
| 53 | nlopt | 0% | 14 알고리즘 |
| 54 | pydmd | 20% (DMD/FbDMD) | 60% (+HODMD/MrDMD/OptDMD/HAVOK/DMDc) |
| 55 | smt | 30% (RBF/Kriging) | 80% (+KPLS/GEKPLS/IDW/QP + LHS/FullFact) |
| 56 | pymor | 0% | POD/DEIM/GramSchmidt |
| 57 | foamlib | 0% | case 파라미터 스윕 + dict 조작 |
| 58 | pymeshlab | 10% (simplify/smooth) | 60% (+Taubin/remesh/curvature/hole) |

**458 passed / 83 commits / v4.2.58**

### 라이브러리 활용도 점검 답변
> **"모든 라이브러리 제대로 잘 이용하고 있는거 맞지?"**

- Round 50 시점: **부분적** (설치된 라이브러리 중 절반만 활용, 자체 구현 과다)
- Round 58 시점: **핵심 라이브러리 8종 전체 활용** (botorch/SALib/nlopt/pydmd/smt/pymor/foamlib/pymeshlab)
- 남은 것: `physicsnemo` (가용성 체크만), `pyCGNS` (h5py 폴백에 의존)
- 미설치: pyspod/pysindy/modred/flowtorch/pysr/shap/torchdiffeq/e3nn/escnn/pygmo/dedalus/jaxfluids/lettuce — 자체 구현 완비로 보완

---

## 🏆 Round 630 Milestone — Commercial Post-Processor Parity

**프로젝트 정체성 확립**: NavierTwin은 CFD **솔버**가 아닌, CFD 결과를 받아
AI/ROM/차원축소로 **의미 있는 데이터를 추출**하는 후처리 도구이다.

### R591–600 (커버리지 강화)
| Round | 영역 | Δ |
|-------|------|---|
| 591–594 | safe_yaml, main CLI, dim_reduction lazy, OpenFOAMReader | 33–47% → 75%+ |
| 595–596 | WNO1D 에러 경로, WaveletDiffusionNO 패킹 | |
| 597–599 | onnx_wrap, device utils, CGNSReader 폴백 체인 | |
| 600 | **커버리지 84% → 85%, 약점 모듈 27 → 16** | gate ratchet |

### R601–605 (신규 ROM 인프라)
- **R601** sparse_sensor — QR-pivot 최적 센서 배치 + 그리디 폴백 + reconstruct
- **R602** SensorDMDPipeline — POD 기저 + 센서 + 재구성 워크플로
- **R603** spectral_energy — 1D/2D 에너지 스펙트럼, Kolmogorov -5/3 적합, 적분 길이
- **R604** MRPOD — 다중 해상도 POD (Gaussian-pyramid + per-scale SVD)
- **R605** IncrementalPOD — Brand 2006 rank-1 SVD 갱신, 망각 인수 지원

### R606–629 (상용 툴 동급 후처리 모듈)
| Round | 모듈 | 상용 툴 대응 |
|-------|------|--------------|
| 606 | reynolds_stats | Tecplot 360 Time-Average / CFD-Post Statistics |
| 607 | psd | MATLAB Signal Processing Toolbox |
| 608 | surface_integrals | Tecplot 360 / CFD-Post Force/Moment |
| 609 | quadrant_pdf | Quadrant analysis (Wallace 1972) + KDE |
| 610 | two_point | Tecplot Two-Point Stats + Taylor microscale |
| 611 | stat_convergence | Fluent Sample Convergence + Geweke |
| 612 | plane_flux | CFD-Post Surface Integral / EnSight Flux |
| 613 | time_interp | Tecplot Time-Aware Sliding |
| 614 | coord_transform | EnSight Cylindrical/Spherical |
| 615 | slice_extract | Tecplot Slice + CFD-Post Line |
| 616 | expression_eval | CFD-Post Custom Expressions (AST sandbox) |
| 617 | phase_lock | Fluent Phase-Locked Sampling |
| 618 | running_moments | EnSight Variable Statistics (Welford+Pébay) |
| 619 | denoise | MATLAB sgolayfilt/hampel + wavelet shrinkage |
| 620 | quantile_stats | Tecplot box-and-whisker + ECDF |
| 621 | eof_analysis | NCL/CDO eofunc + Lumley + North test + Varimax |
| 622 | goodness_of_fit | MATLAB kstest/adtest/chi2gof + Shapiro-Wilk |
| 623 | conditional_sampling | Trigger averaging + 사분면 마스크 |
| 624 | grid_derivatives | Tecplot Calculate Variables (2차/4차 정확) |
| 625 | critical_points | Vector Field Topology (Helman 1991) |
| 626 | anisotropy | Pope §11.5 Lumley triangle + barycentric |
| 627 | morphology | SciPy ndimage 등가 (4-/8-conn) |
| 628 | cell_volume | Tecplot Compute Cell Volume + Volume Integral |
| 629 | truncation_criteria | Eckart-Young + scree + AIC/BIC |

### 상태 (R630 시점)
- **2535 테스트 수집** / **677 commits** / **39 라운드 후처리 패리티 추가**
- ruff 0 errors / coverage 85%+ gate
- 후처리 능력: **상용 툴 (Tecplot 360 / CFD-Post / EnSight) 핵심 기능 동등 수준 도달**
- 차세대 차별화 포인트: AI/ROM 통합 (이미 보유) + 검증 인프라 (R561–590)
