# 2026-07-18 세션 기록 — v5.0~v5.6 CFD 트윈 플랫폼 전환

> 자율 실행(멈추지 않고 로드맵 완료까지) 세션. 총 **58개 커밋**(머지 포함),
> 최종 테스트: **3920 passed, 17 skipped, 11 deselected, 0 failed**.
> 시작 커밋 `79cb533` 직전 → 종료 커밋 `6469f62`.

## 1. 세션 범위

시작 시점에 이미 (이전 세션에서) 완료돼 있던 것:
- Kármán 볼텍스 데모 데이터(실제 LBM 해석, 정상/비정상, 진짜 구멍 — SDF 아님)
- 웹 GUI 패널 정리, 툴팁 시스템, 해상도 다운샘플 UI
- v5 플랫폼 전환 로드맵 문서(`.omc/plans/twin-platform-roadmap.md`) 작성

이 세션에서 다룬 것: 위 로드맵의 실제 구현 — 능력 기반 전략 레지스트리부터
시작해 Route 1(재샘플) 전략 고도화, Route 2(메쉬 네이티브, 재샘플 없음)
전략 3종 신규 배선, 4계층 저장 아키텍처, 그리고 세션 무관 기존 버그 정리까지.

## 2. 성공 — 병합·검증·푸시된 기능 (연대순, 24개 논리적 작업)

각 항목은 별도 git worktree 에서 서브에이전트가 구현 → 메인 세션이 머지 →
스모크 테스트 → 그린이면 푸시, 순서로 처리했다.

| # | 기능 | 핵심 내용 | 대표 커밋 |
|---|------|----------|----------|
| 1 | remap 오차 바닥 | `estimate_remap_floor()` — 공통 격자 왕복 오차를 모델 오차와 분리. **버그 발견+수정**: `vtkValidPointMask` 가 무효점을 NaN 아닌 0-채움으로 표시하는 걸 놓쳐 오차가 res→∞ 에도 0.31 에 고정되던 버그, 필터링 추가로 해결 | `79cb533` |
| 2 | 그룹 스플릿 학습 경로 배선 | `build_geometry_fno_twin(group_split=True)` — held-out 케이스 평가 + 4-way 일반화 라벨(`classify_query_split`). 기본값은 이전과 bit-동일 | `eb99089` |
| 3 | Region growing 벽 선택 | `grow_wall_selection()` — edge-connected BFS로 시드 확장 | `76f59a6` |
| 4 | 모델 등급제 + 차원 분리 | `StrategySpec.tier`(production/domain/experimental), `DataProfile.topological_dim/embedding_dim` | `ed01258` |
| 5 | MPI 배치 CLI | `naviertwin batch-train`, `mpirun -n 2` 실증, mpi4py 없으면 순차 폴백 | `b8b7fb9` |
| 6 | DatasetSignature | topology/coordinate sha256 해시, `assign_geometry_ids()` | `3ccceb7` |
| 7 | 보존량 재샘플 경고 | `flag_conserved_fields()` — rho/mass/flux 등 점보간 부적절 경고 | `4495550` |
| 8 | GeometryFNO held-out 검증 토글 | 시그니처 기반 geometry_id 자동 부여(공통 격자 재샘플 후 뭉치면 케이스=그룹 폴백) | `cd7a2a6` |
| 9 | CGNS ZoneBC 자동 wall 인식 | pyCGNS/h5py 폴백 리더에 **Elements_t 셀 연결성 파싱을 처음으로 추가**(이전엔 점 구름뿐이었음), BC_t → boundary_patches | `4e145a2` |
| 10 | Route 2 스코핑 연구 | torch_geometric 2.7.0/GINO 실증, physicsnemo 는 torch 2.11 비호환 확인 | `fb15700` |
| 11 | **Route 2 첫 배선: mesh_gnn** | GCN 기반, 재샘플 없음, `varying_mesh` 덕타이핑으로 원본 메쉬(진짜 구멍) 그대로 표시 | `1492159` |
| 12 | Zarr 텐서 캐시 | `cases_to_grid_tensors` 결과를 시그니처 키로 캐시 | `abc649e` |
| 13 | **Route 2 두 번째 배선: gino** | neuraloperator GINO, 점군 기반이라 고정 그래프 불필요 | `98891b4` |
| 14 | MLflow 실험 추적 | 학습마다 strategy/params/metrics 기록, 미설치 시 조용한 no-op | `e371b3b` |
| 15 | GUI 테스트 스위트 행(hang) 수정 | **근본원인**: `QApplication.setStyleSheet()` 가 매 `MainWindow` 생성마다 프로세스 내 모든 살아있는 위젯을 재폴리시 → 누적 O(n) 비용. 36개 테스트 무한대기 → 43초 완주 | `958de7e` |
| 16 | DatasetSignature → strategies 통합 | `_same_mesh_points()` 를 해시 기반으로(불일치 시 기존 allclose 폴백, 판정 100% 하위호환) | `e149489` |
| 17 | MeshGraphNets predict() 버그 수정 | override 그래프의 edge_features 를 재계산 안 하고 stale 값 재사용하던 결함 | `0ecffd9` |
| 18 | 보존량 volume-weighted 재샘플 | 점보간 오차 100% → 부피가중 오차 0.002%(체커보드 밀도장 실측) | `f36c7d8` |
| 19 | zarr_reader.py zarr 3.x 호환 | `create_dataset` → `create_array` 폴백(오늘 세션이 zarr 3.x 를 설치해서 생긴 회귀를 오늘 세션이 직접 수정) | `59766d6` |
| 20 | 기존 테스트 실패 10건 정리 | 검증 리포트 계약 누락 키, postproc 스모크의 **연산 순서 의존성**(`load_rom` 이 `save_rom` 산출물에 암묵 의존) 등 | `6a86d3e` |
| 21 | canonical VTKHDF 캐시 | 기존 `.ntwin` 포맷 재사용(새로 안 만듦), mtime+size 캐시 키 | `119cf8f` |
| 22 | 분할 뷰어 시간 동기 | 좌측 timestep 변경 시 우측 트윈도 자동 재예측(정상 트윈은 no-op). 슬라이스 동기는 기능 자체가 없어 보류 | `a1b0d57` |
| 23 | **Route 2 세 번째 배선: mesh_gnn_mp** | MeshGraphNets 를 케이스 세트 파라메트릭으로(`fit()` 안 쓰고 `_model` 직접 구동, 한 스텝 델타 예측으로 정상장 프레이밍) | `4826e02` |
| 24 | UTF-8 로케일 하드닝 | 전체 스위트(3920개) 안에서만 재현되던 마지막 2개 실패 — subprocess/read_text 가 호스트 로케일에 의존하던 걸 명시적 UTF-8 강제로 제거 | `6469f62` |

## 3. 실패·문제 → 해결 과정 (정직하게 기록)

세션 중 실제로 잘못됐거나 막혔던 지점들:

1. **`estimate_remap_floor` 초기 구현 버그**: `vtkValidPointMask` 가 경계
   무효점을 NaN 이 아닌 0-채움으로 표시한다는 걸 놓쳐, 해상도를 아무리
   올려도 오차가 ~0.31 에서 안 떨어지는 버그. 직접 프로빙으로 발견, 필터
   추가로 해결, 회귀 테스트(`test_remap_floor_shrinks_toward_zero_with_resolution`)
   로 고정.

2. **`service.py` 파일 끝 append 지점에서 병렬 에이전트 충돌 2회**:
   MLflow 트래킹 에이전트와 GINO 에이전트가 동시에 파일 끝에 다른 섹션을
   추가 → 머지 충돌. 둘 다 순수 추가라 두 블록 다 유지하는 것으로 해결
   (내용 손실 없음).

3. **GUI 테스트 파일 하나가 무한정 멈춤**: `test_main_window_customer_tools_gui.py`
   가 헤드리스 실행에서 60초 넘게 안 끝남 → 전체 스위트가 이 지점에서 막힘.
   사용자가 별도 세션(칩)으로 동시에 같은 문제를 조사 중이었던 걸 파악하고
   중복 방지를 시도했으나 완전히 막지는 못함 — 두 세션 다 같은 파일을
   고쳤을 가능성. `git log` 로 사전 확인 후 충돌 없이 머지 완료.

4. **Zarr 3.x 업그레이드가 무관한 기존 코드를 깨뜨림**: 텐서 캐시 기능
   때문에 설치한 zarr 3.x 가, 세션과 전혀 무관한 예전 `zarr_reader.py`
   (zarr 2.x `create_dataset` API 사용)를 회귀시킴. 전체 스위트를 실제로
   끝까지 돌려보고서야 발견 — 부분 테스트만 돌렸으면 놓쳤을 것.

5. **로케일 의존 테스트 실패 2건이 격리 실행에서는 재현이 안 됨**:
   `test_verification_emit.py`/`test_readme_quickstart_smoke.py` 가 전체
   3920개 스위트를 통으로 돌릴 때만 실패(`UnicodeDecodeError: 'ascii' codec`),
   단독/부분 실행으로는 여러 방법(강제 LC_ALL=C 포함)으로도 재현 안 됨 —
   진짜 원인(어떤 앞선 테스트가 무엇을 오염시키는지)은 끝내 특정 못 함.
   대신 **원인 불문 항상 안전한 수정**(subprocess/read_text 에 `encoding="utf-8"`
   명시)으로 접근 — 전체 스위트 재실행으로 실제 해결 확인.

6. **서브에이전트의 "이미 통과함" 보고를 그대로 믿지 않음**: 기존 테스트
   정리 에이전트가 "test_verification_emit 은 이미 통과한다"고 보고했지만,
   전체 스위트 재실행에서 여전히 실패 — 부분 검증의 한계를 보여준 사례.
   최종적으로 항상 **전체 스위트 재실행**으로 확인하는 습관이 이 세션에서
   중요했다.

7. **worktree 에 네이티브 커널 `.so` 부재**: 거의 모든 에이전트가 worktree
   에서 `_kernels.cpython-312-x86_64-linux-gnu.so`(gitignore 대상 빌드
   산출물)가 없어 초기 테스트 실패를 겪음 — 매번 메인 체크아웃에서 복사해
   해결. 반복적으로 발생한 환경 문제(코드 버그 아님).

8. **worktree 의 gitdir 이 Windows UNC 경로라 WSL git 이 인식 못 함**:
   거의 모든 에이전트가 Windows git 으로 우회 커밋. 근본 해결은 안 했음
   (세션 범위 밖 — WSL/Windows 경로 브릿징 인프라 이슈).

## 4. 남은 작업

### 진짜 남은 것 (스코프 미확정, 제품 방향 결정 필요)
- **대형 3D/HPC 실측 스케일링** — 지금까지 검증은 소~중규모(수백~수천 점)
  데모뿐. 산업 규모(수백만 셀) 에서 Route 1/2 전략이 버티는지, MPI 배치
  CLI 를 실제 클러스터 잡과 어떻게 엮을지는 실사용 데이터 필요.
- **운영형 디지털 트윈(단계 6)** — 센서 동기화·자료동화·온라인 보정.
  현재 플랫폼은 "대리모델/연산자 학습·검증 플랫폼"이고 이건 완전히 다른
  제품 층(실시간 파이프라인, 신뢰성 요구사항) — 별도 기획 필요.
- **모델 카탈로그 확장**(UPT/DoMINO/AB-UPT/Transolver++) — Production Core
  는 이미 확보(ROM/DMD/GeometryFNO + mesh_gnn/GINO/MGN 이 Route 1/2 커버).

### 알려진 미해결 이슈 (작지만 남아있음)
- **분할 뷰어 슬라이스 동기 없음** — 웹 GUI 에 슬라이스(clip-plane) 기능
  자체가 아직 없어서, 시간 동기만 구현하고 슬라이스는 보류.
- **CGNS 리더의 MIXED/NGON 엘리먼트, 멀티존 미지원** — 만나면 경고 후
  점 구름 폴백(정확도 저하, 크래시는 아님).
- **worktree UNC gitdir 문제** — 매 에이전트가 겪는 반복적 마찰, 인프라
  레벨에서 고칠 여지 있음(세션 범위 밖으로 남겨둠).

### 세션 범위 밖 — 원래부터 있던 훨씬 오래된 백로그
ROADMAP.md 의 v2.0.x~v5.2.0 구간에 GPL/LGPL 라이선스 캐비어트가 붙은 채
`[ ]`로 남아있는 것들(mamba-ssm 상태공간모델, pyPDAF 자료동화, SU2 adjoint,
Firedrake 역문제, RBniCSx/dlrbnicsx 등) — v5 플랫폼 전환과 무관, 애초에
optional 로 표시돼 있던 것들이라 이번 세션에서 손대지 않았다.

## 5. 최종 검증

```
3920 passed, 17 skipped, 11 deselected, 0 failed
(전체 스위트, WSL, QT_QPA_PLATFORM=offscreen, PYVISTA_OFF_SCREEN=true)
```

`main` 브랜치 최신 커밋: `6469f62`. 작업 트리 clean, 남은 worktree 없음.
