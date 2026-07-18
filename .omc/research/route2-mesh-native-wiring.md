# Route 2(메쉬 네이티브 전략) 배선 스코핑 — 조사 보고서

작성일: 2026-07-18. 근거 문서: `.omc/plans/twin-platform-roadmap.md` §5(두 루트),
§5.3(루트 2 카탈로그), §6½ #8(모델 등급제 — Production Core 에 GINO/Transolver/MGN 포함).
조사 범위: 저장소 자산 실사 + WSL 설치 OSS 실사(import·forward 스모크 실측) +
최소 배선 경로 설계 + 리스크. **코드 수정 없음 — 이 문서가 유일한 산출물.**

요약(한 줄): 루트 2 의 가장 싼 첫 타깃은 **기존 GCN surrogate(`GNNSurrogate`)를
케이스-세트 파라메트릭으로 확장한 `mesh_gnn` 전략**이다 — 의존성 추가 0(torch_geometric
2.7.0 설치·동작 실증), 기존 `varying_mesh` 예측 경로(`predict_to_mesh`)에 그대로 꽂혀
**원본 케이스 메쉬 위 표시(재샘플 불필요)** 가 앱 수정 최소로 성립한다. GINO 는
neuralop 2.0.0 에서 forward 동작을 실증했으므로 2번째 타깃(3D·점군)으로 대기.
physicsnemo 2.0.0 은 torch 2.11 과 **호환 깨짐**(import 실패) — Transolver/MGN 을
physicsnemo 경유로 쓰는 길은 현재 막혀 있다.

---

## 1. 저장소 내 기존 자산 실사

### 1.1 `core/gnn/` — 그래프 모델 3종

| 모듈 | 인터페이스(실측) | 용도 | 케이스 세트(형상 가변, μ) 학습에 그대로 쓸 수 있나 |
|---|---|---|---|
| `gnn_surrogate/gnn_surrogate.py` `GNNSurrogate` | `fit({"node_features": (B,N,f_in), "outputs": (B,N,f_out), "edge_index": (2,E)})` / `predict({"x": (N,f_in) or (B,N,f_in), "edge_index"?})` | 노드 특징 → 노드 필드 **회귀** (GCNConv ×3) | **거의**. 회귀형이라 방향은 맞다. 두 가지 부족: ① fit 이 **전 샘플 공유 edge_index 1개**를 전제(형상 가변이면 케이스마다 N·E 가 달라 (B,N,F) 직사각 배열 자체가 불가) ② 노드 특징에 μ/좌표를 넣는 빌더가 없음(호출자 책임). 단, `predict` 는 `edge_index` 를 덮어쓸 수 있고 GCN 가중치는 그래프 크기 무관 → **가중치 구조는 형상 가변 대응 완료**, 입력 배관만 새로 필요. |
| `meshgraphnets/meshgraphnets.py` `MeshGraphNets` | `fit({"trajectories": (n_traj,T+1,N,f), "edge_index": (2,E), "edge_features"?: (E,e)})` / `predict({"x": (N,f), "n_steps": int, "edge_index"?})` | **시계열 롤아웃 전용** — delta 예측 `u_{t+1}=u_t+MGN(u_t)`. Encode-Process-Decode + edge/node MLP 메시지패싱(진짜 MGN 구조) | **아니오(그대로는)**. ① 파라미터 μ 입력 개념이 없음 — "파라미터→필드 회귀"가 아니라 순수 시간발전기 ② 전 trajectory 공유 edge_index(동일 메쉬 전제) ③ predict 에서 edge 를 바꿔도 `_edge_features` 는 fit 때 것을 재사용(형상 바꾸면 조용히 틀림 — 잠복 버그). 비정상 메쉬-네이티브 롤아웃(로드맵 v5.2b 두 번째 칸)의 **뼈대**로는 재활용 가치 있음: 노드 특징에 μ 브로드캐스트를 붙이고 케이스별 그래프 루프로 바꾸면 됨. |
| `graph_transformer/hamlet.py`, `egno/` | (미배선 실험 코드) | graph transformer | 이번 범위에서 제외 — 등급제상 experimental. |

공통 베이스: `core/gnn/base.py` `BaseGNN` (fit/predict 추상, PyG 의존 없는 `Any` 선언).
`core/digital_twin/pipeline_neural.py:204` `GNNPipeline` 이 GNNSurrogate 를 이미 감싸고
있으나 트윈 계약(`predict(params)`)과는 무관한 저수준 래퍼다.

### 1.2 `core/operator_learning/` — 관련 자산

- `fno/case_tensorizer.py` `cases_to_grid_tensors(cases, params, field_names, resolution, param_names)`
  → `{"inputs": (N,H,W,2+k)=[sdf,mask,μ...], "targets": (N,H,W,C), "valid_mask", "grid", "meta"}`.
  **루트 1 전용**(균일 격자 재샘플)이지만, ① 벡터 필드의 성분 전개 규칙(`target_names`,
  `U`→`U_x/U_y`) ② μ 브로드캐스트 채널 규칙이 이미 확립돼 있어 그래프 빌더가 **규약을
  그대로 복제**하면 된다(코드 재사용은 아님 — 출력이 텐서 vs 그래프).
- `deeponet/`, `kan/`, `koopman/` 등은 루트 2 와 직접 무관.

### 1.3 트윈 계약과 앱 배선 — 새 전략이 꽂힐 소켓 (핵심 발견)

- 계약: 엔진은 `predict(params: (k,)|(N,k)) -> field-major 평탄 벡터` + `training_metadata`
  dict + `save/load` + (다중 출력이면) `model.output_fields` (`split_multi_prediction`
  이 자름, `service.py:2253`).
- **`web/app.py:1872` — `training_metadata["varying_mesh"]` 분기가 이미 있다**:
  `varying_mesh=True` 면 앱이 `service.predict_to_mesh(engine, values, self.dataset)`
  (`service.py:2200`) 를 불러 **지금 보고 있는 원본 케이스 메쉬 좌표 위에 예측을 그린다
  — 재샘플 없음.** `predict_to_mesh` 의 요구는 단 하나: `engine.model.predict_at(coords,
  params)`. 즉 메쉬 네이티브 GNN 엔진이 이 덕타이핑을 만족하면 **app.py 예측/뷰어
  경로는 수정 0** 이다. (Physics AI 가 이 소켓의 선례 — `physics_ai_engine.py` +
  `cfd_field_model.predict_at`, `cfd_field_model.py:425`.)
- 전략 노출: `core/digital_twin/strategies.py` `STRATEGIES` 에 `StrategySpec` 1개 추가
  → ②Model 카드에 가능/불가+이유+tier 뱃지가 자동으로 뜬다.
- 학습 빌더 선례: `service.build_physics_ai_twin_from_cases`(`service.py:816`) —
  케이스 목록+params 를 받아 `expand_case_params_over_time`(μ,t 전개, `service.py:690`)
  후 모델 fit, 메타데이터 세팅. 그래프판 빌더는 이 함수를 거의 그대로 본뜬다.
- 전처리 재사용: `core/geometry/mesh_features.py` — `attach_wall_features(dataset,
  wall_surface)` 가 `wall_distance`(항상)/`wall_sdf`(폐곡면일 때) point field 를 붙이고
  `field_names` 에 등록한다(`mesh_features.py:182`). 웹 픽킹 경로
  (`service.attach_wall_distance_from_picks`, `service.py:3045`)도 이미 이걸 쓴다 →
  **그래프 노드 피처로 그대로 소비 가능.**
- 데모 데이터: `make_demo_case_set("karman_shapes")`(`service.py:1308`,
  `web/demo_karman.py`) — 세모/네모/원기둥 **진짜 구멍** 메쉬(케이스마다 N 다름),
  현재 Physics AI 전용. **루트 2 의 존재 이유를 시연하는 준비된 테스트 데이터.**
  기본 크기 nx=400, ny=160 → 케이스당 ~6 만 노드(구멍 제외). `"shapes"` 데모는
  재샘플 후라 루트 2 검증용으로 부적합(메쉬가 전부 동일해짐).

### 1.4 판정 요약

| 질문 | 답 |
|---|---|
| 기존 MGN MVP 는 롤아웃용인가 회귀용인가 | **시계열 롤아웃 전용** — μ 입력 없음. 파라미터→필드 회귀에는 GNNSurrogate 가 맞는 뼈대 |
| 케이스 세트에 그대로 쓸 수 있는 그래프 모델이 있나 | 없음 — 전부 "전 샘플 공유 edge_index" 전제. 케이스별 그래프(list) + μ 노드 피처 배관이 공통 결핍 |
| 메쉬→그래프 변환기가 있나 | 없음(신규 필요). 단 `mesh.extract_all_edges(use_all_points=True)` 가 점 순서를 보존하며 에지를 준다(아래 실측) |
| 예측을 원본 메쉬에 그리는 경로가 있나 | **있음** — `varying_mesh` + `predict_to_mesh` 소켓. 앱 수정 불필요 |

---

## 2. 설치된 OSS 실사 (WSL, 2026-07-18 실측)

| 패키지 | 버전 | 상태 | 비고 |
|---|---|---|---|
| torch | 2.11.0+cu130 | ✅ CUDA 동작(RTX 5070) | |
| **torch_geometric** | **2.7.0** | ✅ **동작 실증** | `GCNConv` forward + `Batch.from_data_list` 로 **크기 다른 그래프 2개 배칭** 확인. `torch_scatter` ✅ / `torch_cluster` ❌(→ PyG `knn_graph` 불가, kNN 은 scipy cKDTree 로 대체) |
| **neuraloperator** | **2.0.0** | ✅ **GINO forward 실증** | `from neuralop.models import GINO` OK. 2D 점군 200pt + latent 16² 격자 forward 0.1 s. 생성자 시그니처가 문서와 다름 — radius 는 `in_gno_radius`/`out_gno_radius`(`gno_radius` 아님). open3d ✅ 설치(이웃탐색 가속, 단 3D 전용 — 2D 는 native 폴백 경고 후 동작) |
| physicsnemo | 2.0.0 | ❌ **깨짐** | `import physicsnemo.models` → `No module named 'torch.distributed.tensor._ops.registration'` — torch 2.11 과 비호환. **Transolver/MGN 사전구현을 physicsnemo 로 쓰는 길은 현재 불가**(torch 다운그레이드는 cu130/5070 지원 때문에 비현실적; physicsnemo 신버전 대기 or 자체 구현) |
| dgl | — | ❌ 미설치 | 필요 없음(PyG 로 충분). pip 설치도 무거움(자체 libdgl 바이너리) — **도입 비추천** |
| deepxde | — | ❌ 미설치 | 루트 2 무관 |
| ezyrb / pydmd / pyvista 0.47.2 | ✅ | | 루트 1 자산(참고) |

사전구현 제공 현황:
- **GINO**: neuralop 2.0.0 에 있음(위 실증). 추가 설치 0.
- **Transolver**: physicsnemo(깨짐) 또는 thuml 연구 리포(클론, pip 아님). 코어 로직은
  단일 파일 수준(Physics-Attention)이라 **자체 이식(vendor)** 이 현실적이나 지금은 후순위.
- **MeshGraphNet**: physicsnemo(깨짐) / PyG 예제. **자체 MVP 가 이미 있으므로 외부 불필요.**
- **CORAL/AROMA**: 클론 전용, 성숙도 낮음 — 등급제상 experimental, 이번 범위 제외.
- pip 설치 가능성 평가(설치 안 함): `torch_cluster` 는 torch 2.11+cu130 휠이 아직 없을
  가능성이 높음(소스 빌드 리스크) — **불필요하게 만들었으니 도입하지 않는다**(kNN 은
  scipy, 메쉬 에지는 VTK 추출).

추가 실측: `pv.ImageData(20,20,1).cast_to_unstructured_grid().extract_all_edges(
use_all_points=True)` → 400 점/760 에지, **점 개수·순서 보존 확인**(`edges.n_points ==
mesh.n_points`), 0.002 s. 이것이 메쉬→edge_index 변환의 근거다.

---

## 3. 최소 배선 경로 설계

### 3.1 데이터 흐름 (케이스 세트 → 그래프 → 트윈 → 원본 메쉬 표시)

```
load_case_set / make_demo_case_set("karman_shapes")
  → datasets: list[CFDDataset] (케이스마다 다른 메쉬, 진짜 구멍 허용), params (N,k)
  → [신규] case_to_graph(dataset_i, μ_i):
       edge_index = extract_all_edges(use_all_points=True)  # 점 순서 보존, 양방향화
       x = [x̂,ŷ,(ẑ)  |  wall_distance/wall_sdf(있으면)  |  μ 브로드캐스트]   # 전 항목 train-셋 정규화
       edge_attr = [Δx,Δy,(Δz),‖Δ‖]                        # MGN 표준 상대좌표 에지 피처
       y = [필드 성분 채널 — case_tensorizer 의 target_names 규약 복제]
  → [신규] CaseSetGNN.fit(list[graph])      # 케이스별 그래프 루프(or PyG Batch)
  → [신규] MeshGNNTwinEngine                # 트윈 계약 어댑터
       .predict(params)                      # 대표(0번) 케이스 그래프 위 field-major 벡터
       .model.predict_at(coords, params)     # ← predict_to_mesh 소켓: coords 를 저장된
                                             #    케이스와 대조(N·해시 일치→저장 에지 재사용,
                                             #    불일치→scipy cKDTree kNN(k=8) 그래프 폴백)
       .training_metadata["varying_mesh"]=True
  → app.predict (수정 불필요) → predict_to_mesh → 원본 케이스 메쉬 point_data 에
    twin_* 부착 → 뷰어 표시.  ★ 재샘플 단계가 어디에도 없음 — Route 2 의 존재 이유.
```

μ 는 **노드 피처 브로드캐스트**로 주입(모든 노드에 동일 값 k 채널) — GeometryFNO 의
μ 채널 규약과 동형이라 사용자 개념 모델이 일치한다. 비정상(μ,t)은
`expand_case_params_over_time` 재사용으로 자연 확장되지만 **1차 범위는 steady 전용**
(GeometryFNO 와 같은 게이트 — 시계열이면 명시 거절).

### 3.2 첫 타깃 비교와 추천

| 후보 | 신규 코드 | 신규 의존성 | 원본 메쉬 표시 | 소표본(5~20) 적합 | 판정 |
|---|---|---|---|---|---|
| **A. 기존 GCN surrogate 를 케이스-세트 파라메트릭으로 확장(`mesh_gnn`)** | 그래프 빌더+모델 확장+엔진 ~3 파일 | **0** (PyG 2.7.0 실증) | ✅ `predict_at` 소켓 직결 | 노드 단위로는 샘플 풍부(케이스×N 노드), 작은 hidden 으로 감당 | **★ 추천 — 1번** |
| B. neuralop GINO 도입 | 어댑터+latent 격자·radius 튜닝 | 0 (설치·forward 실증) | ✅ output_queries=임의 좌표 — 소켓 적합 | 문헌 ~500 샘플 — 소표본에서 정성적 | 2번(3D·점군 필요 시) |
| C. 기존 MGN 재활용 | μ 주입+케이스별 그래프+롤아웃 UI | 0 | △ 롤아웃 결과는 시계열 — 뷰어 확장 필요 | 비정상 궤적 다수 필요 | 3번(비정상 메쉬-네이티브 단계에서) |
| D. Transolver | 자체 이식 필요(physicsnemo 깨짐) | 이식 코드 | ✅ | 수백 샘플 | 보류 — physicsnemo 복구 or vendor 결정 후 |

**추천 근거(A):** ① 의존성·설치 리스크 0 — 이 저장소에서 오늘 실증됨 ② 트윈 계약
소켓(`varying_mesh`+`predict_at`)에 정확히 꽂혀 app.py 수정이 사실상 0 ③
`karman_shapes` 데모가 준비된 검증 데이터(진짜 구멍, 현재 Physics AI 독점 칸에 두
번째 전략을 세움 — 좌표 MLP vs 메시지패싱 비교가 즉시 가능해짐) ④ 실패해도 매몰
비용이 그래프 빌더뿐인데, 그 빌더는 B(GINO 점군)·C(MGN)에도 그대로 필요한 공통
기반이다. **즉 A 는 최악의 경우에도 루트 2 전체의 배관 공사가 된다.**

### 3.3 GINO(2번 타깃) 스케치 — A 완료 후

- 입력: `input_geom` = 케이스 표면/전체 점군(서브샘플), `latent_queries` = 바운딩 박스
  균일 격자(GeometryFNO 의 resolution 개념 재사용), `output_queries` = 예측 대상 좌표
  (= `predict_at` 의 coords — 소켓 그대로), `x` = [sdf, μ...] 점 피처.
- 주의: radius 인자명 `in_gno_radius`/`out_gno_radius`(실측), 2D 는 open3d 미가속
  (native 폴백 — 동작 확인). 3D 부터 open3d 가속.
- 케이스당 점 수를 `max_train_points` 식 서브샘플(Physics AI 선례)로 제한.

---

## 4. 리스크

1. **소표본 과적합(케이스 5~20)** — 최대 리스크. 노드 수준 샘플은 많아도 μ 축 방향
   정보는 케이스 수뿐이다. 완화: ① hidden 32~64·3~4층으로 용량 제한 + weight decay
   ② `group_split.py` 재사용(케이스 단위 held-out, GeometryFNO 선례 그대로 —
   `build_geometry_fno_twin` 의 eval_split 블록 복제) ③ 결과 카드에 held-out rel-L2 와
   "소수 케이스는 정성적" 경고(전략 note) 필수 ④ tier="experimental" 로 등급 정직하게.
2. **메모리(노드 수)** — karman 데모 케이스당 ~6 만 노드. GCN hidden 64 활성값
   ~수십 MB/그래프 수준이라 12 GB GPU 에 충분하나, **전 케이스 텐서를 GPU 에 상주시키지
   말 것**(현 GNNSurrogate 는 `Xt` 전체를 device 로 올림 — 케이스별 lazy 전송으로 변경).
   full-batch GCN 이므로 노드 20 만+ 3D 에선 이웃 샘플링 필요 — 1차 범위 밖으로 명시.
3. **학습 시간(3분 스모크 게이트)** — 그래프당 forward 가 E 에 선형. 스모크는
   `karman_shapes` 축소판(nx≈120, 케이스 3, epochs≈50)으로 CPU 60 s 내를 계약으로.
   기본 데모(6 만 노드×5 케이스×200 epochs)는 GPU 로 1~3 분 예상 — 진행 콜백 필수.
4. **형상 가변 그래프 크기 불일치 배치** — 해소됨: PyG `Batch.from_data_list` 가 크기
   다른 그래프 배칭을 지원(실측). 다만 케이스 수가 5~20 이라 **케이스별 루프(batch=1)로
   시작**해도 성능 문제 없음 — 기존 GNNSurrogate/MGN 의 루프 패턴 유지가 단순하다.
5. **physicsnemo 비호환 잠복** — `export_physicsnemo_module`(`service.py:2283`) 등
   기존 physicsnemo 경로도 torch 2.11 에서 깨져 있을 가능성이 높다(이번 범위 밖,
   별도 확인 권장). Transolver/MGN 계획은 physicsnemo 버전 추적에 종속시키지 말 것.
6. **kNN 폴백의 물리 왜곡** — `predict_at` 이 미지 좌표(저장 케이스와 불일치)에서
   kNN 그래프를 만들면 메쉬 위상과 다른 연결이 생긴다(구멍을 가로지르는 에지 등).
   1차 범위에서는 "보고 있는 케이스 메쉬 = 저장 케이스 중 하나" 가 지배적 사용례라
   해시 일치 경로가 대부분을 처리한다. kNN 경로 사용 시 상태 문구에 명시할 것.
7. **MGN `_edge_features` 재사용 버그(§1.1)** — MGN 재활용 단계에서 반드시 수정.

---

## 5. 다음 에이전트에게 줄 구체 구현 지시서 초안

목표: `mesh_gnn` 전략(케이스-세트 파라메트릭 GCN, 메쉬 네이티브) 배선. steady 전용.
예상 규모: 신규 3 파일 + service/strategies/app 각 소폭. 테스트 ~6개.

### 5.1 신규 `src/naviertwin/core/gnn/case_graph.py`

```python
def mesh_edge_index(mesh: Any) -> NDArray[np.int64]:
    """pyvista 메쉬 → 양방향 edge_index (2, 2E).

    mesh.extract_all_edges(use_all_points=True) 사용 — 점 개수·순서 보존을
    assert 로 강제(edges.n_points == mesh.n_points). lines[:,1:] 를 양방향화
    + 중복 제거. VTK 계열이면 정렬/비정렬 무관.
    """

def case_to_graph(
    dataset: Any,                      # CFDDataset
    mu: NDArray[np.float64],           # (k,)
    field_names: Sequence[str],        # 출력 필드 (벡터는 성분 전개)
    *,
    input_field_names: Sequence[str] = (),  # wall_distance/wall_sdf 등
    norm: dict[str, Any] | None = None,     # train-셋 정규화 상수(예측 시 재사용)
) -> dict[str, Any]:
    """반환: {"points","edge_index","edge_attr","x","y","target_names","norm"}.

    - x = [정규화 좌표(topological_dim 성분) | input_fields | μ(정규화) 브로드캐스트]
    - edge_attr = [Δcoords, ‖Δ‖] (정규화 좌표 기준)
    - y 성분 전개 규칙은 case_tensorizer 의 target_names 규약과 동일하게
      ("U" → "U_x","U_y"[, "U_z"]) — 테스트로 상호 일치 고정.
    - 정규화는 norm 인자가 없으면 이 케이스로 계산해 반환(빌더가 train 케이스
      들로 먼저 계산해 전 케이스에 주입 — group_split 시 train-only 원칙 §6½ #2).
    """
```

### 5.2 신규 `src/naviertwin/core/gnn/gnn_surrogate/case_set_gnn.py`

```python
class CaseSetGNN(BaseOperator):
    """케이스별 그래프 목록으로 학습하는 GCN — GNNSurrogate 의 케이스-세트 판.

    GNNSurrogate 와의 차이: fit 이 공유 edge_index 하나가 아니라
    그래프 dict 목록을 받는다(케이스마다 N, E 가 달라도 됨).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64,
                 n_layers: int = 4, max_epochs: int = 200, lr: float = 1e-3,
                 weight_decay: float = 1e-5, device: str = "auto",
                 seed: int | None = 0) -> None: ...

    def fit(self, dataset: dict[str, Any]) -> None:
        """dataset = {"graphs": list[dict]}  # case_to_graph 반환 dict 목록.

        케이스별 루프(batch=1, 기존 GNNSurrogate 패턴 유지). 텐서는 케이스별로
        to(device) — 전 케이스 GPU 상주 금지(§4 리스크 2). train_losses_ 기록.
        """

    def predict_graph(self, graph: dict[str, Any]) -> NDArray[np.float64]:
        """(N, out_dim) 노드별 예측. edge_attr 미사용 시 무시 가능(GCNConv)."""
```
(선택: GCNConv 대신 SAGEConv/GraphConv — edge_attr 활용은 2차. 1차는 GCNConv 로
GNNSurrogate 와 대칭 유지.)

### 5.3 신규 `src/naviertwin/core/digital_twin/mesh_gnn_engine.py`

```python
class _MeshGNNModelFacade:
    """predict_to_mesh / split_multi_prediction 덕타이핑.

    output_fields: [{field_name, display_name, start, end}, ...]  # 좌표 수 기준
    def predict_at(self, coords, params) -> NDArray:  # field-major 평탄 벡터
        # 1) coords 를 저장 케이스와 대조: n_points 일치 + 좌표 해시(또는
        #    np.allclose 샘플 검사) → 저장 edge_index 재사용   [지배 경로]
        # 2) 불일치 → scipy.spatial.cKDTree kNN(k=8) 그래프 폴백 + 로그 경고
        # μ 는 정규화 후 브로드캐스트. 반환 shape 규약은
        # PhysicsNeMoCFDFieldModel.predict_at 와 동일(채널×N field-major).

class MeshGNNTwinEngine:
    """트윈 계약: predict(params)/training_metadata/save/load/get_params.

    training_metadata 필수 키(선례 build_physics_ai_twin_from_cases 와 동일):
      field_name, field_names, reducer="mesh_gnn", surrogate="case_set_gcn",
      problem_type="steady_sweep", param_names, param_mins, param_maxs,
      varying_mesh=True        # ← app.predict 가 predict_to_mesh 경로를 타는 스위치
    predict(params): 대표(0번 또는 μ-최근접) 케이스 그래프 위 예측 —
      GeometryFNOTwinEngine.nearest_case_index 로직 복제(정규화 μ 거리).
    """
```

### 5.4 수정 `src/naviertwin/web/service.py`

```python
def build_mesh_gnn_twin_from_cases(
    datasets: Sequence[CFDDataset],
    field: str | Sequence[str],
    params: np.ndarray,
    *,
    param_names: Sequence[str] | None = None,
    hidden: int = 64, n_layers: int = 4, max_epochs: int = 200,
    input_field_names: Sequence[str] = (),   # wall_distance 등 — UI 체크박스와 연결
    group_split: bool = False, group_ids: Sequence[int] | None = None,
    val_frac: float = 0.15, test_frac: float = 0.15, split_seed: int = 0,
) -> dict[str, Any]:
    """build_physics_ai_twin_from_cases(service.py:816) 를 본뜬다.

    - 시계열 케이스 섞이면 build_geometry_fno_twin(service.py:984)과 같은
      문구 패턴으로 명시 거절(steady 전용).
    - group_split=True 면 group_train_val_test_split + train-only 정규화(norm 을
      train 케이스로 계산해 val/test case_to_graph 에 주입), held-out rel-L2 를
      eval_split 로 반환(build_geometry_fno_twin 의 eval_split 블록 복제).
    - 반환 dict: engine/field/fields/n_cases/param_names/param_mins/param_maxs/
      train_loss/eval_split — 기존 빌더들과 동일 골격.
    """
```

### 5.5 수정 `src/naviertwin/core/digital_twin/strategies.py`

`STRATEGIES` 에 추가(§_check 에 mesh_gnn 분기 — operator 분기 패턴 복제):

```python
StrategySpec(
    key="mesh_gnn", name="메쉬 GNN (그래프 회귀)",
    needs_identical_mesh=False, needs_uniform_grid=False,
    supports_case_sets=True, supports_time_in_sweep=False,
    single_case_needs_steps=2, min_snapshots=3,
    note="메쉬를 그래프로 직접 학습 — 재샘플 없이 원본 격자(진짜 구멍 포함) 위 "
         "예측. 케이스 3개 이상, 소수 케이스는 정성적.",
    tier="experimental",
)
```
`_check`: 케이스 세트에서 `n_time_steps > 1` 거절, `n_cases < 3` 거절, 단일 케이스 거절
(operator 와 동일 패턴). **단일 케이스 직접 학습은 범위 밖.**

### 5.6 수정 `src/naviertwin/web/app.py`

- ②Model 전략 버튼에 `mesh_gnn` 추가(3783행 근처 — key 루프에 편승하면 수정 최소),
  `_build_twin` 디스패치(1147행)에 `elif method == "mesh_gnn": self._build_mesh_gnn_twin()`.
- `_build_mesh_gnn_twin`: `_build_geometry_fno_twin`(1445행) 골격 복제 —
  `service.build_mesh_gnn_twin_from_cases` 호출, 결과 카드에 train_loss/held-out.
- **predict 경로 수정 없음** — `varying_mesh=True` 가 기존 분기(1872행)를 탄다.
  상태 문구가 "보고 있는 형상 위에 표시" 로 이미 정확하다.

### 5.7 테스트 계약 — `tests/test_mesh_gnn_twin.py`

torch_geometric 없으면 `pytest.importorskip("torch_geometric")`.

1. `test_mesh_edge_index_preserves_point_order` — ImageData(8,8,1)→UG:
   edge_index max < n_points, 양방향(각 (i,j)에 (j,i) 존재), 점 순서 보존.
2. `test_case_to_graph_mu_broadcast_and_targets` — 벡터 필드 "U" 성분 전개가
   `cases_to_grid_tensors` 의 target_names 와 동일 문자열, μ 채널이 전 노드 동일.
3. `test_case_set_gnn_fit_predict_varying_sizes` — 노드 수 다른 그래프 3개(예:
   demo_karman.grid_with_hole 축소판 or 합성) fit → predict_graph shape (N_i, C).
4. `test_mesh_gnn_engine_contract` — predict(params) 1D field-major 길이 = C×N₀;
   `engine.model.output_fields` start/end 정합; `training_metadata["varying_mesh"] is True`;
   save/load 후 예측 bit-동일.
5. `test_predict_to_mesh_on_original_case_mesh` — **루트 2 의 존재 증명 테스트**:
   구멍 뚫린 케이스 로드 → 엔진 학습 → `service.predict_to_mesh(engine, μ, case)` →
   반환 dataset 의 n_points == 원본 케이스 n_points(재샘플 없음), twin_* 필드 부착.
6. `test_mesh_gnn_smoke_time_budget` — 케이스 3×~800 노드×50 epochs, CPU 60 s 이내
   (전체 스위트 3 분 게이트 보호). 학습 loss 감소 단조성까지는 요구하지 않되
   최종 loss < 초기 loss 확인.
7. (권장) `test_mesh_gnn_group_split_holdout` — group_split=True 로 held-out rel-L2 가
   유한값, train_idx∩test_idx=∅.

검증(구현 후): `wsl -d ubuntu -- bash -lc 'cd ~/work/claude_code/NavierTwin &&
PYTHONPATH=src python3 -m pytest tests/test_mesh_gnn_twin.py -x -q'` + 브라우저에서
karman_shapes 데모 로드 → mesh_gnn 학습 → 예측이 구멍 뚫린 원본 메쉬 위에 표시되는지
스크린샷 확인. Physics AI 와 같은 케이스에서 rel-L2 비교 수치를 결과 카드로 남길 것.

### 5.8 범위 밖(명시)

- 비정상 메쉬-네이티브(MGN 재활용 — §1.1 의 `_edge_features` 버그 수정 포함), GINO
  어댑터(§3.3), Transolver vendor 결정, 단일 케이스 mesh_gnn, 이웃 샘플링(대형 3D),
  edge_attr 활용 모델(SAGE/NNConv) — 전부 후속 티켓.
