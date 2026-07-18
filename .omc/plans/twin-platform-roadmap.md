# NavierTwin v5 — 일반 CFD 트윈 플랫폼 로드맵

작성 기준일: 2026-07-18. 현재 앱 v4.2.0 + 17 rounds(웹 GUI, 카르만 실LBM 데모까지).
이 문서는 사용자가 제시한 "만들고 싶은 앱" 비전 대비 현재 앱의 문제점 · 해결방법 ·
아키텍처 보완 · 단계별 로드맵을 정리한다. 근거는 3-갈래 조사(현재 코드 감사 +
최신 방법/OSS 연구 + 포맷/BC/SDF 라이브러리 연구).

관련 문서: [[model-taxonomy-plan]] (4-계열 분류·패널 정리의 선행 결정 기록).

---

## §1. 목표 비전 — 데이터 축 매트릭스

사용자 비전을 축으로 정리하면, 트윈이 다뤄야 하는 데이터 공간은:

```
{정상 | 비정상(+시간축 t)}
  × {한 형상 | 여러 형상}
  × {한 조건 | 여러 조건(경계조건)}
  × {정렬격자 | 비정렬격자}
  × {1D | 2D | 3D}
```

즉 **형상 × 조건 × 시간**의 3축 곱 위에서, 격자 종류와 차원이 자유로워야 한다.
여기에:
- **입력**: 물리장(p, U, T…)은 파일에서. 경계조건은 앱 내 사용자 입력. wall face는
  뷰어에서 마우스 클릭 지정. 앱 내 전처리로 wall-SDF·형상특징 계산.
- **학습**: OSS 다수 전략. 학습 전 강제 해상도 낮추기(사용자가 비율 지정).
  균일격자/동일형상 요구 전략은 자동 재샘플(빈 셀 0-채움 허용).
- **결과**: 실제 vs 트윈 비교 + 에러장. 좌(실제)/우(트윈) 분할 뷰어. 실제 없는
  외삽 결과도(에러 계산 없이).

---

## §2. 현재 아키텍처와 근본 문제

**핵심 전제(문제의 뿌리):** 모든 리더는 데이터를 PyVista `UnstructuredGrid`로
정규화하지만(`_mesh_utils.py`, `CFDDataset.mesh`), 학습 경로에서 곧바로 갈린다:
- **Physics AI(신경장)** 만 비정형 메쉬를 좌표로 직접 소비 → 격자 무관.
- **ROM/POD, coarsen, resample, FNO** 는 전부 균일 `ImageData` 격자를 강제.

이 전제가 비전과 충돌하는 지점:
- 형상 가변 → ROM은 공통 균일격자 재샘플 없이는 불가(벽이 가짜 empty가 됨).
- 3D/비정형 → 균일격자 변환이 비싸거나 형상을 뭉갬.
- **경계조건·wall face 개념이 데이터 모델에 아예 없음** → 전처리(wall-SDF)의 토대 부재.

결론: 아키텍처를 **"균일격자 중심"에서 "메쉬 네이티브 + 능력(capability) 기반 전략
라우팅"** 으로 전환해야 한다. 전략마다 필요 조건(균일격자? 동일메쉬? 3D? 시간?)을
선언하고, 앱이 데이터 특성에 맞춰 라우팅/적응한다.

---

## §3. 갭 분석 (비전 3영역)

### 영역 1 — 데이터 로드

| 비전 요구 | 현재 | 갭/문제 |
|---|---|---|
| 정상 {1형상×N조건 / N형상×1조건 / N형상×N조건} | `load_case_set` (파일=조건), 형상가변은 resample 또는 Physics AI | 조건이 **CSV 파일**에서만 옴 — 앱 내 입력 없음 |
| 비정상 = 정상 + 시간축 t | 단일형상 시계열은 지원(t=파라미터) | **비정상×다케이스 붕괴**: case-set이 각 케이스의 **마지막 스텝만** 보존(`service.py:510-511`). (형상×조건×시간) 곱이 없음 |
| 1D/2D/3D | 리더+신경장은 원리상 차원 무관, 2D 자동 | **3D를 실제로 태우는 경로 없음**(솔버·데모 전부 1D/2D), 뷰어 카메라가 flat=2D 가정 |
| VTK/CGNS 등 다양한 포맷 | VTK/VTU/PVD/STL, OpenFOAM, CGNS(4단 폴백), Fluent, Gmsh, SU2 | EnSight/Tecplot/Exodus/XDMF 미연결. parquet/zarr 리더는 등록 안 됨 |
| BC 앱 내 입력 + wall face 마우스 클릭 | **없음** | 최대 갭. BC 데이터 모델 0, 뷰어 face 픽킹 0, 파일 patch 메타(OpenFOAM boundary/CGNS ZoneBC) **로드 시 폐기** |
| 앱 내 SDF/특징 전처리 | SDF는 **균일격자에서만**(EDT on ImageData, resample 시). core/geometry는 원시 도형 SDF만, 파이프라인 미연결 | 임의 메쉬 wall-distance 불가. wall이 뭔지 모르니 wall-SDF 자체 불가 |

추가 발견: **벡터장이 크기로 뭉개짐** — `_field_to_scalar_snapshots`가
`norm(axis=2)`. U가 방향을 잃음 → 비전의 "p, U, T" 중 U가 반쪽.

### 영역 2 — 트윈 학습

| 비전 요구 | 현재 | 갭/문제 |
|---|---|---|
| OSS 다수 전략 | 웹에서 실제 선택 가능: ROM(POD+RBF/kriging), Physics AI(torch MLP), DMD. FNO는 별도 ⑥랩 | core엔 FNO/DeepONet/GNN/Koopman/Mamba/diffusion 등 다수 존재하나 **트윈 전략으로 미연결**. 형상가변용 GINO/Transolver/MeshGraphNet **없음** |
| 학습 전 강제 해상도 낮추기(사용자 비율) | `coarsen_dataset` + 미리보기 슬라이더 있음 | **균일 ImageData로만** 출력 — 비정형 유지 감소(decimation) 불가 |
| 균일격자/동일형상 전략 자동 재샘플(빈셀 0) | `resample_cases_to_common_grid`(SDF+0채움) | 균일 Cartesian만. 공통 **비정형** 학습메쉬 옵션 없음 |
| 정상/비정상 × 정렬/비정렬 × 동일/다른형상 × 1/2/3D 전면 | 비정형+형상가변 = **Physics AI만**; ROM은 거절; FNO N/A. 3D 미검증 | 매트릭스 다수 셀이 Physics AI 하나에 의존 |

### 영역 3 — 트윈 결과

| 비전 요구 | 현재 | 갭/문제 |
|---|---|---|
| 실제 vs 트윈 비교 + 에러 | `compare_models` 리더보드(RMSE/R²/rel-L2, 학습 파라미터 재구성 오차) | 임의 예측점의 held-out 비교 아님. case-set은 리더보드 제외 |
| 좌(실제)/우(트윈) 분할 뷰어 | **없음** — 단일 plotter, 필드 토글만 | 분할 뷰/카메라 동기 0 |
| 차이(에러)장 | **없음** — |real−twin| 셀별 장 계산·표시 없음 | 신규 필요 |
| 외삽 결과(에러 없이) | DMD 예보 + Physics AI `predict_to_mesh` | Physics AI/DMD 한정, ROM 불가. "외삽 구간—에러 N/A" 상태 라벨 없음 |

---

## §4. 해결 아키텍처

### 4.1 데이터 모델 — 축을 1급으로 (모든 것의 척추)

현재 `CFDDataset`(mesh/time_steps/field_names/metadata)은 단일 케이스용. 신규:

```
TwinDataset
  cases: list[TwinCase]
  axes:  {geometry: N형상, conditions: [조건파라미터명], time: bool}
TwinCase
  geometry_id: int            # 같은 형상끼리 묶는 키
  mesh:        UnstructuredGrid | ImageData
  conditions:  dict[str,float] # 앱 내 입력 or CSV or 파일 메타
  fields:      {name -> (n_time, n_points, n_comp)}   # 시간축 보존!
  boundary:    {patch_name -> face_ids, 각 patch의 BC type}  # 신규
  features:    {sdf, wall_distance, normals, ...}     # 전처리 산물
```

핵심: **시간축을 붕괴시키지 않는다**(비정상×다케이스 해결). 입력은 (좌표, 조건,
[t]) → 출력 필드. 벡터장은 성분 보존(방향 유지).

### 4.2 능력 기반 전략 레지스트리

각 전략이 요구/지원을 선언 → 앱이 데이터 특성에 맞춰 라우팅·적응:

```
StrategySpec
  needs_uniform_grid: bool     # True면 자동 resample(0채움)
  needs_identical_mesh: bool   # True면 동일형상만 또는 resample
  supports_varying_geometry: bool
  supports_unstructured: bool
  supports_3d: bool
  supports_time: bool          # 시계열/예보
  min_samples: int             # 데이터 요구량 힌트
```

`recommend_method`를 이 레지스트리 기반으로 교체 → 데이터 로드 시 "이 데이터엔
가능한 전략 N개" 를 정확히 제시(현재는 ImageData 여부 + 스텝수 휴리스틱).

### 4.3 경계조건 서브시스템 (최대 갭 → 최우선)

1. **파일 patch 메타 보존**: OpenFOAM은 `POpenFOAMReader.patch_array_names` +
   `boundary` 멀티블록 유지(현재 `.combine()`으로 폐기). CGNS는 `pyCGNS` 있으면
   `ZoneBC_t` 파싱해 wall face 범위 복원(vtkCGNSReader는 family 블록으로 노출).
2. **앱 내 BC 입력**: 조건 파라미터 폼(속도/압력/온도 등) — 케이스별 `conditions`.
3. **마우스 클릭 wall face 지정**: trame **server 모드**에서 PyVista 픽킹이 그대로
   동작(브라우저 이벤트가 서버 인터랙터로 전달). `extract_surface()`(→
   `vtkOriginalCellIds`) 후 `enable_element_picking(mode='cell')` 또는
   `enable_cell_picking(through=False)`, 픽 결과의 `orig_extract_id`로 원본 face
   역추적. patch 메타 없는 파일(raw VTU/STL)의 폴백.
4. **wall-SDF/거리 전처리**: wall이 정해지면 `mesh.compute_implicit_distance(wall_surf)`
   가 임의 메쉬에서 wall-distance를 준다(균일격자 EDT 대체, 신규 deps 0). 비수밀
   기하는 `libigl.signed_distance`(winding number, 견고). 추가 특징:
   `compute_normals`, `curvature`, `compute_cell_sizes`, 위치 인코딩.

### 4.4 분할 뷰어 + 에러장

- **듀얼 `plotter_ui` 두 pane**(좌 실제/우 트윈), 카메라 링크. "분할" 토글 버튼.
- **차이장** `|real − twin|`을 셀별로 계산해 필드로 부착, 컬러맵. 요약 에러(RMSE/
  rel-L2/max)도.
- **외삽 모드**: ground truth 없으면 에러 pane/장을 숨기고 "외삽 — 에러 N/A" 라벨.
  DMD 예보뿐 아니라 모든 전략의 미지 조건/형상 예측에 일반화.

---

## §5. 전략 카탈로그 — 두 개의 루트

**핵심 재구성(사용자 지적 반영):** 형상 가변을 다루는 길은 두 갈래이고, **루트 1이
훨씬 크고 성숙하다.** 지난 종합은 루트 2(메쉬 네이티브)에 편향돼 ROM 계열을
과소평가했다.

- **루트 1 — 공통 격자 재샘플(0/빈셀 채움 허용).** 형상이 **파라미터 벡터** 또는
  **SDF/마스크 채널**이 되는 순간, 고전·파라메트릭 ROM 생태계 전체 + 격자 기반 딥
  서로게이트 전체가 열린다. 성숙·저비용·소데이터 친화. `shapes` 데모가 이미 이 길.
- **루트 2 — 메쉬 네이티브(재샘플 회피).** GINO/Transolver/MeshGraphNet/CORAL/
  Physics AI. 진짜 구멍·위상 변화·대형 3D에 필요하지만 무겁고 데이터 많이 든다.

두 루트는 공존하고, 능력 레지스트리(§4.2)가 데이터 특성에 따라 고른다.

### 5.1 루트 1 — 공통 격자: 고전·파라메트릭 ROM

기존 POD+RBF/Kriging+PyDMD는 이 공간의 한 귀퉁이일 뿐. 열리는 것들:

| 전략 | 형상 진입 방식 | 정상/비정상 | 데이터 | OSS (pip?) | 우선 |
|---|---|---|---|---|---|
| **EZyRB** (POD-NN, POD-GPR(UQ), KNN, AE) | 파라미터 μ | 둘 다 | 소~중 | `ezyrb` ✅ | **1 — 최고 ROI** |
| **ParametricDMD** | μ 잠재 보간 | 비정상 | 시계열/μ | `pydmd` ✅(이미 의존) | **2 — 거의 무료** |
| POD + GP (GPyTorch/BoTorch) — 예측 분산(UQ) | μ | 둘 다 | 소~중 | `gpytorch`,`botorch` ✅ | 중 |
| POD + 희소 PCE (+Sobol 민감도) | 독립 μ | 둘 다 | 소~중 | `chaospy`,`UQpy` ✅ | 중 |
| **클러스터/로컬 ROM** (전역 POD 실패 시) | μ 클러스터+로컬기저 | 둘 다 | 중 | `scikit-learn`+기존POD ✅(DIY) | 중 |
| **Grassmann 기저 보간** (교과서적 정답) | **기저 보간** over μ | 둘 다 | 중(μ별 기저) | `pymanopt`+~50줄 ✅* | 중 |
| 메쉬 모핑 + POD (PyGeM→EZyRB) | 모핑 파라미터 μ | 둘 다 | 소~중 | `pygem`,`ezyrb` ✅ | 위상불변 시 |
| Operator Inference (물리구조 보존) | 아핀 μ 임베딩 | 비정상 | 스냅+시간도함수 | `opinf` ✅ | 낮 |
| 이차 매니폴드 / 등록기반 ROM (이류지배) | μ / 공간사상 | 비정상 | 중~고 | 연구코드 ⚠️ | 낮(연구) |

\* pymanopt는 pip, 보간층은 자체 구현. 고전 RB(pyMOR/RBniCS)는 **침투식**(PDE 약형
필요)이라 스냅샷 기반 앱과 안 맞음 — pyMOR의 데이터 기반부(DMD/NN/VKOGA)만 적합.

### 5.2 루트 1 — 공통 격자: 격자 기반 딥 서로게이트 (형상 = SDF 채널)

0-채움 공통 격자는 케이스별 **SDF/마스크**를 자연 생성 → 이미지→이미지 문제:

| 전략 | 형상 진입 | 정상/비정상 | 데이터 | OSS (pip?) | 우선 |
|---|---|---|---|---|---|
| **FNO + SDF/마스크/BC 입력 채널** | 추가 입력 채널 | 정상(비정상=자기회귀) | 수백~1k | `neuraloperator` ✅ **이미 의존, 모델변경 0** | **★ 가장 싼 승리** |
| **U-Net + SDF 채널** (DeepCFD/Thuerey식) | SDF 채널 | 정상 | 수백~수천 | 기존 `unet.py` 재활용 | 낮은 노력 |
| **CAE DL-ROM** (POD-DL-ROM) | μ 또는 SDF채널→CAE | 둘 다(one-shot) | 수백 | `dlroms`(git)/기존 POD+conv_ae | 중 |
| **잠재 동역학** (latent-DMD→LSTM/Transformer) | 인코더+μ | 비정상 롤아웃 | 수백 궤적 | PyDMD+conv_ae ✅ | 중 |
| **POD-DeepONet** (가장 소데이터) | branch 입력(μ/SDF) | 정상(시간=trunk) | 소~수백 | `deepxde` ✅/기존 deeponet | 중 |
| 조건부 확산/cGAN (불확실성) | SDF 마스크 조건 | 정상+UQ | 1k+ | `tum-pbs` 클론 ⚠️ | 낮(UQ목표시) |
| PINO (물리제약 FNO) | SDF채널+PDE잔차 | 둘 다 | 소(잔차가 데이터 대체) | `neuraloperator`/`nvidia-physicsnemo` ✅ | 소데이터시 |

**증거:** DeepCFD(SDF 3채널→u,v,p, arXiv:2004.08826), Thuerey(마스크+자유류 채널,
AIAA'20) — 둘 다 "0-채움 = 형상 인코딩" 전제를 검증. FNO는 `in_channels`만 키우면
됨(`neuralop_fno.py`가 이미 노출).

### 5.3 루트 2 — 메쉬 네이티브 (재샘플 회피)

진짜 구멍·위상 변화·대형 3D용. 기존 Physics AI(신경장 MLP)가 이 철학의 시작.

| 전략 | 채우는 칸 | OSS (pip?) | 근거 |
|---|---|---|---|
| **GINO** | 3D·비정형·형상가변(SDF+점군) | `neuraloperator` ✅ | NeurIPS'23, ~500샘플 |
| **MeshGraphNet** | 비정형 비정상 롤아웃 | `torch_geometric`/`physicsnemo` ✅ | 시간축 비정형 성숙답 |
| **Transolver** | 형상가변 2D/3D SOTA | `physicsnemo`/`thuml` ✅ | ICML'24 |
| **CORAL/AROMA** | 신경장 오퍼레이터 | 클론 ⚠️ | Physics AI 직접 진화 |

### 5.4 기하 인코딩 세 개의 문 — 데이터 모델을 결정한다

형상이 방법에 들어가는 방식이 근본적으로 셋이고, 각각 데이터 모델/UI가 다르다:

- **문 A — 저차원 파라미터 벡터 μ** (고전 ROM의 집). §5.1 거의 전부 + POD-DeepONet의
  branch. **UI/데이터 모델:** 케이스 = `{μ ∈ R^p, 스냅샷}`. 명명 파라미터 스키마 +
  샘플링 패널. 가장 싸고 일반적 — 기존 POD+RBF/Kriging 배선과 일치. **기본 채택.**
- **문 B — 기저/부분공간 보간** (Grassmann). UI는 A와 같으나, μ별 POD 기저를 저장하고
  부분공간을 보간. **저장 포맷 변경**(전역 Φ 하나가 아니라 케이스별 기저).
- **문 C — 이미지형 SDF/마스크 채널** (딥 서로게이트). **고전/선형 ROM은 이미지 채널을
  파라미터로 못 먹는다** — 벡터가 필요. 그래서 다리가 필요:
  - **(C→A 다리) POD-of-SDF:** SDF/마스크 필드 자체에 POD를 걸어 상위 기하 잠재좌표를
    μ로 사용. **손으로 형상 파라미터를 안 붙여도** 자유형상·위상변화가 모든 문-A ROM을
    구동하게 된다. **강력 추천** — SDF 채널과 파라미터 벡터 세계를 통합.
  - **(순수 C)** CNN/U-Net/FNO/DeepONet + SDF 입력 — 딥 서로게이트 스택(PyTorch, 셀별
    필드 I/O)과 별도 UI(케이스별 SDF 필드).

**설계 결론:** **문 A + C→A(POD-of-SDF) 다리**에 집중하면, 하나의 데이터 모델
(μ = 명시 파라미터 및/또는 SDF 잠재좌표) + 하나의 스냅샷 저장소가 §5.1·5.2 대부분을
덮고 SDF 채널과도 호환된다. 순수 SDF-채널 신경 서로게이트(문 C)는 위상변화·파라미터화
곤란 기하가 실제 필요할 때만.

**위상 변화가 분기점:** 메쉬 모핑(PyGeM)은 **위상 불변**만(두께·각도·범프). 구멍
추가/물체 병합 등 위상이 바뀌면 모핑 불가 → 0-채움 공통격자 + POD-of-SDF 또는 SDF 채널.

### 5.5 벤치마크 타깃

실 리더보드 매핑: **AirfRANS(2D 형상가변, 1000샘플)** → **ShapeNet-Car(3D 중간)** →
**DrivAerNet++(3D 대형)**.

---

## §6. 단계별 로드맵

의존 순서: 데이터 모델 → BC/전처리 → 전략 → 결과 뷰어. 각 단계는 독립 검증 가능.

### v5.0 — 데이터 모델 척추 (모든 것 선행) — **진행 중 (M1 완료)**
- ✅ **M1 (2026-07-18):** 능력 기반 `StrategySpec` 레지스트리(`web/strategies.py`) +
  `recommend_method` 교체 — ②Model 카드가 로드 시점에 가능/불가+이유 표시.
  **시간축 보존**: `load_case_set` 이 마지막 스텝 붕괴를 중단(PVD 우선 규칙 포함),
  ROM/Physics AI 케이스 빌더가 (μ, t) 로 전개 학습(`expand_case_params_over_time`),
  ③Twin 에 μ·t 슬라이더 자동 생성. `sweep_unsteady` 데모(문제 유형 C). DMD 가
  케이스 세트에서 첫 케이스만으로 조용히 학습하던 함정 차단. 비정상+형상가변
  재샘플은 시간을 버리므로 명시 거절. 202 테스트.
- ✅ **M2 (2026-07-18):** 벡터장 성분 보존 — Physics AI 가 U 를 U_x/U_y/U_z 채널로
  학습(방향 유지), U_mag 는 성분에서 파생. 부호 재현 테스트(크기 학습으론 불가능한
  일)로 검증. 발견: 작은 MLP 의 스펙트럴 바이어스 절벽(24/60ep R²≈0 vs 64/400ep
  0.81) — 테스트에 문서화.
- ✅ **M3 (2026-07-18, v5.2 항목):** ParametricDMD — 비정상 스윕의 (μ, t) 예보.
  partitioned(케이스별 DMD, μ별 고유 주파수 안전) + ezyrb POD/RBF 계수 보간.
  진행파 데모 적합 2e-15, 미지 μ 보간 + 학습 구간 밖 t 예보 브라우저 검증.
  신규 의존성: ezyrb.
- ✅ **GUI 패리티 1차 (2026-07-18):** 레지스트리를 core/digital_twin/strategies.py
  로 이동(웹 shim 유지), 데스크톱 Model 패널에 전략 어드바이저, 도구 메뉴에
  데모 데이터 로드(시계열 4종 — 데스크톱 최초의 인앱 데모).
- 남은 것: `TwinDataset`/`TwinCase` 정식 구조(현재는 list[CFDDataset]+전개 헬퍼).
  데스크톱: 케이스 세트 데모/(μ,t) 트윈 패널 다중 슬라이더는 후속.

### v5.1 — 경계조건 & 전처리 (최우선 갭)
- OpenFOAM boundary 블록 / CGNS ZoneBC patch 메타 보존.
- 앱 내 조건 입력 폼 + **마우스 클릭 wall face 지정**(trame server + element picking).
- 메쉬 네이티브 **wall-SDF/거리**(`compute_implicit_distance` + `libigl` 백엔드) +
  법선/곡률/셀크기 특징. `core/geometry/sdf.py` 옆 신규 `mesh_sdf` 모듈.
- 검증: raw 메쉬 로드 → 벽 클릭 → wall-distance 필드 생성 → 뷰어 표시.

### v5.2 — 형상가변 트윈 전략 (루트 1 우선 — 싼 승리부터)
공통 격자 재샘플이 이미 있으므로(§5) 여기부터가 저비용·고가치. 순서:
- **★ FNO + SDF/마스크/BC 입력 채널** — 가장 싼 승리. `NeuralOpFNO`가 `in_channels`를
  이미 노출하므로 모델 변경 0, 데이터 배관만(v5.1의 SDF 재사용). DeepCFD/Thuerey 검증.
- **EZyRB** 배선 — POD-NN/POD-GPR(UQ)/KNN을 기존 POD+RBF/Kriging 옆에. 최고 ROI.
  ✅ 완료(부분): `ezyrb_gpr`(POD-GPR, uq_mean_std 메타데이터)·`ezyrb_ann`(POD-NN)을
  `core/surrogate/ezyrb_surrogate.py` + TwinEngine + web SURROGATES/UI 에 배선.
  KNN 은 후속.
- **ParametricDMD** 켜기(비정상 파라미터 스윕) — pydmd 이미 의존.
- **POD-of-SDF → μ 다리**(§5.4) — 자유형상이 손 파라미터 없이 문-A ROM 구동.
- **U-Net + SDF 채널**(기존 unet.py) — 벽면 급구배 강함, FNO와 데이터 파이프 공유.
- 전략 선택 UI를 능력 매트릭스로 필터(데이터에 가능한 것만 노출).
- 검증: 형상가변 케이스셋으로 FNO+SDF·EZyRB 학습, held-out 형상 예측.

### v5.2b — 루트 2 (메쉬 네이티브 — 진짜 구멍·위상변화·대형 3D)
루트 1로 부족한 경우(위상 변화, 재샘플이 벽을 뭉갬)만:
- **GINO**(neuraloperator) → 3D·비정형·형상가변(SDF+점군).
- **MeshGraphNet**(torch_geometric) — 비정형 비정상 롤아웃.
- (후보) Transolver, CORAL/AROMA.

### v5.3 — 해상도/재샘플 일반화
- 비정형 유지 **decimation**(균일격자 강제 대신) — 사용자 비율 슬라이더 확장.
- 공통 **비정형** 학습메쉬 재샘플 옵션(현재 Cartesian만).
- 검증: 대형 3D 메쉬 → 비율 낮추기 → 메모리/점수 미리보기 정확.

### v5.4 — 결과: 분할 뷰어 + 에러장 + 외삽
- 듀얼 pane 링크 카메라(좌 실제/우 트윈) + "분할" 버튼.
- 셀별 |real−twin| 차이장 + 요약 에러. held-out 비교(재구성 오차 아님).
- 외삽 모드(에러 N/A) 전 전략 일반화.
- 검증: 학습에 없던 조건 예측 → 좌우 비교 → 차이장; 외삽 → 에러 숨김.

### v5.5 — 3D 검증 + 벤치마크
- 실 3D 케이스 end-to-end. AirfRANS(2D) → ShapeNet-Car(3D) 벤치 수치.
- 3D 뷰어 카메라/클리핑, 대형 데이터 성능.

### 후보(추가 계열, 우선순위 낮음)
- Transolver(SOTA 정확도), CORAL/AROMA(신경장 진화), Geom-DeepONet(파라메트릭 CAD),
  EnSight/Tecplot/Exodus/XDMF 리더, Koopman/SINDy 해석.

---

## §7. 리스크 · 결정 필요

- **"가짜 empty" 텐션**: 비전은 학습용으로 0-채움을 명시 허용(ROM/FNO 경로). 동시에
  진짜 구멍(GINO/MeshGraphNet/Physics AI)도 지원. → 아키텍처가 **둘 다** 지원
  (전략 레지스트리가 결정). 모순 아님.
- **pyCGNS 설치**: CGNS C lib + HDF5 빌드 필요, Windows pip 실패 잦음 → **WSL/conda**
  에서. 프로젝트가 WSL 전용([[runs-under-wsl]])이라 자연스러움.
- **서버 렌더 GL 컨텍스트**: headless WSL 픽킹에 `vtk-osmesa` 또는 `start_xvfb()`
  필요(현재도 오프스크린 렌더 사용 중이라 확장).
- **데이터 요구량**: GINO ~수백, Transolver/MeshGraphNet 수백~수천 샘플. 현재 데모는
  6케이스 — 실사용은 사용자 데이터 규모에 의존. 데모/벤치로 검증하되 과대광고 금지.
- **범위**: v5.0~v5.1이 토대(BC/데이터모델). 나머지는 그 위에서 병렬 가능. 한 번에
  다 하지 않는다.
