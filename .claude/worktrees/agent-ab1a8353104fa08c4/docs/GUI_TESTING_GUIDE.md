# NavierTwin GUI 검증 가이드 — **결과 파일 로드 기반**

실제 CFD 결과 파일 (.vtu / .vtk / .foam / .cgns / .npz / .csv)을 로드해
모든 기능이 정상 작동하는지 단계별로 검증한다.

## 0. 검증용 데이터 생성

테스트 파일이 없으면 합성 데이터를 자동 생성:

```bash
PYTHONPATH=src python3 scripts/make_test_dataset.py /tmp/naviertwin_demo
```

생성되는 파일:
- **`/tmp/naviertwin_demo/cavity.vtu`** — 20×20 cavity flow (U / p / T / wallShearStress, 400 pts)
- **`/tmp/naviertwin_demo/cavity_time.npz`** — 시공간 행렬 (n_t=50 × n_x=400, rank-5 신호)
- **`/tmp/naviertwin_demo/cavity_probe.csv`** — 단일 프로브 시계열 (n=2000, 5+17Hz, change point + spike)

> 본인의 OpenFOAM/Fluent 결과를 쓰려면 위 합성 파일 대신 자기 .foam / .cgns / .vtu를 직접 사용. 지원 확장자: `.foam .cgns .vtk .vtu .vtp .stl .ply .msh .cas .dat .su2 .openfoam .ntwin`.

## 1. GUI 실행

```bash
PYTHONPATH=src python3 -m naviertwin --gui
# 또는 설치 후
naviertwin --gui
```

언어 전환: 메뉴 `보기 (V)` → `English / 한국어`.

---

## 시나리오 1 — `cavity.vtu` 1개 파일로 6개 탭 검증

### ① Import 탭 — 파일 로드

| 단계 | 동작 | 검증 |
|------|------|------|
| 1 | **`파일 선택`** → `/tmp/naviertwin_demo/cavity.vtu` | 경로 표시 |
| 2 | **`Readiness 점검`** | "OK: VTKReader" |
| 3 | **`데이터 로드`** | Analyze 탭 3D 뷰어에 mesh/field 표시 |

**기대 출력**: n_points=400, fields=`T`, `U`, `p`, `wallShearStress`.

---

### ② Analyze 탭 — Q-criterion / FFT / y+ / 해석해

데이터셋 자동 주입.

| op | 입력 | 검증 |
|----|------|------|
| **Q-criterion** | velocity_combo=`U` → `분석 실행` | 결과 텍스트에 "Q field added" |
| **λ₂** | 동일 | mesh.point_data["lambda2"] 추가 |
| **FFT/PSD** | dt=0.01, field=`U` | 주파수 표 출력 |
| **y+** | rho=1.225, mu=1e-5, wallShearStress 사용 | y+ 분포 평균/min/max |
| **해석해 비교** | Couette, μ=1, h=1, U_top=1 | "정확도 R²>0.99" |

---

### ③ Reduce 탭 — POD / Randomized / Incremental / MRPOD

> 단일 .vtu (정적 1 스냅샷)는 POD에 부족. **시간 시리즈 .ntwin** 또는 **⑩ Post-Tools의 `eof` op** (시공간 분석 즉시 가능) 사용 권장.

| 단계 | 동작 | 검증 |
|------|------|------|
| 1 | reducer=`POD`, n_modes=5 | 입력 활성화 |
| 2 | **`축소 실행`** | "누적 에너지 X%, 소요 Y초" |

---

### ④ Model 탭 — 모델 학습

| 단계 | 동작 | 검증 |
|------|------|------|
| 1 | model=`Kriging` | 파라미터 노출 |
| 2 | **`모델 학습`** | Loss curve 갱신 → ⑦ Compare 자동 채워짐 |
| 3 | **`후보 추천`** (BO) | "다음 평가점: x=..." |
| 4 | **`연산자 학습`** (FNO/DeepONet) | PyTorch 있으면 학습 |

---

### ⑤ Twin 탭 — 디지털 트윈

| 단계 | 동작 | 검증 |
|------|------|------|
| 1 | **`예측 실행`** | 학습된 surrogate 예측 → R² |
| 2 | **`최적화 실행`** (BO) | 수렴 곡선 |
| 3 | **`동화 quick-check`** (EnKF) | "RMSE: X → Y (감소)" |
| 4 | **`저장`** → `/tmp/cavity.h5` | HDF5 파일 생성 |
| 5 | **`로드`** | 파이프라인 복원 OK |

---

### ⑥ Export 탭

| 단계 | 동작 | 검증 |
|------|------|------|
| 1 | format=`VTK` → 경로 → **`내보내기`** | ParaView로 열림 |
| 2 | format=`CSV` | 엑셀로 열림 |
| 3 | format=`ONNX` | `onnx.checker.check_model` 통과 |
| 4 | format=`PDF Report` | 그림 + 메트릭 표 |

---

### ⑦ Compare 탭 — view-only

④ Model에서 2개 이상 모델 학습 후 자동 갱신.

---

### ⑧ Simulation 탭 — 라이브 데모

dataset과 무관 — 자체 시뮬레이션 (LBM/Streaming/RL/Burgers).

---

### ⑨ Explain 탭

학습된 모델 후:
- **SHAP** → 변수 기여도 막대 차트
- **Symbolic** → `y ≈ 2x₀ + 0.3 sin(x₁)` LaTeX
- **Attention** → Transformer heatmap

---

## 시나리오 2 — Post-Tools 핵심 op 검증 (cavity.vtu 로드 후)

⑩ Post-Tools 탭은 ① Import 후 자동으로 dataset 주입. 데이터 라벨이 "로드됨 (400 pts, …, T, U, p)"로 변경되고 **필드 콤보** 활성화.

### dataset 입력으로 작동하는 op (25개) — 권장 검증

| op | 카테고리 | 검증 차트 |
|----|----------|-----------|
| **psd_welch** | spectral | log-log PSD |
| **kolmogorov_slope** | spectral | E(k) loglog + slope |
| **eof** | rom | 4-모드 subplot (n_modes=5) |
| **reynolds_stats** | statistics | mean/RMS/TKE 텍스트 |
| **two_point_acf** | statistics | R(r) + 적분 길이 마커 |
| **box_stats** | statistics | broken_barh + outlier |
| **quadrant_analysis** | statistics | Q1-Q4 4-bar |
| **change_points** | anomaly | segment means bar |
| **anomaly_mahalanobis** | anomaly | scores ndarray |
| **denoise** | preprocessing | 평활 신호 line |
| **safe_eval** | preprocessing | `sqrt(u**2+v**2)` 등 |
| **ts_features** | features | 18 특성 dict |
| **gof_normality** | validation | A² + 임계값 |
| **stat_convergence** | statistics | Geweke z + ESS |
| **quantile** | statistics | 분위값 |
| **morphology_components** | topology | 마스크 + components |
| **grid_derivatives** | topology | gradient/laplacian heatmap |
| **helmholtz_decomp** | topology | 4 imshow |
| **anisotropy_state** | turbulence | Lumley state |
| **conditional_sampling** | statistics | 트리거 평균 |
| **coord_transform** | topology | 좌표 변환 |
| **time_interp** | preprocessing | 보간 신호 |
| **plane_flux** | integrals | 평면 통과 flux |
| **surface_forces** | integrals | F/M/lift/drag |
| **line_probe** | preprocessing | 라인 샘플 |

### 합성/외부 입력만 가능한 op (27개) — Demo 모드 전용

`save_rom`, `load_rom`, `gappy_reconstruct`, `basis_interpolate`, `mass_search`, `find_motifs`, `mode_summary`, `subspace_drift`, `rom_residual`, `rom_envelope`, `surrogate_metrics`, `residual_diagnostics`, `ensemble_average`, `bic_model_average`, `stacking`, `morris_sensitivity`, `permutation_importance`, `batch_predict`, `trajectory_clustering`, `cell_volume_integrals`, `auto_report_field/probe`, `phase_average`, `running_moments`, `acoustic_strouhal`, `pod_truncation`, `critical_points`

→ 이 op은 dataset과 무관한 합성 데이터 (예: 두 기저 행렬, 트래젝토리)가 필요. **dataset 로드 안 한 상태에서 `Demo 실행`** 으로 검증.

### 절차 (cavity.vtu 로드 후)

1. **필드 콤보**: `(자동 선택)` → `U` → `p` 순서로 변경하며 결과 비교
2. **`Demo 실행`** 버튼이 `실행 (로드 데이터)` 으로 변경됨 확인
3. 위 25개 op 차례 실행, 결과 텍스트의 입력 라벨이 "로드 데이터셋" 인지 확인
4. **`CSV`** export → `/tmp/post_result.csv` → 엑셀로 열어 컬럼 확인
5. **`차트 이미지 저장`** → PNG/SVG/PDF 선택
6. **`카테고리 일괄 실행`**: `statistics` 선택 → markdown 요약 확인

---

## 시나리오 3 — 프로브 CSV로 시계열 op 검증 (CLI)

`cavity_probe.csv`는 5Hz + 17Hz + 중간 평균 변화 + spike를 포함. GUI 직접 입력이 어려우면 CLI로 facade 호출:

```bash
PYTHONPATH=src python3 -c "
import numpy as np
from naviertwin.core.post_process_facade import PostProcessFacade
data = np.loadtxt('/tmp/naviertwin_demo/cavity_probe.csv',
                  delimiter=',', skiprows=1)
t, sig = data[:, 0], data[:, 1]

facade = PostProcessFacade()

# PSD: 5Hz, 17Hz 피크
r = facade.run('psd_welch', signal=sig, fs=100.0, nperseg=256)
top = sorted(np.argsort(r['psd'])[-3:], key=lambda i: r['psd'][i], reverse=True)
print('PSD 상위 피크:', [round(r['frequency'][i], 2) for i in top])

# Change points: ~1000
print('변화점:', facade.run('change_points', signal=sig,
       n_changepoints=1, method='binary')['changepoints'])

# Anomaly: spike → crest factor 큼
print('crest_factor:', round(facade.run('ts_features', signal=sig)['features']['crest_factor'], 2))
"
```

기대:
```
PSD 상위 피크: [5.0, 17.0, ...]
변화점: [995] (또는 1000 근처)
crest_factor: > 5
```

이 결과를 GUI Post-Tools에서 동일 op + 같은 신호로 실행해 차트가 일치하는지 비교.

---

## 시나리오 4 — 회귀 자동 검증

직접 클릭 대신 자동:

```bash
QT_QPA_PLATFORM=offscreen pytest \
  tests/test_postproc_*.py \
  tests/test_main_window_*.py \
  tests/test_post_process_*.py \
  -q
```

기대: **180+ tests pass, 0 failed**.

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| 3D 뷰어가 정적 이미지로 보임 | headless/offscreen fallback 상태. 실제 마우스 회전은 display 환경 + `pyvistaqt` 필요 |
| 3D 뷰어 비어있음 | `pip install pyvista pyvistaqt vtk` 후 실제 display에서 실행 |
| AE/VAE 비활성 | `pip install torch` |
| FNO/DeepONet 학습 안 됨 | PyTorch CUDA 없으면 CPU (시간 걸림) |
| Post-Tools 차트 안 보임 | `pip install matplotlib` |
| WNO 결과 skip | `pip install pywavelets` |
| OpenFOAM 직접 로드 실패 | `.foam` 더미 파일 필요 — 케이스 디렉토리에 빈 파일 생성 |

---

## 검증 체크리스트 (요약)

- [ ] `scripts/make_test_dataset.py` 실행 → 3 파일 생성
- [ ] GUI 실행, 10개 탭 모두 표시 확인
- [ ] ① Import → cavity.vtu 로드 → 400 pts, 4 fields 표시
- [ ] ② Analyze → Q-criterion / FFT / y+ / 해석해 모두 실행
- [ ] ④ Model → Kriging 학습 → ⑦ Compare 자동 갱신
- [ ] ⑤ Twin → 예측 + 최적화 + HDF5 저장/로드
- [ ] ⑥ Export → VTK / CSV / 보고서 PDF
- [ ] ⑨ Explain → SHAP / Symbolic / Attention
- [ ] ⑩ Post-Tools → 25 dataset op 모두 통과 (라이브 검증 결과: **25/25 OK**)
- [ ] CSV/JSON/NPZ/PNG export 4종 모두 파일 생성
- [ ] 카테고리 일괄 실행 markdown 요약 표시
- [ ] 사용자 프리셋 저장 → 재선택 시 폼 복원
- [ ] 이력 보기 다이얼로그 → 더블클릭 재실행
- [ ] 메뉴 View → English 전환 → 모든 라벨 영어
