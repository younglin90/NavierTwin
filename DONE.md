# DONE — 자율 루프 목표 달성 (2026-07-02)

## 목표
웹 GUI(trame)에서 **CFD 벤치마크 데이터를 손쉽게 로드하고, 학습 라이브러리(신경 연산자)로
학습시켜 빠른 예측 결과**를 얻을 수 있다.

## 완료 판단 (최종 실행 결과)
- `pytest tests/test_web_*.py` — **74개 전부 통과** (service 37 / app 18 / async 3 / bench 16)
- `ruff check src/naviertwin/web/` — 통과
- `naviertwin web` 서버 기동 — HTTP 200, 로그 에러 0

## 반복 이력

### 반복 1 — AI Bench 패널 (1D 벤치마크 + PDEBench + FNO1D)
- `src/naviertwin/web/bench.py` 신설 (Qt/GL 비의존 서비스 계층):
  - `generate_operator_dataset("burgers"|"heat")` — core 솔버(`solve_burgers_1d`/`solve_heat_1d`)로
    (u0 → uT) 연산자 학습 쌍 즉석 생성 (다운로드 불필요)
  - `load_pdebench_hdf5(path)` — **PDEBench(NeurIPS 2022) 포맷 HDF5** 직접 로드
    (`tensor` (B,T,N) 또는 임의 3D float dataset, h5py 만 필요 — pdebench 패키지 불필요)
  - `train_operator` — `core.operator_learning.fno.FNO1D` 학습, train/test 분할,
    RMSE·상대 L2·**단일 예측 지연시간(ms)** 측정, epoch 손실 이력 반환
  - `evaluate_sample` — 샘플별 참값 vs 예측 + 지연시간
- `app.py` — "⑧ AI Bench" 패널: 데이터셋 생성/로드 → FNO 학습(손실곡선 차트 모달) →
  샘플 비교 차트. 모든 무거운 연산은 기존 비동기 래퍼(진행바, 비차단)로 실행.

### 반복 2 — 2D 실유동 확장 (Cavity LBM + FNO2D)
- `generate_operator_dataset("cavity2d")` — 리드 드리븐 캐비티 **LBM 시계열**에서
  (u(t) → u(t+Δt)) 시간전진 페어 생성, 채널 [ux, uy], shape (B, ny, nx, 2)
- `train_operator(operator="auto")` — 입력 차원으로 FNO1D(3D)/FNO2D(4D) 자동 선택
- 2D 평가 차트: true |u| / FNO pred / |error| 3패널 imshow

### 반복 3 — 문서화·봉인
- README 웹 GUI/AI Bench 항목 갱신, 프로젝트 메모리 갱신, 본 문서 작성

## 변경 파일
- 신규: `src/naviertwin/web/bench.py`, `tests/test_web_bench.py`
- 수정: `src/naviertwin/web/app.py` (⑧ 패널 + 콜백 + 차트 2종), `README.md`

## 측정된 "빠른 결과" (WSL, CPU)
- FNO1D (heat, N=32): 추론 ~2.6 ms/샘플
- FNO2D (cavity 24×24×2ch): 추론 ~2.7–4.8 ms/샘플, 손실 4.5e-3 → 1.7e-4 (6 epochs)

## 후속 후보 (범위 외 기록)
- AirfRANS/CFDBench 로더 — pip 패키지 설치·수 GB 다운로드가 필요해 자율 범위에서 제외
  (환경 변경은 사용자 승인 사안). `pip install airfrans` 후 로더 추가는 소규모 작업.
- 학습된 연산자 저장/재사용 (torch state_dict → Export 패널 통합)
- DeepONet/TFNO 등 추가 연산자 선택지
