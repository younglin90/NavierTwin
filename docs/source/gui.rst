GUI 사용법
===========

10 개 탭 구조
------------

.. list-table::
   :header-rows: 1

   * - 탭
     - 기능
   * - ① Import
     - CFD 파일 읽기, 포맷 자동 감지, 타임스텝/필드 선택
   * - ② Analyze
     - Q-criterion, λ₂, FFT/PSD, y+, 해석해 비교, SPOD, SINDy, Wavelet/STFT, BL/nondim/FTLE/PGD/entropy quick checks
   * - ③ Reduce
     - POD / AE / VAE / GNN-AE 차원축소 + 에너지 곡선
   * - ④ Model
     - Kriging / RBF / FNO / DeepONet / UNet / WNO / TFNO 학습, Active Learning 후보 추천, Online Update
   * - ⑤ Twin
     - 파라미터 → 실시간 예측, TwinEngine 저장/로드, inverse-design 최적화, NSGA-II/SIMP 설계 quick-check, 4D-Var/PF/UKF 동화 quick-check
   * - ⑥ Export
     - .ntwin / VTK / CSV / 고객 보고서 / ONNX / TorchScript 모델 아티팩트
   * - ⑦ Compare
     - 여러 모델 RMSE/R² 바 차트
   * - ⑧ Simulation
     - LBM cavity / StreamingTwin / RL / Burgers FVM 런처
   * - ⑨ Explain
     - Kernel SHAP feature attribution, symbolic expression fit, Attention weight matrix/top-k token viz
   * - ⑩ Post-Tools
     - 29개 상용 후처리 parity facade: 통계, PSD, 적분, flux, interpolation, derivatives, EOF, topology 등 실행

단축키
-------

- ``Ctrl+O`` — 파일 열기
- ``Ctrl+S`` — 프로젝트 저장 (.ntwin)
- ``Ctrl+Q`` — 종료
- ``Ctrl+1..9`` — 상위 9개 탭 전환
- **보기** 메뉴 — 모든 탭 전환, 다크/라이트 테마, 한국어/영어 전환

도구 메뉴
---------

- **벤치마크 실행** — Burgers benchmark smoke를 GUI에서 실행
- **파이프라인 데모 실행** — 합성 end-to-end 데모 산출물과 HTML 보고서 생성
- **CSV 스냅샷으로 트윈 생성** — CSV 시퀀스에서 ``engine.pkl``/metrics/report 산출물 생성 후 Twin/Export 패널에 자동 연결
- **저장된 트윈 예측** — ``engine.pkl`` 과 쉼표 구분 파라미터로 예측을 실행하고 CSV로 저장
- **배포 트윈 디렉토리 예측** — 검증/추출된 산출물 디렉토리에서 ``engine.pkl`` 을 찾아 예측 실행
- **배포 트윈 지연시간 측정** — 단일 파라미터 또는 CSV 배치 입력으로 p95/throughput SLO를 측정
- **저장된 트윈 검증** — ``engine.pkl`` 예측장을 기준 CSV snapshot과 비교하고, RMSE/R²/relative-L2 기준값으로 acceptance gate 실행
- **배포 트윈 디렉토리 검증** — 검증/추출된 산출물 디렉토리를 기준 snapshot과 비교
- **트윈 산출물 패키징** — build/validate 산출물을 고객 전달용 ZIP과 무결성 manifest로 묶고, 별도 validation JSON 포함 가능
- **트윈 패키지 정보 보기** — ZIP을 추출하지 않고 delivery metadata, build metric, parameter contract를 조회
- **트윈 패키지 검증** — 고객 전달 ZIP 내부 ``MANIFEST.json`` 의 bytes/SHA256을 다시 대조
- **트윈 패키지 검증 후 추출** — 검증 성공 시 새 디렉토리 또는 빈 디렉토리로 안전 추출
- **트윈 패키지 원샷 수락 검사** — ZIP 검증, 안전 추출, ``sample_params.csv`` 예측, ``delivery.json`` latency SLO gate, acceptance JSON/Markdown 저장을 한 번에 실행
- **지원 번들 생성** — 환경 진단, 현재 Import 탭 preflight, 최근 또는 수동 선택한 acceptance JSON/Markdown을 redaction 후 고객 지원 ZIP으로 묶음
- **API 서버 시작/중지** — FastAPI 서버를 GUI 백그라운드 프로세스로 관리

언어 전환
---------

메뉴 **보기 → 언어** 에서 한국어/영어 실시간 전환. ``utils/i18n.py`` 의 Translator 사용.
