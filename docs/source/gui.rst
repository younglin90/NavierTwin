GUI 사용법
===========

8 개 탭 구조
------------

.. list-table::
   :header-rows: 1

   * - 탭
     - 기능
   * - ① Import
     - CFD 파일 읽기, 포맷 자동 감지, 타임스텝/필드 선택
   * - ② Analyze
     - Q-criterion, FFT/PSD, y+, 해석해 비교
   * - ③ Reduce
     - POD / AE / VAE / GNN-AE 차원축소 + 에너지 곡선
   * - ④ Model
     - Kriging / RBF / Ensemble / FNO / DeepONet / UNet / WNO / TFNO 학습
   * - ⑤ Twin
     - 파라미터 → 실시간 예측 + VTK 시각화
   * - ⑥ Export
     - ONNX / TorchScript / .ntwin / Jinja2 HTML 보고서
   * - ⑦ Compare
     - 여러 모델 RMSE/R² 바 차트
   * - ⑧ Simulation
     - LBM cavity / StreamingTwin / RL / Burgers FVM 런처

단축키
-------

- ``Ctrl+O`` — 파일 열기
- ``Ctrl+S`` — 프로젝트 저장 (.ntwin)
- ``Ctrl+Q`` — 종료
- ``Ctrl+1..6`` — 탭 전환

언어 전환
---------

메뉴 **보기 → 언어** 에서 한국어/영어 실시간 전환. ``utils/i18n.py`` 의 Translator 사용.
