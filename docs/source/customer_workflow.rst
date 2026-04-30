Customer Workflow
=================

목표
----

CFD 결과 snapshot에서 고객 전달 가능한 디지털 트윈 ZIP까지 만드는 표준 절차입니다.
steady-state, unsteady, parametric sweep 결과 모두 같은 흐름으로 취급합니다.

1. 입력 데이터 readiness 확인
-----------------------------

.. code-block:: bash

   naviertwin preflight case/reference.su2 --json --output /tmp/naviertwin-preflight.json

기대 결과: 지원 포맷, mesh 크기, field 목록, NaN/Inf 여부, readiness score를 확인합니다.

2. ROM/surrogate 후보 비교
--------------------------

.. code-block:: bash

   naviertwin model-sweep --reducers pod --n-modes 2,3,5 --surrogates rbf,kriging --json

기대 결과: RMSE 기준 랭킹으로 초기 모델 family와 mode 수를 좁힙니다.

3. CFD/CSV snapshot에서 트윈 생성
--------------------------------

.. code-block:: bash

   naviertwin build-twin --csv-snapshots "case/snapshots/*.csv" --field-column U --outdir /tmp/naviertwin-twin --n-modes 3 --surrogate rbf --json

기대 결과: ``engine.pkl``, ``pipeline.h5``, ``metrics.json``, ``manifest.json``,
``report.html`` 이 생성됩니다. ``manifest.json`` 은 주요 artifact bytes와 SHA256을
포함해 납품 파일 무결성을 추적합니다.

4. 저장된 트윈 예측
-------------------

.. code-block:: bash

   naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json

기대 결과: 입력 파라미터에서 거의 실시간으로 복원된 field vector를 CSV로 저장합니다.

5. 독립 검증 및 acceptance gate
------------------------------

.. code-block:: bash

   naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/validation/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json

기대 결과: RMSE, R², relative L2, max error와 worst-sample 정보를 기록합니다.
설정한 threshold를 넘기면 종료 코드 1로 실패하므로 CI나 납품 승인 게이트에 사용할 수 있습니다.

6. 고객 전달 ZIP 생성
---------------------

.. code-block:: bash

   naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --json

기대 결과: 트윈 산출물과 validation report를 하나의 ZIP으로 묶고, ZIP 내부
``MANIFEST.json`` 에 각 파일의 bytes와 SHA256을 기록합니다.

7. 전달 ZIP 무결성 확인
-----------------------

.. code-block:: bash

   naviertwin verify-twin-package --package /tmp/naviertwin-twin.zip --json

기대 결과: 고객에게 전달하기 전이나 전달받은 뒤 ZIP 내부 ``MANIFEST.json`` 과
실제 archive entry bytes/SHA256을 다시 대조합니다.

GUI 대응 흐름
-------------

동일한 작업은 데스크톱 GUI의 **도구** 메뉴에서도 실행할 수 있습니다.

- **CSV 스냅샷으로 트윈 생성**: CSV snapshot 선택 → field 컬럼 입력 → 산출물 폴더 선택
- **저장된 트윈 예측**: ``engine.pkl`` 선택 → 파라미터 입력 → 예측 CSV 저장
- **저장된 트윈 검증**: ``engine.pkl`` 선택 → 검증 CSV snapshot 선택 → validation JSON 저장
- **트윈 산출물 패키징**: build-twin 산출물 폴더 선택 → 고객 전달 ZIP 저장
- **트윈 패키지 검증**: 고객 전달 ZIP 선택 → archive manifest 무결성 확인
