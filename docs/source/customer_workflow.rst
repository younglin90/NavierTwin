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
포함해 납품 파일 무결성을 추적하고, ``parameter_contract`` 로 입력 파라미터
차원/이름/학습 데이터 관측 범위를 기록합니다.

4. 저장된 트윈 예측
-------------------

.. code-block:: bash

   naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json
   naviertwin predict-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --output /tmp/naviertwin-prediction.csv --json

기대 결과: ``engine.pkl`` 직접 경로 또는 검증/추출된 artifact 디렉토리에서
입력 파라미터에 대한 field vector를 거의 실시간으로 복원해 CSV로 저장합니다.
artifact에 ``parameter_contract`` 가 있으면 예측 전에 입력 차원을 선제 검증합니다.

5. 지연시간 벤치마크
--------------------

.. code-block:: bash

   naviertwin benchmark-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --warmup 2 --repeat 20 --max-p95-ms 100 --min-throughput-hz 10 --output /tmp/naviertwin-latency.json --json

기대 결과: 고객 PC에서 반복 예측의 min/mean/p50/p95/p99/max latency와 throughput을 기록합니다.
설정한 p95/throughput SLO를 만족하지 못하면 종료 코드 1로 실패하므로 납품 성능 게이트에 사용할 수 있습니다.
예측과 동일한 입력 contract preflight가 latency 측정 전에 실행됩니다.

6. 독립 검증 및 acceptance gate
------------------------------

.. code-block:: bash

   naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/validation/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json
   naviertwin validate-twin --artifacts-dir /tmp/naviertwin-deploy --csv-snapshots "case/validation/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json

기대 결과: RMSE, R², relative L2, max error와 worst-sample 정보를 기록합니다.
설정한 threshold를 넘기면 종료 코드 1로 실패하므로 CI나 납품 승인 게이트에 사용할 수 있습니다.

7. 고객 전달 ZIP 생성
---------------------

.. code-block:: bash

   naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --max-p95-ms 100 --min-throughput-hz 10 --json

기대 결과: 트윈 산출물과 validation report를 하나의 ZIP으로 묶고, ZIP 내부
``README.txt`` 에 고객 실행 안내를, ``delivery.json`` 에 기계 판독용 메타데이터를,
``MANIFEST.json`` 에 각 파일의 bytes와 SHA256을 기록합니다. 전달 metadata에는
고객이 예상 입력 파라미터를 확인할 수 있도록 ``parameter_contract`` 도 포함됩니다.
contract에 파라미터 이름이 있으면 ``sample_params.csv`` 도 함께 생성되어
README/delivery 명령이 다차원 입력에서도 그대로 실행됩니다.
``latency_slo`` 정책도 함께 기록되어 고객 수락 검사에서 기본 성능 기준으로
사용됩니다. 기준을 넣고 싶지 않으면 ``--no-latency-slo`` 를 사용합니다.

8. 전달 ZIP 구성 조회
---------------------

.. code-block:: bash

   naviertwin inspect-twin-package --package /tmp/naviertwin-twin.zip --json

기대 결과: ZIP을 추출하지 않고 무결성 검증 결과, build metric, 포함 파일,
validation 포함 여부, README/delivery metadata 존재 여부와 ``parameter_contract`` 를 조회합니다.

9. 전달 ZIP 무결성 확인
-----------------------

.. code-block:: bash

   naviertwin verify-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-deploy --json

기대 결과: 고객에게 전달하기 전이나 전달받은 뒤 ZIP 내부 ``MANIFEST.json`` 과
실제 archive entry bytes/SHA256을 다시 대조합니다. ``--extract-to`` 를 지정하면
검증을 통과한 경우에만 새 디렉토리 또는 빈 디렉토리로 안전하게 추출합니다.

10. 전달 ZIP 원샷 acceptance smoke
---------------------------------

.. code-block:: bash

   naviertwin accept-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-accepted --output /tmp/naviertwin-acceptance.json --summary-output /tmp/naviertwin-acceptance.md --json

기대 결과: 고객이 받은 ZIP 하나로 무결성 검증, 안전 추출, delivery metadata
조회, ``sample_params.csv`` 기반 샘플 예측, latency SLO 측정을 한 번에 수행합니다.
``delivery.json`` 의 ``latency_slo`` 가 기본 기준으로 적용되고, 필요하면
``--max-p95-ms`` 또는 ``--min-throughput-hz`` 같은 CLI threshold로 override할 수
있습니다. 기준이 실패하면 종료 코드 1로 실패하므로 납품 승인 acceptance gate에
바로 연결할 수 있습니다. ``--summary-output`` 을 지정하면 JSON을 열지 않아도
통과/실패, 예측 shape, latency 통계, SLO check를 확인할 수 있는 Markdown
수락 요약 리포트도 생성됩니다.

11. 고객 지원 번들에 acceptance evidence 첨부
------------------------------------------

.. code-block:: bash

   naviertwin support-bundle --outdir /tmp/naviertwin-support --acceptance-json /tmp/naviertwin-acceptance.json --acceptance-summary /tmp/naviertwin-acceptance.md --zip

기대 결과: 환경 진단, optional preflight, acceptance JSON/Markdown 요약을
redaction 후 하나의 ``support-bundle.zip`` 으로 묶습니다. 고객이 실패한
acceptance 결과를 그대로 첨부해 보내면 개발/지원팀은 재현 정보, SLO 실패 항목,
패키지 metadata를 한 번에 확인할 수 있습니다. 번들 루트의 ``README.txt`` 는
상태, 포함 파일, 경고/오류, 먼저 열 파일을 요약합니다. ``metadata.json`` 은
고객 절대경로 대신 schema version, 입력 제공 여부, 파일 확장자, 비가역 경로
해시만 기록합니다.

12. 고객 지원 번들 수신 검증
--------------------------

.. code-block:: bash

   naviertwin inspect-support-bundle /tmp/naviertwin-support/support-bundle.zip --json

기대 결과: 받은 support bundle ZIP 또는 디렉토리를 추출하지 않고 읽어
``metadata.json`` 과 ``MANIFEST.json`` 의 bytes/SHA256을 대조합니다. 누락 파일,
변조 파일, manifest 불일치가 있으면 non-zero로 실패하므로 지원 티켓 접수 시
첫 단계 검증으로 사용할 수 있습니다.

GUI 대응 흐름
-------------

동일한 작업은 데스크톱 GUI의 **도구** 메뉴에서도 실행할 수 있습니다.

- **CSV 스냅샷으로 트윈 생성**: CSV snapshot 선택 → field 컬럼 입력 → 산출물 폴더 선택
- **저장된 트윈 예측**: ``engine.pkl`` 선택 → 파라미터 입력 → 예측 CSV 저장
- **배포 트윈 디렉토리 예측**: 검증/추출된 artifact 디렉토리 선택 → 파라미터 입력 → 예측 CSV 저장
- **배포 트윈 지연시간 측정**: 검증/추출된 artifact 디렉토리 선택 → 파라미터/반복 횟수 입력 → latency JSON 저장
- **저장된 트윈 검증**: ``engine.pkl`` 선택 → 검증 CSV snapshot 선택 → validation JSON 저장
- **배포 트윈 디렉토리 검증**: 검증/추출된 artifact 디렉토리 선택 → 검증 CSV snapshot 선택 → validation JSON 저장
- **트윈 산출물 패키징**: build-twin 산출물 폴더 선택 → 고객 전달 ZIP 저장
- **트윈 패키지 정보 보기**: 고객 전달 ZIP 선택 → delivery metadata와 build metric 확인
- **트윈 패키지 검증**: 고객 전달 ZIP 선택 → archive manifest 무결성 확인
- **트윈 패키지 검증 후 추출**: 고객 전달 ZIP 선택 → 배포 디렉토리 선택 → 검증 성공 시 안전 추출
- **트윈 패키지 원샷 수락 검사**: 고객 전달 ZIP 선택 → 추출 디렉토리/SLO 입력 → 검증, 샘플 예측, latency gate, acceptance JSON/Markdown 저장
- **지원 번들 생성**: 최근 또는 수동 선택한 acceptance JSON/Markdown과 현재 Import 탭 preflight 대상을 포함해 고객 지원 ZIP 생성
