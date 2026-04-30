CLI Reference
=============

NavierTwin의 고객-facing CLI는 GUI 실행, 데이터 readiness 점검, 진단 번들 생성,
릴리스 검증, API 서버 실행을 같은 entry point에서 제공합니다.

Version
-------

.. code-block:: bash

   naviertwin --version

Expected: exits with code 0 and prints the installed package version, for example
``naviertwin X.Y.Z``.

benchmark
---------

.. code-block:: bash

   naviertwin benchmark --kind burgers

Expected: runs the selected local benchmark script. It exits non-zero if the
requested benchmark script is unavailable.

server
------

.. code-block:: bash

   naviertwin server --host 0.0.0.0 --port 8000

Expected: starts the FastAPI service. It exits with code 1 if ``uvicorn`` is not
installed.

pipeline
--------

.. code-block:: bash

   naviertwin pipeline --reducer pod --n-modes 5 --surrogate kriging

Expected: runs a synthetic end-to-end pipeline and prints a JSON payload with
``status`` and validation metrics.

pipeline-demo
-------------

.. code-block:: bash

   naviertwin pipeline-demo --outdir /tmp/naviertwin-pipeline-demo

Expected: writes ``metrics.json`` and ``report.html`` to the output directory.
It returns code 2 for a clean dependency/runtime error rather than a traceback.

model-sweep
-----------

.. code-block:: bash

   naviertwin model-sweep --reducers pod --n-modes 2,3,5 --surrogates rbf,kriging --json

Expected: evaluates the configured ROM/surrogate candidates on the same
synthetic CFD-like snapshot set, sorts them by validation RMSE, and prints a
ranked table or JSON payload with ``best`` and ``rows``.

build-twin
----------

.. code-block:: bash

   naviertwin build-twin --csv-snapshots "case/snapshots/*.csv" --field-column U --outdir /tmp/naviertwin-twin --json

Expected: loads a CFD reader input or CSV snapshot sequence, trains a
``NavierTwinPipeline`` with a validation split, and writes ``metrics.json``,
``manifest.json``, ``pipeline.h5``, loadable ``engine.pkl``, and ``report.html``.
The manifest records artifact bytes and SHA256 hashes for delivery auditing.

predict-twin
------------

.. code-block:: bash

   naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json

Expected: loads a saved ``TwinEngine``, evaluates the input parameters, and
prints prediction shape/preview metadata while optionally writing the predicted
field matrix to CSV.

validate-twin
-------------

.. code-block:: bash

   naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json

Expected: reloads a saved ``TwinEngine``, predicts every supplied validation
parameter row, compares the reconstructed field against CFD/CSV reference
snapshots, writes RMSE/R²/relative-L2/max-error metrics, and exits non-zero when
configured acceptance thresholds fail.

package-twin
------------

.. code-block:: bash

   naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --json

Expected: packages ``engine.pkl``, ``manifest.json``, ``metrics.json``,
``pipeline.h5``, ``report.html``, and optional validation JSON into a delivery
ZIP with an archive ``MANIFEST.json`` containing bytes and SHA256 hashes.
Before packaging, ``manifest.json`` integrity records are checked against the
current files so tampered build artifacts fail fast.

preflight
---------

.. code-block:: bash

   naviertwin preflight tests/fixtures/tiny_square.su2 --json --output /tmp/naviertwin-preflight.json

Expected: prints and optionally writes a readiness JSON report. Missing input
paths return code 1 with ``status: error``.

support-bundle
--------------

.. code-block:: bash

   naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2 --zip

Expected: writes ``doctor.json``, optional ``preflight.json``, ``metadata.json``,
and when ``--zip`` is used, ``support-bundle.zip`` with ``MANIFEST.json``.

autorefine
----------

.. code-block:: bash

   naviertwin autorefine --iterations 1 --dry-run

Expected: analyzes the project roadmap and emits an automation report without
modifying files in dry-run mode.

update-check
------------

.. code-block:: bash

   naviertwin update-check --metadata examples/release-metadata.example.json

Expected: reads local release metadata and reports whether an update is
available for the selected channel.

doctor
------

.. code-block:: bash

   naviertwin doctor --json --output /tmp/naviertwin-doctor.json

Expected: prints and optionally writes an environment diagnostic report.
``status`` may be ``warn`` on machines without CUDA.
