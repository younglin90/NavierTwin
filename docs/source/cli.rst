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
