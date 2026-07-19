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
   naviertwin server --host 0.0.0.0 --port 8443 --workers 4 \
     --ssl-certfile server.crt --ssl-keyfile server.key

Expected: starts the FastAPI service. It exits with code 1 if ``uvicorn`` is not
installed. API keys, shared rate limiting, and request size limits use the
``NAVIERTWIN_API_*`` and ``NAVIERTWIN_RATE_LIMIT_*`` environment variables in
:doc:`api/api`. Enable ``--proxy-headers`` only for proxies listed by
``--forwarded-allow-ips``.

web
---

.. code-block:: bash

   naviertwin web --host 127.0.0.1 --port 8080 --no-browser

Expected: starts the trame-based browser GUI. It exits with code 1 if the
optional ``trame`` dependency is not installed.

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

batch-train
-----------

.. code-block:: bash

   naviertwin batch-train --config jobs.json --json
   mpirun -n 4 naviertwin batch-train --config jobs.json

Expected: runs the headless batch twin-training orchestrator. The JSON config
contains a ``jobs`` list where each job selects ``kind`` (``rom`` or
``physics``), a synthetic ``demo`` dataset (or a ``data_path`` input), the
target ``field``, and training sizes (``n_modes`` or ``epochs``). Jobs are
distributed round-robin as ``jobs[rank::size]`` across MPI ranks; without
``mpi4py`` or ``mpirun`` the command degrades gracefully to a sequential
rank 0 / size 1 run. Each rank writes ``batch_results_rank{rank}.json`` with
per-job ``name``/``status``/``rmse`` (or ``train_loss``)/``elapsed_s``
summaries, and when MPI is active rank 0 also gathers a merged
``batch_results.json``. MPI is initialized only on this headless path — never
inside the desktop or web GUI event loops. The command exits 1 if any job
fails and 2 on config or runtime errors.

launch-ddp
----------

.. code-block:: bash

   naviertwin launch-ddp --entrypoint train.py --nproc-per-node auto --dry-run -- --epochs 20

Expected: validates the distributed launch configuration and runs the Python
entrypoint through ``torchrun``. ``auto`` selects the visible CUDA device count,
falling back to one process on CPU-only hosts. ``--dry-run`` prints the exact
command without starting workers. Multi-node launches use ``--nnodes``,
``--node-rank``, ``--master-addr``, and ``--master-port``.

plan-scale
----------

.. code-block:: bash

   naviertwin plan-scale --cases 12 --time-steps 40 --points 2000000 --cells 1800000 --route route2 --workers 4 --ram-gb 32 --vram-gb 12 --json

Expected: estimates total CFD bytes separately from bounded host/device working
sets, then reports point chunk size, case batch size, GPU microbatch, gradient
accumulation, and MPI ranks. Omit ``--ram-gb``/``--vram-gb`` to inspect current
machine resources. The command exits 1 when even the minimum point chunk cannot
fit and 2 for invalid resource or workload values.

build-twin
----------

.. code-block:: bash

   naviertwin build-twin --csv-snapshots "case/snapshots/*.csv" --field-column U --outdir /tmp/naviertwin-twin --json

Expected: loads a CFD reader input or CSV snapshot sequence, trains a
``NavierTwinPipeline`` with a validation split, and writes ``metrics.json``,
``manifest.json``, ``pipeline.h5``, loadable ``engine.pkl``, and ``report.html``.
The manifest records artifact bytes/SHA256 hashes and a ``parameter_contract``
with expected input dimension, parameter names, and observed training ranges.

predict-twin
------------

.. code-block:: bash

   naviertwin predict-twin --engine /tmp/naviertwin-twin/engine.pkl --params 0.25 --output /tmp/naviertwin-prediction.csv --json
   naviertwin predict-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --output /tmp/naviertwin-prediction.csv --json

Expected: loads a saved ``TwinEngine`` directly or from an extracted artifact
directory containing ``engine.pkl``, evaluates the input parameters, and prints
prediction shape/preview metadata while optionally writing the predicted field
matrix to CSV. If ``manifest.json`` beside ``engine.pkl`` includes a
``parameter_contract``, input width is checked before prediction so customer
integration errors fail with a clear dimension mismatch.

benchmark-twin
---------------

.. code-block:: bash

   naviertwin benchmark-twin --artifacts-dir /tmp/naviertwin-deploy --params 0.25 --warmup 2 --repeat 20 --max-p95-ms 100 --min-throughput-hz 10 --output /tmp/naviertwin-latency.json --json

Expected: runs warmup predictions, measures repeated prediction latency, and
reports min/mean/p50/p95/p99/max milliseconds plus approximate throughput. When
``--max-mean-ms``, ``--max-p50-ms``, ``--max-p95-ms``, ``--max-p99-ms``, or
``--min-throughput-hz`` is set, the JSON includes an ``acceptance`` block and
the command exits 1 if the delivered twin misses the configured performance SLO.
The same optional ``parameter_contract`` preflight used by ``predict-twin`` runs
before latency measurement.

validate-twin
-------------

.. code-block:: bash

   naviertwin validate-twin --engine /tmp/naviertwin-twin/engine.pkl --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json
   naviertwin validate-twin --artifacts-dir /tmp/naviertwin-deploy --csv-snapshots "case/snapshots/*.csv" --field-column U --max-rmse 0.05 --min-r2 0.98 --output /tmp/naviertwin-validation.json --json

Expected: reloads a saved ``TwinEngine`` directly or from an artifact directory,
predicts every supplied validation parameter row, compares the reconstructed
field against CFD/CSV reference snapshots, writes RMSE/R²/relative-L2/max-error
metrics, and exits non-zero when configured acceptance thresholds fail.

package-twin
------------

.. code-block:: bash

   naviertwin package-twin --artifacts-dir /tmp/naviertwin-twin --include-validation /tmp/naviertwin-validation.json --output /tmp/naviertwin-twin.zip --max-p95-ms 100 --min-throughput-hz 10 --json

Expected: packages ``engine.pkl``, ``manifest.json``, ``metrics.json``,
``pipeline.h5``, ``report.html``, and optional validation JSON into a delivery
ZIP. The archive also includes ``README.txt`` for customer handoff instructions
and ``delivery.json`` for machine-readable package metadata. ``MANIFEST.json``
contains bytes and SHA256 hashes for every archived entry. Before packaging,
``manifest.json`` integrity records are checked against the current files so
tampered build artifacts fail fast. The delivery metadata also echoes the
parameter contract so recipients can inspect expected input names/ranges without
loading Python code. When a contract with names is available, the ZIP includes
``sample_params.csv`` and README/delivery commands use ``--params-csv`` plus
``--param-columns`` so multi-parameter twins have copy-pasteable inputs.
``delivery.json`` also records an optional ``latency_slo`` policy. The default
package command writes p95 <= 100 ms and throughput >= 10 Hz unless
``--no-latency-slo`` is used; explicit packaging flags can override those
handoff criteria.

inspect-twin-package
--------------------

.. code-block:: bash

   naviertwin inspect-twin-package --package /tmp/naviertwin-twin.zip --json

Expected: reads the delivery ZIP without extracting it, verifies archive
integrity, and reports ``delivery.json`` metadata such as package format,
build metrics, packaged files, generated entries, available commands, and
whether validation data/README are present. Newer packages also expose
``parameter_contract``; older packages without it remain inspectable.

verify-twin-package
-------------------

.. code-block:: bash

   naviertwin verify-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-deploy --json

Expected: reads the delivery ZIP ``MANIFEST.json`` and verifies each archived
entry's bytes/SHA256. It also checks that ``engine.pkl`` and ``manifest.json``
are covered by the archive manifest. Duplicate entries, unsafe archive paths,
unmanifested files, and integrity mismatches exit non-zero. When ``--extract-to``
is provided, extraction happens only after verification succeeds and the target
directory is new or empty.

accept-twin-package
-------------------

.. code-block:: bash

   naviertwin accept-twin-package --package /tmp/naviertwin-twin.zip --extract-to /tmp/naviertwin-accepted --output /tmp/naviertwin-acceptance.json --summary-output /tmp/naviertwin-acceptance.md --json

Expected: runs the customer handoff acceptance smoke in one command. It verifies
``MANIFEST.json`` integrity, safely extracts the ZIP, inspects delivery
metadata, loads ``sample_params.csv`` or a contract-derived example input, runs
a sample prediction, then benchmarks prediction latency with the configured SLO
thresholds. If ``delivery.json`` contains ``latency_slo``, those thresholds are
used by default; explicit ``--max-*`` or ``--min-throughput-hz`` flags override
package policy. Older packages without policy remain valid and only gate
latency when thresholds are passed on the CLI. Package or prediction failures
exit 1, SLO misses also exit 1, and runtime/setup errors exit 2. The JSON report
contains ``verification``, ``inspection``, ``prediction``, ``benchmark``, and
top-level ``acceptance`` blocks that can be attached to customer delivery
records. ``--summary-output`` writes a Markdown checklist with the same verdict,
prediction shape, effective SLO, latency statistics, and pass/fail rows for
handoff reviews that do not require reading JSON.

preflight
---------

.. code-block:: bash

   naviertwin preflight tests/fixtures/tiny_square.su2 --json --output /tmp/naviertwin-preflight.json

Expected: prints and optionally writes a readiness JSON report. Missing input
paths return code 1 with ``status: error``.

support-bundle
--------------

.. code-block:: bash

   naviertwin support-bundle --outdir /tmp/naviertwin-support --preflight tests/fixtures/tiny_square.su2 --acceptance-json /tmp/naviertwin-acceptance.json --acceptance-summary /tmp/naviertwin-acceptance.md --zip

Expected: writes ``doctor.json``, optional ``preflight.json``, optional
``acceptance.json``/``acceptance.md``, human-readable ``README.txt``,
``metadata.json``, and when ``--zip`` is used, ``support-bundle.zip`` with
``MANIFEST.json``. Acceptance artifacts and the read-first summary are redacted
before being copied into the bundle. ``metadata.json`` records schema version,
input presence, file suffixes, the ZIP filename, and non-reversible path
hashes, not customer absolute paths.

inspect-support-bundle
----------------------

.. code-block:: bash

   naviertwin inspect-support-bundle /tmp/naviertwin-support/support-bundle.zip --json

Expected: reads an existing support-bundle directory or ZIP without extracting
it, verifies ``metadata.json`` artifact hashes, verifies ZIP ``MANIFEST.json``
when present, and prints a summary for support triage. The command exits non-zero
when required files are missing or bytes/SHA256 do not match.

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
   naviertwin update-check --metadata examples/release-metadata.example.json --verify-artifact NavierTwinSetup.exe

Expected: verifies signed local release metadata and reports whether an update
is available for the selected channel. The JSON includes the validated
installer ``url`` and ``sha256`` so support or the GUI can hand off the exact
``NavierTwinSetup.exe`` download without implementing an in-app self-updater.
With ``--verify-artifact``, the downloaded installer is hashed locally and
compared against the signed metadata SHA-256 before the customer runs it.
If the signed metadata includes ``installer_signing`` identity fields, the JSON
also reports Windows Authenticode publisher/thumbprint verification status
(``unavailable`` on non-Windows smoke environments).
Release maintainers can generate the signed metadata with
``python scripts/sign_release_metadata.py --input release-unsigned.json --output release.json --key-id naviertwin-release-2026q2``.
The Ed25519 private key is read from ``NAVIER_TWIN_RELEASE_PRIVATE_KEY_B64`` or
``--private-key-file`` and is never stored in the repository.

feature-pack
------------

.. code-block:: bash

   naviertwin feature-pack list --json
   naviertwin feature-pack download --pack gpu-extras --install
   naviertwin feature-pack install --archive naviertwin-featurepack-gpu-extras.zip

Expected: lists, downloads, or installs large optional feature packs. The
``install`` action validates the archive layout (and an optional SHA256)
before activating the pack for subsequent runs.

doctor
------

.. code-block:: bash

   naviertwin doctor --json --output /tmp/naviertwin-doctor.json

Expected: prints and optionally writes an environment diagnostic report.
``status`` may be ``warn`` on machines without CUDA.
