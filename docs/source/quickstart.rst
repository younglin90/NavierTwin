빠른 시작 (Quickstart)
========================

최소 예제: POD + Kriging 디지털 트윈
-------------------------------------

.. code-block:: python

   import numpy as np
   from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

   rng = np.random.default_rng(0)
   X = rng.standard_normal((100, 30))  # (n_features, n_snapshots)

   pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=5, surrogate_kind="kriging")
   pipe.load_snapshots(X, field_name="U")
   pipe.reduce()
   params = np.linspace(0, 1, 30).reshape(-1, 1)
   pipe.fit_surrogate(params)
   metrics = pipe.validate(params[-8:], pipe.state.coeffs[-8:])
   pipe.export_report("report.html", project="Demo")

Burgers + FNO 연산자 학습
-------------------------

.. code-block:: python

   from naviertwin.core.digital_twin.pipeline_neural import NeuralOperatorPipeline
   from naviertwin.core.benchmarks.dataset_catalog import generate_burgers_dataset

   data = generate_burgers_dataset(n_samples=30, n_x=64, seed=0)
   X = data["snapshots"][:, :, None].astype("float32")
   pipe = NeuralOperatorPipeline(
       kind="fno1d", in_ch=1, out_ch=1,
       modes=8, width=16, n_layers=2, max_epochs=20,
   )
   pipe.fit(X, X)  # 자기회귀 예시

Streaming Digital Twin 실시간 동화
-----------------------------------

.. code-block:: python

   import numpy as np
   from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

   rng = np.random.default_rng(0)
   A = np.eye(3) * 0.95
   twin = StreamingDigitalTwin(
       state_dim=3, n_ensemble=40,
       model_fn=lambda x: A @ x,
       H=np.eye(3), R=0.01 * np.eye(3), rng=rng,
   )
   twin.initialize(rng.standard_normal((40, 3)))
   for _ in range(20):
       twin.step()
       twin.assimilate(np.zeros(3))
   est = twin.estimate()

GUI 실행
---------

.. code-block:: bash

   naviertwin --gui

REST 서버 실행
---------------

.. code-block:: bash

   naviertwin server --host 0.0.0.0 --port 8000
