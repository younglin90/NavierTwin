naviertwin.core.post_process_facade
===================================

PostProcessFacade는 GUI/CLI/스크립트에서 같은 이름으로 호출할 수 있는
상용 후처리 parity API입니다. ``list_operations()`` 로 사용 가능한 연산을
나열하고, ``describe(op_name)`` 으로 파라미터/반환값을 확인한 뒤,
``run(op_name, **kwargs)`` 로 실행합니다.

대표 기능 범위
--------------

- 통계/난류: Reynolds stats, quadrant analysis, running moments, quantile,
  statistical convergence, anisotropy state
- 스펙트럼/ROM: Welch PSD, Kolmogorov slope, EOF, POD truncation
- 샘플링/전처리: line/plane probes, denoise, time interpolation,
  safe expression evaluation, coordinate transform
- 적분/기하: surface forces, plane flux, tetrahedral cell volume integrals
- 이상/위상/위상학: change points, Mahalanobis anomaly, critical points,
  morphology components, goodness-of-fit diagnostics

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from naviertwin.core.post_process_facade import PostProcessFacade

   facade = PostProcessFacade()
   assert "psd_welch" in facade.list_operations()
   info = facade.describe("psd_welch")
   result = facade.run(
       "psd_welch",
       signal=np.sin(np.linspace(0, 4 * np.pi, 200)),
       fs=100.0,
   )

   print(info["category"], result["frequency"].shape, result["psd"].shape)

API
---

.. automodule:: naviertwin.core.post_process_facade
   :members:
   :undoc-members:
   :show-inheritance:
