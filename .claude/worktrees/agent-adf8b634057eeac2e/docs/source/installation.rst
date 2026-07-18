설치 (Installation)
====================

빠른 설치
---------

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[core,dev]"

선택 의존성
-----------

.. code-block:: bash

   pip install -e ".[full]"

``[full]`` 에는 gmsh, pymeshlab, dedalus, foamlib, pycgns, torch_geometric 등이 포함됩니다.

필수 vs 선택
------------

===============================  ========================================
필수 (``[core]``)                PySide6, pyvista, meshio, h5py, numpy,
                                 scipy, torch, scikit-learn, smt, SALib,
                                 jinja2, weasyprint, onnx, fastapi
선택 (``[full]``)                gmsh, pymeshlab, dedalus, foamlib,
                                 pyCGNS, torch_geometric, e3nn/escnn,
                                 deepxde, neuraloperator, pygmo, shap, pysr
개발 (``[dev]``)                 pytest, pytest-cov, ruff, isort, mypy
===============================  ========================================

Windows 설치 주의
-----------------

- CUDA PyTorch 사용 시 공식 wheel (``pip install torch --index-url https://download.pytorch.org/whl/cu121``)
- Gmsh 는 conda-forge 권장: ``conda install -c conda-forge gmsh``
- foamlib 는 OpenFOAM 이 PATH 에 있을 때 최대 활용 가능
