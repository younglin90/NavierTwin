naviertwin.core.export
============================================================

Customer artifact export APIs for saving NavierTwin datasets and moving trained
models into deployment runtimes. The package root exposes the stable artifact
surface lazily so importing ``naviertwin.core.export`` does not eagerly import
Torch, ONNX, or other backend stacks.

.. automodule:: naviertwin.core.export
   :members:
   :undoc-members:
   :show-inheritance:

NavierTwin Dataset Format
------------------------------------------------------------

.. automodule:: naviertwin.core.export.ntwin_format
   :members:
   :undoc-members:
   :show-inheritance:

ONNX Export
------------------------------------------------------------

.. automodule:: naviertwin.core.export.onnx_export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.export.onnx_wrap
   :members:
   :undoc-members:
   :show-inheritance:

TorchScript Export
------------------------------------------------------------

.. automodule:: naviertwin.core.export.torchscript_export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.export.torchscript_verify
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.export.torchscript_wrap
   :members:
   :undoc-members:
   :show-inheritance:

FMI/FMU Export
------------------------------------------------------------

.. automodule:: naviertwin.core.export.fmu_export
   :members:
   :undoc-members:
   :show-inheritance:

Quantization And Deployment Stubs
------------------------------------------------------------

.. automodule:: naviertwin.core.export.quantize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.export.coreml_stub
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.export.tflite_stub
   :members:
   :undoc-members:
   :show-inheritance:
