naviertwin.core.tools
============================================================

Mesh generation, mesh cleanup, and quality inspection helpers. The root package
keeps optional heavy dependencies lazy: Gmsh and PyMeshLab are only required
when calling the relevant generation or cleanup functions, while quality
inspection remains available in the core install path.

.. automodule:: naviertwin.core.tools
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Generation
------------------------------------------------------------

.. automodule:: naviertwin.core.tools.mesh_generator
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Cleanup And Quality
------------------------------------------------------------

.. automodule:: naviertwin.core.tools.mesh_processor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.tools.mesh_quality
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.tools.laplacian_smooth
   :members:
   :undoc-members:
   :show-inheritance:

Extraction And Sampling
------------------------------------------------------------

.. automodule:: naviertwin.core.tools.clip_plane
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.tools.surface_extract
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.tools.stream_seeds
   :members:
   :undoc-members:
   :show-inheritance:
