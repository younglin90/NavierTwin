naviertwin.core.solver_interfaces
============================================================

Built-in reference solvers and solver-adapter helpers used by demos, REST
routes, GUI simulation workflows, and regression benchmarks. The package root
exports dependency-light NumPy implementations for quick synthetic data
generation, while optional external wrappers stay in their own modules.

.. automodule:: naviertwin.core.solver_interfaces
   :members:
   :undoc-members:
   :show-inheritance:

Reference PDE Solvers
------------------------------------------------------------

.. automodule:: naviertwin.core.solver_interfaces.pde_solvers
   :members:
   :undoc-members:
   :show-inheritance:

Finite-Volume Advection
------------------------------------------------------------

.. automodule:: naviertwin.core.solver_interfaces.fvm_advection
   :members:
   :undoc-members:
   :show-inheritance:

Lattice Boltzmann
------------------------------------------------------------

.. automodule:: naviertwin.core.solver_interfaces.lbm_d2q9
   :members:
   :undoc-members:
   :show-inheritance:

Smoothed Particle Hydrodynamics
------------------------------------------------------------

.. automodule:: naviertwin.core.solver_interfaces.sph
   :members:
   :undoc-members:
   :show-inheritance:

Optional External Solver Adapters
------------------------------------------------------------

.. automodule:: naviertwin.core.solver_interfaces.lettuce_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.solver_interfaces.flowtorch_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: naviertwin.core.solver_interfaces.jax_fluids_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
