"""Physics-informed neural solver wrappers and PhysicsNEMO adapters."""

from naviertwin.core.physnemo.cfd_field_model import PhysicsNeMoCFDFieldModel
from naviertwin.core.physnemo.dd_pinn import DomainDecompPINN
from naviertwin.core.physnemo.deep_ritz import DeepRitzSolver
from naviertwin.core.physnemo.physicsnemo_model import (
    load_checkpoint,
    physicsnemo_available,
    save_checkpoint,
    wrap_as_physicsnemo_module,
)
from naviertwin.core.physnemo.physnemo_wrapper import PhysicsNEMOWrapper
from naviertwin.core.physnemo.pina_wrapper import PINNSolver

__all__ = [
    "DeepRitzSolver",
    "DomainDecompPINN",
    "PINNSolver",
    "PhysicsNeMoCFDFieldModel",
    "PhysicsNEMOWrapper",
    "load_checkpoint",
    "physicsnemo_available",
    "save_checkpoint",
    "wrap_as_physicsnemo_module",
]
