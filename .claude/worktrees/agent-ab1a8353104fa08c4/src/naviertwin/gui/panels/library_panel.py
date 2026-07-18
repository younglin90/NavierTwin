"""Library/capability browser panel.

This panel exposes the shipped core API surface that is otherwise hidden behind
specialized workflow tabs. It intentionally runs only small deterministic demos;
production workflows still live in their domain-specific tabs.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from naviertwin.utils.feature_packs import (
    FEATURE_PACKS,
    activate_installed_feature_packs,
    default_release_asset_url,
    feature_pack_status,
    get_feature_pack_spec,
    install_feature_pack_archive,
    install_feature_pack_online,
    recommended_pack_for_modules,
)


@dataclass(frozen=True)
class LibraryStatus:
    """Runtime availability status describing an optional/core package."""

    name: str
    module: str
    purpose: str
    available: bool
    version: str
    install_hint: str


@dataclass(frozen=True)
class CapabilitySpec:
    """GUI-visible capability metadata."""

    id: str
    category: str
    name: str
    description: str
    module: str
    gui_route: str
    dependencies: tuple[str, ...] = ()
    demo_id: str | None = None

    @property
    def demo_supported(self) -> bool:
        return self.demo_id is not None


@dataclass(frozen=True)
class CoreApiItem:
    """Statically discovered public core API item."""

    category: str
    kind: str
    name: str
    module: str
    signature: str
    file_path: str
    line: int
    doc: str
    gui_route: str

    @property
    def import_path(self) -> str:
        return f"{self.module}.{self.name}"


DemoRunner = Callable[[], dict[str, Any]]


_LIBRARIES: tuple[tuple[str, str, str, str], ...] = (
    ("NumPy", "numpy", "array/numerics", "pip install numpy"),
    ("SciPy", "scipy", "scientific algorithms", "pip install scipy"),
    ("scikit-learn", "sklearn", "fallback surrogate/UQ", "pip install scikit-learn"),
    ("h5py", "h5py", ".ntwin HDF5 storage", "pip install h5py"),
    ("PySide6", "PySide6", "desktop GUI", "pip install PySide6"),
    ("VTK", "vtk", "3D mesh rendering backend", "pip install vtk"),
    ("PyVista", "pyvista", "CFD mesh visualization", "pip install pyvista"),
    ("pyvistaqt", "pyvistaqt", "native Qt 3D viewer", "pip install pyvistaqt"),
    ("meshio", "meshio", "mesh format import/export", "pip install meshio"),
    ("PyTorch", "torch", "AI/ROM/operator learning", "Feature Pack: ml-cpu"),
    (
        "PyTorch Geometric",
        "torch_geometric",
        "GNN/MeshGraphNets",
        "Feature Pack: ml-cpu",
    ),
    ("PyWavelets", "pywt", "WNO/wavelet analysis", "Feature Pack: ml-cpu"),
    (
        "NVIDIA PhysicsNeMo",
        "physicsnemo",
        "PhysicsNeMo module/PINN integration",
        "Feature Pack: physicsnemo",
    ),
    ("FastAPI", "fastapi", "REST serving", "Feature Pack: serving"),
    ("uvicorn", "uvicorn", "REST server runtime", "Feature Pack: serving"),
    ("SHAP", "shap", "external SHAP backend", "Feature Pack: ml-cpu"),
    ("captum", "captum", "PyTorch explainability", "Feature Pack: ml-cpu"),
    ("SALib", "SALib", "Sobol/Morris sensitivity", "pip install SALib"),
    ("SMT", "smt", "RBF/Kriging surrogate backend", "pip install smt"),
    ("pyCGNS", "CGNS.MAP", "CGNS tree reader", "pip install pyCGNS"),
    ("Gmsh", "gmsh", "mesh generation", "pip install gmsh"),
    ("PyMeshLab", "pymeshlab", "mesh processing", "pip install pymeshlab"),
)


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_version(module: str) -> str:
    candidates = {
        "PySide6": "PySide6",
        "sklearn": "scikit-learn",
        "torch_geometric": "torch-geometric",
        "pywt": "PyWavelets",
        "CGNS.MAP": "pyCGNS",
        "physicsnemo": "nvidia-physicsnemo",
    }
    dist = candidates.get(module, module.split(".", 1)[0])
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "-"


def list_library_statuses() -> list[LibraryStatus]:
    """Return runtime dependency availability without importing heavy packages."""
    statuses: list[LibraryStatus] = []
    library_index = 0
    while library_index < len(_LIBRARIES):
        name, module, purpose, hint = _LIBRARIES[library_index]
        library_index += 1
        available = _module_available(module)
        statuses.append(
            LibraryStatus(
                name=name,
                module=module,
                purpose=purpose,
                available=available,
                version=_package_version(module) if available else "-",
                install_hint=hint,
            )
        )
    return statuses


def list_capability_specs() -> list[CapabilitySpec]:
    """Return all GUI-routable capabilities plus post-processing operations."""
    specs: list[CapabilitySpec] = [
        CapabilitySpec(
            "workflow.import",
            "Workflow",
            "CFD Import / ReaderFactory",
            "VTK, VTU, OpenFOAM, CGNS, Fluent, SU2, Gmsh 등 CFD 입력을 읽습니다.",
            "naviertwin.core.cfd_reader",
            "Import",
        ),
        CapabilitySpec(
            "workflow.analyze",
            "Workflow",
            "Flow Analysis Viewer",
            "Q-criterion, lambda2, y+, FFT/PSD, SPOD, SINDy, 해석해 비교를 실행합니다.",
            "naviertwin.gui.panels.analyze_panel",
            "Analyze",
        ),
        CapabilitySpec(
            "workflow.reduce",
            "Workflow",
            "ROM Reduction",
            "POD, Randomized POD, DMD, Incremental/MR-POD 기반 차원축소를 실행합니다.",
            "naviertwin.gui.panels.reduce_panel",
            "Reduce",
        ),
        CapabilitySpec(
            "workflow.model",
            "Workflow",
            "Surrogate / Operator Learning",
            "RBF/Kriging, FNO/TFNO/DeepONet/UNet/WNO, PhysicsNeMo/PINN 학습 GUI입니다.",
            "naviertwin.gui.panels.model_panel",
            "Model",
        ),
        CapabilitySpec(
            "workflow.twin",
            "Workflow",
            "Digital Twin Engine",
            "학습된 reducer+surrogate로 예측, 최적화, 동화 quick-check를 실행합니다.",
            "naviertwin.gui.panels.twin_panel",
            "Twin",
        ),
        CapabilitySpec(
            "workflow.postproc",
            "Workflow",
            "Post-Tools Facade",
            "후처리/ROM/AI 진단 연산을 Facade 기반으로 실행합니다.",
            "naviertwin.core.post_process_facade",
            "Post-Tools",
        ),
        CapabilitySpec(
            "physnemo.pinn",
            "PhysicsNeMo / PINN",
            "PhysicsNeMo wrapper PINN",
            "physicsnemo 설치를 감지하고 현재 래퍼 경로로 PINN Poisson demo를 실행합니다.",
            "naviertwin.core.physnemo.physnemo_wrapper",
            "Model",
            ("torch",),
            "physnemo.pinn",
        ),
        CapabilitySpec(
            "physnemo.ddpinn",
            "PhysicsNeMo / PINN",
            "Domain-Decomposition PINN",
            "1D 영역을 분할해 PINN을 순차 학습하는 demo를 실행합니다.",
            "naviertwin.core.physnemo.dd_pinn",
            "Model",
            ("torch",),
            "physnemo.ddpinn",
        ),
        CapabilitySpec(
            "physnemo.module",
            "PhysicsNeMo / PINN",
            "PhysicsNeMo Module wrapper",
            "PyTorch nn.Module을 PhysicsNeMo Module로 감싸 저장/로드 경로를 확인합니다.",
            "naviertwin.core.physnemo.physicsnemo_model",
            "Model",
            ("torch", "physicsnemo"),
            "physnemo.module",
        ),
        CapabilitySpec(
            "surrogate.rbf",
            "Surrogate",
            "RBF Surrogate",
            "SMT가 있으면 SMT RBF, 없으면 sklearn/numpy fallback으로 학습합니다.",
            "naviertwin.core.surrogate.rbf_surrogate",
            "Model",
            (),
            "surrogate.rbf",
        ),
        CapabilitySpec(
            "surrogate.kriging",
            "Surrogate",
            "Kriging / Gaussian Process",
            "Kriging/GP surrogate와 예측 분산 demo를 실행합니다.",
            "naviertwin.core.surrogate.kriging_surrogate",
            "Model",
            (),
            "surrogate.kriging",
        ),
        CapabilitySpec(
            "reduction.pod",
            "Reduction",
            "POD / Randomized POD / CPOD",
            "선형 ROM 축소와 제약 보존 POD demo를 실행합니다.",
            "naviertwin.core.dimensionality_reduction.linear",
            "Reduce",
            (),
            "reduction.pod",
        ),
        CapabilitySpec(
            "reduction.ae",
            "Reduction",
            "Autoencoder / VAE",
            "비선형 PyTorch 차원축소 demo를 실행합니다.",
            "naviertwin.core.dimensionality_reduction.nonlinear",
            "Reduce",
            ("torch",),
            "reduction.ae",
        ),
        CapabilitySpec(
            "operator.fno1d",
            "Operator Learning",
            "FNO1D",
            "작은 1D Fourier Neural Operator 학습/예측 demo를 실행합니다.",
            "naviertwin.core.operator_learning.fno.fno",
            "Model",
            ("torch",),
            "operator.fno1d",
        ),
        CapabilitySpec(
            "operator.deeponet",
            "Operator Learning",
            "DeepONet",
            "branch/trunk DeepONet 학습/예측 demo를 실행합니다.",
            "naviertwin.core.operator_learning.deeponet.deeponet",
            "Model",
            ("torch",),
            "operator.deeponet",
        ),
        CapabilitySpec(
            "operator.unet2d",
            "Operator Learning",
            "UNet2D",
            "2D field-to-field U-Net demo를 실행합니다.",
            "naviertwin.core.operator_learning.unet.unet",
            "Model",
            ("torch",),
            "operator.unet2d",
        ),
        CapabilitySpec(
            "operator.wno1d",
            "Operator Learning",
            "WNO1D",
            "PyWavelets 기반 Wavelet Neural Operator demo를 실행합니다.",
            "naviertwin.core.operator_learning.fno.wno",
            "Model",
            ("torch", "pywt"),
            "operator.wno1d",
        ),
        CapabilitySpec(
            "gnn.surrogate",
            "GNN",
            "GNN Surrogate",
            "PyTorch Geometric 기반 node-level surrogate demo를 실행합니다.",
            "naviertwin.core.gnn.gnn_surrogate.gnn_surrogate",
            "Model",
            ("torch", "torch_geometric"),
            "gnn.surrogate",
        ),
        CapabilitySpec(
            "timeseries.koopman",
            "Time Series / Koopman",
            "LSTM + KNO",
            "시계열 예측과 Koopman latent dynamics demo를 실행합니다.",
            "naviertwin.core.time_series",
            "Model",
            ("torch",),
            "timeseries.koopman",
        ),
        CapabilitySpec(
            "generative.diffusion",
            "Generative",
            "Diffusion PDE",
            "작은 DDPM-style PDE field 생성 demo를 실행합니다.",
            "naviertwin.core.generative.diffusion_pde.diffusion_pde",
            "Model",
            ("torch",),
            "generative.diffusion",
        ),
        CapabilitySpec(
            "assimilation.enkf",
            "Twin / Assimilation",
            "EnKF + Particle Filter",
            "데이터 동화 필터 demo를 실행합니다.",
            "naviertwin.core.data_assimilation",
            "Twin",
            (),
            "assimilation.enkf",
        ),
        CapabilitySpec(
            "optimization.bo",
            "Optimization / UQ",
            "Bayesian Optimization + MC",
            "Bayesian optimizer와 Monte Carlo uncertainty propagation demo를 실행합니다.",
            "naviertwin.core.optimization",
            "Twin",
            ("sklearn",),
            "optimization.bo",
        ),
        CapabilitySpec(
            "simulation.solvers",
            "Simulation",
            "LBM + FVM",
            "LBM D2Q9와 1D FVM advection demo를 실행합니다.",
            "naviertwin.core.solver_interfaces",
            "Simulation",
            (),
            "simulation.solvers",
        ),
        CapabilitySpec(
            "explain.symbolic",
            "Explainability",
            "SHAP fallback + Symbolic regression",
            "내장 KernelSHAP와 symbolic polynomial regression demo를 실행합니다.",
            "naviertwin.core.explainability",
            "Explain",
            (),
            "explain.symbolic",
        ),
        CapabilitySpec(
            "applied.calculators",
            "Applied",
            "Fan / Duct / Pump calculators",
            "현장 계산용 fan affinity, duct pressure loss, pump operating point demo입니다.",
            "naviertwin.core.applied",
            "Twin",
            (),
            "applied.calculators",
        ),
    ]
    try:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        operations = facade.list_operations()
        op_index = 0
        while op_index < len(operations):
            op_name = operations[op_index]
            op_index += 1
            info = facade.describe(op_name)
            specs.append(
                CapabilitySpec(
                    id=f"postproc:{op_name}",
                    category=f"Post-Tools / {info['category']}",
                    name=op_name,
                    description=str(info["description"]),
                    module="naviertwin.core.post_process_facade",
                    gui_route="Post-Tools",
                    demo_id=f"postproc:{op_name}",
                )
            )
    except Exception:  # noqa: BLE001
        pass
    return specs


def list_core_api_items(root: Path | None = None) -> list[CoreApiItem]:
    """Statically enumerate public APIs under ``naviertwin.core``.

    The scanner intentionally uses AST instead of imports so optional heavy
    backends such as VTK, torch, pyCGNS, or PhysicsNeMo are not loaded just to
    render the GUI catalog.
    """
    core_root = root or (Path(__file__).resolve().parents[2] / "core")
    if not core_root.exists():
        return []

    items: list[CoreApiItem] = []
    seen: set[tuple[str, str]] = set()
    py_files = sorted(core_root.rglob("*.py"))
    file_index = 0
    while file_index < len(py_files):
        py_file = py_files[file_index]
        file_index += 1
        if "__pycache__" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        module = _module_name_from_core_file(core_root, py_file)
        category = _category_from_core_file(core_root, py_file)
        definitions = _collect_public_definitions(tree)
        exported = _extract_all_names(tree)
        imported_exports = _collect_imported_export_names(tree)

        names = exported if exported else sorted(definitions)
        name_index = 0
        while name_index < len(names):
            name = names[name_index]
            name_index += 1
            if name.startswith("_"):
                continue

            definition = definitions.get(name)
            if definition is not None:
                kind, signature, line, doc = definition
            elif py_file.name == "__init__.py" and name in imported_exports:
                kind = "export"
                signature = f"{name}"
                line = imported_exports[name]
                doc = ""
            else:
                continue

            key = (module, name)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                CoreApiItem(
                    category=category,
                    kind=kind,
                    name=name,
                    module=module,
                    signature=signature,
                    file_path=str(py_file),
                    line=line,
                    doc=doc,
                    gui_route=_gui_route_for_module(module),
                )
            )

    return sorted(items, key=lambda item: (item.category, item.module, item.name))


def _module_name_from_core_file(core_root: Path, py_file: Path) -> str:
    rel = py_file.relative_to(core_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["naviertwin", "core", *parts])


def _category_from_core_file(core_root: Path, py_file: Path) -> str:
    rel = py_file.relative_to(core_root)
    first = rel.parts[0]
    return "core" if first == "__init__.py" else first


def _collect_public_definitions(
    tree: ast.Module,
) -> dict[str, tuple[str, str, int, str]]:
    definitions: dict[str, tuple[str, str, int, str]] = {}
    node_index = 0
    while node_index < len(tree.body):
        node = tree.body[node_index]
        node_index += 1
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            definitions[node.name] = (
                "class",
                _signature_for_class(node),
                node.lineno,
                ast.get_docstring(node) or "",
            )
        elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            definitions[node.name] = (
                "async function",
                _signature_for_function(node.name, node.args, node.returns),
                node.lineno,
                ast.get_docstring(node) or "",
            )
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            definitions[node.name] = (
                "function",
                _signature_for_function(node.name, node.args, node.returns),
                node.lineno,
                ast.get_docstring(node) or "",
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assigned_names = _assigned_public_names(node)
            name_index = 0
            while name_index < len(assigned_names):
                name = assigned_names[name_index]
                name_index += 1
                definitions.setdefault(
                    name,
                    ("constant", name, getattr(node, "lineno", 1), ""),
                )
    return definitions


def _assigned_public_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]

    names: list[str] = []
    target_index = 0
    while target_index < len(targets):
        target = targets[target_index]
        target_index += 1
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            names.append(target.id)
    return names


def _extract_all_names(tree: ast.Module) -> list[str]:
    node_index = 0
    while node_index < len(tree.body):
        node = tree.body[node_index]
        node_index += 1
        if not isinstance(node, ast.Assign):
            continue
        target_index = 0
        while target_index < len(node.targets):
            target = node.targets[target_index]
            target_index += 1
            if isinstance(target, ast.Name) and target.id == "__all__":
                return _literal_str_sequence(node.value)
    return []


def _literal_str_sequence(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        names: list[str] = []
        element_index = 0
        while element_index < len(node.elts):
            element = node.elts[element_index]
            element_index += 1
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.append(element.value)
        return names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_str_sequence(node.left) + _literal_str_sequence(node.right)
    return []


def _collect_imported_export_names(tree: ast.Module) -> dict[str, int]:
    imports: dict[str, int] = {}
    node_index = 0
    while node_index < len(tree.body):
        node = tree.body[node_index]
        node_index += 1
        if not isinstance(node, ast.ImportFrom):
            continue
        alias_index = 0
        while alias_index < len(node.names):
            alias = node.names[alias_index]
            alias_index += 1
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            if not name.startswith("_"):
                imports[name] = node.lineno
    return imports


def _signature_for_class(node: ast.ClassDef) -> str:
    init = None
    child_index = 0
    while child_index < len(node.body):
        child = node.body[child_index]
        child_index += 1
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            init = child
            break
    if init is None:
        return f"{node.name}()"
    return _signature_for_function(node.name, init.args, None)


def _signature_for_function(
    name: str,
    args: ast.arguments,
    returns: ast.expr | None,
) -> str:
    params: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    positional_pairs = list(zip(positional, defaults))
    positional_index = 0
    while positional_index < len(positional_pairs):
        arg, default = positional_pairs[positional_index]
        positional_index += 1
        if arg.arg in {"self", "cls"}:
            continue
        params.append(_format_arg(arg, default is not None))

    if args.vararg is not None:
        params.append(f"*{_format_arg(args.vararg, False)}")
    elif args.kwonlyargs:
        params.append("*")

    kwonly_pairs = list(zip(args.kwonlyargs, args.kw_defaults))
    kwonly_index = 0
    while kwonly_index < len(kwonly_pairs):
        arg, default = kwonly_pairs[kwonly_index]
        kwonly_index += 1
        params.append(_format_arg(arg, default is not None))

    if args.kwarg is not None:
        params.append(f"**{_format_arg(args.kwarg, False)}")

    ret = f" -> {_safe_unparse(returns)}" if returns is not None else ""
    return f"{name}({', '.join(params)}){ret}"


def _format_arg(arg: ast.arg, has_default: bool) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {_safe_unparse(arg.annotation)}"
    if has_default:
        text += "=..."
    return text


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "..."


def _gui_route_for_module(module: str) -> str:
    route_rules = [
        (".cfd_reader", "Import"),
        (".post_process_facade", "Post-Tools"),
        (".dimensionality_reduction", "Reduce"),
        (".surrogate", "Model"),
        (".operator_learning", "Model"),
        (".gnn", "Model"),
        (".physicsnemo", "Model"),
        (".time_series", "Model"),
        (".generative", "Model"),
        (".data_assimilation", "Twin"),
        (".optimization", "Twin"),
        (".solver_interfaces", "Simulation"),
        (".explainability", "Explain"),
        (".export", "Export"),
    ]
    route_index = 0
    while route_index < len(route_rules):
        needle, route = route_rules[route_index]
        route_index += 1
        if needle in module:
            return route
    return "Library"


def run_capability_demo(capability_id: str) -> dict[str, Any]:
    """Run a small deterministic smoke demo linked to a capability."""
    if capability_id.startswith("postproc:"):
        op_name = capability_id.split(":", 1)[1]
        return _run_postproc_demo(op_name)
    runner = _DEMO_RUNNERS.get(capability_id)
    if runner is None:
        raise KeyError(f"demo not registered: {capability_id}")
    return runner()


def _run_postproc_demo(op_name: str) -> dict[str, Any]:
    from naviertwin.core.post_process_facade import PostProcessFacade
    from naviertwin.gui.panels.postproc_panel import PostProcessPanel

    kwargs = PostProcessPanel._build_smoke_kwargs(op_name)
    result = PostProcessFacade().run(op_name, **kwargs)
    return {"operation": op_name, "result": result}


def _demo_applied_calculators() -> dict[str, Any]:
    from naviertwin.core.applied.centrifugal_pump import operating_point
    from naviertwin.core.applied.fan_affinity import scale_Q_H_P
    from naviertwin.core.applied.hvac_duct import duct_velocity, total_pressure_loss

    return {
        "fan_scaled_Q_H_P": scale_Q_H_P(Q1=10.0, H1=20.0, P1=300.0, N1=1000.0, N2=1500.0),
        "duct_velocity": duct_velocity(mdot=2.0, rho=1.2, A=0.4),
        "duct_pressure_loss": total_pressure_loss(L=10.0, D=0.3, rho=1.2, U=5.0, K_total=2.0),
        "pump_operating_point": operating_point(sys_a=5.0, sys_b=1.5, pump_a=30.0, pump_b=2.0),
    }


def _demo_surrogate_rbf() -> dict[str, Any]:
    from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (24, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    model = RBFSurrogate(d0=1.0)
    model.fit(X, y)
    pred = model.predict(X[:4])
    return {"backend": model.get_params().get("backend"), "prediction": pred}


def _demo_surrogate_kriging() -> dict[str, Any]:
    from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (16, 2))
    y = np.cos(X[:, 0]) + X[:, 1]
    model = KrigingSurrogate()
    model.fit(X, y)
    pred, var = model.predict_with_variance(X[:3])
    return {"prediction": pred, "variance": var, "backend": model.get_params().get("backend")}


def _demo_reduction_pod() -> dict[str, Any]:
    from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD
    from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
    from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 18))
    pod = SnapshotPOD(n_modes=4)
    pod.fit(X)
    rpod = RandomizedPOD(n_modes=4, random_state=0)
    rpod.fit(X)
    centered = X - X.mean(axis=0, keepdims=True)
    cpod = ConstrainedPOD(n_modes=3, C=np.ones((1, centered.shape[0])), d=np.zeros(1))
    cpod.fit(centered)
    return {
        "pod_modes": pod.n_components,
        "randomized_modes": rpod.n_components,
        "cpod_constraint_residual": float(np.abs(np.ones((1, 30)) @ cpod.reconstruct(centered)).max()),
    }


def _demo_reduction_ae() -> dict[str, Any]:
    from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import Autoencoder
    from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 16))
    ae = Autoencoder(latent_dim=3, hidden_dims=[16, 8], max_epochs=2)
    ae.fit(X)
    vae = VAE(latent_dim=2, hidden_dims=[12, 6], max_epochs=2)
    vae.fit(X)
    return {
        "ae_latent_shape": ae.encode(X).shape,
        "ae_reconstruction_shape": ae.reconstruct(X).shape,
        "vae_sample_shape": vae.sample(n_samples=3, seed=0).shape,
    }


def _demo_fno1d() -> dict[str, Any]:
    from naviertwin.core.operator_learning.fno.fno import FNO1D

    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 16, 1)).astype(np.float32)
    Y = np.sin(X).astype(np.float32)
    op = FNO1D(in_channels=1, out_channels=1, modes=4, width=6, n_layers=1, max_epochs=1)
    op.fit({"inputs": X, "outputs": Y})
    return {"prediction_shape": op.predict({"x": X[:2]}).shape, "loss": op.train_losses_}


def _demo_deeponet() -> dict[str, Any]:
    from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

    rng = np.random.default_rng(0)
    branch = rng.standard_normal((12, 8)).astype(np.float32)
    trunk = rng.standard_normal((5, 1)).astype(np.float32)
    y = rng.standard_normal((12, 5)).astype(np.float32)
    op = DeepONet(branch_in=8, trunk_in=1, hidden=8, latent=4, max_epochs=1)
    op.fit({"branch_inputs": branch, "trunk_inputs": trunk, "outputs": y})
    return {"prediction_shape": op.predict({"branch_inputs": branch[:3]}).shape}


def _demo_unet2d() -> dict[str, Any]:
    from naviertwin.core.operator_learning.unet.unet import UNet2D

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 16, 16, 1)).astype(np.float32)
    op = UNet2D(in_channels=1, out_channels=1, base_ch=4, max_epochs=1)
    op.fit({"inputs": X, "outputs": X ** 2})
    return {"prediction_shape": op.predict({"x": X[:1]}).shape}


def _demo_wno1d() -> dict[str, Any]:
    from naviertwin.core.operator_learning.fno.wno import WNO1D

    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 32, 1)).astype(np.float32)
    op = WNO1D(in_channels=1, out_channels=1, width=4, level=1, n_layers=1, max_epochs=1)
    op.fit({"inputs": X, "outputs": X ** 2})
    return {"prediction_shape": op.predict({"x": X[:2]}).shape}


def _demo_gnn_surrogate() -> dict[str, Any]:
    from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate

    rng = np.random.default_rng(0)
    n_nodes = 10
    X = rng.standard_normal((4, n_nodes, 2)).astype(np.float32)
    edge = np.stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]).astype(np.int64)
    op = GNNSurrogate(in_dim=2, out_dim=2, hidden=8, n_layers=1, max_epochs=1)
    op.fit({"node_features": X, "outputs": X ** 2, "edge_index": edge})
    return {"prediction_shape": op.predict({"x": X[:1]}).shape}


def _demo_timeseries_koopman() -> dict[str, Any]:
    from naviertwin.core.operator_learning.koopman.kno import KNO
    from naviertwin.core.time_series.lstm.lstm import LSTMForecaster

    rng = np.random.default_rng(0)
    seqs = rng.standard_normal((3, 14, 2)).astype(np.float32)
    lstm = LSTMForecaster(n_features=2, hidden=6, lookback=4, max_epochs=1)
    lstm.fit({"sequences": seqs})
    kno = KNO(n_features=2, latent=3, hidden=6, max_epochs=1)
    kno.fit({"sequences": seqs})
    return {
        "lstm_rollout_shape": lstm.predict(seqs[0, :4], n_steps=3).shape,
        "kno_matrix_shape": kno.koopman_matrix().shape,
    }


def _demo_generative_diffusion() -> dict[str, Any]:
    from naviertwin.core.generative.diffusion_pde.diffusion_pde import DiffusionPDE

    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 6)).astype(np.float32)
    model = DiffusionPDE(n_features=6, hidden=8, n_steps=5, max_epochs=1)
    model.fit(X)
    return {"sample_shape": model.sample(n_samples=3, seed=0).shape}


def _demo_assimilation() -> dict[str, Any]:
    from naviertwin.core.data_assimilation.enkf import EnKF
    from naviertwin.core.data_assimilation.particle_filter import ParticleFilter

    rng = np.random.default_rng(0)
    truth = np.array([1.0, -0.5])
    ens = rng.standard_normal((40, 2))
    enkf = EnKF(H=np.eye(2), R=0.02 * np.eye(2))
    updated = enkf.analysis(ens, truth, rng=rng)
    pf = ParticleFilter(n_particles=80, state_dim=2)
    pf.initialize(rng.standard_normal((80, 2)))
    pf.update(truth, np.eye(2), 0.05 * np.eye(2))
    return {
        "enkf_mean_error": float(np.linalg.norm(updated.mean(axis=0) - truth)),
        "pf_estimate": pf.estimate(),
    }


def _demo_optimization() -> dict[str, Any]:
    from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer
    from naviertwin.core.optimization.mc_propagation import propagate_mc

    def obj(x: np.ndarray) -> float:
        return float((x[0] - 0.2) ** 2 + (x[1] + 0.1) ** 2)

    opt = BayesianOptimizer(
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        n_initial=4,
        max_iter=3,
        seed=0,
    )
    x_best, f_best = opt.minimize(obj)
    rng = np.random.default_rng(0)
    mc = propagate_mc(lambda X: X[:, 0] ** 2, rng.standard_normal((100, 1)))
    return {"x_best": x_best, "f_best": f_best, "mc_mean": mc["mean"]}


def _demo_simulation_solvers() -> dict[str, Any]:
    from naviertwin.core.solver_interfaces.fvm_advection import fvm_upwind_1d
    from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

    x = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    times, U = fvm_upwind_1d(np.sin(x), T=0.2)
    snaps = LBMD2Q9(nx=12, ny=12, tau=0.8, u_top=0.03).run(n_steps=8, record_every=4)
    return {"fvm_shape": U.shape, "fvm_last_time": times[-1], "lbm_snapshots": snaps.shape}


def _demo_explainability() -> dict[str, Any]:
    from naviertwin.core.explainability.shap_explainer import KernelSHAP
    from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (20, 2))

    def predict(v: np.ndarray) -> np.ndarray:
        return 2.0 * v[:, 0] - 0.5 * v[:, 1]

    shap = KernelSHAP(predict, background=X[:8], n_samples=20, seed=0).explain(X[:2])
    reg = SymbolicRegressor(max_degree=2, threshold=1e-6)
    reg.fit(X, predict(X))
    return {"shap_shape": shap.shape, "symbolic_expression": reg.expression_}


def _demo_physnemo_pinn() -> dict[str, Any]:
    import torch

    from naviertwin.core.physnemo.physnemo_wrapper import PhysicsNEMOWrapper

    def residual(model: object, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

    wrapper = PhysicsNEMOWrapper(equation="poisson_1d", hidden=8, max_epochs=8)
    col = np.linspace(0, 1, 16, dtype=np.float32).reshape(-1, 1)
    bc = {
        "x": np.array([[0.0], [1.0]], dtype=np.float32),
        "u": np.array([[0.0], [0.0]], dtype=np.float32),
    }
    wrapper.fit(residual, col, bc)
    return {"physicsnemo_import_available": wrapper.available, "u_mid": wrapper.predict(np.array([[0.5]]))}


def _demo_physnemo_ddpinn() -> dict[str, Any]:
    import torch

    from naviertwin.core.physnemo.dd_pinn import DomainDecompPINN

    def residual(model: object, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

    dd = DomainDecompPINN(n_sub=2, n_collocation=16, hidden=8, max_epochs=16)
    bc = {
        "x": np.array([[0.0], [1.0]], dtype=np.float32),
        "u": np.array([[0.0], [0.0]], dtype=np.float32),
    }
    dd.fit(residual, bc)
    return {"is_fitted": dd.is_fitted, "prediction": dd.predict(np.array([[0.25], [0.75]]))}


def _demo_physnemo_module() -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from naviertwin.core.physnemo.physicsnemo_model import (
        physicsnemo_available,
        save_checkpoint,
        wrap_as_physicsnemo_module,
    )

    if not physicsnemo_available():
        raise RuntimeError("physicsnemo 미설치: pip install nvidia-physicsnemo")
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    wrapped = wrap_as_physicsnemo_module(model, name="naviertwin_gui_demo")
    y = wrapped(torch.ones(2, 2))
    out = Path(tempfile.gettempdir()) / "naviertwin_physicsnemo_demo.pt"
    save_checkpoint(wrapped, out)
    return {"output_shape": tuple(y.shape), "checkpoint": str(out), "exists": out.exists()}


_DEMO_RUNNERS: dict[str, DemoRunner] = {
    "applied.calculators": _demo_applied_calculators,
    "surrogate.rbf": _demo_surrogate_rbf,
    "surrogate.kriging": _demo_surrogate_kriging,
    "reduction.pod": _demo_reduction_pod,
    "reduction.ae": _demo_reduction_ae,
    "operator.fno1d": _demo_fno1d,
    "operator.deeponet": _demo_deeponet,
    "operator.unet2d": _demo_unet2d,
    "operator.wno1d": _demo_wno1d,
    "gnn.surrogate": _demo_gnn_surrogate,
    "timeseries.koopman": _demo_timeseries_koopman,
    "generative.diffusion": _demo_generative_diffusion,
    "assimilation.enkf": _demo_assimilation,
    "optimization.bo": _demo_optimization,
    "simulation.solvers": _demo_simulation_solvers,
    "explain.symbolic": _demo_explainability,
    "physnemo.pinn": _demo_physnemo_pinn,
    "physnemo.ddpinn": _demo_physnemo_ddpinn,
    "physnemo.module": _demo_physnemo_module,
}


class _FeaturePackInstallWorker(QObject):
    """``install_feature_pack_online`` 을 백그라운드 QThread 에서 실행.

    Qt signals 로 main thread 의 LibraryPanel 슬롯에 결과를 돌려준다.
    """

    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, pack_id: str) -> None:
        super().__init__()
        self._pack_id = pack_id

    @Slot()
    def run(self) -> None:
        try:
            status = install_feature_pack_online(self._pack_id)
        except Exception as exc:  # noqa: BLE001 - 사용자에게 표시할 메시지로 변환
            self.failed.emit(self._pack_id, f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(status)


class LibraryPanel(QWidget):
    """Runtime library status and full capability entrypoint panel."""

    capability_done = Signal(str, object)
    navigate_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._capabilities = list_capability_specs()
        self._libraries = list_library_statuses()
        self._core_api_items = list_core_api_items()
        self._selected_capability: CapabilitySpec | None = None
        self._selected_api: CoreApiItem | None = None
        self._selected_feature_pack_id: str | None = None
        self._visible_api_items: list[CoreApiItem] = []
        self._dataset: object | None = None
        # 온라인 설치 백그라운드 작업 상태.
        self._install_in_progress: bool = False
        self._install_thread: object | None = None
        self._install_worker: object | None = None
        self._setup_ui()
        self._refresh_library_table()
        self._refresh_capability_list()
        self._refresh_pack_table()

    def set_dataset(self, dataset: object) -> None:
        """Attach dataset context used by dataset-aware capability demos."""
        self._dataset = dataset
        n_points = getattr(dataset, "n_points", "?")
        n_cells = getattr(dataset, "n_cells", "?")
        self._context_label.setText(f"데이터셋: {n_points} pts, {n_cells} cells 연결됨")

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Library / Capabilities")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        self._context_label = QLabel("데이터셋: 없음")
        self._context_label.setObjectName("subtitleLabel")
        left_layout.addWidget(self._context_label)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("기능/모듈 검색")
        self._search_edit.textChanged.connect(self._refresh_capability_list)
        left_layout.addWidget(self._search_edit)

        self._category_combo = QComboBox()
        self._category_combo.addItem("전체")
        categories = sorted(set(map(lambda cap: cap.category, self._capabilities)))
        category_index = 0
        while category_index < len(categories):
            category = categories[category_index]
            category_index += 1
            self._category_combo.addItem(category)
        self._category_combo.currentTextChanged.connect(self._refresh_capability_list)
        left_layout.addWidget(self._category_combo)

        self._capability_list = QListWidget()
        self._capability_list.currentItemChanged.connect(self._on_capability_selected)
        left_layout.addWidget(self._capability_list, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        detail_group = QGroupBox("선택 기능")
        detail_layout = QVBoxLayout(detail_group)
        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(180)
        detail_layout.addWidget(self._detail_text)

        button_row = QHBoxLayout()
        self._run_btn = QPushButton("데모 실행")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._run_selected_demo)
        self._run_btn.setEnabled(False)
        button_row.addWidget(self._run_btn)

        self._navigate_btn = QPushButton("해당 GUI 탭으로 이동")
        self._navigate_btn.clicked.connect(self._navigate_selected)
        self._navigate_btn.setEnabled(False)
        button_row.addWidget(self._navigate_btn)

        self._feature_pack_online_btn = QPushButton("Feature Pack 온라인 설치")
        self._feature_pack_online_btn.setObjectName("primaryButton")
        self._feature_pack_online_btn.setToolTip(
            "PyPI 에서 직접 다운로드해 설치합니다 (인터넷 필요, 최대 수 분 소요)."
        )
        self._feature_pack_online_btn.clicked.connect(
            self._install_selected_feature_pack_online
        )
        self._feature_pack_online_btn.setEnabled(False)
        button_row.addWidget(self._feature_pack_online_btn)

        self._feature_pack_download_btn = QPushButton("브라우저로 ZIP 받기")
        self._feature_pack_download_btn.clicked.connect(self._open_selected_feature_pack_download)
        self._feature_pack_download_btn.setEnabled(False)
        button_row.addWidget(self._feature_pack_download_btn)

        self._feature_pack_install_btn = QPushButton("받은 ZIP 설치")
        self._feature_pack_install_btn.clicked.connect(self._install_selected_feature_pack_zip)
        self._feature_pack_install_btn.setEnabled(False)
        button_row.addWidget(self._feature_pack_install_btn)
        detail_layout.addLayout(button_row)

        # Feature Pack 전반 상태 + 개별 설치/재설치 (현재 선택과 무관한 한눈에 보기).
        pack_overview_group = QGroupBox("Feature Pack 설치 상태")
        pack_overview_layout = QVBoxLayout(pack_overview_group)
        self._pack_table = QTableWidget(0, 4)
        self._pack_table.setHorizontalHeaderLabels(
            ["Pack ID", "상태", "누락 모듈", "설치/재설치"]
        )
        self._pack_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._pack_table.verticalHeader().setVisible(False)
        pack_overview_layout.addWidget(self._pack_table)
        detail_layout.addWidget(pack_overview_group)

        right_layout.addWidget(detail_group)

        libs_group = QGroupBox("현재 사용 가능한 라이브러리")
        libs_layout = QVBoxLayout(libs_group)
        self._library_table = QTableWidget(0, 5)
        self._library_table.setHorizontalHeaderLabels(
            ["Library", "Status", "Version", "Purpose", "Install hint"]
        )
        self._library_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        libs_layout.addWidget(self._library_table)
        right_layout.addWidget(libs_group, stretch=1)

        api_group = QGroupBox("Core API 전체 목록")
        api_layout = QVBoxLayout(api_group)
        api_hint = QLabel("행 선택: import 경로/파일 확인, 더블클릭: 관련 GUI 탭 이동")
        api_hint.setObjectName("subtitleLabel")
        api_layout.addWidget(api_hint)

        self._api_table = QTableWidget(0, 6)
        self._api_table.setHorizontalHeaderLabels(
            ["Category", "Type", "Name", "Module", "GUI", "Signature"]
        )
        self._api_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._api_table.cellClicked.connect(self._on_api_selected)
        self._api_table.cellDoubleClicked.connect(self._navigate_selected_api)
        api_layout.addWidget(self._api_table)

        api_button_row = QHBoxLayout()
        self._api_route_btn = QPushButton("선택 API 관련 탭으로 이동")
        self._api_route_btn.clicked.connect(self._navigate_selected_api)
        self._api_route_btn.setEnabled(False)
        api_button_row.addWidget(self._api_route_btn)
        api_layout.addLayout(api_button_row)
        right_layout.addWidget(api_group, stretch=1)

        result_group = QGroupBox("실행 결과")
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        result_layout.addWidget(self._result_text)
        right_layout.addWidget(result_group, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([340, 900])

    def _refresh_library_table(self) -> None:
        self._library_table.setRowCount(len(self._libraries))
        row = 0
        while row < len(self._libraries):
            status = self._libraries[row]
            values = [
                status.name,
                "OK" if status.available else "MISSING",
                status.version,
                status.purpose,
                "" if status.available else status.install_hint,
            ]
            col = 0
            while col < len(values):
                value = values[col]
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setForeground(
                        Qt.GlobalColor.green if status.available else Qt.GlobalColor.yellow
                    )
                self._library_table.setItem(row, col, item)
                col += 1
            row += 1

    def _refresh_capability_list(self) -> None:
        category = self._category_combo.currentText() if hasattr(self, "_category_combo") else "전체"
        query = self._search_edit.text().strip().lower() if hasattr(self, "_search_edit") else ""
        previous = self._selected_capability.id if self._selected_capability is not None else None

        self._capability_list.blockSignals(True)
        self._capability_list.clear()
        selected_row = -1
        cap_index = 0
        while cap_index < len(self._capabilities):
            cap = self._capabilities[cap_index]
            cap_index += 1
            if category not in ("", "전체") and cap.category != category:
                continue
            haystack = f"{cap.id} {cap.category} {cap.name} {cap.description} {cap.module}".lower()
            if query and query not in haystack:
                continue
            item = QListWidgetItem(f"[{cap.category}] {cap.name}")
            item.setData(Qt.ItemDataRole.UserRole, cap.id)
            self._capability_list.addItem(item)
            if cap.id == previous:
                selected_row = self._capability_list.count() - 1
        self._capability_list.blockSignals(False)

        if self._capability_list.count() == 0:
            self._selected_capability = None
            self._render_capability_detail(None)
            self._refresh_api_table()
            return
        self._capability_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        self._on_capability_selected(self._capability_list.currentItem(), None)
        self._refresh_api_table()

    def _on_capability_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            self._selected_capability = None
            self._render_capability_detail(None)
            return
        cap_id = str(current.data(Qt.ItemDataRole.UserRole))
        self._selected_capability = next(
            filter(lambda cap: cap.id == cap_id, self._capabilities),
            None,
        )
        self._render_capability_detail(self._selected_capability)

    def _render_capability_detail(self, cap: CapabilitySpec | None) -> None:
        if cap is None:
            self._detail_text.setPlainText("기능을 선택하세요.")
            self._run_btn.setEnabled(False)
            self._navigate_btn.setEnabled(False)
            self._selected_feature_pack_id = None
            self._feature_pack_download_btn.setEnabled(False)
            self._feature_pack_install_btn.setEnabled(False)
            self._feature_pack_online_btn.setEnabled(False)
            return

        dep_status = self._dependency_summary(cap.dependencies)
        missing_deps = list(filter(lambda dep: not _module_available(dep), cap.dependencies))
        pack_id = recommended_pack_for_modules(missing_deps)
        self._selected_feature_pack_id = pack_id
        pack_hint = None
        if pack_id is not None:
            pack_spec = get_feature_pack_spec(pack_id)
            pack_hint = {
                "id": pack_spec.id,
                "name": pack_spec.name,
                "download_url": default_release_asset_url(pack_id),
            }
        detail = {
            "id": cap.id,
            "category": cap.category,
            "name": cap.name,
            "description": cap.description,
            "module": cap.module,
            "gui_route": cap.gui_route,
            "dependencies": dep_status,
            "feature_pack_hint": pack_hint,
            "demo_supported": cap.demo_supported,
        }
        self._detail_text.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2))
        deps_ok = self._dependencies_available(cap.dependencies)
        self._run_btn.setEnabled(cap.demo_supported and deps_ok)
        if cap.demo_supported and not deps_ok:
            self._run_btn.setToolTip("필수 optional dependency가 설치되어 있지 않습니다.")
        else:
            self._run_btn.setToolTip("")
        self._navigate_btn.setEnabled(bool(cap.gui_route))
        self._feature_pack_download_btn.setEnabled(pack_id is not None)
        self._feature_pack_install_btn.setEnabled(pack_id is not None)
        self._feature_pack_online_btn.setEnabled(
            pack_id is not None and not self._install_in_progress
        )

    def _refresh_api_table(self) -> None:
        if not hasattr(self, "_api_table"):
            return

        query = self._search_edit.text().strip().lower() if hasattr(self, "_search_edit") else ""
        selected = self._selected_api.import_path if self._selected_api is not None else None
        self._visible_api_items = list(
            filter(lambda item: not query or query in _api_haystack(item), self._core_api_items)
        )

        self._api_table.blockSignals(True)
        self._api_table.setRowCount(len(self._visible_api_items))
        selected_row = -1
        row = 0
        while row < len(self._visible_api_items):
            item = self._visible_api_items[row]
            values = [
                item.category,
                item.kind,
                item.name,
                item.module,
                item.gui_route,
                item.signature,
            ]
            col = 0
            while col < len(values):
                value = values[col]
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(_api_tooltip(item))
                self._api_table.setItem(row, col, table_item)
                col += 1
            if item.import_path == selected:
                selected_row = row
            row += 1
        self._api_table.blockSignals(False)

        if selected_row >= 0:
            self._api_table.selectRow(selected_row)
        elif self._visible_api_items:
            self._selected_api = self._visible_api_items[0]
            self._api_table.selectRow(0)
        else:
            self._selected_api = None
        self._api_route_btn.setEnabled(self._selected_api is not None)

    def _on_api_selected(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._visible_api_items):
            self._selected_api = None
            self._api_route_btn.setEnabled(False)
            return
        self._selected_api = self._visible_api_items[row]
        self._api_route_btn.setEnabled(True)
        self._render_api_detail(self._selected_api)

    def _render_api_detail(self, item: CoreApiItem) -> None:
        detail = {
            "kind": item.kind,
            "name": item.name,
            "signature": item.signature,
            "import_path": item.import_path,
            "module": item.module,
            "gui_route": item.gui_route,
            "file": item.file_path,
            "line": item.line,
            "doc": item.doc,
        }
        self._result_text.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2))

    def _dependency_summary(self, dependencies: tuple[str, ...]) -> dict[str, str]:
        if not dependencies:
            return {"required": "none"}
        return dict(
            map(lambda dep: (dep, "OK" if _module_available(dep) else "MISSING"), dependencies)
        )

    def _dependencies_available(self, dependencies: tuple[str, ...]) -> bool:
        return all(map(_module_available, dependencies))

    def _run_selected_demo(self) -> None:
        cap = self._selected_capability
        if cap is None or cap.demo_id is None:
            self._result_text.setPlainText("실행 가능한 데모가 없습니다.")
            return
        try:
            result = run_capability_demo(cap.demo_id)
            self._result_text.setPlainText(_format_result(result))
            self.capability_done.emit(cap.id, result)
        except Exception as exc:  # noqa: BLE001
            self._result_text.setPlainText(f"[ERROR] {cap.id}\n{exc}")

    def _open_selected_feature_pack_download(self) -> None:
        pack_id = self._selected_feature_pack_id
        if pack_id is None:
            return
        QDesktopServices.openUrl(QUrl(default_release_asset_url(pack_id)))

    def _install_selected_feature_pack_zip(self) -> None:
        pack_id = self._selected_feature_pack_id
        if pack_id is None:
            return
        archive, _filter = QFileDialog.getOpenFileName(
            self,
            "Feature Pack ZIP 선택",
            "",
            "Feature Pack ZIP (*.zip)",
        )
        if not archive:
            return
        try:
            status = install_feature_pack_archive(archive, expected_pack_id=pack_id)
            activate_installed_feature_packs()
            self._libraries = list_library_statuses()
            self._refresh_library_table()
            self._render_capability_detail(self._selected_capability)
            self._refresh_pack_table()
            self._result_text.setPlainText(_format_result({"feature_pack": status}))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Feature Pack 설치 실패", str(exc))

    def _install_selected_feature_pack_online(self) -> None:
        """선택된 capability 의 권장 pack 을 PyPI 에서 바로 설치."""
        pack_id = self._selected_feature_pack_id
        if pack_id is None:
            return
        self._kick_off_online_install(pack_id)

    def _kick_off_online_install(self, pack_id: str) -> None:
        """백그라운드 QThread 로 ``install_feature_pack_online`` 실행."""
        if self._install_in_progress:
            QMessageBox.information(
                self,
                "Feature Pack 설치 진행 중",
                "다른 Feature Pack 설치가 진행 중입니다. 완료 후 다시 시도하세요.",
            )
            return
        try:
            spec = get_feature_pack_spec(pack_id)
        except KeyError:
            QMessageBox.warning(self, "알 수 없는 Feature Pack", pack_id)
            return
        confirm = QMessageBox.question(
            self,
            "Feature Pack 온라인 설치",
            (
                f"'{spec.name}' 을 PyPI 에서 다운로드해 설치합니다.\n\n"
                f"패키지: {', '.join(spec.packages)}\n"
                f"용량: 수백 MB ~ 1 GB. 인터넷 + 디스크 공간이 필요합니다.\n\n"
                "계속하시겠습니까?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        from PySide6.QtCore import QThread

        thread = QThread(self)
        worker = _FeaturePackInstallWorker(pack_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_online_install_finished)
        worker.failed.connect(self._on_online_install_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._install_thread = thread
        self._install_worker = worker
        self._set_install_in_progress(True, pack_id)
        thread.start()

    def _set_install_in_progress(self, busy: bool, pack_id: str | None = None) -> None:
        self._install_in_progress = busy
        self._feature_pack_online_btn.setEnabled(
            (not busy) and (self._selected_feature_pack_id is not None)
        )
        if busy and pack_id is not None:
            self._result_text.setPlainText(
                f"[설치 진행] '{pack_id}' 다운로드 + 설치 중 ... (수 분 소요 가능)"
            )

    @Slot(object)
    def _on_online_install_finished(self, status: object) -> None:
        activate_installed_feature_packs()
        self._libraries = list_library_statuses()
        self._refresh_library_table()
        self._render_capability_detail(self._selected_capability)
        self._refresh_pack_table()
        self._set_install_in_progress(False)
        self._result_text.setPlainText(_format_result({"feature_pack": status}))
        QMessageBox.information(
            self,
            "Feature Pack 설치 완료",
            "설치가 완료되었습니다. 새 모듈은 다음 GUI 작업부터 사용 가능합니다.",
        )

    @Slot(str, str)
    def _on_online_install_failed(self, pack_id: str, error_text: str) -> None:
        self._set_install_in_progress(False)
        self._refresh_pack_table()
        self._result_text.setPlainText(
            f"[ERROR] '{pack_id}' 설치 실패\n{error_text}"
        )
        QMessageBox.critical(
            self,
            "Feature Pack 설치 실패",
            (
                f"'{pack_id}' 설치 중 오류가 발생했습니다.\n\n"
                f"{error_text}\n\n"
                "인터넷 연결 / 프록시 / 디스크 공간을 확인하고 다시 시도하세요."
            ),
        )

    def _refresh_pack_table(self) -> None:
        """전체 Feature Pack 상태를 한눈에 표시 + 각 행에 설치 버튼."""
        if not hasattr(self, "_pack_table"):
            return
        pack_ids = list(FEATURE_PACKS.keys())
        self._pack_table.setRowCount(len(pack_ids))
        row = 0
        while row < len(pack_ids):
            pack_id = pack_ids[row]
            try:
                st = feature_pack_status(pack_id)
            except Exception:
                row += 1
                continue
            installed = bool(st.get("installed"))
            missing = st.get("missing_modules") or []
            unreadable = st.get("unreadable_paths") or []

            if unreadable and not installed:
                # ProgramData 등 system 경로가 권한 부족으로 못 읽힘 — 사용자가
                # GUI 의 재설치 버튼을 누르면 LOCALAPPDATA 로 새로 설치되어 해결됨.
                status_text = "권한 문제 — 재설치 필요"
                color = Qt.GlobalColor.red
                button_label = "내 계정에 재설치"
                tooltip = (
                    "인스톨러가 ProgramData 에 설치한 site 디렉토리를 현재 "
                    "사용자가 읽지 못합니다.\n"
                    "권한 부족 경로:\n  " + "\n  ".join(unreadable) +
                    "\n\n[내 계정에 재설치] 누르면 LOCALAPPDATA 에 새로 설치합니다."
                )
            elif installed and not missing:
                status_text = "설치됨"
                color = Qt.GlobalColor.green
                button_label = "재설치"
                tooltip = ""
            elif installed and missing:
                status_text = "부분 설치 (모듈 누락)"
                color = Qt.GlobalColor.yellow
                button_label = "재설치"
                tooltip = f"누락된 모듈: {', '.join(missing)}"
            else:
                status_text = "미설치"
                color = Qt.GlobalColor.yellow
                button_label = "설치"
                tooltip = ""

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(color)
            if tooltip:
                status_item.setToolTip(tooltip)
            self._pack_table.setItem(row, 0, QTableWidgetItem(pack_id))
            self._pack_table.setItem(row, 1, status_item)
            self._pack_table.setItem(
                row, 2,
                QTableWidgetItem(", ".join(missing) if missing else "-")
            )
            btn = QPushButton(button_label)
            btn.setEnabled(not self._install_in_progress)
            if tooltip:
                btn.setToolTip(tooltip)
            btn.clicked.connect(
                lambda _checked=False, pid=pack_id: self._kick_off_online_install(pid)
            )
            self._pack_table.setCellWidget(row, 3, btn)
            row += 1

    def _navigate_selected(self) -> None:
        cap = self._selected_capability
        if cap is not None and cap.gui_route:
            self.navigate_requested.emit(cap.gui_route)

    def _navigate_selected_api(self, *_args: object) -> None:
        item = self._selected_api
        if item is not None and item.gui_route:
            self.navigate_requested.emit(item.gui_route)


def _api_haystack(item: CoreApiItem) -> str:
    return (
        f"{item.category} {item.kind} {item.name} {item.module} "
        f"{item.signature} {item.doc}"
    ).lower()


def _api_tooltip(item: CoreApiItem) -> str:
    return f"{item.import_path}\n{item.file_path}:{item.line}"


def _format_result(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if value.size else None,
            "max": float(np.nanmax(value)) if value.size else None,
            "mean": float(np.nanmean(value)) if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return dict(map(lambda item: (str(item[0]), _json_safe(item[1])), value.items()))
    if isinstance(value, (list, tuple)):
        return list(map(_json_safe, value))
    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    return value


__all__ = [
    "CapabilitySpec",
    "CoreApiItem",
    "LibraryPanel",
    "LibraryStatus",
    "list_capability_specs",
    "list_core_api_items",
    "list_library_statuses",
    "run_capability_demo",
]
