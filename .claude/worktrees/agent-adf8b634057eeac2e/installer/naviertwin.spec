# -*- mode: python ; coding: utf-8 -*-
# NavierTwin PyInstaller 스펙 파일
# 빌드: pyinstaller installer/naviertwin.spec
# 출력: dist/NavierTwin/

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

def _resolve_project_root():
    """Resolve the project root robustly across relative/absolute PyInstaller calls."""
    candidates = [
        Path.cwd(),
        Path(SPECPATH),
        Path(SPECPATH).parent,
        Path(SPECPATH).parent.parent,
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "src" / "naviertwin" / "main.py").exists():
            return root
    raise FileNotFoundError(
        "Cannot resolve NavierTwin project root; src/naviertwin/main.py not found"
    )


ROOT = _resolve_project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BUILD_PROFILE = os.environ.get("NAVIER_TWIN_BUILD_PROFILE", "desktop").strip().lower()
if BUILD_PROFILE not in {"desktop", "full"}:
    raise ValueError("NAVIER_TWIN_BUILD_PROFILE must be either 'desktop' or 'full'")

# ──────────────────────────────────────────────────────────────────────
# 숨겨진 임포트 (런타임에 동적으로 로드되는 모듈)
# ──────────────────────────────────────────────────────────────────────
hidden_imports = [
    # PyVista / VTK
    "vtkmodules.util.numpy_support",
    "vtkmodules.vtkCommonCore",
    "vtkmodules.vtkCommonDataModel",
    "vtkmodules.vtkFiltersCore",
    "vtkmodules.vtkFiltersGeometry",
    "vtkmodules.vtkFiltersGeneral",
    "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkIOGeometry",
    "vtkmodules.vtkIOLegacy",
    "vtkmodules.vtkIOPLY",
    "vtkmodules.vtkIOXML",
    "vtkmodules.vtkRenderingCore",
    "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkRenderingOpenGL2",
    "pyvista",
    "pyvistaqt",
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    # Scientific
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.linalg",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.gaussian_process",
    "sklearn.decomposition",
    "pandas",
    "pip",
    "pip._internal",
    "pip._internal.cli.main",
    "pydmd",
    "SALib",
    "smt",
    "smt.sampling_methods",
    "smt.surrogate_models",
    # HDF5
    "h5py",
    # Matplotlib Qt backend is optional at runtime but should be bundled
    # whenever it is installed on the Windows release builder.
    "matplotlib.backends.backend_qtagg",
    "matplotlib.figure",
    # GUI modules reached by lazy imports / optional panels.
    "naviertwin.gui.main_window",
    "naviertwin.gui.panels.simulation_panel",
    "naviertwin.gui.panels.postproc_panel",
    "naviertwin.gui.widgets.model_compare_widget",
    "naviertwin.gui.widgets.vtk_viewer",
    "naviertwin.gui.wizard.tutorial_wizard",
]

if BUILD_PROFILE == "full":
    hidden_imports += [
        "fastapi",
        "onnx",
        "pydmd",
        "smt",
        "torch",
        "uvicorn",
        "weasyprint",
    ]

# The installer can install selected optional feature packs during Setup by
# running ``NavierTwin.exe --install-feature-pack``.  That code path needs a
# complete pip runtime, including vendored distlib wrapper resources.
pip_datas, pip_binaries, pip_hidden_imports = collect_all("pip", include_py_files=True)
hidden_imports += pip_hidden_imports

# ──────────────────────────────────────────────────────────────────────
# 데이터 파일 (QSS, 설정, 리소스)
# ──────────────────────────────────────────────────────────────────────
datas = list(pip_datas)
binaries = list(pip_binaries)

for qss_file in (SRC / "naviertwin" / "gui" / "styles").glob("*.qss"):
    datas.append((str(qss_file), "naviertwin/gui/styles"))

for locale_file in (SRC / "naviertwin" / "gui" / "styles" / "i18n").glob("*.json"):
    datas.append((str(locale_file), "naviertwin/gui/styles/i18n"))

resources_dir = ROOT / "resources"
if resources_dir.exists():
    datas.append((str(resources_dir), "resources"))

# ──────────────────────────────────────────────────────────────────────
# 바이너리 제외 (불필요한 크기 절감)
# ──────────────────────────────────────────────────────────────────────
excludes = [
    # GUI installer must not bundle developer/test stacks.
    "tkinter",
    "PyQt5",
    "PyQt6",
    "wx",
    "IPython",
    "jupyter",
    "notebook",
    "test",
    "tests",
    "pytest",
    "mypy",
    "setuptools.tests",
    # Server/cloud/PDF/reporting packages are optional and very large.
    "aiohttp",
    "boto3",
    "botocore",
    "fastapi",
    "google",
    "grpc",
    "psycopg2",
    "s3transfer",
    "sqlalchemy",
    "uvicorn",
    "watchfiles",
    "weasyprint",
    # PyVista's browser/Jupyter viewer stack is not used by the Qt desktop app.
    "trame",
    "trame_client",
    "trame_server",
    "trame_vtk",
    "trame_vuetify",
    "wslink",
    "multidict",
    "propcache",
    "yarl",
    # API/schema support is excluded from the size-optimized desktop profile.
    "attrs",
    "jsonschema",
    "jsonschema_specifications",
    "pydantic",
    "pydantic_core",
    "rpds",
    "shap",
    "captum",
    "pywt",
    # Heavy optional scientific backends. They can be enabled via full profile.
    "botorch",
    "gmsh",
    "llvmlite",
    "netCDF4",
    "nlopt",
    "numba",
    "onnx",
    "openmdao",
    "pymeshlab",
    "pyspod",
    "torch",
    "torch_geometric",
    "torchdiffeq",
    "xarray",
    "zarr",
    # Unused Qt stacks that dominate installer size when collected wholesale.
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
]

if BUILD_PROFILE == "full":
    excludes = [
        item
        for item in excludes
        if item
        not in {
            "fastapi",
            "onnx",
            "pydmd",
            "smt",
            "torch",
            "torch_geometric",
            "torchdiffeq",
            "uvicorn",
            "weasyprint",
            "attrs",
            "jsonschema",
            "jsonschema_specifications",
            "pydantic",
            "pydantic_core",
            "rpds",
            "shap",
            "captum",
            "pywt",
        }
    ]

# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
a = Analysis(
    [str(SRC / "naviertwin" / "gui_entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)


def _drop_desktop_bundle_item(item):
    """Return True for optional desktop bundle artifacts safe to prune."""
    target = str(item[0]).replace("\\", "/")
    source = str(item[1]).replace("\\", "/") if len(item) > 1 else ""
    paths = (target, source)
    drop_prefixes = (
        "trame_client",
        "trame_vtk",
        "trame_vuetify",
        "trimesh",
        "jsonschema",
        "jsonschema_specifications",
        "pydantic",
        "pydantic_core",
        "rpds",
        "shap",
        "captum",
        "pywt",
        "attrs",
        "multidict",
        "propcache",
        "yarl",
    )
    drop_suffixes = (
        "PySide6/Qt6Pdf.dll",
        "PySide6/Qt6Qml.dll",
        "PySide6/Qt6QmlMeta.dll",
        "PySide6/Qt6QmlModels.dll",
        "PySide6/Qt6QmlWorkerScript.dll",
        "PySide6/Qt6Quick.dll",
        "PySide6/Qt6VirtualKeyboard.dll",
    )
    for path in paths:
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in drop_prefixes):
            return True
        if any(path.endswith(suffix) for suffix in drop_suffixes):
            return True
        if path.startswith("PySide6/translations/") and not (
            path.endswith("_ko.qm") or path.endswith("_en.qm")
        ):
            return True
    return False


if BUILD_PROFILE == "desktop":
    a.binaries = [item for item in a.binaries if not _drop_desktop_bundle_item(item)]
    a.datas = [item for item in a.datas if not _drop_desktop_bundle_item(item)]

# ──────────────────────────────────────────────────────────────────────
# PYZ (Python 아카이브)
# ──────────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ──────────────────────────────────────────────────────────────────────
# EXE (실행 파일)
# ──────────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NavierTwin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # Windows GUI 모드 (콘솔 창 없음)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="installer/naviertwin.ico",  # 아이콘 파일 경로 (준비 시 활성화)
)

# ──────────────────────────────────────────────────────────────────────
# COLLECT (--onedir 출력 디렉토리)
# ──────────────────────────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NavierTwin",
    distpath=str(ROOT / "dist"),
)
