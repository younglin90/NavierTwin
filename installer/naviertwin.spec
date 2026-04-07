# -*- mode: python ; coding: utf-8 -*-
# NavierTwin PyInstaller 스펙 파일
# 빌드: pyinstaller installer/naviertwin.spec
# 출력: installer/dist/naviertwin/

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # 프로젝트 루트
SRC = ROOT / "src"

# ──────────────────────────────────────────────────────────────────────
# 숨겨진 임포트 (런타임에 동적으로 로드되는 모듈)
# ──────────────────────────────────────────────────────────────────────
hidden_imports = [
    # PyVista / VTK
    "vtkmodules.all",
    "vtkmodules.util.numpy_support",
    "vtkmodules.util.data_model",
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
    # HDF5
    "h5py",
    # NavierTwin 모듈
    "naviertwin",
    "naviertwin.core",
    "naviertwin.core.cfd_reader",
    "naviertwin.core.cfd_reader.reader_factory",
    "naviertwin.core.cfd_reader.openfoam_reader",
    "naviertwin.core.cfd_reader.vtk_reader",
    "naviertwin.core.export.ntwin_format",
    "naviertwin.core.flow_analysis.vortex.q_criterion",
    "naviertwin.core.flow_analysis.statistics.fft_psd",
    "naviertwin.core.flow_analysis.boundary_layer.yplus",
    "naviertwin.core.flow_analysis.modal.dmd",
    "naviertwin.core.dimensionality_reduction.linear.pod",
    "naviertwin.core.dimensionality_reduction.linear.randomized_svd",
    "naviertwin.core.surrogate.rbf_surrogate",
    "naviertwin.core.surrogate.kriging_surrogate",
    "naviertwin.core.validation.metrics",
    "naviertwin.core.digital_twin.twin_engine",
    "naviertwin.gui.main_window",
    "naviertwin.gui.panels.import_panel",
    "naviertwin.gui.panels.analyze_panel",
    "naviertwin.gui.panels.reduce_panel",
    "naviertwin.gui.panels.model_panel",
    "naviertwin.gui.panels.twin_panel",
    "naviertwin.gui.panels.export_panel",
    "naviertwin.gui.widgets.vtk_viewer",
    "naviertwin.utils.config",
    "naviertwin.utils.logger",
]

# SMT (선택적)
try:
    import smt
    hidden_imports += collect_submodules("smt")
except ImportError:
    pass

# PyDMD (선택적)
try:
    import pydmd
    hidden_imports += collect_submodules("pydmd")
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────
# 데이터 파일 (QSS, 설정, 리소스)
# ──────────────────────────────────────────────────────────────────────
datas = [
    # 다크 테마 QSS
    (str(SRC / "naviertwin" / "gui" / "styles" / "dark_theme.qss"),
     "naviertwin/gui/styles"),
    # 기본 설정 파일 (있는 경우)
    # (str(ROOT / "config" / "default.json"), "config"),
]

# VTK 데이터 파일
try:
    datas += collect_data_files("vtkmodules")
except Exception:
    pass

# PySide6 데이터
try:
    datas += collect_data_files("PySide6")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────
# 바이너리 제외 (불필요한 크기 절감)
# ──────────────────────────────────────────────────────────────────────
excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "wx",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "test",
    "tests",
    "pytest",
]

# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
a = Analysis(
    [str(SRC / "naviertwin" / "main.py")],
    pathex=[str(SRC)],
    binaries=[],
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
    name="naviertwin",
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
    name="naviertwin",
    distpath=str(ROOT / "installer" / "dist"),
)
