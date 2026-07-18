"""Native extension build hooks for NavierTwin.

The Python package remains the public API.  C++ kernels are optional
accelerators: if a platform cannot build them, setuptools can still produce a
usable pure-Python wheel.
"""

from __future__ import annotations

from setuptools import Extension, setup

try:
    import numpy as np
    import pybind11
except Exception:  # pragma: no cover - build environment fallback
    ext_modules = []
else:
    ext_modules = [
        Extension(
            "naviertwin._native._kernels",
            ["src/naviertwin/_native/kernels.cpp"],
            include_dirs=[pybind11.get_include(), np.get_include()],
            language="c++",
            optional=True,
            extra_compile_args=["/std:c++17"] if __import__("os").name == "nt" else ["-std=c++17", "-O3"],
        )
    ]


setup(ext_modules=ext_modules)
