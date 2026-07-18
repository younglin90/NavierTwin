"""Sphinx 문서 빌드 설정."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_project_version() -> str:
    """Read the package version without importing the package."""
    pyproject = ROOT / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("Could not find project.version in pyproject.toml")


# src/ 를 path 에 추가 — editable install 안 되어 있어도 API 문서화
sys.path.insert(0, str(ROOT / "src"))

# -- Project info ----------------------------------------------------------
project = "NavierTwin"
author = "NavierTwin Contributors"
copyright = "2026, NavierTwin Contributors"

try:
    from naviertwin import __version__ as release
except ImportError:
    release = _read_project_version()

version = ".".join(release.split(".")[:2])

# -- General configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "ko"

# -- HTML output -----------------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]

# -- Intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pyvista": ("https://docs.pyvista.org", None),
}

# -- Todo -----------------------------------------------------------------
todo_include_todos = True
