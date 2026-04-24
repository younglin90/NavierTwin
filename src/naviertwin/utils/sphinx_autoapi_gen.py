"""Sphinx + autoapi conf.py generator.

Examples:
    >>> from naviertwin.utils.sphinx_autoapi_gen import sphinx_conf
    >>> "autoapi" in sphinx_conf()
    True
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
project = {project!r}
extensions = ["autoapi.extension"]
autoapi_dirs = [{src_path!r}]
autoapi_type = "python"
html_theme = "alabaster"
"""


def sphinx_conf(project: str = "NavierTwin", src_path: str = "../src") -> str:
    return _TEMPLATE.format(project=project, src_path=src_path)


def write_sphinx_conf(
    path: str | Path = "docs/conf.py",
    project: str = "NavierTwin",
    src_path: str = "../src",
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sphinx_conf(project, src_path))


__all__ = ["sphinx_conf", "write_sphinx_conf"]
