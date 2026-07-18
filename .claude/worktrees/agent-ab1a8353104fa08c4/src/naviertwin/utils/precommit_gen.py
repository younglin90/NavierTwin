"""Pre-commit config generator (ruff + mypy).

Examples:
    >>> from naviertwin.utils.precommit_gen import precommit_yaml
    >>> "ruff" in precommit_yaml()
    True
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v{ruff}
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v{mypy}
    hooks:
      - id: mypy
        additional_dependencies: [numpy, types-PyYAML]
"""


def precommit_yaml(ruff: str = "0.6.0", mypy: str = "1.10.0") -> str:
    return _TEMPLATE.format(ruff=ruff, mypy=mypy)


def write_precommit(path: str | Path = ".pre-commit-config.yaml") -> None:
    Path(path).write_text(precommit_yaml())


__all__ = ["precommit_yaml", "write_precommit"]
