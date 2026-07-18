"""GitHub Actions CI YAML generator.

Examples:
    >>> from naviertwin.utils.github_ci_gen import ci_yaml
    >>> "pytest" in ci_yaml()
    True
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: {versions}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{{{ matrix.python-version }}}}
      - run: pip install -e .[dev]
      - run: ruff check src/ tests/
      - run: pytest tests/ -q
"""


def ci_yaml(versions: list[str] | None = None) -> str:
    versions = versions or ["3.10", "3.11", "3.12"]
    return _TEMPLATE.format(versions=str(versions).replace("'", '"'))


def write_ci(path: str | Path = ".github/workflows/ci.yml") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ci_yaml())


__all__ = ["ci_yaml", "write_ci"]
