"""CFD reader package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_cfd_reader_root_exports_advanced_helpers() -> None:
    """Package root should expose customer-facing CFD format helper APIs."""
    import naviertwin.core.cfd_reader as cfd_reader

    expected = {
        "read_foam_dict": "naviertwin.core.cfd_reader.foamlib_case",
        "set_foam_value": "naviertwin.core.cfd_reader.foamlib_case",
        "modify_transport_properties": "naviertwin.core.cfd_reader.foamlib_case",
        "parameter_sweep": "naviertwin.core.cfd_reader.foamlib_case",
        "sample_field_at_points": "naviertwin.core.cfd_reader.foamlib_case",
        "has_h5py": "naviertwin.core.cfd_reader.cgns_advanced",
        "iter_zones": "naviertwin.core.cfd_reader.cgns_advanced",
        "list_zones": "naviertwin.core.cfd_reader.cgns_advanced",
        "parse_section_ids": "naviertwin.core.cfd_reader.fluent_cas_ext",
        "section_count": "naviertwin.core.cfd_reader.fluent_cas_ext",
        "list_zone_names": "naviertwin.core.cfd_reader.fluent_cas_ext",
    }

    assert len(cfd_reader.__all__) == len(set(cfd_reader.__all__))
    assert set(expected).issubset(set(cfd_reader.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(cfd_reader, symbol) is getattr(source_module, symbol)


def test_cfd_reader_root_does_not_eagerly_import_foamlib_backend() -> None:
    """Root import should not require the optional foamlib case-edit backend."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.core.cfd_reader; "
        "raise SystemExit(1 if 'foamlib' in sys.modules else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0
