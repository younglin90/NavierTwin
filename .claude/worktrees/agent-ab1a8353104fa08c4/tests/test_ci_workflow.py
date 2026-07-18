"""CI workflow release-smoke regression tests."""

from __future__ import annotations

from pathlib import Path


def _ci_text() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ci_uses_core_dev_install_and_headless_env() -> None:
    """CI must install runtime deps and run Qt-capable tests headlessly."""
    text = _ci_text()

    assert 'python -m pip install -e ".[core,dev]"' in text
    assert "QT_QPA_PLATFORM: offscreen" in text
    assert "MPLCONFIGDIR: /tmp/mpl" in text
    assert "NAVIER_TWIN_LOG_DIR: /tmp/naviertwin/logs" in text


def test_ci_uses_release_smoke_not_full_matrix_pytest() -> None:
    """PR CI should use the curated release smoke instead of a huge suite."""
    text = _ci_text()

    assert "python scripts/release_smoke.py" in text
    assert "python scripts/installer_smoke.py" in text
    assert 'pytest -q -m "not optional"' not in text
    assert "naviertwin --help" in text
    assert "python -m naviertwin.main --help" in text


def test_ci_validates_wheel_artifact() -> None:
    """CI should verify wheel metadata and bundled runtime assets."""
    text = _ci_text()

    assert "python scripts/wheel_smoke.py --install-smoke" in text


def test_ci_validates_sdist_artifact() -> None:
    """CI should verify source distribution metadata and bundled runtime assets."""
    text = _ci_text()

    assert "python scripts/sdist_smoke.py --install-smoke" in text


def test_verify_all_includes_customer_smoke_gates() -> None:
    """Aggregate verification should not omit customer-facing smoke gates."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "verify_all.sh").read_text(encoding="utf-8")

    assert "scripts/release_smoke.py" in text
    assert "scripts/installer_smoke.py" in text
    assert "scripts/wheel_smoke.py --outdir /tmp/naviertwin-wheel-smoke-verifyall --install-smoke" in text
    assert "scripts/sdist_smoke.py --outdir /tmp/naviertwin-sdist-smoke-verifyall --install-smoke" in text
    assert "smoke.json" in text
