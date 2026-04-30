"""REST API package root public API tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for API surface tests")


def test_api_root_exports_fastapi_entrypoints() -> None:
    """Package root should expose the customer-facing REST app surface."""
    import naviertwin.api as api
    from naviertwin.api.server import (
        CouetteReq,
        LBMReq,
        PODReq,
        TwinBenchmarkReq,
        TwinBuildReq,
        TwinPackageAcceptReq,
        TwinPackageCreateReq,
        TwinPackageInspectReq,
        TwinPackageVerifyReq,
        TwinPredictReq,
        create_app,
    )

    expected = {
        "create_app",
        "app",
        "CouetteReq",
        "PoiseuilleReq",
        "PODReq",
        "BayesianOptReq",
        "TwinBuildReq",
        "TwinPredictReq",
        "TwinBenchmarkReq",
        "TwinPackageAcceptReq",
        "TwinPackageCreateReq",
        "TwinPackageInspectReq",
        "TwinPackageVerifyReq",
        "LBMReq",
    }

    assert expected.issubset(set(api.__all__))
    assert api.create_app is create_app
    assert api.CouetteReq is CouetteReq
    assert api.PODReq is PODReq
    assert api.TwinBuildReq is TwinBuildReq
    assert api.TwinPredictReq is TwinPredictReq
    assert api.TwinBenchmarkReq is TwinBenchmarkReq
    assert api.TwinPackageAcceptReq is TwinPackageAcceptReq
    assert api.TwinPackageCreateReq is TwinPackageCreateReq
    assert api.TwinPackageInspectReq is TwinPackageInspectReq
    assert api.TwinPackageVerifyReq is TwinPackageVerifyReq
    assert api.LBMReq is LBMReq
    assert api.app is not None


def test_api_root_create_app_exposes_advertised_routes() -> None:
    """Root create_app import path should produce the documented REST routes."""
    from naviertwin.api import create_app

    app = create_app()
    routes = {route.path for route in app.routes if hasattr(route, "path")}

    assert {
        "/health",
        "/reduce",
        "/reduce/pod",
        "/twin/build",
        "/twin/predict",
        "/twin/benchmark",
        "/twin/package",
        "/twin/package/inspect",
        "/twin/package/verify",
        "/twin/package/accept",
        "/analytic/couette",
        "/analytic/poiseuille_2d",
        "/simulate/lbm_cavity",
        "/optimize/bayesian",
    }.issubset(routes)
