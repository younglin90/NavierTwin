"""NavierTwin REST API public API (FastAPI 기반, optional).

사용:
    >>> from naviertwin.api import create_app
    >>> app = create_app()
    >>> app.title
    'NavierTwin API'
"""

from naviertwin.api.server import (
    BayesianOptReq,
    CouetteReq,
    LBMReq,
    PODReq,
    PoiseuilleReq,
    TwinBenchmarkReq,
    TwinBuildReq,
    TwinPackageAcceptReq,
    TwinPackageCreateReq,
    TwinPackageInspectReq,
    TwinPackageVerifyReq,
    TwinPredictReq,
    app,
    create_app,
)

__all__ = [
    "BayesianOptReq",
    "CouetteReq",
    "LBMReq",
    "PODReq",
    "PoiseuilleReq",
    "TwinBuildReq",
    "TwinBenchmarkReq",
    "TwinPackageAcceptReq",
    "TwinPackageCreateReq",
    "TwinPackageInspectReq",
    "TwinPackageVerifyReq",
    "TwinPredictReq",
    "app",
    "create_app",
]
