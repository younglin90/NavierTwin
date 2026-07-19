"""NavierTwin REST API public API (FastAPI 기반, optional).

사용:
    >>> from naviertwin.api import create_app
    >>> app = create_app()
    >>> app.title
    'NavierTwin API'
"""

from naviertwin.api.operations import (
    APIMetrics,
    APISettings,
    SlidingWindowRateLimiter,
    SQLiteWindowRateLimiter,
    api_key_sha256,
)
from naviertwin.api.server import (
    BayesianOptReq,
    CouetteReq,
    LBMReq,
    PODReq,
    PoiseuilleReq,
    PreflightReq,
    TwinBenchmarkReq,
    TwinBuildReq,
    TwinPackageAcceptReq,
    TwinPackageCreateReq,
    TwinPackageInspectReq,
    TwinPackageVerifyReq,
    TwinPredictReq,
    TwinStreamAlignReq,
    TwinStreamCloseReq,
    TwinStreamInitReq,
    TwinStreamObserveBatchReq,
    TwinStreamObserveLineReq,
    TwinStreamObserveReq,
    TwinStreamSensorReq,
    TwinStreamStepReq,
    app,
    create_app,
)

__all__ = [
    "APIMetrics",
    "APISettings",
    "SlidingWindowRateLimiter",
    "SQLiteWindowRateLimiter",
    "api_key_sha256",
    "BayesianOptReq",
    "CouetteReq",
    "LBMReq",
    "PODReq",
    "PoiseuilleReq",
    "PreflightReq",
    "TwinBuildReq",
    "TwinBenchmarkReq",
    "TwinPackageAcceptReq",
    "TwinPackageCreateReq",
    "TwinPackageInspectReq",
    "TwinPackageVerifyReq",
    "TwinPredictReq",
    "TwinStreamInitReq",
    "TwinStreamAlignReq",
    "TwinStreamCloseReq",
    "TwinStreamObserveBatchReq",
    "TwinStreamObserveLineReq",
    "TwinStreamObserveReq",
    "TwinStreamSensorReq",
    "TwinStreamStepReq",
    "app",
    "create_app",
]
