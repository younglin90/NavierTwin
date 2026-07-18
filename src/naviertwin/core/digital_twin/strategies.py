"""능력(capability) 기반 트윈 전략 레지스트리 — v5.0 데이터 모델 척추.

전략마다 "무엇이 필요하고 무엇을 지원하는가"를 선언하고, 로드된 데이터의 특성
(:class:`DataProfile`)과 대조해 **가능/불가 + 이유**를 돌려준다.

왜 필요한가: 기존 ``recommend_method`` 는 (ImageData 여부, 스텝 수) 두 가지만 보는
휴리스틱이라 형상 가변·케이스 세트·차원 같은 축을 몰랐다. 트윈 전략이 늘수록
"이 데이터엔 뭐가 되는가"를 사람이 외워야 했고, 안 되는 조합은 학습 버튼을 눌러야
에러로 알 수 있었다. 여기서는 그 판정을 데이터 로드 시점에 정확히 내려 UI 가
미리 보여준다 (로드맵 §4.2).

이 모듈은 Qt/torch/trame 에 의존하지 않는다 — 순수 판정 로직이라 core 에 두고
데스크톱(gui)과 웹(web)이 공유한다 (웹 shim: ``naviertwin.web.strategies``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "STRATEGIES",
    "TIER_LABELS",
    "DataProfile",
    "StrategySpec",
    "profile_data",
    "recommend",
    "strategy_report",
]


@dataclass(frozen=True)
class DataProfile:
    """로드된 데이터의 학습 관점 특성 — 전략 판정의 입력.

    Attributes:
        n_cases: 케이스 수 (단일 데이터셋이면 1).
        n_time_steps: 케이스당 최소 타임스텝 수.
        total_snapshots: 전 케이스 스냅샷 합 (= 학습 샘플 수).
        identical_mesh: 모든 케이스가 같은 격자를 공유하는가.
        uniform_grid: 균일 격자(ImageData)인가.
        n_points: 대표(첫) 케이스의 점 수.
        dims: 공간 차원 (1/2/3 — 바운딩 박스의 유한 두께 축 수).
        n_params: 케이스 파라미터(μ) 차원 (케이스 세트가 아니면 0).
        topological_dim: 셀이 실제로 span 하는 차원 — 바운딩 박스에서 두께 0
            축을 뺀 개수 (2D 평면 메쉬=2, 볼륨=3). ``dims`` 와 같은 값이지만
            embedding_dim 과의 대비를 위해 이름을 명시한다 (리뷰 #10).
        embedding_dim: 좌표가 사는 공간의 차원 — 점 좌표 배열의 성분 수
            (PyVista 는 항상 3). 2D 평면 메쉬도 3D 공간에 놓이므로 보통 3.
    """

    n_cases: int
    n_time_steps: int
    total_snapshots: int
    identical_mesh: bool
    uniform_grid: bool
    n_points: int
    dims: int
    n_params: int
    topological_dim: int = 3
    embedding_dim: int = 3


@dataclass(frozen=True)
class StrategySpec:
    """트윈 전략 하나의 요구/지원 선언.

    Attributes:
        key: 앱의 ``nt_model_method`` 값 ("rom" | "physics" | "dynamics" | "operator").
        name: 표시 이름.
        needs_identical_mesh: 케이스 간 동일 격자가 필수인가 (POD 스냅샷 쌓기).
        needs_uniform_grid: 균일 격자(ImageData)가 필수인가 (FFT 등).
        supports_case_sets: 파라미터 스윕(케이스 세트) 학습을 지원하는가.
        supports_time_in_sweep: 케이스 세트에서 시간축(t 파라미터)까지 지원하는가.
        single_case_needs_steps: 단일 케이스일 때 필요한 최소 타임스텝.
        min_snapshots: 학습에 필요한 최소 총 스냅샷.
        note: 사람용 한 줄 설명 (툴팁).
        tier: 모델 등급 (리뷰 #8) — "production"(검증됨) | "domain"(도메인
            특화) | "experimental"(실험적). UI 가 뱃지로 표시해 사용자가
            성숙도를 학습 전에 알게 한다.
    """

    key: str
    name: str
    needs_identical_mesh: bool
    needs_uniform_grid: bool
    supports_case_sets: bool
    supports_time_in_sweep: bool
    single_case_needs_steps: int
    min_snapshots: int
    note: str
    tier: str = "production"


#: tier 값 → 한국어 라벨 (UI 뱃지 텍스트).
TIER_LABELS: dict[str, str] = {
    "production": "검증됨",
    "domain": "도메인 특화",
    "experimental": "실험적",
}


# 현재 앱이 실제로 배선한 4개 전략의 선언. 새 전략(EZyRB, ParametricDMD, GINO,
# FNO+SDF …)을 붙일 때는 여기에 spec 하나를 더하는 것이 첫 단계다.
STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        key="rom",
        name="축소+보간 (ROM)",
        needs_identical_mesh=True,
        needs_uniform_grid=False,
        supports_case_sets=True,
        supports_time_in_sweep=True,
        single_case_needs_steps=2,
        min_snapshots=2,
        note="POD 로 압축 후 계수 보간 — 적은 스냅샷에 안정적. 케이스 간 동일 격자 필수.",
        tier="production",
    ),
    StrategySpec(
        key="physics",
        name="직접 회귀 (Physics AI)",
        needs_identical_mesh=False,
        needs_uniform_grid=False,
        supports_case_sets=True,
        supports_time_in_sweep=True,
        single_case_needs_steps=2,
        min_snapshots=2,
        note="좌표 기반 신경장 — 격자 무관, 형상 가변(케이스마다 다른 격자) 가능.",
        tier="production",
    ),
    StrategySpec(
        key="dynamics",
        name="동역학 예보 (DMD)",
        needs_identical_mesh=True,
        needs_uniform_grid=False,
        supports_case_sets=True,  # ParametricDMD (v5.2) — 비정상 스윕 예보
        supports_time_in_sweep=True,
        single_case_needs_steps=4,
        min_snapshots=4,
        note="시간 전이 규칙을 학습해 학습 구간 밖 t 까지 예보 — 비정상 스윕은 "
        "ParametricDMD(케이스별 DMD + μ 보간). 저랭크 선형 동역학일 때만 맞으니 "
        "적합도(재구성 오차) 확인 필수.",
        tier="production",
    ),
    StrategySpec(
        key="operator",
        name="신경 연산자 (FNO)",
        # 케이스 세트는 GeometryFNO(FNO+SDF) 가 내부에서 공통 격자로 재샘플·
        # 텐서화하므로 동일/균일 격자 제약이 없다. 단일 케이스 직접 학습은
        # 여전히 미배선(⑥연산자 랩 전용) — _check 가 먼저 거절한다.
        needs_identical_mesh=False,
        needs_uniform_grid=False,
        supports_case_sets=True,  # 정상 스윕 전용 — GeometryFNO(FNO+SDF, v5.2)
        supports_time_in_sweep=False,
        single_case_needs_steps=2,
        min_snapshots=100,  # 문헌 기준 규모 — 케이스 세트 분기는 별도 판정
        note="함수→함수 연산자 — 정상 케이스 세트는 SDF 채널(GeometryFNO)로 "
        "형상 가변까지 학습(공통 격자 자동 재샘플). 샘플 수백 장이 문헌 기준 — "
        "소수 케이스는 정성적. 단일 케이스 직접 학습은 ⑥연산자 랩 전용.",
        # 소표본 few-shot 한계가 note 에 명시된 대로 — 결과가 정성적 수준이라
        # 아직 "검증됨" 이 아니다 (리뷰 #8).
        tier="experimental",
    ),
)


def _same_mesh_points(cases: Sequence[Any]) -> bool:
    """점 좌표가 전부 일치하는가 — core 자급자족 판정 (web 의존 금지)."""
    try:
        import numpy as np

        ref = np.asarray(cases[0].mesh.points)
        for case in cases[1:]:
            pts = np.asarray(case.mesh.points)
            if pts.shape != ref.shape or not np.allclose(pts, ref):
                return False
        return True
    except Exception:  # noqa: BLE001 — 판정 실패는 보수적으로 False
        return False


def profile_data(
    dataset: Any, case_datasets: Sequence[Any] | None = None
) -> DataProfile:
    """데이터셋(+케이스 목록)에서 :class:`DataProfile` 을 계산한다.

    Args:
        dataset: 대표 데이터셋 (뷰어에 올라간 것).
        case_datasets: 케이스 세트면 전체 케이스 목록, 아니면 None/빈 목록.
    """
    cases = list(case_datasets) if case_datasets else [dataset]
    steps = [max(1, int(getattr(c, "n_time_steps", 1))) for c in cases]

    identical = True
    if len(cases) > 1:
        # 점 수가 하나라도 다르면 확실히 다른 격자 — 좌표 비교보다 싸고 충분하다
        # (같은 점 수의 다른 격자는 service.meshes_are_identical 이 정밀 판정).
        counts = {int(getattr(c, "n_points", 0)) for c in cases}
        if len(counts) > 1:
            identical = False
        else:
            identical = _same_mesh_points(cases)

    try:
        import pyvista as pv

        uniform = isinstance(dataset.mesh, pv.ImageData)
    except Exception:  # noqa: BLE001
        uniform = False

    # topological vs embedding 차원 분리 (리뷰 #10):
    #   topological_dim — 셀이 실제로 span 하는 차원 (두께 0 축 제외).
    #   embedding_dim — 좌표가 사는 공간 (점 좌표 성분 수, PyVista 는 항상 3).
    dims = 3
    try:
        bounds = dataset.mesh.bounds
        spans = [abs(bounds[2 * i + 1] - bounds[2 * i]) for i in range(3)]
        scale = max(max(spans), 1e-12)
        dims = max(1, sum(1 for s in spans if s > 1e-9 * scale))
    except Exception:  # noqa: BLE001
        pass

    embedding = 3
    try:
        embedding = int(dataset.mesh.points.shape[1])
    except Exception:  # noqa: BLE001
        pass

    return DataProfile(
        n_cases=len(cases),
        n_time_steps=min(steps),
        total_snapshots=sum(steps),
        identical_mesh=identical,
        uniform_grid=uniform,
        n_points=int(getattr(dataset, "n_points", 0)),
        dims=dims,
        n_params=0,  # 호출자가 케이스 파라미터를 알면 replace 로 채운다
        topological_dim=dims,
        embedding_dim=embedding,
    )


def _check(spec: StrategySpec, p: DataProfile) -> tuple[bool, str]:
    """전략 하나를 데이터 프로파일과 대조 — (가능 여부, 이유)."""
    if p.n_cases > 1:
        if not spec.supports_case_sets:
            return False, "케이스 세트(파라미터 스윕) 학습은 아직 미지원입니다."
        if spec.key == "operator":
            # GeometryFNO(FNO+SDF): 내부에서 공통 격자로 재샘플·텐서화하므로
            # 동일/균일 격자 제약이 없다. min_snapshots(문헌 규모)도 여기서는
            # 적용하지 않는다 — 소수 케이스 동작을 허용하되 note 의 few-shot
            # 경고("소수 케이스는 정성적")가 그 한계를 알린다.
            if p.n_time_steps > 1:
                return False, (
                    "비정상(시간축) 케이스 세트의 GeometryFNO 는 아직 "
                    "미지원입니다 — ROM/Physics AI/ParametricDMD 를 쓰세요."
                )
            if p.n_cases < 3:
                return False, (
                    f"케이스 {p.n_cases}개 — GeometryFNO 학습에는 최소 3개가 "
                    "필요합니다 (문헌 기준은 수백 장)."
                )
            return True, spec.note
        if spec.needs_identical_mesh and not p.identical_mesh:
            return False, (
                "케이스마다 격자가 달라 불가 — 동일 격자가 필요합니다. "
                "공통 격자로 재샘플하거나 Physics AI 를 쓰세요."
            )
        if p.n_time_steps > 1 and not spec.supports_time_in_sweep:
            return False, "비정상(시간축) 케이스 세트는 이 전략이 아직 미지원입니다."
        if spec.key == "dynamics" and p.n_time_steps < spec.single_case_needs_steps:
            return False, (
                f"케이스당 타임스텝 {p.n_time_steps}개 — 동역학 예보(ParametricDMD)에는 "
                f"케이스당 최소 {spec.single_case_needs_steps}개가 필요합니다 "
                "(정상 스윕이면 ROM/Physics AI 를 쓰세요)."
            )
    else:
        if p.n_time_steps < spec.single_case_needs_steps:
            return False, (
                f"타임스텝 {p.n_time_steps}개 — 이 전략은 최소 "
                f"{spec.single_case_needs_steps}개가 필요합니다."
            )
        if spec.key == "operator":
            return False, (
                "로드한 데이터 직접 학습은 아직 미지원 — ⑥연산자 랩의 벤치마크로 "
                "실험하세요 (FNO+SDF 채널 배선 예정)."
            )

    if spec.needs_uniform_grid and not p.uniform_grid:
        return False, "균일 격자(ImageData)가 필요합니다."
    if p.total_snapshots < spec.min_snapshots:
        return False, (
            f"스냅샷 {p.total_snapshots}개 — 최소 {spec.min_snapshots}개가 필요합니다."
        )
    return True, spec.note


def strategy_report(profile: DataProfile) -> dict[str, dict[str, Any]]:
    """모든 전략의 가능/불가 판정 — UI(②Model 카드)가 그대로 쓴다.

    Returns:
        ``{key: {"ok": bool, "reason": str, "name": str, "tier": str,
        "tier_label": str}}`` — tier 는 모델 등급(리뷰 #8), tier_label 은
        한국어 뱃지 텍스트.
    """
    report: dict[str, dict[str, Any]] = {}
    for spec in STRATEGIES:
        ok, reason = _check(spec, profile)
        report[spec.key] = {
            "ok": ok,
            "reason": reason,
            "name": spec.name,
            "tier": spec.tier,
            "tier_label": TIER_LABELS.get(spec.tier, spec.tier),
        }
    return report


def recommend(profile: DataProfile) -> dict[str, str]:
    """프로파일에서 최선 전략 추천 — (method, 한국어 reason).

    규칙(우선순위):
      1. 아무것도 불가하면 "none".
      2. 형상 가변(격자 다름) → Physics AI (유일하게 가능).
      3. 케이스 세트 → 스냅샷 적으면 ROM, 아니면 ROM 기본 + Physics AI 언급.
      4. 단일 시계열 → ROM 기본 (스텝 많고 균일 격자면 operator 언급).
    """
    report = strategy_report(profile)
    feasible = [k for k, v in report.items() if v["ok"]]
    header = (
        f"현재 데이터: 케이스 {profile.n_cases}개 × 타임스텝 {profile.n_time_steps}개 "
        f"· {profile.n_points:,} 포인트 · {profile.dims}D"
    )
    if not feasible:
        return {
            "method": "none",
            "reason": f"{header} — 학습 가능한 전략이 없습니다. "
            + report["rom"]["reason"],
        }
    if profile.n_cases > 1 and not profile.identical_mesh:
        # 정상 스윕이면 GeometryFNO(SDF 채널)도 가능해졌다 — "만 가능" 이라고
        # 말하지 않되 기본 추천은 소표본에 더 안전한 Physics AI 로 유지한다.
        operator_note = (
            " 신경 연산자(GeometryFNO·SDF 채널)도 시험할 수 있습니다 — "
            "소수 케이스에서는 정성적입니다."
            if report["operator"]["ok"]
            else ""
        )
        return {
            "method": "physics",
            "reason": f"{header} · 케이스마다 격자가 다름(형상 가변) — 좌표 기반 "
            "Physics AI 를 추천합니다. ROM 이 필요하면 공통 격자로 재샘플하세요."
            + operator_note,
        }
    if profile.n_cases > 1:
        time_note = (
            " 시간축이 있어 (μ, t) 로 함께 학습됩니다." if profile.n_time_steps > 1 else ""
        )
        return {
            "method": "rom",
            "reason": f"{header} — 파라미터 스윕은 스냅샷이 적어도 ROM 이 안정적입니다."
            + time_note,
        }
    if "dynamics" in feasible and profile.n_time_steps >= 8:
        extra = " 미래 예보가 필요하면 동역학(DMD)도 후보입니다 — 적합도 확인 필수."
    else:
        extra = ""
    return {
        "method": "rom",
        "reason": f"{header} — 단일 시계열은 축소+보간(ROM)이 표준입니다.{extra}",
    }
