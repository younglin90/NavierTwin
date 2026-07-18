"""필드 의미론 — 보존량(conserved quantity) 필드 판별.

해상도 낮추기·공통 격자 재샘플은 VTK ``sample()`` 점 보간을 쓴다. 온도·압력
같은 intensive 량은 점 보간이 타당하지만, 밀도·질량 유량·운동량 같은 보존량은
셀 부피 가중 없이 점 값만 보간하므로 재샘플 후 적분 총량이 보존되지 않는다.
총량을 보존하려면 셀 부피(면적) 겹침 가중으로 재분배하는 conservative
remapping(ESMPy/MEDCoupling 계열)이 필요하다 — 이 모듈은 그 구분을 위한
경고 판별만 담당한다 (conservative remap 구현은 후속 단계).
"""

from __future__ import annotations

from collections.abc import Sequence

# 보존량으로 의심되는 필드명 패턴 (소문자, 부분 일치).
#
# 보수적으로 명확한 것만 넣는다 — 과잉 경고보다 과소 경고가 낫다.
# 특히 "Q" 단독은 Q-criterion 과 혼동되므로 제외하고, 유량은
# "flux"/"flow_rate" 처럼 명시적인 이름만 잡는다. "energy" 단독도
# 비열·비내부에너지(intensive) 필드와 혼동될 수 있어 "total_energy" 만 잡는다.
CONSERVED_FIELD_PATTERNS: tuple[str, ...] = (
    "rho",  # 밀도(rho, rhoU, rhoE ...) — 부피 적분하면 질량/운동량/에너지
    "density",
    "mass",  # mass_flux, mass_flow, massFlowRate ...
    "flux",  # mass_flux, volumetric_flux, heat_flux(면적분량) ...
    "flow_rate",
    "flowrate",
    "momentum",
    "total_energy",
    "totalenergy",
    "enthalpy_flow",
)


def flag_conserved_fields(field_names: Sequence[str]) -> list[str]:
    """보존량으로 의심되는 필드명을 골라 반환한다.

    점 보간 재샘플은 셀 부피 가중 없이 점 값만 새 격자로 옮기므로, 부피(면적)
    적분으로 의미가 정의되는 보존량은 재샘플 후 총량(총질량·총유량 등)이
    보존되지 않는다. 총량 보존이 필요하면 격자 겹침 부피로 가중 재분배하는
    conservative remapping(ESMPy/MEDCoupling 계열)을 써야 한다.

    Args:
        field_names: 검사할 필드명 목록.

    Returns:
        ``CONSERVED_FIELD_PATTERNS`` 와 대소문자 무시 부분 일치하는 필드명
        목록 (입력 순서 유지).
    """
    flagged: list[str] = []
    for name in field_names:
        lowered = str(name).lower()
        if any(pattern in lowered for pattern in CONSERVED_FIELD_PATTERNS):
            flagged.append(str(name))
    return flagged
