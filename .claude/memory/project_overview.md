---
name: NavierTwin Project Overview
description: CFD 후처리 결과를 AI/ROM/Operator Learning으로 디지털 트윈 변환하는 Windows 데스크톱 툴의 핵심 개요
type: project
---

NavierTwin은 CFD 후처리 결과 데이터를 AI/ROM/Operator Learning을 통해 디지털 트윈으로 변환하는 Windows 데스크톱 툴.
비상업용(오픈소스), Python 3.10+, PySide6 GUI.

**Why:** 엔지니어 일반 사용자가 복잡한 CFD 데이터를 로컬 GPU로 학습·예측·시각화할 수 있도록.

**How to apply:** 모든 코드는 이 목적에 맞게 작성. GUI는 6단계 워크플로우 패널 구성.

**GUI 워크플로우:** [1.Import] → [2.Analyze] → [3.Reduce] → [4.Model] → [5.Twin] → [6.Export]

**지원 CFD 포맷:**
- 비상업: OpenFOAM, SU2, Code_Saturne/Nektar++ (XDMF/HDF5)
- 상업: Fluent (.cas/.dat), CFX, STAR-CCM+, Tecplot
- 범용: VTK/VTU, CGNS, EnSight Gold, HDF5/XDMF

**내부 포맷 전략:** 전체 → VTK UnstructuredGrid → 내부 HDF5 정규화 (.ntwin)

**레이아웃:** 좌측 설정패널 + 우측 3D뷰어 + 하단 로그/진행률 + 모델비교 대시보드
