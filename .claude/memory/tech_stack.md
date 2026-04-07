---
name: NavierTwin Tech Stack
description: 프로젝트 전체 기술 스택 — GUI, 3D, ML, CFD I/O, 패키징 등
type: project
---

| 레이어 | 기술 |
|--------|------|
| GUI | PySide6 (Qt6), QSS 다크테마, i18n(한/영) |
| 3D 시각화 | PyVista + pyvistaqt |
| CFD I/O | meshio, fluidfoam, h5py |
| 내부 포맷 | HDF5 (.ntwin) — 메쉬+필드+메타+모델가중치 |
| 차원축소(선형) | NumPy/SciPy |
| 차원축소(비선형) | PyTorch |
| ROM/모달 | PyDMD, NumPy |
| Surrogate | SMT, scikit-learn, PyTorch |
| Operator Learning | neuraloperator(FNO/TFNO), deepxde(DeepONet), PyTorch(U-Net) |
| Koopman | PyTorch (KNO, IKNO, FlowDMD) |
| State Space Model | mamba-ssm, PyTorch (MNO, DeepOMamba) |
| 생성 모델 | PyTorch (Diffusion Model, Score-based) |
| GNN | PyTorch Geometric (GNN surrogate, MeshGraphNets, EGNO) |
| 시계열 | PyTorch (LSTM, Transformer, Neural ODE — torchdiffeq, Mamba) |
| Equivariant NN | e3nn, PyTorch |
| PINN | NVIDIA PhysicsNEMO |
| 데이터동화 | filterpy, NumPy |
| 설명가능성 | SHAP, captum |
| 모델 내보내기 | ONNX, TorchScript |
| API 서버 | FastAPI (선택) |
| 패키징 | PyInstaller + Inno Setup |
| 테스트 | pytest |
| 린터 | ruff |
| 패키지 관리 | pyproject.toml (setuptools) |
