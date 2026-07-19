"""내보내기 패널 — .ntwin HDF5, VTK, CSV 내보내기 및 프로젝트 저장/복원.

Signals:
    export_done(str): 내보내기 완료 시 저장된 파일 경로 발생.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.data_model import TwinProject


class ExportPanel(QWidget):
    """내보내기 탭 패널.

    Signals:
        export_done: 내보내기 완료 시 파일 경로와 함께 발생.
    """

    export_done = Signal(str)
    project_loaded = Signal(object, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._canonical_project: TwinProject | None = None
        self._engine: Optional[object] = None
        self._model_artifact: Optional[object] = None
        self._model_sample_input: Optional[object] = None
        self._last_project_load_error: str | None = None
        self._last_project_load_warning: str | None = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Export")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel("CFD 데이터, 모델, 프로젝트를 다양한 포맷으로 내보냅니다.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # 출력 경로
        path_group = QGroupBox("출력 경로")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("저장할 파일 경로...")
        path_layout.addWidget(self._path_edit)
        browse_btn = QPushButton("찾기")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)
        layout.addWidget(path_group)

        # 내보내기 옵션
        options_group = QGroupBox("내보내기 옵션")
        options_form = QFormLayout(options_group)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            ".ntwin (HDF5 — 프로젝트 전체)",
            ".vtu (VTK UnstructuredGrid)",
            ".vtk (레거시 VTK)",
            ".csv (점 데이터)",
            ".html/.pdf (고객 보고서)",
            ".onnx (ONNX PyTorch 모델)",
            ".pt (TorchScript 모델)",
            ".fmu (FMI/FMU 2.0 Co-Simulation)",
        ])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        options_form.addRow("포맷:", self._format_combo)

        self._include_mesh_cb = QCheckBox("메쉬 포함")
        self._include_mesh_cb.setChecked(True)
        options_form.addRow("", self._include_mesh_cb)

        self._include_model_cb = QCheckBox("모델(TwinEngine) 포함")
        self._include_model_cb.setChecked(True)
        options_form.addRow("", self._include_model_cb)

        self._compress_cb = QCheckBox("HDF5 압축 (gzip level 4)")
        self._compress_cb.setChecked(True)
        options_form.addRow("", self._compress_cb)

        layout.addWidget(options_group)

        # 프로젝트 저장/복원
        project_group = QGroupBox("프로젝트 (.ntwin)")
        project_layout = QHBoxLayout(project_group)

        self._save_project_btn = QPushButton("프로젝트 저장")
        self._save_project_btn.clicked.connect(self._save_project)
        project_layout.addWidget(self._save_project_btn)

        self._load_project_btn = QPushButton("프로젝트 열기")
        self._load_project_btn.clicked.connect(self._load_project)
        project_layout.addWidget(self._load_project_btn)

        layout.addWidget(project_group)

        # 내보내기 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._export_btn = QPushButton("내보내기")
        self._export_btn.setObjectName("primaryButton")
        self._export_btn.clicked.connect(self._export)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group, stretch=1)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """내보낼 CFDDataset을 설정한다."""
        self._dataset = dataset
        # dataset이 바뀌면 기존 engine은 같은 데이터 기준이 아닐 수 있으므로 초기화한다.
        self._engine = None
        self._log(f"Dataset 설정: {dataset.n_points} pts, {dataset.n_cells} cells")
        self._log("TwinEngine 상태 초기화")

    def set_canonical_project(self, project: TwinProject | None) -> None:
        """Set the canonical manifest saved beside the array container."""
        self._canonical_project = project

    def set_engine(self, engine: object) -> None:
        """내보낼 TwinEngine을 설정한다."""
        self._engine = engine
        self._model_artifact = engine
        self._model_sample_input = None
        self._log(f"TwinEngine 설정: {type(engine).__name__}")

    def set_model(self, model: object, sample_input: object | None = None) -> None:
        """ONNX/TorchScript로 내보낼 PyTorch 모델 산출물을 설정한다."""
        self._model_artifact = model
        self._model_sample_input = sample_input
        self._log(f"모델 산출물 설정: {type(model).__name__}")

    def load_project_path(self, path: Path) -> bool:
        """지정한 .ntwin 프로젝트 파일을 로드하고 workflow 복원 신호를 발생시킨다."""
        self._last_project_load_error = None
        self._last_project_load_warning = None
        try:
            from naviertwin.core.export.ntwin_format import load_dataset

            dataset = load_dataset(path)
            metadata = self._read_metadata_sidecar(path)

            engine: object | None = None
            model_path = path.with_suffix(".engine.pkl")
            if model_path.exists():
                from naviertwin.core.digital_twin.twin_engine import TwinEngine

                try:
                    engine = TwinEngine.load(model_path)
                    self._log(f"✓ TwinEngine 로드: {model_path}")
                except Exception as exc:  # noqa: BLE001
                    try:
                        from naviertwin.core.digital_twin.physics_ai_engine import (
                            PhysicsAITwinEngine,
                        )

                        engine = PhysicsAITwinEngine.load(model_path)
                        self._log(f"✓ Physics AI TwinEngine 로드: {model_path}")
                    except Exception as physics_exc:  # noqa: BLE001
                        self._last_project_load_warning = (
                            "TwinEngine 로드 실패: "
                            f"{exc}; Physics AI 엔진 로드 실패: {physics_exc}"
                        )
                        self._log(f"[WARN] {self._last_project_load_warning}")
            else:
                self._log("ℹ TwinEngine 파일 없음 (.engine.pkl)")

            self._dataset = dataset
            self._engine = engine
            self._path_edit.setText(str(path))
            self._log(f"✓ 프로젝트 로드: {path} ({dataset.n_points} pts)")

            if metadata is not None:
                self._attach_project_metadata(metadata)
                self._apply_loaded_metadata_to_dataset(metadata)
            if metadata is not None and self._engine is not None:
                self._attach_project_metadata(metadata)
            self.project_loaded.emit(self._dataset, self._engine)
            return True
        except Exception as exc:
            self._last_project_load_error = str(exc)
            self._log(f"[ERROR] 프로젝트 로드 실패: {exc}")
            return False

    def last_project_load_error(self) -> str | None:
        """마지막 .ntwin 로드 실패 메시지를 반환한다."""
        return self._last_project_load_error

    def last_project_load_warning(self) -> str | None:
        """마지막 .ntwin 부분 복구 경고 메시지를 반환한다."""
        return self._last_project_load_warning

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _browse_path(self) -> None:
        fmt_idx = self._format_combo.currentIndex()
        filters = [
            "NavierTwin Project (*.ntwin)",
            "VTK UnstructuredGrid (*.vtu)",
            "VTK Legacy (*.vtk)",
            "CSV (*.csv)",
            "Report (*.html *.pdf)",
            "ONNX Model (*.onnx)",
            "TorchScript Model (*.pt)",
            "FMU (*.fmu)",
        ]
        path, _ = QFileDialog.getSaveFileName(self, "저장 경로 선택", "", filters[fmt_idx])
        if path:
            self._path_edit.setText(path)

    def _on_format_changed(self, idx: int) -> None:
        is_hdf5 = idx == 0
        self._include_model_cb.setEnabled(is_hdf5)
        self._compress_cb.setEnabled(is_hdf5)

    def _export(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            self._log("[WARN] 출력 경로를 입력하세요.")
            return

        path = Path(path_str)
        fmt_idx = self._format_combo.currentIndex()

        try:
            if fmt_idx == 0:
                self._export_ntwin(path)
            elif fmt_idx in (1, 2):
                self._export_vtk(path)
            elif fmt_idx == 3:
                self._export_csv(path)
            elif fmt_idx == 4:
                self._export_report(path)
            elif fmt_idx == 5:
                self._export_onnx(path)
            elif fmt_idx == 6:
                self._export_torchscript(path)
            else:
                self._export_fmu(path)
        except Exception as exc:
            self._log(f"[ERROR] 내보내기 실패: {exc}")

    def _export_ntwin(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        from naviertwin.core.export.ntwin_format import save_dataset

        compression = "gzip" if self._compress_cb.isChecked() else None
        save_dataset(self._dataset, path, compression=compression)
        self._log(f"✓ .ntwin 저장: {path}")
        if self._canonical_project is not None:
            from naviertwin.core.data_model import save_project_manifest

            manifest_path = path.with_suffix(".manifest.json")
            save_project_manifest(self._canonical_project, manifest_path)
            self._log(f"✓ canonical manifest 저장: {manifest_path}")
        metadata = self._build_project_metadata(path)
        self._write_metadata_sidecar(path, metadata)
        self._attach_project_metadata(metadata)
        if self._include_model_cb.isChecked():
            if self._engine is None:
                self._log("[WARN] 모델 포함이 선택됐지만 TwinEngine이 없습니다.")
            elif hasattr(self._engine, "save"):
                model_path = path.with_suffix(".engine.pkl")
                self._engine.save(model_path)  # type: ignore[union-attr]
                self._log(f"✓ TwinEngine 저장: {model_path}")
                metadata["has_engine"] = True
                metadata["engine_path"] = str(model_path)
                self._write_metadata_sidecar(path, metadata)
                self._attach_project_metadata(metadata)
            else:
                self._log("[WARN] 현재 엔진은 save()를 지원하지 않습니다.")
        self.export_done.emit(str(path))

    def _export_vtk(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return

        mesh = self._dataset.mesh
        mesh.save(str(path))
        self._log(f"✓ VTK 저장: {path}")
        self.export_done.emit(str(path))

    def _export_csv(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        import numpy as np

        mesh = self._dataset.mesh
        pts = mesh.points
        header = "x,y,z"
        rows = [pts]
        field_index = 0
        while field_index < len(self._dataset.field_names):
            name = self._dataset.field_names[field_index]
            if name in mesh.point_data:
                arr = np.array(mesh.point_data[name])
                if arr.ndim == 1:
                    header += f",{name}"
                    rows.append(arr.reshape(-1, 1))
                else:
                    column_index = 0
                    while column_index < arr.shape[1]:
                        i = column_index
                        header += f",{name}_{i}"
                        column_index += 1
                    rows.append(arr)
            field_index += 1
        reshaped_rows = []
        row_index = 0
        while row_index < len(rows):
            reshaped_rows.append(rows[row_index].reshape(len(pts), -1))
            row_index += 1
        data = np.hstack(reshaped_rows)
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
        self._log(f"✓ CSV 저장: {path} ({len(pts)} rows)")
        self.export_done.emit(str(path))

    def _export_report(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return

        from naviertwin.core.report.generator import ReportGenerator

        data = self._build_report_data(path)
        generator = ReportGenerator()
        if path.suffix.lower() == ".pdf":
            out = generator.render_pdf(data, path)
        else:
            if path.suffix.lower() not in {".html", ".htm"}:
                path = path.with_suffix(".html")
            out = generator.render_html(data, path)
        self._log(f"✓ 보고서 저장: {out}")
        self.export_done.emit(str(out))

    def _export_onnx(self, path: Path) -> None:
        if path.suffix.lower() != ".onnx":
            path = path.with_suffix(".onnx")
        try:
            model, sample_input = self._prepare_torch_export()
        except RuntimeError as exc:
            self._log(f"[WARN] {exc}")
            return
        out = self._export_to_onnx(model, sample_input, path)
        self._log(f"✓ ONNX 모델 저장: {out}")
        self.export_done.emit(str(out))

    def _export_torchscript(self, path: Path) -> None:
        if path.suffix.lower() != ".pt":
            path = path.with_suffix(".pt")
        try:
            model, sample_input = self._prepare_torch_export()
        except RuntimeError as exc:
            self._log(f"[WARN] {exc}")
            return
        out = self._export_to_torchscript(model, path, sample_input)
        self._log(f"✓ TorchScript 모델 저장: {out}")
        self.export_done.emit(str(out))

    def _export_fmu(self, path: Path) -> None:
        if path.suffix.lower() != ".fmu":
            path = path.with_suffix(".fmu")
        if self._engine is None:
            self._log("[WARN] 내보낼 TwinEngine이 없습니다.")
            return
        out = self._export_to_fmu(self._engine, path)
        self._log(f"✓ FMI/FMU 모델 저장: {out}")
        self.export_done.emit(str(out))

    def _prepare_torch_export(self) -> tuple[object, object]:
        """현재 모델 산출물에서 torch.nn.Module과 trace sample을 추출한다."""
        module, source = self._resolve_torch_module()
        sample_input = self._resolve_sample_input(module, source)
        return module, sample_input

    def _resolve_torch_module(self) -> tuple[object, object]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch가 필요합니다.") from exc

        source = self._model_artifact or self._engine
        if source is None:
            raise RuntimeError("내보낼 모델이 없습니다. Model 탭에서 신경 연산자를 먼저 학습하세요.")

        candidates = [source]
        attrs = ("surrogate", "model", "_model", "module", "net", "network")
        attr_index = 0
        while attr_index < len(attrs):
            attr = attrs[attr_index]
            value = getattr(source, attr, None)
            if value is not None:
                candidates.append(value)
                nested = getattr(value, "_model", None)
                if nested is not None:
                    candidates.append(nested)
            attr_index += 1

        candidate_index = 0
        while candidate_index < len(candidates):
            candidate = candidates[candidate_index]
            if isinstance(candidate, torch.nn.Module):
                return candidate, source
            candidate_index += 1
        raise RuntimeError("PyTorch nn.Module을 찾을 수 없습니다.")

    def _resolve_sample_input(self, module: object, source: object) -> object:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch가 필요합니다.") from exc

        explicit = self._model_sample_input
        if explicit is None:
            owners = (source, module)
            owner_index = 0
            attrs = ("sample_input", "example_input", "example_input_array", "_sample_input")
            while owner_index < len(owners):
                owner = owners[owner_index]
                attr_index = 0
                while attr_index < len(attrs):
                    attr = attrs[attr_index]
                    value = getattr(owner, attr, None)
                    if value is not None:
                        explicit = value
                        break
                    attr_index += 1
                if explicit is not None:
                    break
                owner_index += 1
        if explicit is not None:
            return self._move_sample_input(explicit, module)

        param = next(module.parameters(), None)  # type: ignore[attr-defined]
        device = param.device if param is not None else torch.device("cpu")
        dtype = param.dtype if param is not None and param.dtype.is_floating_point else torch.float32
        name = type(source).__name__.lower()
        in_channels = int(getattr(source, "in_channels", 1) or 1)

        if "1d" in name:
            n = max(16, int(getattr(source, "modes", 8) or 8) * 2)
            return torch.zeros((1, n, in_channels), device=device, dtype=dtype)
        if "2d" in name or "unet" in name or "tfno" in name:
            return torch.zeros((1, 16, 16, in_channels), device=device, dtype=dtype)

        first_linear = None
        modules = tuple(module.modules())  # type: ignore[attr-defined]
        module_index = 0
        while module_index < len(modules):
            candidate = modules[module_index]
            if isinstance(candidate, torch.nn.Linear):
                first_linear = candidate
                break
            module_index += 1
        if first_linear is not None:
            return torch.zeros((1, int(first_linear.in_features)), device=device, dtype=dtype)
        raise RuntimeError("trace용 sample_input을 추론할 수 없습니다.")

    def _move_sample_input(self, value: object, module: object) -> object:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch가 필요합니다.") from exc

        param = next(module.parameters(), None)  # type: ignore[attr-defined]
        device = param.device if param is not None else torch.device("cpu")
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            moved_items = []
            item_index = 0
            while item_index < len(value):
                moved_items.append(self._move_sample_input(value[item_index], module))
                item_index += 1
            return tuple(moved_items)
        if isinstance(value, list):
            moved_items = []
            item_index = 0
            while item_index < len(value):
                moved_items.append(self._move_sample_input(value[item_index], module))
                item_index += 1
            return tuple(moved_items)
        return value

    def _export_to_onnx(self, model: object, sample_input: object, path: Path) -> Path:
        from naviertwin.core.export.onnx_export import export_to_onnx

        return export_to_onnx(model, sample_input, path)

    def _export_to_torchscript(
        self,
        model: object,
        path: Path,
        sample_input: object,
    ) -> Path:
        from naviertwin.core.export.torchscript_export import export_to_torchscript

        trace_input = sample_input if isinstance(sample_input, tuple) else (sample_input,)
        return export_to_torchscript(model, path, sample_input=trace_input, mode="trace")

    def _export_to_fmu(self, engine: object, path: Path) -> Path:
        from naviertwin.core.export.fmu_export import export_to_fmu

        return export_to_fmu(engine, path).path

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "project.ntwin", "NavierTwin Project (*.ntwin)"
        )
        if not path:
            return
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        try:
            self._path_edit.setText(path)
            self._export_ntwin(Path(path))
        except Exception as exc:
            self._log(f"[ERROR] 프로젝트 저장 실패: {exc}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "NavierTwin Project (*.ntwin)"
        )
        if path:
            self.load_project_path(Path(path))

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _build_project_metadata(self, path: Path) -> dict[str, Any]:
        """현재 프로젝트 상태를 JSON 직렬화 가능한 메타데이터로 구성한다."""
        assert self._dataset is not None
        metadata: dict[str, Any] = {
            "project_path": str(path),
            "include_model": self._include_model_cb.isChecked(),
            "has_engine": self._engine is not None and self._include_model_cb.isChecked(),
            "engine_path": None,
            "dataset": {
                "n_points": int(self._dataset.n_points),
                "n_cells": int(self._dataset.n_cells),
                "n_time_steps": int(self._dataset.n_time_steps),
                "field_names": list(self._dataset.field_names),
            },
        }
        if self._engine is not None:
            metadata["engine"] = self._extract_engine_metadata(self._engine)
        return metadata

    def _extract_engine_metadata(self, engine: object) -> dict[str, Any]:
        """TwinEngine의 안전한 메타데이터를 추출한다."""
        reducer = getattr(engine, "reducer", None)
        surrogate = getattr(engine, "surrogate", None)
        return {
            "reducer_type": getattr(engine, "reducer_type", None),
            "surrogate_type": getattr(engine, "surrogate_type", None),
            "n_modes": int(getattr(engine, "n_modes", 0) or 0),
            "reducer": getattr(reducer, "training_metadata", None),
            "surrogate": getattr(surrogate, "training_metadata", None),
        }

    def _build_report_data(self, path: Path) -> dict[str, Any]:
        """현재 GUI 상태를 ReportGenerator 입력 데이터로 변환한다."""
        assert self._dataset is not None
        project_meta = self._project_metadata()
        model_info: dict[str, Any] = {
            "n_points": int(self._dataset.n_points),
            "n_cells": int(self._dataset.n_cells),
            "n_time_steps": int(self._dataset.n_time_steps),
            "fields": ", ".join(self._dataset.field_names) or "none",
        }
        if self._engine is not None:
            model_info.update(self._extract_engine_metadata(self._engine))
        elif isinstance(project_meta.get("engine"), Mapping):
            model_info.update(dict(project_meta["engine"]))

        return {
            "project": path.stem or "NavierTwin Report",
            "summary": (
                "NavierTwin CFD digital-twin report: "
                f"{self._dataset.n_points} points, {self._dataset.n_cells} cells, "
                f"{self._dataset.n_time_steps} time steps"
            ),
            "metrics": self._report_metrics(project_meta),
            "model_info": model_info,
            "figures": [],
            "notes": "Generated from the NavierTwin desktop Export panel.",
        }

    def _project_metadata(self) -> dict[str, Any]:
        """dataset/engine에 부착된 프로젝트 메타데이터를 반환한다."""
        dataset_meta = getattr(self._dataset, "metadata", None)
        if isinstance(dataset_meta, Mapping):
            project_meta = dataset_meta.get("project_metadata")
            if isinstance(project_meta, Mapping):
                return dict(project_meta)

        engine_meta = getattr(self._engine, "project_metadata", None)
        if isinstance(engine_meta, Mapping):
            return dict(engine_meta)
        return {}

    def _report_metrics(self, project_meta: Mapping[str, Any]) -> dict[str, float]:
        """프로젝트/엔진 메타데이터에서 보고서용 수치 지표를 추출한다."""
        candidates: list[object] = [
            project_meta.get("metrics"),
            project_meta.get("validation_metrics"),
        ]
        engine_meta = project_meta.get("engine")
        if isinstance(engine_meta, Mapping):
            candidates.append(engine_meta.get("metrics"))
            surrogate_meta = engine_meta.get("surrogate")
            if isinstance(surrogate_meta, Mapping):
                candidates.append(surrogate_meta.get("validation_metrics"))
                candidates.append(surrogate_meta.get("metrics"))

        surrogate = getattr(self._engine, "surrogate", None)
        surrogate_meta = getattr(surrogate, "training_metadata", None)
        if isinstance(surrogate_meta, Mapping):
            candidates.append(surrogate_meta.get("validation_metrics"))
            candidates.append(surrogate_meta.get("metrics"))

        candidate_index = 0
        while candidate_index < len(candidates):
            candidate = candidates[candidate_index]
            metrics = self._numeric_mapping(candidate)
            if metrics:
                return metrics
            candidate_index += 1
        return {}

    @staticmethod
    def _numeric_mapping(value: object) -> dict[str, float]:
        """유한한 수치 항목만 dict[str, float]로 반환한다."""
        if not isinstance(value, Mapping):
            return {}
        out: dict[str, float] = {}
        items = tuple(value.items())
        item_index = 0
        while item_index < len(items):
            key, raw = items[item_index]
            if not isinstance(key, str):
                item_index += 1
                continue
            try:
                number = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                item_index += 1
                continue
            if math.isfinite(number):
                out[key] = number
            item_index += 1
        return out

    def _write_metadata_sidecar(self, path: Path, metadata: dict[str, Any]) -> None:
        """프로젝트 메타데이터 sidecar JSON을 기록한다."""
        sidecar = path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"✓ 메타데이터 저장: {sidecar}")

    def _read_metadata_sidecar(self, path: Path) -> dict[str, Any] | None:
        """프로젝트 메타데이터 sidecar JSON을 읽는다."""
        sidecar = path.with_suffix(".meta.json")
        if not sidecar.exists():
            return None
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._log(f"✓ 메타데이터 로드: {sidecar}")
                return data
        except Exception as exc:
            self._log(f"[WARN] 메타데이터 로드 실패: {exc}")
        return None

    def _attach_project_metadata(self, metadata: dict[str, Any]) -> None:
        """로드/저장된 메타데이터를 dataset/engine 객체에 부착한다."""
        if self._dataset is not None:
            try:
                self._dataset.metadata["project_metadata"] = metadata
            except Exception:
                pass
        if self._engine is not None:
            try:
                setattr(self._engine, "project_metadata", metadata)
            except Exception:
                pass

    def _apply_loaded_metadata_to_dataset(self, metadata: dict[str, Any]) -> None:
        """sidecar 메타데이터를 dataset 메타데이터에 반영한다."""
        if self._dataset is not None:
            try:
                self._dataset.metadata["project_metadata"] = metadata
            except Exception:
                pass
