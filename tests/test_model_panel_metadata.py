from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from naviertwin.gui.panels.model_panel import ModelPanel


def test_set_loaded_metadata_restores_nested_engine_surrogate_values(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel.set_loaded_metadata(
        {
            "engine": {
                "n_modes": 11,
                "surrogate": {
                    "n_params": 4,
                    "n_outputs": 9,
                    "n_samples": 33,
                },
            }
        }
    )

    assert panel._n_params_spin.value() == 4
    assert panel._n_outputs_spin.value() == 9
    assert panel._n_samples_spin.value() == 33


def test_set_loaded_metadata_prefers_top_level_over_nested(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel.set_loaded_metadata(
        {
            "n_modes": 18,
            "n_params": 6,
            "n_outputs": 7,
            "n_samples": 44,
            "engine": {
                "n_modes": 3,
                "surrogate": {
                    "n_params": 2,
                    "n_outputs": 5,
                    "n_samples": 12,
                },
            },
        }
    )

    assert panel._n_params_spin.value() == 6
    assert panel._n_outputs_spin.value() == 7
    assert panel._n_samples_spin.value() == 44


def test_set_loaded_metadata_invalid_or_non_positive_keeps_defaults(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    assert panel._n_samples_spin.value() == 20
    assert panel._n_params_spin.value() == 2
    assert panel._n_outputs_spin.value() == 5


def test_explainability_metadata_background_is_capped(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)
    X_train = np.arange(120, dtype=float).reshape(40, 3)

    metadata = panel._build_explainability_metadata(X_train)

    background = metadata["background"]
    assert isinstance(background, np.ndarray)
    assert background.shape == (32, 3)
    assert metadata["feature_names"] == ["param_0", "param_1", "param_2"]
    assert metadata["output_index"] == 0

    panel.set_loaded_metadata(
        {
            "n_modes": 0,
            "n_params": -1,
            "n_outputs": "bad",
            "n_samples": None,
            "engine": {
                "n_modes": -2,
                "surrogate": {
                    "n_params": 0,
                    "n_outputs": -3,
                    "n_samples": "NaN",
                },
            },
        }
    )

    assert panel._n_samples_spin.value() == 20
    assert panel._n_params_spin.value() == 2
    assert panel._n_outputs_spin.value() == 5
