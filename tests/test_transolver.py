"""Transolver mesh-native operator and service wiring tests."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from naviertwin.core.cfd_reader.base import CFDDataset
from naviertwin.core.digital_twin.transolver_engine import TransolverTwinEngine
from naviertwin.core.gnn.transolver import CaseSetTransolver
from naviertwin.web.service import build_transolver_twin_from_cases


def _graphs() -> list[dict[str, np.ndarray]]:
    graphs = []
    for count, parameter in ((11, 1.0), (17, 2.0), (23, 3.0)):
        coordinate = np.linspace(0.0, 1.0, count, dtype=np.float32)
        x = np.column_stack([coordinate, np.full(count, parameter, np.float32)])
        y = (coordinate * parameter + 0.1).reshape(-1, 1)
        graphs.append({"x": x, "y": y})
    return graphs


def test_case_set_transolver_varying_points_and_pickle(tmp_path) -> None:
    graphs = _graphs()
    model = CaseSetTransolver(
        2,
        1,
        hidden=16,
        n_layers=2,
        n_slices=4,
        n_heads=4,
        max_epochs=8,
        device="cpu",
    )
    model.fit({"graphs": graphs})

    assert model.train_losses_[-1] < model.train_losses_[0]
    assert model.predict_graph(graphs[0]).shape == (11, 1)
    assert model.predict_graph(graphs[-1]).shape == (23, 1)

    import pickle

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.predict_graph(graphs[1]), model.predict_graph(graphs[1])
    )


def _case(dimensions: tuple[int, int, int], parameter: float) -> CFDDataset:
    mesh = pv.ImageData(dimensions=dimensions).cast_to_unstructured_grid()
    points = np.asarray(mesh.points)
    mesh.point_data["p"] = points[:, 0] + parameter * points[:, 1]
    return CFDDataset(mesh, [0.0], ["p"], {})


def test_transolver_service_engine_contract(tmp_path) -> None:
    cases = [_case((5 + index, 4, 1), float(index)) for index in range(3)]
    result = build_transolver_twin_from_cases(
        cases,
        "p",
        np.asarray([[0.0], [1.0], [2.0]]),
        param_names=["reynolds"],
        hidden=16,
        n_layers=1,
        n_slices=4,
        n_heads=4,
        max_epochs=3,
        device="cpu",
    )
    engine = result["engine"]

    assert isinstance(engine, TransolverTwinEngine)
    assert engine.training_metadata["varying_mesh"] is True
    assert engine.training_metadata["architecture"] == "physics_attention_slices"
    prediction = engine.predict(np.asarray([0.5]))
    assert prediction.shape == (cases[0].n_points,)
    assert np.isfinite(prediction).all()

    path = tmp_path / "transolver.pkl"
    engine.save(path)
    restored = TransolverTwinEngine.load(path)
    np.testing.assert_array_equal(restored.predict(np.asarray([0.5])), prediction)
