"""Unit tests for tool adapters and knowledge base manager."""

from pathlib import Path

import pytest

from acpt.knowledge import KBManager
from acpt.tools import GNNPredictor, GradientDescentSolver


@pytest.fixture()
def kb_descriptor(tmp_path: Path) -> dict:
    return {
        "index_type": "file",
        "path": tmp_path / "kb" / "ris.json",
    }


def test_gradient_descent_solver_invocation():
    solver = GradientDescentSolver(learning_rate=0.2, iterations=5)
    initial = 1.0
    gradient = 0.5
    payload = solver.invoke({"initial": initial, "gradient": gradient})

    expected = initial - solver.learning_rate * gradient * solver.iterations
    assert payload["result"]["solution"] == pytest.approx(expected)
    assert payload["diagnostics"]["iterations"] == 5
    assert payload["diagnostics"]["learning_rate"] == pytest.approx(0.2)
    assert solver.metadata()["method"] == "gradient_descent"


def test_gnn_predictor_outputs_score():
    predictor = GNNPredictor()
    payload = predictor.invoke({"nodes": [{"id": 1}, {"id": 2}], "baseline": 0.4})

    assert payload["result"]["score"] == pytest.approx(0.42, rel=1e-3)
    assert payload["diagnostics"]["graph_size"] == 2
    assert predictor.metadata()["model"].startswith("gnn")


def test_kb_manager_initialize_and_retrieve(kb_descriptor: dict):
    manager = KBManager(kb_descriptor)
    manager.initialize(seed_documents=["doc-1", "doc-2"])

    retrieved = manager.retrieve("query", k=1)
    assert retrieved == ["doc-1"]

    manager.add_document("doc-3")
    assert manager.retrieve("query", k=3)[-1] == "doc-3"

    embedding = manager.embed("doc-3")
    assert embedding[0] == len("doc-3")
