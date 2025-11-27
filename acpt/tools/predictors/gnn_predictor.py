"""Graph neural network predictor stub implementing the ToolInterface."""

from __future__ import annotations

from typing import Any, Dict, List

from acpt.core.interfaces import ToolInterface


class GNNPredictor(ToolInterface):
    """Placeholder predictor returning heuristic scores for inference requests."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "object"},
            },
            "baseline": {"type": "number"},
        },
        "required": ["nodes"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "graph_size": {"type": "integer"},
                    "baseline": {"type": "number"},
                },
                "required": ["graph_size"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, model_name: str = "gnn-surrogate-v0") -> None:
        self.model_name = model_name

    def name(self) -> str:
        return "predictor.gnn"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "nodes" not in inputs:
            raise ValueError("GNNPredictor expects 'nodes' field per schema")

        nodes: List[Dict[str, Any]] = list(inputs.get("nodes", []))
        graph_size = len(nodes)
        baseline = float(inputs.get("baseline", 0.5))
        score = baseline + 0.01 * graph_size

        return {
            "result": {"score": round(score, 3)},
            "diagnostics": {
                "graph_size": graph_size,
                "baseline": baseline,
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "predictor",
            "model": self.model_name,
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }
