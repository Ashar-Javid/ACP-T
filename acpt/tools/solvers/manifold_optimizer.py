"""Manifold optimizer projecting control vectors onto a unit sphere."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from acpt.core.interfaces import ToolInterface


class ManifoldOptimizer(ToolInterface):
    """Projects a vector along a gradient direction onto the unit sphere manifold."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "vector": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
            },
            "gradient": {
                "type": "array",
                "items": {"type": "number"},
            },
            "step_size": {"type": "number", "minimum": 0.0},
        },
        "required": ["vector"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "projection": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "updated": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["projection"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "norm_before": {"type": "number"},
                    "norm_after": {"type": "number"},
                    "step_size": {"type": "number"},
                },
                "required": ["norm_after", "step_size"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size = step_size

    def name(self) -> str:
        return "optimizer.manifold"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "vector" not in inputs:
            raise ValueError("ManifoldOptimizer expects 'vector' field per schema")

        vector_raw: Iterable[float] = inputs.get("vector", [])  # type: ignore[assignment]
        gradient_raw: Iterable[float] = inputs.get("gradient", [])  # type: ignore[assignment]
        step = float(inputs.get("step_size", self.step_size))

        vector: List[float] = [float(value) for value in vector_raw]
        gradients: List[float] = [float(value) for value in gradient_raw]
        if not gradients and vector:
            gradients = [0.0] * len(vector)

        updated: List[float] = []
        for idx, value in enumerate(vector):
            grad = gradients[idx] if idx < len(gradients) else gradients[-1]
            updated.append(float(value) + step * float(grad))

        if len(updated) < len(vector):
            updated.extend(vector[len(updated):])

        norm_before = math.sqrt(sum(value * value for value in vector)) or 1.0
        norm_after = math.sqrt(sum(value * value for value in updated)) or 1.0
        projected = [value / norm_after for value in updated]

        return {
            "result": {
                "projection": projected,
                "updated": updated,
            },
            "diagnostics": {
                "norm_before": norm_before,
                "norm_after": norm_after,
                "step_size": step,
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "optimizer",
            "manifold": "unit_sphere",
            "defaults": {"step_size": self.step_size},
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }
