"""Simple gradient descent solver wrapper implementing the ToolInterface."""

from __future__ import annotations

from typing import Any, Dict

from acpt.core.interfaces import ToolInterface


class GradientDescentSolver(ToolInterface):
    """Lightweight gradient descent wrapper producing standardized outputs."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "initial": {"type": "number"},
            "gradient": {"type": "number"},
            "learning_rate": {"type": "number", "minimum": 0.0},
            "iterations": {"type": "integer", "minimum": 0},
        },
        "required": ["initial", "gradient"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {"solution": {"type": "number"}},
                "required": ["solution"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "iterations": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "initial": {"type": "number"},
                    "gradient": {"type": "number"},
                },
                "required": ["iterations", "learning_rate"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, learning_rate: float = 0.1, iterations: int = 10) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations

    def name(self) -> str:
        return "solver.gradient_descent"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "initial" not in inputs or "gradient" not in inputs:
            raise ValueError("GradientDescentSolver expects 'initial' and 'gradient' fields per schema")

        value = float(inputs.get("initial"))
        gradient = float(inputs.get("gradient"))
        learning_rate = float(inputs.get("learning_rate", self.learning_rate))
        iterations = int(inputs.get("iterations", self.iterations))

        iterations = max(0, iterations)
        learning_rate = max(0.0, learning_rate)

        updated = value
        for _ in range(iterations):
            updated -= learning_rate * gradient

        return {
            "result": {"solution": updated},
            "diagnostics": {
                "iterations": iterations,
                "learning_rate": learning_rate,
                "initial": value,
                "gradient": gradient,
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "solver",
            "method": "gradient_descent",
            "defaults": {
                "learning_rate": self.learning_rate,
                "iterations": self.iterations,
            },
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }
