"""RIS phase optimizer implementing standardized tool signatures.

Example
-------
>>> optimizer = RISPhaseOptimizer()
>>> optimizer.invoke({"phase": 0.2, "gradient": 0.05, "learning_rate": 0.1, "iterations": 3})
{'result': {'phase': 0.185, 'delta': -0.015, 'trace': [0.195, 0.19, 0.185]}, 'diagnostics': {'iterations': 3, 'learning_rate': 0.1, 'initial_phase': 0.2, 'gradient': 0.05}}
"""

from __future__ import annotations

from typing import Any, Dict, List

from acpt.core.interfaces import ToolInterface


class RISPhaseOptimizer(ToolInterface):
    """Gradient-based optimizer tailored for RIS phase adjustments."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "phase": {"type": "number"},
            "gradient": {"type": "number"},
            "learning_rate": {"type": "number", "minimum": 0.0},
            "iterations": {"type": "integer", "minimum": 0},
        },
        "required": ["phase", "gradient"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "phase": {"type": "number"},
                    "delta": {"type": "number"},
                    "trace": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["phase"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "iterations": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "initial_phase": {"type": "number"},
                    "gradient": {"type": "number"},
                },
                "required": ["iterations", "learning_rate"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, learning_rate: float = 0.12, iterations: int = 5) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations

    def name(self) -> str:
        return "optimizer.ris_phase"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "phase" not in inputs or "gradient" not in inputs:
            raise ValueError("RISPhaseOptimizer expects 'phase' and 'gradient' fields per schema")

        phase = float(inputs.get("phase"))
        gradient = float(inputs.get("gradient"))
        learning_rate = float(inputs.get("learning_rate", self.learning_rate))
        iterations = int(inputs.get("iterations", self.iterations))

        learning_rate = max(0.0, learning_rate)
        iterations = max(0, iterations)

        current = phase
        trace: List[float] = []
        for _ in range(iterations):
            current -= learning_rate * gradient
            trace.append(current)

        result = {
            "phase": current,
            "delta": current - phase,
            "trace": trace,
        }
        diagnostics = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "initial_phase": phase,
            "gradient": gradient,
        }
        return {"result": result, "diagnostics": diagnostics}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "optimizer",
            "intent": "ris_phase_adjustment",
            "defaults": {
                "learning_rate": self.learning_rate,
                "iterations": self.iterations,
            },
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }
