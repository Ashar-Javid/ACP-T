"""Power allocation helper computing normalized budgets.

Example
-------
>>> allocator = PowerAllocator(total_power=2.0)
>>> allocator.invoke({"weights": [0.25, 0.75]})
{'result': {'allocation': [0.5, 1.5]}, 'diagnostics': {'total_power': 2.0, 'weights': [0.25, 0.75], 'weight_sum': 1.0}}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from acpt.core.interfaces import ToolInterface


class PowerAllocator(ToolInterface):
    """Distributes power across channels based on weights."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "weights": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
            },
            "total_power": {"type": "number", "minimum": 0.0},
        },
        "required": ["weights"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "allocation": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
                "required": ["allocation"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "total_power": {"type": "number"},
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "weight_sum": {"type": "number"},
                },
                "required": ["total_power", "weights"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, total_power: float = 1.0) -> None:
        self.total_power = total_power

    def name(self) -> str:
        return "allocator.power"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "weights" not in inputs:
            raise ValueError("PowerAllocator expects 'weights' field per schema")

        weights_raw: Iterable[float] = inputs.get("weights", [])  # type: ignore[assignment]
        total_power = float(inputs.get("total_power", self.total_power))
        weights: List[float] = [float(weight) for weight in weights_raw]
        weight_sum = sum(weights) or 1.0
        allocation = [weight / weight_sum * total_power for weight in weights]

        return {
            "result": {"allocation": allocation},
            "diagnostics": {
                "total_power": total_power,
                "weights": weights,
                "weight_sum": weight_sum,
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "allocator",
            "total_power": self.total_power,
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }
