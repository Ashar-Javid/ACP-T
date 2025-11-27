"""UAV trajectory smoothing solver with standardized signatures.

Example
-------
>>> solver = UAVTrajectorySolver(max_step=10.0)
>>> solver.invoke({"waypoints": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 25.0, "y": 0.0, "z": 0.0}]})
{'result': {'waypoints': [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 10.0, 'y': 0.0, 'z': 0.0}]}, 'diagnostics': {'total_distance': 10.0, 'original_distance': 25.0, 'max_step': 10.0, 'violations': 1}}
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from acpt.core.interfaces import ToolInterface


class UAVTrajectorySolver(ToolInterface):
    """Enforces maximum segment length constraints on UAV waypoints."""

    INPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "waypoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                    "required": ["x", "y", "z"],
                },
                "minItems": 1,
            },
            "max_step": {"type": "number", "minimum": 0.0},
        },
        "required": ["waypoints"],
        "additionalProperties": True,
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                            },
                            "required": ["x", "y", "z"],
                        },
                    },
                },
                "required": ["waypoints"],
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "total_distance": {"type": "number"},
                    "original_distance": {"type": "number"},
                    "max_step": {"type": "number"},
                    "violations": {"type": "integer"},
                },
                "required": ["total_distance", "max_step"],
            },
        },
        "required": ["result", "diagnostics"],
    }

    def __init__(self, max_step: float = 25.0) -> None:
        self.max_step = max_step

    def name(self) -> str:
        return "solver.uav_trajectory"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "waypoints" not in inputs:
            raise ValueError("UAVTrajectorySolver expects 'waypoints' field per schema")

        raw_waypoints: Iterable[Dict[str, Any]] = inputs.get("waypoints", [])  # type: ignore[assignment]
        waypoints: List[Dict[str, float]] = []
        for wp in raw_waypoints:
            waypoints.append({
                "x": float(wp.get("x", 0.0)),
                "y": float(wp.get("y", 0.0)),
                "z": float(wp.get("z", 0.0)),
            })

        if not waypoints:
            raise ValueError("UAVTrajectorySolver requires at least one waypoint")

        max_step = float(inputs.get("max_step", self.max_step))
        max_step = max(max_step, 0.0)

        optimized: List[Dict[str, float]] = [waypoints[0]]
        original_distance = 0.0
        total_distance = 0.0
        violations = 0

        for index in range(1, len(waypoints)):
            prev = optimized[-1]
            target = waypoints[index]
            dx = target["x"] - prev["x"]
            dy = target["y"] - prev["y"]
            dz = target["z"] - prev["z"]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            original_distance += distance

            if max_step > 0.0 and distance > max_step:
                scale = max_step / distance
                clipped = {
                    "x": prev["x"] + dx * scale,
                    "y": prev["y"] + dy * scale,
                    "z": prev["z"] + dz * scale,
                }
                optimized.append(clipped)
                total_distance += max_step
                violations += 1
            else:
                optimized.append(target)
                total_distance += distance

        diagnostics = {
            "total_distance": total_distance,
            "original_distance": original_distance,
            "max_step": max_step,
            "violations": violations,
        }
        return {"result": {"waypoints": optimized}, "diagnostics": diagnostics}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "solver",
            "intent": "uav_path_planning",
            "defaults": {"max_step": self.max_step},
            "input_schema": self.INPUT_SCHEMA,
            "output_schema": self.OUTPUT_SCHEMA,
        }

