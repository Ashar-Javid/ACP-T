"""RIS-specific agent built on the BaseAgent orchestration layer."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from acpt.agents.base_agent import BaseAgent
from acpt.tools import GradientDescentSolver, ManifoldOptimizer, RISPhaseOptimizer


def _gd_solver_factory() -> GradientDescentSolver:
    return GradientDescentSolver(learning_rate=0.12, iterations=5)


def _manifold_factory() -> ManifoldOptimizer:
    return ManifoldOptimizer(step_size=0.05)


def _ris_phase_factory() -> RISPhaseOptimizer:
    return RISPhaseOptimizer(learning_rate=0.12, iterations=5)


TOOL_FACTORIES = {
    "gd_solver": _gd_solver_factory,
    "solver.gradient_descent": _gd_solver_factory,
    "manifold_optimizer": _manifold_factory,
    "optimizer.manifold": _manifold_factory,
    "ris_phase_optimizer": _ris_phase_factory,
    "optimizer.ris_phase": _ris_phase_factory,
}


_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "ris_phase_update",
    "fields": {
        "phase": {"type": "float", "description": "Target RIS phase shift in radians."},
        "policy": {"type": "string", "enum": ["direct", "manifold_projection"]},
        "projection": {"type": "array", "items": {"type": "float"}},
    },
}


class RISAgent(BaseAgent):
    """RIS control agent leveraging RAG, tool routing, and LLM planning."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="agent.ris",
            intent="ris_phase_optimization",
            llm_spec={
                "model": "qwen-32b",
                "device": "cerebras",
                "infer_params": {"temperature": 0.0, "max_tokens": 256},
            },
            tool_factories=TOOL_FACTORIES,
        )

    # ------------------------------------------------------------------
    # BaseAgent hook implementations

    @property
    def action_schema(self) -> Mapping[str, Any]:
        return _ACTION_SCHEMA

    def _build_prompt(self, observations: Mapping[str, Any], documents: Sequence[str]) -> str:
        summary_parts = []
        for key, value in sorted(observations.items()):
            if isinstance(value, (int, float, str)):
                summary_parts.append(f"{key}={value}")
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                summary_parts.append(f"{key}=len{len(value)}")
        summary = ", ".join(summary_parts) or "no_observations"
        docs_summary = " | ".join(documents) if documents else "no_docs"
        return (
            "You are optimizing an RIS phase profile."
            f" Observations: {summary}."
            f" Retrieved knowledge: {docs_summary}."
            " Produce numeric estimates and reference tool results when useful."
        )

    def _prepare_tool_calls(self, observations: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        calls: list[Dict[str, Any]] = []
        phase_gradient = observations.get("phase_gradient")
        if phase_gradient is not None:
            calls.append(
                {
                    "name": "optimizer.ris_phase",
                    "params": {
                        "phase": float(observations.get("phase", 0.0)),
                        "gradient": float(phase_gradient),
                        "learning_rate": 0.12,
                        "iterations": 5,
                    },
                }
            )
        phase_vector = observations.get("phase_vector")
        if isinstance(phase_vector, Sequence) and not isinstance(phase_vector, (str, bytes, bytearray)):
            gradient_vec = observations.get("phase_direction")
            if not isinstance(gradient_vec, Sequence) or isinstance(gradient_vec, (str, bytes, bytearray)):
                gradient_vec = phase_vector
            calls.append(
                {
                    "name": "optimizer.manifold",
                    "params": {
                        "vector": [float(x) for x in phase_vector],
                        "gradient": [float(x) for x in gradient_vec],
                        "step_size": 0.05,
                    },
                }
            )
        return calls

    def _interpret_reasoning(
        self,
        reasoning: Mapping[str, Any],
        observations: Mapping[str, Any],
    ) -> Dict[str, Any]:
        estimates = super()._interpret_reasoning(reasoning, observations)
        phase_value = estimates.get("phase")
        if phase_value is None:
            solution = estimates.pop("solution", None)
            if solution is not None:
                try:
                    estimates["phase"] = float(solution)
                except (TypeError, ValueError):
                    self._logger.debug("RISAgent: ignoring non-numeric phase solution: %s", solution)
        projection = estimates.pop("projection", None)
        if projection is not None:
            if isinstance(projection, Sequence) and not isinstance(projection, (str, bytes, bytearray)):
                estimates["manifold_projection"] = [float(x) for x in projection]
        estimates.pop("updated", None)
        return estimates

    def _fallback_estimates(self, observations: Mapping[str, Any]) -> Dict[str, Any]:
        estimates: Dict[str, Any] = {}
        phase_gradient = observations.get("phase_gradient")
        if phase_gradient is not None:
            response = self.use_tool(
                "optimizer.ris_phase",
                {
                    "phase": float(observations.get("phase", 0.0)),
                    "gradient": float(phase_gradient),
                    "learning_rate": 0.12,
                    "iterations": 5,
                },
            )
            result = response.get("result", {})
            solution = result.get("phase") if isinstance(result, Mapping) else None
            if solution is not None:
                try:
                    estimates["phase"] = float(solution)
                except (TypeError, ValueError):
                    self._logger.debug("RISAgent: ignoring fallback solution: %s", solution)
            diagnostics = response.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)

        phase_vector = observations.get("phase_vector")
        if isinstance(phase_vector, Sequence) and not isinstance(phase_vector, (str, bytes, bytearray)):
            gradient_vec = observations.get("phase_direction")
            if not isinstance(gradient_vec, Sequence) or isinstance(gradient_vec, (str, bytes, bytearray)):
                gradient_vec = phase_vector
            response = self.use_tool(
                "optimizer.manifold",
                {
                    "vector": [float(x) for x in phase_vector],
                    "gradient": [float(x) for x in gradient_vec],
                    "step_size": 0.05,
                },
            )
            result = response.get("result", {})
            projected = result.get("projection") if isinstance(result, Mapping) else None
            if isinstance(projected, Sequence) and not isinstance(projected, (str, bytes, bytearray)):
                estimates["manifold_projection"] = [float(x) for x in projected]
            diagnostics = response.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)
        return estimates

    def _build_actions(
        self,
        observations: Mapping[str, Any],
        estimates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        phase = float(estimates.get("phase", observations.get("phase", 0.0)))
        projection = estimates.get("manifold_projection")
        action: Dict[str, Any] = {
            "phase": phase,
            "policy": "manifold_projection" if projection else "direct",
        }
        if projection:
            action["projection"] = [float(x) for x in projection]
        return {"ris_phase_update": action}

    def _retrieve_documents(self) -> Sequence[str]:
        if self._kb is None:
            return []
        return self._kb.retrieve("ris", k=3)
