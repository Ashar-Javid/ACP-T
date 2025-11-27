"""NOMA scheduling agent coordinating multi-user resource allocations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from acpt.agents.base_agent import BaseAgent
from acpt.tools import GNNPredictor, GradientDescentSolver, PowerAllocator


def _gnn_factory() -> GNNPredictor:
    return GNNPredictor()


def _gd_factory() -> GradientDescentSolver:
    return GradientDescentSolver(learning_rate=0.15, iterations=4)


def _allocator_factory() -> PowerAllocator:
    return PowerAllocator(total_power=1.0)


TOOL_FACTORIES = {
    "gnn_predictor": _gnn_factory,
    "predictor.gnn": _gnn_factory,
    "gd_solver": _gd_factory,
    "solver.gradient_descent": _gd_factory,
    "power_allocator": _allocator_factory,
    "allocator.power": _allocator_factory,
}


_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "noma_resource_plan",
    "fields": {
        "fairness_score": {"type": "float", "description": "Predicted fairness metric."},
        "power_budget": {"type": "float", "description": "Aggregate power budget post-optimization."},
        "allocation": {"type": "array", "items": {"type": "float"}},
    },
}


class NOMAAgent(BaseAgent):
    """NOMA resource allocation agent leveraging shared BaseAgent utilities."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="agent.noma",
            intent="noma_resource_plan",
            llm_spec={
                "model": "qwen-32b",
                "device": "cerebras",
                "infer_params": {"temperature": 0.05, "max_tokens": 320},
            },
            tool_factories=TOOL_FACTORIES,
        )

    # ------------------------------------------------------------------
    # BaseAgent hook implementations

    @property
    def action_schema(self) -> Mapping[str, Any]:
        return _ACTION_SCHEMA

    def _build_prompt(self, observations: Mapping[str, Any], documents: Sequence[str]) -> str:
        obs_parts = []
        for key, value in sorted(observations.items()):
            if key == "graph":
                obs_parts.append("graph=present")
            elif isinstance(value, (int, float, str)):
                obs_parts.append(f"{key}={value}")
        summary = ", ".join(obs_parts) or "no_observations"
        docs_summary = " | ".join(documents) if documents else "no_docs"
        return (
            "You schedule NOMA resources balancing fairness and throughput."
            f" Observations: {summary}."
            f" Knowledge base: {docs_summary}."
            " Return fairness, power budgets, and allocation hints."
        )

    def _prepare_tool_calls(self, observations: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        calls: list[Dict[str, Any]] = []
        graph = observations.get("graph")
        if graph:
            calls.append({"name": "predictor.gnn", "params": {"nodes": graph, "baseline": 0.55}})
        power_gradient = observations.get("power_gradient")
        if power_gradient is not None:
            calls.append(
                {
                    "name": "gd_solver",
                    "params": {
                        "initial": float(observations.get("power", 1.0)),
                        "gradient": float(power_gradient),
                        "learning_rate": 0.15,
                        "iterations": 4,
                    },
                }
            )
        weights = observations.get("weights")
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, bytearray)):
            calls.append({"name": "allocator.power", "params": {"weights": [float(w) for w in weights]}})
        return calls

    def _interpret_reasoning(
        self,
        reasoning: Mapping[str, Any],
        observations: Mapping[str, Any],
    ) -> Dict[str, Any]:
        estimates = super()._interpret_reasoning(reasoning, observations)
        score = estimates.pop("score", None)
        if score is not None:
            try:
                estimates["fairness_score"] = float(score)
            except (TypeError, ValueError):
                self._logger.debug("NOMAAgent: ignoring non-numeric score: %s", score)
        solution = estimates.pop("solution", None)
        if solution is not None:
            try:
                estimates["power_budget"] = float(solution)
            except (TypeError, ValueError):
                self._logger.debug("NOMAAgent: ignoring non-numeric solution: %s", solution)
        allocation = estimates.pop("allocation", None)
        if isinstance(allocation, Sequence) and not isinstance(allocation, (str, bytes, bytearray)):
            estimates["allocation"] = [float(x) for x in allocation]
        estimates.pop("graph_size", None)
        return estimates

    def _fallback_estimates(self, observations: Mapping[str, Any]) -> Dict[str, Any]:
        estimates: Dict[str, Any] = {}
        graph = observations.get("graph")
        if graph:
            response = self.use_tool("predictor.gnn", {"nodes": graph, "baseline": 0.55})
            result = response.get("result") if isinstance(response, Mapping) else None
            diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
            score = result.get("score") if isinstance(result, Mapping) else None
            if score is not None:
                try:
                    estimates["fairness_score"] = float(score)
                except (TypeError, ValueError):
                    self._logger.debug("NOMAAgent: ignoring fallback fairness: %s", score)
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)

        power_gradient = observations.get("power_gradient")
        if power_gradient is not None:
            response = self.use_tool(
                "gd_solver",
                {
                    "initial": float(observations.get("power", 1.0)),
                    "gradient": float(power_gradient),
                    "learning_rate": 0.15,
                    "iterations": 4,
                },
            )
            result = response.get("result") if isinstance(response, Mapping) else None
            diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
            solution = result.get("solution") if isinstance(result, Mapping) else None
            if solution is not None:
                try:
                    estimates["power_budget"] = float(solution)
                except (TypeError, ValueError):
                    self._logger.debug("NOMAAgent: ignoring fallback solution: %s", solution)
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)

        weights = observations.get("weights")
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, bytearray)):
            response = self.use_tool("allocator.power", {"weights": [float(w) for w in weights]})
            result = response.get("result") if isinstance(response, Mapping) else None
            diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
            allocation = result.get("allocation") if isinstance(result, Mapping) else None
            if isinstance(allocation, Sequence) and not isinstance(allocation, (str, bytes, bytearray)):
                estimates["allocation"] = [float(x) for x in allocation]
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)
        return estimates

    def _build_actions(
        self,
        observations: Mapping[str, Any],
        estimates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        action: Dict[str, Any] = {}
        fairness = estimates.get("fairness_score")
        if fairness is not None:
            action["fairness_score"] = float(fairness)
        power_budget = estimates.get("power_budget")
        if power_budget is not None:
            action["power_budget"] = float(power_budget)
        allocation = estimates.get("allocation")
        if allocation is not None:
            action["allocation"] = [float(x) for x in allocation]
        return {"noma_resource_plan": action}

    def _retrieve_documents(self) -> Sequence[str]:
        if self._kb is None:
            return []
        return self._kb.retrieve("noma", k=3)
