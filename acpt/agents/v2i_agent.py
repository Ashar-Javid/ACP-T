"""V2I link adaptation agent powered by the shared BaseAgent."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from acpt.agents.base_agent import BaseAgent
from acpt.tools import GNNPredictor, PowerAllocator


def _gnn_factory() -> GNNPredictor:
    return GNNPredictor()


def _allocator_factory() -> PowerAllocator:
    return PowerAllocator(total_power=1.0)


TOOL_FACTORIES = {
    "gnn_predictor": _gnn_factory,
    "predictor.gnn": _gnn_factory,
    "power_allocator": _allocator_factory,
    "allocator.power": _allocator_factory,
}


_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "v2i_link_plan",
    "fields": {
        "link_score": {"type": "float", "description": "Predicted reliability of the link."},
        "power_allocation": {"type": "array", "items": {"type": "float"}},
    },
}


class V2IAgent(BaseAgent):
    """Vehicle-to-infrastructure agent using RAG, tools, and LLM reasoning."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="agent.v2i",
            intent="v2i_link_adaptation",
            llm_spec={
                "model": "qwen-32b",
                "device": "cerebras",
                "infer_params": {"temperature": 0.1, "max_tokens": 384},
            },
            tool_factories=TOOL_FACTORIES,
        )

    # ------------------------------------------------------------------
    # BaseAgent hook implementations

    @property
    def action_schema(self) -> Mapping[str, Any]:
        return _ACTION_SCHEMA

    def _build_prompt(self, observations: Mapping[str, Any], documents: Sequence[str]) -> str:
        obs_summary = []
        for key, value in sorted(observations.items()):
            if key == "graph":
                obs_summary.append("graph=present")
            elif isinstance(value, (int, float, str)):
                obs_summary.append(f"{key}={value}")
        summary = ", ".join(obs_summary) or "no_observations"
        docs_summary = " | ".join(documents) if documents else "no_docs"
        return (
            "You coordinate V2I link adaptation decisions."
            f" Observations: {summary}."
            f" Knowledge base documents: {docs_summary}."
            " Provide a link score and any supporting tool outputs."
        )

    def _prepare_tool_calls(self, observations: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        calls: list[Dict[str, Any]] = []
        graph = observations.get("graph")
        if graph:
            calls.append({"name": "gnn_predictor", "params": {"nodes": graph, "baseline": 0.6}})
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
                estimates["link_score"] = float(score)
            except (TypeError, ValueError):
                self._logger.debug("V2IAgent: ignoring non-numeric score: %s", score)
        allocation = estimates.pop("allocation", None)
        if isinstance(allocation, Sequence) and not isinstance(allocation, (str, bytes, bytearray)):
            estimates["power_allocation"] = [float(x) for x in allocation]
        estimates.pop("graph_size", None)
        return estimates

    def _fallback_estimates(self, observations: Mapping[str, Any]) -> Dict[str, Any]:
        estimates: Dict[str, Any] = {}
        graph = observations.get("graph")
        if graph:
            response = self.use_tool("gnn_predictor", {"nodes": graph, "baseline": 0.6})
            result = response.get("result") if isinstance(response, Mapping) else None
            diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
            score = result.get("score") if isinstance(result, Mapping) else None
            if score is not None:
                try:
                    estimates["link_score"] = float(score)
                except (TypeError, ValueError):
                    self._logger.debug("V2IAgent: ignoring fallback score: %s", score)
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)
        weights = observations.get("weights")
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, bytearray)):
            response = self.use_tool("allocator.power", {"weights": [float(w) for w in weights]})
            result = response.get("result") if isinstance(response, Mapping) else None
            diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
            allocation = result.get("allocation") if isinstance(result, Mapping) else None
            if isinstance(allocation, Sequence) and not isinstance(allocation, (str, bytes, bytearray)):
                estimates["power_allocation"] = [float(x) for x in allocation]
            if isinstance(diagnostics, Mapping):
                self._last_tool_diagnostics.append(diagnostics)
        return estimates

    def _build_actions(
        self,
        observations: Mapping[str, Any],
        estimates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        action: Dict[str, Any] = {}
        score = estimates.get("link_score")
        if score is not None:
            action["link_score"] = float(score)
        allocation = estimates.get("power_allocation")
        if allocation is not None:
            action["power_allocation"] = [float(x) for x in allocation]
        return {"v2i_link_plan": action}

    def _retrieve_documents(self) -> Sequence[str]:
        if self._kb is None:
            return []
        return self._kb.retrieve("v2i", k=3)
