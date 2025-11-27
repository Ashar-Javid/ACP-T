"""Metric-aware coordinator agent orchestrating multi-agent proposals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from acpt.core.interfaces import CoordinatorInterface
from acpt.core.runtime.registry import Registry, RegistryError
from acpt.core.runtime.protocol_manager import ProtocolManager
from acpt.knowledge import KBManager
from acpt.utils import compute_weighted_utility, get_logger, normalize_weights, rank_candidates


class CoordinatorAgent(CoordinatorInterface):
    """Coordinator that merges agent proposals according to configurable metrics."""

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "energy": 0.4,
        "fairness": 0.3,
        "latency": 0.3,
    }

    def __init__(
        self,
        registry: Registry,
        protocol_manager: ProtocolManager,
        *,
        default_metric_weights: Optional[Mapping[str, float]] = None,
        optimizer_tool: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._protocol = protocol_manager
        self._metric_weights_cfg = dict(default_metric_weights or self._DEFAULT_WEIGHTS)
        self._metric_weights = dict(self._metric_weights_cfg)
        self._metrics = list(self._metric_weights.keys())
        self._optimizer_tool = optimizer_tool
        self._kb: Optional[KBManager] = None
        self._logger = get_logger(self.__class__.__name__)
        self._agent_capabilities: Dict[str, Mapping[str, Any]] = {}
        self._last_plan: Optional[Dict[str, Any]] = None
        self._task_context = "network_optimization"

    def id(self) -> str:
        return "coordinator.orchestrator"

    def init_rag(self, global_kb_descriptor: Dict[str, Any]) -> None:
        self._kb = KBManager(global_kb_descriptor)
        self._kb.initialize(
            [
                "Network policy: prioritize energy efficiency while maintaining fairness.",
                "Telemetry summary: historical latency kept under 15ms for critical links.",
            ]
        )

    def configure_metrics(self, metrics: Any) -> None:
        weights: Dict[str, float]
        if isinstance(metrics, Mapping):
            weights = {str(k): float(v) for k, v in metrics.items()}
            metric_list = list(weights.keys())
        elif isinstance(metrics, str):
            metric_list = [metrics]
            weights = {metrics: self._metric_weights_cfg.get(metrics, 1.0)}
        elif isinstance(metrics, Iterable):
            metric_list = [str(item) for item in metrics]
            weights = {metric: self._metric_weights_cfg.get(metric, 1.0) for metric in metric_list}
        else:
            raise TypeError("metrics must be a string, iterable of strings, or mapping")

        if not metric_list:
            metric_list = list(self._metric_weights_cfg.keys())
            weights = dict(self._metric_weights_cfg)

        self._metrics = metric_list
        self._metric_weights = normalize_weights(weights)
        self._logger.info("Configured metrics: %s", self._metric_weights)

    def aggregate_proposals(
        self, proposals: Dict[str, Dict[str, Any]], *, task: Optional[str] = None
    ) -> Dict[str, Any]:
        self._task_context = task or self._task_context
        documents = self._kb.retrieve(self._task_context, k=3) if self._kb else []

        filtered: Dict[str, Dict[str, Any]] = {}
        utilities: Dict[str, float] = {}
        for agent_id, proposal in proposals.items():
            capabilities = self._capabilities_for(agent_id)
            if task and task != "network_optimization":
                intent = capabilities.get("intent") or proposal.get("intent")
                if intent and intent != task:
                    continue

            estimates = proposal.get("estimates", {})
            metrics_subset = {metric: float(estimates.get(metric, 0.0)) for metric in self._metrics}
            utility = compute_weighted_utility(metrics_subset, self._metric_weights)
            filtered[agent_id] = {
                "proposal": proposal,
                "capabilities": capabilities,
                "metrics": metrics_subset,
            }
            utilities[agent_id] = utility

        if not filtered:
            plan = {
                "task": self._task_context,
                "ranked_candidates": [],
                "allocations": {},
                "actions": {},
                "telemetry": {"documents": documents, "utilities": {}},
            }
            self._last_plan = plan
            return plan

        ranked = list(rank_candidates({aid: data["metrics"] for aid, data in filtered.items()}, self._metric_weights))
        selected_agent, selected_score = ranked[0]

        allocations: Dict[str, Dict[str, Any]] = {}
        actions: Dict[str, Dict[str, Any]] = {}
        for agent_id, score in ranked:
            proposal_actions = filtered[agent_id]["proposal"].get("actions") or {}
            allocations[agent_id] = {
                "approved": agent_id == selected_agent,
                "utility": score,
                "intent": filtered[agent_id]["proposal"].get("intent"),
            }
            if agent_id == selected_agent:
                actions[agent_id] = dict(proposal_actions)
            else:
                actions[agent_id] = {}

        optimizer_result = None
        if self._optimizer_tool:
            try:
                tool = self._registry.handler(self._optimizer_tool)
            except RegistryError:
                tool = None
            if tool is not None:
                try:
                    optimizer_result = tool.invoke({"utilities": utilities})
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._logger.warning("Optimizer tool failed: %s", exc)

        plan = {
            "task": self._task_context,
            "ranked_candidates": ranked,
            "allocations": allocations,
            "actions": actions,
            "telemetry": {
                "documents": documents,
                "utilities": utilities,
                "optimizer": optimizer_result,
                "selected": {"agent": selected_agent, "score": selected_score},
            },
        }

        self._last_plan = plan
        return plan

    def commit_plan(self, plan: Dict[str, Any]) -> None:
        allocations = plan.get("allocations", {})
        for agent_id, decision in allocations.items():
            if not decision.get("approved"):
                continue

            message = {
                "jsonrpc": "2.0",
                "method": f"agent:{agent_id}.commit",
                "params": {"decision": decision},
            }
            response = self._protocol.send_rpc(message)
            if response.get("error"):
                self._logger.warning("Commit failed for %s: %s", agent_id, response["error"])
            else:
                self._logger.info("Commit ack for %s: %s", agent_id, response.get("result"))

    @property
    def metric_weights(self) -> Mapping[str, float]:
        return dict(self._metric_weights)

    @property
    def last_plan(self) -> Optional[Dict[str, Any]]:
        return self._last_plan

    def _capabilities_for(self, agent_id: str) -> Mapping[str, Any]:
        if agent_id in self._agent_capabilities:
            return self._agent_capabilities[agent_id]
        try:
            handler = self._registry.handler(agent_id)
        except RegistryError:
            self._agent_capabilities[agent_id] = {}
            return {}
        if hasattr(handler, "capabilities"):
            capabilities = handler.capabilities()
            self._agent_capabilities[agent_id] = capabilities or {}
            return self._agent_capabilities[agent_id]
        self._agent_capabilities[agent_id] = {}
        return {}
