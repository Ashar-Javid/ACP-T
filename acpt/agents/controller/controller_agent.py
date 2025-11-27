"""High-level controller agent coordinating optimization objectives in ACP-T."""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, Mapping, Optional

from acpt.knowledge import KBManager
from acpt.utils import generate_checksum, get_logger


ToolCallable = Callable[[Dict[str, Any]], Dict[str, Any]]


OBJECTIVE_ALIASES = {
    "EE": "energy",
    "ENERGY": "energy",
    "FAIRNESS": "fairness",
    "SINR": "sinr",
    "THROUGHPUT": "throughput",
}


class ControllerAgent:
    """Objective-driven controller producing step-wise coordination plans."""

    def __init__(self, *, optimization_objective: str = "EE") -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._objective = self._normalize_objective(optimization_objective)
        self._kb: Optional[KBManager] = None
        self._step_index = 0
        self._tool_routes: Dict[str, Dict[str, ToolCallable]] = {}
        self._agent_capabilities: Dict[str, Mapping[str, Any]] = {}
        self._last_plan: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Initialization helpers

    def init_rag(self, kb_descriptor: Mapping[str, Any]) -> None:
        """Attach a private knowledge base used for environment interpretation."""

        self._kb = KBManager(dict(kb_descriptor))
        self._kb.initialize()
        self._logger.info("Controller RAG initialized at %s", self._kb.storage_path)

    def set_objective(self, objective: str) -> None:
        """Update the optimization objective used when generating plans."""

        self._objective = self._normalize_objective(objective)
        self._logger.info("Controller objective set to %s", self._objective)

    def register_agent_capabilities(self, agent_id: str, capabilities: Mapping[str, Any]) -> None:
        """Store capability metadata for downstream decision heuristics."""

        self._agent_capabilities[agent_id] = dict(capabilities)

    def register_tool_route(
        self,
        agent_id: str,
        tool_name: str,
        handler: ToolCallable | Any,
    ) -> None:
        """Register a callable used to route tool executions for *agent_id*.

        The handler may be a ``ToolInterface`` instance (exposing ``invoke``) or
        a plain callable accepting/returning ``Dict[str, Any]`` payloads.
        """

        callback: Optional[ToolCallable]
        if callable(getattr(handler, "invoke", None)):
            callback = handler.invoke  # type: ignore[assignment]
        elif callable(handler):  # pragma: no branch - fallback callables
            callback = handler  # type: ignore[assignment]
        else:
            raise TypeError("handler must be a ToolInterface or callable")

        routes = self._tool_routes.setdefault(agent_id, {})
        routes[tool_name] = callback  # type: ignore[assignment]
        routes[tool_name.lower()] = callback  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Planning loop

    def plan(
        self,
        context_snapshot: Mapping[str, Any],
        *,
        optimization_objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce a structured plan and action dictionary for the next step."""

        objective = self._objective
        if optimization_objective is not None:
            objective = self._normalize_objective(optimization_objective)

        observations = self._extract_observations(context_snapshot)
        metrics = self._extract_metrics(context_snapshot)
        documents = self._retrieve_documents(objective)

        if not observations:
            self._logger.warning("No observations present in context; emitting idle plan")
            idle_plan = self._build_idle_plan(objective, metrics, documents, context_snapshot)
            self._last_plan = idle_plan
            return idle_plan

        scorecard = self._score_agents(observations, objective, metrics)
        selected_agent = self._select_agent(scorecard)

        actions = self._build_actions(selected_agent, observations, objective)

        plan = {
            "step": self._step_index,
            "objective": objective,
            "selected_agent": selected_agent,
            "scorecard": scorecard,
            "actions": actions,
            "metrics": metrics,
            "rag_documents": documents,
            "context_hash": generate_checksum(dict(context_snapshot)),
        }

        self._step_index += 1
        self._last_plan = plan
        return plan

    # ------------------------------------------------------------------
    # Tool routing

    def use_tool(self, agent_id: str, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route a tool invocation for *agent_id* if a handler is registered."""

        toolset = self._tool_routes.get(agent_id, {})
        handler = toolset.get(tool_name) or toolset.get(tool_name.lower())
        if handler is None:
            raise KeyError(f"No tool route configured for agent '{agent_id}' and tool '{tool_name}'.")
        return handler(dict(inputs))

    # ------------------------------------------------------------------
    # Accessors

    @property
    def last_plan(self) -> Optional[Dict[str, Any]]:
        """Return the most recent plan emitted by the controller."""

        return self._last_plan

    # ------------------------------------------------------------------
    # Internal helpers

    def _extract_observations(self, snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        env_state = snapshot.get("env_state", {})
        latest = env_state.get("latest") if isinstance(env_state, Mapping) else None
        observations = latest.get("observations") if isinstance(latest, Mapping) else None
        return {k: dict(v) for k, v in observations.items()} if isinstance(observations, Mapping) else {}

    def _extract_metrics(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        metrics = snapshot.get("metrics", {})
        if not isinstance(metrics, Mapping):
            return {}
        latest = metrics.get("latest")
        if isinstance(latest, Mapping) and "metrics" in latest:
            inner = latest.get("metrics")
            if isinstance(inner, Mapping):
                return {str(key): inner[key] for key in inner.keys()}
        return {str(key): metrics[key] for key in metrics.keys()}

    def _retrieve_documents(self, objective: str) -> list[str]:
        if self._kb is None:
            return []
        topic = OBJECTIVE_ALIASES.get(objective, objective)
        return self._kb.retrieve(topic, k=3)

    def _score_agents(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        objective: str,
        metrics: Mapping[str, Any],
    ) -> Dict[str, float]:
        observations_dict = {agent_id: dict(obs) for agent_id, obs in observations.items()}
        objective_upper = objective.upper()

        if objective_upper == "EE":
            return {
                agent_id: -float(obs.get("energy_cost", obs.get("power", 1.0)))
                for agent_id, obs in observations_dict.items()
            }

        if objective_upper == "FAIRNESS":
            snr_values = [float(obs.get("SNR", 0.0)) for obs in observations_dict.values()]
            reference = statistics.mean(snr_values) if snr_values else 0.0
            return {
                agent_id: -abs(float(obs.get("SNR", 0.0)) - reference)
                for agent_id, obs in observations_dict.items()
            }

        if objective_upper == "SINR":
            return {
                agent_id: float(obs.get("SNR", obs.get("sinr", 0.0)))
                for agent_id, obs in observations_dict.items()
            }

        if objective_upper == "THROUGHPUT":
            throughput_metric = metrics.get("throughput")
            if isinstance(throughput_metric, Mapping):
                return {agent_id: float(throughput_metric.get(agent_id, 0.0)) for agent_id in observations_dict}
            return {
                agent_id: float(obs.get("throughput", obs.get("SNR", 0.0)))
                for agent_id, obs in observations_dict.items()
            }

        return {
            agent_id: float(obs.get("utility", obs.get("SNR", 0.0)))
            for agent_id, obs in observations_dict.items()
        }

    def _select_agent(self, scorecard: Mapping[str, float]) -> Optional[str]:
        if not scorecard:
            return None
        best_agent = max(scorecard.items(), key=lambda item: item[1])[0]
        return best_agent

    def _build_actions(
        self,
        selected_agent: Optional[str],
        observations: Mapping[str, Mapping[str, Any]],
        objective: str,
    ) -> Dict[str, Dict[str, Any]]:
        actions: Dict[str, Dict[str, Any]] = {}
        for agent_id, obs in observations.items():
            setpoints = self._recommend_setpoints(agent_id, obs, objective)
            actions[agent_id] = {
                "activate": agent_id == selected_agent,
                "objective": objective,
                "setpoints": setpoints,
            }
        return actions

    def _recommend_setpoints(
        self,
        agent_id: str,
        observation: Mapping[str, Any],
        objective: str,
    ) -> Dict[str, Any]:
        payload = {
            "objective": objective,
            "observation": dict(observation),
            "step": self._step_index,
        }

        toolset = self._tool_routes.get(agent_id, {})
        preferred_tool = None
        objective_upper = objective.upper()
        if objective_upper in {"EE", "SINR"}:
            preferred_tool = toolset.get("optimizer") or toolset.get("gd_solver")
        elif objective_upper in {"THROUGHPUT", "FAIRNESS"}:
            preferred_tool = toolset.get("predictor") or toolset.get("gnn_predictor")

        if preferred_tool is None:
            preferred_tool = (
                toolset.get(objective.lower())
                or toolset.get(f"opt_{objective.lower()}")
                or toolset.get("default")
            )

        if preferred_tool is not None:
            try:
                result = preferred_tool(payload)
                if isinstance(result, Mapping):
                    return dict(result)
            except Exception as exc:  # pragma: no cover - tool failure path
                self._logger.warning("Tool route failed for %s/%s: %s", agent_id, objective, exc)

        return {
            "target_power": observation.get("power"),
            "target_snr": observation.get("SNR"),
            "notes": f"heuristic_targets_for_{objective.lower()}",
        }

    def _build_idle_plan(
        self,
        objective: str,
        metrics: Mapping[str, Any],
        documents: list[str],
        snapshot: Mapping[str, Any],
    ) -> Dict[str, Any]:
        plan = {
            "step": self._step_index,
            "objective": objective,
            "selected_agent": None,
            "scorecard": {},
            "actions": {},
            "metrics": dict(metrics),
            "rag_documents": documents,
            "context_hash": generate_checksum(dict(snapshot)),
        }
        self._step_index += 1
        return plan

    @staticmethod
    def _normalize_objective(objective: str) -> str:
        normalized = objective.strip().upper() if objective else "EE"
        return normalized if normalized in OBJECTIVE_ALIASES else normalized
