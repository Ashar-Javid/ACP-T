"""Runtime orchestrator wiring together registry, agents, and environment."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple
from acpt.core.runtime.protocol_manager import ProtocolManager
from acpt.core.runtime.registry import Registry
from acpt.environments import BaseEnvironment
from acpt.core.interfaces.environment_interface import Transition
from acpt.utils import get_logger, load_config, persist_step

if TYPE_CHECKING:
    from acpt.agents.coordinator_agent import CoordinatorAgent


class Orchestrator:
    """Bootstraps the ACP runtime and executes a simplified coordination loop."""

    def __init__(
        self,
        wiring_path: str,
        env_config_path: str,
        *,
        steps: int = 3,
        task: str = "network_optimization",
        coordinator_metrics: Optional[Any] = None,
    ) -> None:
        self._wiring_path = Path(wiring_path)
        self._env_config_path = Path(env_config_path)
        self._steps = steps
        self._task = task
        self._coordinator_metrics = coordinator_metrics
        self._logger = get_logger(self.__class__.__name__)

    def run(self, steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute the coordination loop for *steps* iterations and return history."""

        wiring = load_config(str(self._wiring_path))
        env_spec = load_config(str(self._env_config_path)).get("environment", {})

        registry = Registry()
        protocol = ProtocolManager(registry)

        tools = self._register_tools(wiring.get("tools", []), registry)
        agents = self._register_agents(wiring.get("agents", {}), registry)
        coordinator, coordinator_manifest = self._initialise_coordinator(
            wiring.get("coordinator", {}), registry, protocol
        )

        registry.register(coordinator_manifest, handler=coordinator)

        if self._coordinator_metrics is not None:
            coordinator.configure_metrics(self._coordinator_metrics)
        else:
            metrics = coordinator_manifest.get("metrics")
            if metrics is not None:
                coordinator.configure_metrics(metrics)

        environment = self._instantiate_environment(env_spec)

        history = []
        observations, reward, done, info = self._normalize_transition(environment.reset())
        for step_idx in range(steps or self._steps):
            proposals = self._collect_proposals(agents, observations)
            plan = coordinator.aggregate_proposals(proposals, task=self._task)
            coordinator.commit_plan(plan)

            actions = self._derive_actions(plan, agents)
            observations, reward, done, info = self._normalize_transition(environment.step(actions))
            self._dispatch_feedback(agents, observations)

            entry = {
                "step": step_idx,
                "plan": plan,
                "observations": observations,
                "reward": reward,
                "info": info,
            }
            persisted = persist_step(entry)
            history.append(persisted)

            utilities = plan.get("telemetry", {}).get("utilities", {})
            self._logger.info(
                "Step %s | selected=%s | utilities=%s",
                step_idx,
                plan.get("telemetry", {}).get("selected", {}).get("agent"),
                utilities,
            )

            if done:
                break

        return {
            "history": history,
            "metrics": coordinator.metric_weights,
            "tools": tuple(tools.keys()),
        }

    def _register_tools(self, tool_specs: Any, registry: Registry) -> Dict[str, Any]:
        registered = {}
        for spec in tool_specs or []:
            module = self._import(spec["module"])
            cls = getattr(module, spec["class"])
            tool_id = spec["name"]
            manifest = {
                "id": tool_id,
                "type": "tool",
                "llm_spec": {"model": "n/a", "device": "cpu"},
                "capabilities": ["tool"],
                "factory": cls,
            }
            registry.register(manifest)
            registered[tool_id] = cls
        return registered

    def _register_agents(self, agent_specs: Mapping[str, Any], registry: Registry) -> Dict[str, Any]:
        agents: Dict[str, Any] = {}
        for alias, spec in agent_specs.items():
            module = self._import(spec["module"])
            agent_cls = getattr(module, spec["class"])
            agent = agent_cls()
            manifest = self._prepare_agent_manifest(alias, spec.get("manifest", {}), agent)
            registry.register(manifest, handler=agent)
            rag_descriptor = manifest.get("rag_descriptor")
            if rag_descriptor:
                agent.init_rag(rag_descriptor)
            agents[manifest["id"]] = agent
        return agents

    def _initialise_coordinator(
        self,
        coordinator_spec: Mapping[str, Any],
        registry: Registry,
        protocol: ProtocolManager,
    ) -> Tuple["CoordinatorAgent", Dict[str, Any]]:
        module_name = coordinator_spec.get("module", "acpt.agents.coordinator_agent")
        class_name = coordinator_spec.get("class", "CoordinatorAgent")
        module = self._import(module_name)
        coordinator_cls = getattr(module, class_name)

        defaults = coordinator_spec.get("default_metric_weights")
        optimizer_tool = coordinator_spec.get("optimizer_tool")
        coordinator = coordinator_cls(
            registry,
            protocol,
            default_metric_weights=defaults,
            optimizer_tool=optimizer_tool,
        )

        manifest = self._prepare_coordinator_manifest(coordinator_spec.get("manifest", {}), coordinator)
        metrics_override = coordinator_spec.get("metrics")
        if metrics_override is not None:
            manifest["metrics"] = metrics_override
        rag_descriptor = manifest.get("rag_descriptor")
        if rag_descriptor:
            coordinator.init_rag(rag_descriptor)
        return coordinator, manifest

    def _instantiate_environment(self, env_spec: Mapping[str, Any]) -> BaseEnvironment:
        module = self._import(env_spec.get("module"))
        env_cls = getattr(module, env_spec.get("class"))
        config = env_spec.get("config", {})
        args: Sequence[Any] = tuple(env_spec.get("args", []))
        kwargs: Dict[str, Any] = dict(env_spec.get("kwargs", {}))

        if config and not args and not kwargs:
            if isinstance(config, Mapping):
                kwargs.update(config)
            else:
                args = (config,)

        try:
            return env_cls(*args, **kwargs)
        except TypeError:
            # Fallback for legacy environments expecting a single config mapping.
            return env_cls(config)

    def _collect_proposals(self, agents: Mapping[str, Any], observations: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        proposals: Dict[str, Dict[str, Any]] = {}
        for agent_id, agent in agents.items():
            agent.observe(observations.get(agent_id, {}))
            proposals[agent_id] = agent.propose()
        return proposals

    def _derive_actions(self, plan: Mapping[str, Any], agents: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        actions: Dict[str, Dict[str, Any]] = {}
        plan_actions = plan.get("actions", {})
        for agent_id in agents:
            actions[agent_id] = dict(plan_actions.get(agent_id, {}) or {})
        return actions

    @staticmethod
    def _dispatch_feedback(agents: Mapping[str, Any], observations: Mapping[str, Any]) -> None:
        for agent_id, agent in agents.items():
            agent.feedback({"observation": observations.get(agent_id, {})})

    def _prepare_agent_manifest(self, alias: str, manifest: Mapping[str, Any], agent: Any) -> Dict[str, Any]:
        data = dict(manifest)
        data.setdefault("id", alias)
        data.setdefault("type", "agent")
        data.setdefault("llm_spec", agent.llm_spec())

        rag_descriptor = data.get("rag_descriptor")
        if isinstance(rag_descriptor, Mapping):
            resolved = dict(rag_descriptor)
            path_hint = resolved.get("path") or resolved.get("prefix")
            if path_hint:
                resolved["path"] = self._resolve_path(self._wiring_path, path_hint)
                resolved.setdefault("prefix", resolved["path"])
            data["rag_descriptor"] = resolved

        return data

    def _prepare_coordinator_manifest(self, manifest: Mapping[str, Any], coordinator: CoordinatorAgent) -> Dict[str, Any]:
        data = dict(manifest)
        data.setdefault("id", coordinator.id())
        data.setdefault("type", "coordinator")
        data.setdefault(
            "llm_spec",
            {
                "model": "qwen-32b",
                "device": "cerebras",
                "infer_params": {"temperature": 0.0, "max_tokens": 128},
            },
        )

        rag_descriptor = data.get("rag_descriptor")
        if isinstance(rag_descriptor, Mapping):
            resolved = dict(rag_descriptor)
            path_hint = resolved.get("path") or resolved.get("prefix")
            if path_hint:
                resolved["path"] = self._resolve_path(self._wiring_path, path_hint)
                resolved.setdefault("prefix", resolved["path"])
            data["rag_descriptor"] = resolved

        return data

    @staticmethod
    def _import(module_name: str):
        if not module_name:
            raise ValueError("Module name must be provided in configuration.")
        return import_module(module_name)

    @staticmethod
    def _resolve_path(anchor: Path, candidate: str) -> str:
        path = Path(candidate)
        if not path.is_absolute():
            path = anchor.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def _normalize_transition(result: Any) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        if isinstance(result, Transition):
            return (
                dict(result.state),
                dict(result.reward),
                bool(result.done),
                dict(result.info),
            )
        return dict(result or {}), {}, False, {}
