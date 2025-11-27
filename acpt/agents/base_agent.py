"""Shared base class for LLM-driven ACP agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from acpt.core.interfaces import AgentInterface
from acpt.agents.llm_agent import LLMReasoner
from acpt.knowledge import KBManager
from acpt.utils import get_logger


ToolFactory = Callable[[], Any]


class BaseAgent(AgentInterface, ABC):
    """Provides shared orchestration logic for specialized ACP agents."""

    def __init__(
        self,
        *,
        agent_id: str,
        intent: str,
        llm_spec: Mapping[str, Any],
        tool_factories: Mapping[str, ToolFactory],
    ) -> None:
        self._agent_id = agent_id
        self._intent = intent
        self._llm_spec = dict(llm_spec)
        self._tool_factories = {alias: factory for alias, factory in tool_factories.items()}
        self._observations: Dict[str, Any] = {}
        self._telemetry: Dict[str, Any] = {}
        self._kb: Optional[KBManager] = None
        self._logger = get_logger(agent_id)
        self._last_tool_diagnostics: List[Mapping[str, Any]] = []
        self._reasoner = LLMReasoner(
            {
                "provider": "cerebras",
                "model": self._llm_spec.get("model", "qwen-32b"),
                "device": self._llm_spec.get("device", "cerebras"),
                "infer_params": self._llm_spec.get("infer_params", {}),
            }
        )
        for alias, factory in self._tool_factories.items():
            self._reasoner.register_tool(alias, lambda payload, factory=factory: factory().invoke(payload))

    # ------------------------------------------------------------------
    # AgentInterface contract

    def id(self) -> str:
        return self._agent_id

    def llm_spec(self) -> Dict[str, Any]:
        return dict(self._llm_spec)

    def init_rag(self, kb_descriptor: Dict[str, Any]) -> None:
        self._kb = KBManager(kb_descriptor)
        self._kb.initialize()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "rag": True,
            "tools": sorted(set(self._tool_factories.keys())),
            "intent": self._intent,
            "schema": self.action_schema,
        }

    def observe(self, obs: Dict[str, Any]) -> None:
        self._observations.update(obs)

    def propose(self) -> Dict[str, Any]:
        documents = list(self._retrieve_documents())
        prompt = self._build_prompt(self._observations, documents)

        tool_calls = self._prepare_tool_calls(self._observations)
        reasoner_context: Dict[str, Any] = {"messages": documents}
        if tool_calls:
            reasoner_context["tool_calls"] = tool_calls

        reasoning = self._reasoner.reason(prompt, reasoner_context)
        estimates = self._interpret_reasoning(reasoning, self._observations)

        if not estimates:
            estimates = self._fallback_estimates(self._observations)

        actions = self._build_actions(self._observations, estimates)

        transcript = reasoning.get("transcript", []) or []
        final_response = ""
        if transcript:
            last = transcript[-1]
            if isinstance(last, Mapping):
                final_response = str(last.get("response", ""))
        if not final_response:
            consensus = reasoning.get("consensus")
            if isinstance(consensus, Mapping):
                final_response = str(consensus.get("analysis") or consensus.get("decision") or "")
        if "model=" not in final_response:
            model_name = self._llm_spec.get("model", "unknown")
            suffix = final_response or "no_response"
            final_response = f"model={model_name} | {suffix}"

        payload = {
            "agent_id": self._agent_id,
            "intent": self._intent,
            "schema": self.action_schema,
            "observations": self._sanitize_observations(self._observations),
            "estimates": estimates,
            "actions": actions,
            "context": {
                "rag_documents": list(documents),
                "messages": list(documents),
                "tool_calls": list(tool_calls) if tool_calls else [],
            },
            "llm": {
                "model": self._llm_spec.get("model"),
                "consensus": reasoning.get("consensus"),
                "transcript": reasoning.get("transcript"),
                "candidates": reasoning.get("candidates"),
                "response": final_response,
            },
            "tool_diagnostics": list(self._last_tool_diagnostics),
        }
        return payload

    def commit(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self._telemetry = {"decision": decision}
        return {"status": "ack", "decision": decision}

    def use_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        factory = self._tool_factories.get(tool_name)
        if factory is None:
            factory = self._tool_factories.get(tool_name.lower())
        if factory is None:
            raise ValueError(f"Tool '{tool_name}' is not registered for {self._agent_id}.")
        tool = factory()
        return tool.invoke(dict(inputs))

    def feedback(self, telemetry: Dict[str, Any]) -> None:
        self._telemetry.update(telemetry)

    # ------------------------------------------------------------------
    # Hooks for subclasses

    @property
    @abstractmethod
    def action_schema(self) -> Mapping[str, Any]:
        """Return a lightweight schema describing action output."""

    @abstractmethod
    def _build_prompt(self, observations: Mapping[str, Any], documents: Sequence[str]) -> str:
        """Construct the reasoning prompt."""

    def _prepare_tool_calls(self, observations: Mapping[str, Any]) -> List[Dict[str, Any]]:
        return []

    def _interpret_reasoning(
        self,
        reasoning: Mapping[str, Any],
        observations: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._last_tool_diagnostics = []
        estimates: Dict[str, Any] = {}
        for result in reasoning.get("tool_results", []) or []:
            if isinstance(result, Mapping):
                payload = result
                if "result" in result and isinstance(result["result"], Mapping):
                    payload = result["result"]
                for key, value in payload.items():
                    if key != "diagnostics":
                        estimates[str(key)] = value
                diagnostics = result.get("diagnostics")
                if isinstance(diagnostics, Mapping):
                    self._last_tool_diagnostics.append(diagnostics)
        consensus = reasoning.get("consensus")
        if isinstance(consensus, Mapping):
            estimates.setdefault("decision", consensus.get("decision"))
        return {k: v for k, v in estimates.items() if v is not None}

    def _fallback_estimates(self, observations: Mapping[str, Any]) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def _build_actions(
        self,
        observations: Mapping[str, Any],
        estimates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Construct standardized action payloads."""

    # ------------------------------------------------------------------
    # Helpers

    def _retrieve_documents(self) -> Sequence[str]:
        if self._kb is None:
            return []
        topic = self._intent or self._agent_id
        return self._kb.retrieve(topic, k=3)

    @staticmethod
    def _sanitize_observations(observations: Mapping[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in observations.items():
            if isinstance(value, (int, float, str, bool)):
                sanitized[key] = value
            elif isinstance(value, Mapping):
                sanitized[key] = BaseAgent._sanitize_observations(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                sanitized[key] = [item for item in value if isinstance(item, (int, float, str, bool))][:32]
        return sanitized
