"""LLM-aware agent interface for ACP-T runtime components."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentInterface(ABC):
    """Contract for LLM-powered agents interacting with the ACP orchestrator."""

    @abstractmethod
    def id(self) -> str:
        """Return the unique identifier for the agent instance."""

    @abstractmethod
    def llm_spec(self) -> Dict[str, Any]:
        """Describe the backing LLM (model, device, and inference parameters)."""

    @abstractmethod
    def init_rag(self, kb_descriptor: Dict[str, Any]) -> None:
        """Initialize retrieval-augmented generation context from a knowledge base descriptor."""

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Advertise agent capabilities and feature flags."""

    @abstractmethod
    def observe(self, obs: Dict[str, Any]) -> None:
        """Consume latest observations emitted by the environment or coordinator."""

    @abstractmethod
    def propose(self) -> Dict[str, Any]:
        """Generate a structured proposal (plan, action candidates, metadata)."""

    @abstractmethod
    def commit(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Commit the approved decision and return resulting telemetry."""

    @abstractmethod
    def use_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a named tool with the provided input payload."""

    @abstractmethod
    def feedback(self, telemetry: Dict[str, Any]) -> None:
        """Receive feedback signals (rewards, diagnostics) for post-action learning."""
