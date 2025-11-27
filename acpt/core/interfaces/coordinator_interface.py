"""Coordinator interface coordinating agent proposals with programmable metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class CoordinatorInterface(ABC):
    """Metric-aware coordinator contract for aggregating agent proposals."""

    @abstractmethod
    def id(self) -> str:
        """Return the unique identifier for the coordinator instance."""

    @abstractmethod
    def init_rag(self, global_kb_descriptor: Dict[str, Any]) -> None:
        """Initialize shared retrieval context for the coordinator."""

    @abstractmethod
    def configure_metrics(self, metrics: Union[str, List[str]]) -> None:
        """Set one or more utility metrics that govern aggregation strategy."""

    @abstractmethod
    def aggregate_proposals(self, proposals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compose agent proposals into a joint plan based on configured metrics."""

    @abstractmethod
    def commit_plan(self, plan: Dict[str, Any]) -> None:
        """Commit the chosen plan and dispatch actions downstream."""
