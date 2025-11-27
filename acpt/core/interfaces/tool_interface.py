"""Tool interface definition for external solvers and predictors."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ToolInterface(ABC):
    """Standardized contract for tool adapters consumed by ACP agents."""

    @abstractmethod
    def name(self) -> str:
        """Return the canonical tool identifier."""

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with provided inputs and return a structured result."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Expose tool capabilities, schema information, and configuration hints."""
