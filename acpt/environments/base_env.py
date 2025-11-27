"""Base interface for ACP simulation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEnvironment(ABC):
    """Abstract environment declaring the lifecycle for ACP simulators."""

    @abstractmethod
    def reset(self) -> Dict[str, Dict[str, Any]]:
        """Reset the environment state and return initial observations per agent."""

    @abstractmethod
    def step(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Advance the environment using the provided agent actions."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return static metadata describing RATs, channel models, and configuration."""
