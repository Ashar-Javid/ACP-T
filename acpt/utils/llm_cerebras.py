"""Cerebras Qwen-32B adapter (stubbed) for ACP runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .llm_client import create_llm_client
from .logging_utils import get_logger


@dataclass
class CerebrasConfig:
    """Configuration for the Cerebras LLM adapter."""

    model: str = "qwen-32b"
    device: str = "cerebras"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    batch_size: int = 1
    allow_streaming: bool = True
    latency_budget_ms: Optional[float] = 500.0
    mock_mode: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CerebrasLLMAdapter:
    """Adapter encapsulating Cerebras inference integration details."""

    def __init__(self, config: Optional[CerebrasConfig] = None, *, use_mock: Optional[bool] = None) -> None:
        self.config = config or CerebrasConfig()
        self._use_mock = self.config.mock_mode if use_mock is None else use_mock
        self._logger = get_logger(self.__class__.__name__)
        self._mock_client: Optional[Any] = None

    def generate(
        self,
        prompt: str,
        *,
        context: Optional[Iterable[str]] = None,
        max_tokens: Optional[int] = None,
        infer_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a response string for *prompt* using Cerebras or the mock engine."""

        params = infer_params or {}
        if max_tokens is not None:
            params = dict(params)
            params.setdefault("max_tokens", max_tokens)

        if self._use_mock:
            return self._mock_generate(prompt, context=context, infer_params=params)

        return self._invoke_cerebras(prompt, context=context, params=params)

    def stream_generate(
        self,
        prompt: str,
        *,
        context: Optional[Iterable[str]] = None,
        max_tokens: Optional[int] = None,
        infer_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Yield tokens/chunks from the model output."""

        response = self.generate(
            prompt,
            context=context,
            max_tokens=max_tokens,
            infer_params=infer_params,
        )
        for token in response.split():
            yield token

    def configure(self, **overrides: Any) -> None:
        """Update adapter configuration at runtime."""

        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        if "mock_mode" in overrides:
            self._use_mock = bool(self.config.mock_mode)

    def diagnostics(self) -> Dict[str, Any]:
        """Return adapter state for logging/observability."""

        return {
            "model": self.config.model,
            "device": self.config.device,
            "endpoint": self.config.endpoint,
            "batch_size": self.config.batch_size,
            "mock_mode": self._use_mock,
            "latency_budget_ms": self.config.latency_budget_ms,
        }

    # Internal helpers -------------------------------------------------

    def _mock_generate(
        self,
        prompt: str,
        *,
        context: Optional[Iterable[str]],
        infer_params: Dict[str, Any],
    ) -> str:
        if self._mock_client is None:
            spec = {
                "model": self.config.model,
                "device": self.config.device,
                "infer_params": {"temperature": 0.0, **infer_params},
            }
            self._mock_client = create_llm_client(spec)
        response = self._mock_client.generate(prompt, context=context)
        return f"[mock]{response}"

    def _invoke_cerebras(
        self,
        prompt: str,
        *,
        context: Optional[Iterable[str]],
        params: Dict[str, Any],
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._logger.info("Dispatching prompt to Cerebras (attempt %s)", attempt)
                # Placeholder for actual Cerebras inference call.
                # Replace with SDK/HTTP invocation when integrating with real hardware.
                time.sleep(0.01)
                joined_context = "|".join(context or [])
                return f"cerebras:{self.config.model}:{hash(prompt + joined_context) % 10_000}"[: max(params.get("max_tokens", 128), 16)]
            except Exception as exc:  # pragma: no cover - placeholder for future errors
                last_error = exc
                self._logger.warning("Cerebras call failed (attempt %s/%s): %s", attempt, self.config.max_retries, exc)
                time.sleep(min(0.5 * attempt, 2.0))
        raise RuntimeError(f"Cerebras inference failed after {self.config.max_retries} attempts") from last_error


__all__ = ["CerebrasConfig", "CerebrasLLMAdapter"]
