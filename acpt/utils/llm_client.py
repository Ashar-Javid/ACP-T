"""LLM client factory supporting Cerebras, OpenAI, Gemini, and OpenRouter."""

from __future__ import annotations

import os
from hashlib import md5
from typing import Any, Dict, Iterable, Optional, Sequence

from acpt.utils.logging_utils import get_logger


_LOGGER = get_logger("LLMClient")


def _compose_prompt(prompt: str, context: Optional[Iterable[str]]) -> str:
    context_items: Sequence[str] = list(map(str, context or []))
    if not context_items:
        return prompt
    merged = "\n".join(context_items)
    return f"{prompt}\n\nContext:\n{merged}"


class _StubLLMClient:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.model = spec.get("model", "qwen-32b")
        self.device = spec.get("device", spec.get("provider", "cerebras"))
        self.infer_params = spec.get("infer_params", {})

    def generate(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        signature = f"{prompt}|{'|'.join(map(str, context or []))}"
        digest = md5(signature.encode("utf-8")).hexdigest()[:12]
        temperature = self.infer_params.get("temperature", 0.0)
        return f"model={self.model} device={self.device} temp={temperature} reply={digest}"


class _RESTLLMClient:
    def __init__(self, spec: Dict[str, Any], api_key: str) -> None:
        self.model = spec.get("model", "qwen-32b")
        self.device = spec.get("device", spec.get("provider", "cloud"))
        self.infer_params = spec.get("infer_params", {})
        self.api_key = api_key

    def _request(self, *_: Any, **__: Any) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        try:
            return self._request(prompt, context=context)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _LOGGER.warning("LLM API request failed (%s); falling back to stub", exc)
            stub = _StubLLMClient({"model": self.model, "device": self.device, "infer_params": self.infer_params})
            return stub.generate(prompt, context=context)


class _CerebrasClient(_RESTLLMClient):
    def _request(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("requests package is required for Cerebras API access") from exc

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an ACP-T research assistant."},
                {"role": "user", "content": _compose_prompt(prompt, context)},
            ],
            "max_tokens": self.infer_params.get("max_tokens", 512),
            "temperature": self.infer_params.get("temperature", 0.1),
        }
        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class _OpenAIClient(_RESTLLMClient):
    def _request(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("requests package is required for OpenAI API access") from exc

        payload = {
            "model": self.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an ACP-T research assistant."},
                {"role": "user", "content": _compose_prompt(prompt, context)},
            ],
            "temperature": self.infer_params.get("temperature", 0.1),
            "max_tokens": self.infer_params.get("max_tokens", 512),
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class _OpenRouterClient(_RESTLLMClient):
    def _request(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("requests package is required for OpenRouter API access") from exc

        payload = {
            "model": self.model or "openrouter/openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an ACP-T research assistant."},
                {"role": "user", "content": _compose_prompt(prompt, context)},
            ],
            "temperature": self.infer_params.get("temperature", 0.1),
            "max_tokens": self.infer_params.get("max_tokens", 512),
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class _GeminiClient(_RESTLLMClient):
    def _request(self, prompt: str, *, context: Optional[Iterable[str]] = None) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("requests package is required for Gemini API access") from exc

        model_name = self.model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _compose_prompt(prompt, context)},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.infer_params.get("temperature", 0.1),
                "maxOutputTokens": self.infer_params.get("max_tokens", 512),
            },
        }
        response = requests.post(url, params=params, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts]
        return "\n".join(filter(None, texts)).strip()


_PROVIDER_CLIENTS = {
    "cerebras": _CerebrasClient,
    "openai": _OpenAIClient,
    "openrouter": _OpenRouterClient,
    "gemini": _GeminiClient,
}

_API_ENV_VARS = {
    "cerebras": "CEREBRAS_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _resolve_api_key(provider: str) -> Optional[str]:
    env_name = _API_ENV_VARS.get(provider)
    if not env_name:
        return None
    return os.getenv(env_name)


def create_llm_client(spec: Dict[str, Any]):
    provider = str(spec.get("provider", "cerebras")).lower()
    api_key = _resolve_api_key(provider)
    client_cls = _PROVIDER_CLIENTS.get(provider)
    if api_key and client_cls is not None:
        try:
            return client_cls(spec, api_key)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _LOGGER.warning("Failed to initialise %s client (%s); using stub", provider, exc)
    return _StubLLMClient(spec)
