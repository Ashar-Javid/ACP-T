"""LLM-backed reasoning engine with tool routing and self-consistency checks."""

from __future__ import annotations

import collections
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from acpt.core.interfaces import ToolInterface
from acpt.utils import get_logger
from acpt.utils.llm_client import create_llm_client


ReasoningEntry = Dict[str, Any]


_DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "cerebras",
    "model": "qwen-32b",
    "device": "cerebras",
    "infer_params": {"temperature": 0.1, "max_tokens": 384},
    "reasoning": {
        "max_steps": 3,
        "self_consistency_votes": 3,
    },
}


class LLMReasoner:
    """Reasoning engine that orchestrates multi-step prompts and tool usage."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        tools: Optional[Mapping[str, ToolInterface | Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
    ) -> None:
        self._config = self._merge_config(config)
        self._logger = get_logger(self.__class__.__name__)
        self._client = self._build_client(self._config)
        self._reasoning_cfg = dict(self._config.get("reasoning", {}))
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        if tools:
            for name, tool in tools.items():
                self.register_tool(name, tool)

    # ------------------------------------------------------------------
    # Public API

    def register_tool(
        self,
        tool_name: str,
        tool: ToolInterface | Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register *tool* under *tool_name* for downstream invocations."""

        if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
            handler = tool.invoke  # type: ignore[assignment]
        elif callable(tool):
            handler = tool  # type: ignore[assignment]
        else:
            raise TypeError("Tool must be callable or implement ToolInterface.invoke().")

        self._tools[tool_name] = handler
        self._tools[tool_name.lower()] = handler

    def reason(
        self,
        prompt: str,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run a reasoning pass and return a canonicalized result."""

        context_messages, tool_calls = self._normalize_context(context)
        tool_results = [self.call_tool(call["name"], call.get("params", {})) for call in tool_calls]

        augmented_context = context_messages + [str(result) for result in tool_results]
        transcript: List[ReasoningEntry] = []
        max_steps = int(self._reasoning_cfg.get("max_steps", 3))

        for step in range(max_steps):
            step_prompt = self._format_step_prompt(prompt, step, transcript, augmented_context)
            response = self._client.generate(step_prompt, context=augmented_context)
            transcript.append({
                "step": step,
                "prompt": step_prompt,
                "response": response,
            })
            augmented_context.append(response)

        final_prompt = self._format_final_prompt(prompt, transcript)
        votes = int(self._reasoning_cfg.get("self_consistency_votes", 3))
        raw_candidates = [self._client.generate(final_prompt, context=augmented_context) for _ in range(votes)]
        canonical_candidates = [self.post_process(candidate) for candidate in raw_candidates]

        consensus, agreement = self._resolve_consensus(canonical_candidates)

        result = {
            "prompt": prompt,
            "context": context_messages,
            "tool_results": tool_results,
            "transcript": transcript,
            "candidates": canonical_candidates,
            "consensus": consensus,
            "consistency": {
                "votes": votes,
                "agreement": agreement,
            },
        }

        return result

    def call_tool(self, tool_name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Invoke a registered tool and return its response."""

        handler = self._tools.get(tool_name) or self._tools.get(tool_name.lower())
        if handler is None:
            raise KeyError(f"Tool '{tool_name}' is not registered with the reasoner.")

        try:
            return handler(dict(params))
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.warning("Tool '%s' failed: %s", tool_name, exc)
            return {"error": str(exc)}

    def post_process(self, llm_output: Any) -> Dict[str, Any]:
        """Canonicalize raw LLM output into a structured dictionary."""

        if isinstance(llm_output, Mapping):
            data = {str(key): llm_output[key] for key in llm_output.keys()}
        else:
            text = str(llm_output).strip()
            data = {
                "analysis": text,
                "action": "recommend",
                "decision": text.split(" ")[0] if text else "n/a",
                "confidence": 0.5,
            }

        data.setdefault("analysis", data.get("decision", ""))
        data.setdefault("action", "recommend")
        data.setdefault("confidence", 0.5)
        data.setdefault("decision", data["analysis"])
        return data

    # ------------------------------------------------------------------
    # Internal helpers

    def _merge_config(self, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = {**_DEFAULT_CONFIG}
        if config:
            for key, value in config.items():
                if key == "reasoning" and isinstance(value, Mapping):
                    merged.setdefault("reasoning", {}).update(value)
                else:
                    merged[key] = value
        return merged

    def _build_client(self, cfg: Mapping[str, Any]) -> LLMClient:
        provider = str(cfg.get("provider", "cerebras")).lower()
        spec = {
            "model": cfg.get("model", _DEFAULT_CONFIG["model"]),
            "device": cfg.get("device", provider),
            "infer_params": cfg.get("infer_params", _DEFAULT_CONFIG["infer_params"]),
        }

        if provider == "openai":
            spec.setdefault("model", "gpt-4o-mini")
        elif provider == "openrouter":
            spec.setdefault("model", "anthropic/claude-3-sonnet")
        elif provider == "cerebras":
            spec.setdefault("device", "cerebras")

        return create_llm_client(spec)

    def _normalize_context(self, context: Optional[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        if context is None:
            return [], []

        messages: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(context, Mapping):
            messages_ext = context.get("messages") or context.get("docs") or []
            tool_calls = list(context.get("tool_calls", []))
            if isinstance(messages_ext, SequenceABC) and not isinstance(messages_ext, (str, bytes, bytearray)):
                messages.extend(str(item) for item in messages_ext)
            else:
                messages.append(str(messages_ext))
        elif isinstance(context, (list, tuple, set)):
            messages.extend(str(item) for item in context)
        else:
            messages.append(str(context))

        return messages, tool_calls

    def _format_step_prompt(
        self,
        prompt: str,
        step: int,
        transcript: Sequence[ReasoningEntry],
        context: Sequence[str],
    ) -> str:
        history = "\n".join(entry["response"] for entry in transcript)
        context_block = "\n".join(context)
        return (
            f"[Step {step}] Analyze the problem step-by-step.\n"
            f"Context: {context_block}\n"
            f"History: {history}\n"
            f"Prompt: {prompt}\n"
            "Provide a concise reasoning update."
        )

    def _format_final_prompt(self, prompt: str, transcript: Sequence[ReasoningEntry]) -> str:
        analysis = "\n".join(f"Step {entry['step']}: {entry['response']}" for entry in transcript)
        return (
            "Synthesize the final decision based on the following reasoning steps:\n"
            f"{analysis}\n"
            f"Original prompt: {prompt}\n"
            "Respond with a short decision and confidence indicator."
        )

    def _resolve_consensus(self, candidates: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        if not candidates:
            return {}, 0.0

        counter: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for candidate in candidates:
            key = str(candidate.get("decision"))
            counter[key].append(candidate)

        best_key, entries = max(counter.items(), key=lambda item: len(item[1]))
        agreement = len(entries) / max(len(candidates), 1)
        # return first candidate for deterministic behaviour
        return entries[0], agreement
