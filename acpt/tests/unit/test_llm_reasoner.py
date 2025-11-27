"""Unit tests for the LLMReasoner orchestration engine."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from acpt.agents.llm_agent import LLMReasoner


class EchoTool:
    """Simple tool used to validate routing behaviour."""

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"echo": payload, "invoked": True}


def test_reasoning_pipeline_runs_with_tool_calls(tmp_path):
    reasoner = LLMReasoner(tools={"echo": EchoTool()})
    prompt = "Optimize beamforming weights"
    context = {"messages": ["channel=rician"], "tool_calls": [{"name": "echo", "params": {"value": 1}}]}

    result = reasoner.reason(prompt, context)

    assert result["prompt"] == prompt
    assert result["tool_results"][0]["invoked"] is True
    assert result["consensus"]["action"] == "recommend"
    assert result["consistency"]["votes"] >= 1


def test_call_tool_missing_raises():
    reasoner = LLMReasoner()
    with pytest.raises(KeyError):
        reasoner.call_tool("missing", {})


def test_provider_switch_uses_override():
    reasoner = LLMReasoner({"provider": "openai", "model": "gpt-4o"})
    result = reasoner.reason("Compute fairness metric")
    assert result["consensus"]["analysis"]
