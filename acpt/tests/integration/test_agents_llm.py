"""Integration tests for single-file LLM-backed agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import pytest

from acpt.agents import NOMAAgent, RISAgent, V2IAgent


def _descriptor(tmp_path: Path, name: str) -> Dict[str, object]:
    return {
        "index_type": "file",
        "path": tmp_path / f"{name}.json",
    }


@pytest.mark.parametrize(
    "agent_cls,obs,intent,estimate_keys",
    [
        (RISAgent, {"phase": 0.4, "phase_gradient": 0.2}, "ris_phase_optimization", {"phase"}),
        (
            V2IAgent,
            {"graph": [{"id": 1}, {"id": 2}, {"id": 3}]},
            "v2i_link_adaptation",
            {"link_score"},
        ),
        (
            NOMAAgent,
            {"graph": [{"id": "u1"}, {"id": "u2"}], "power": 1.2, "power_gradient": 0.3},
            "noma_resource_plan",
            {"fairness_score", "power_budget"},
        ),
    ],
)
def test_agent_propose_flow(tmp_path: Path, agent_cls: Type, obs: Dict[str, object], intent: str, estimate_keys: set[str]):
    agent = agent_cls()
    agent.init_rag(_descriptor(tmp_path, agent.id()))

    agent.observe(obs)
    proposal = agent.propose()

    assert proposal["intent"] == intent
    assert proposal["context"]["rag_documents"], "RAG context should not be empty"
    assert "model=" in proposal["llm"]["response"]

    if estimate_keys:
        assert estimate_keys.issubset(proposal["estimates"].keys())

    commit_ack = agent.commit({"action": "deploy"})
    assert commit_ack["status"] == "ack"

    agent.feedback({"reward": 0.9})
