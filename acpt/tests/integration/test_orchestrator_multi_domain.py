"""Integration test for the multi-domain composite environment."""

from __future__ import annotations

from pathlib import Path

import yaml

from acpt.core.runtime import Orchestrator


def _agent_manifest(agent_id: str) -> dict[str, object]:
    rel_path = Path("kb") / f"{agent_id.replace('.', '_')}.json"
    return {
        "id": agent_id,
        "type": "agent",
        "llm_spec": {
            "model": "qwen-32b",
            "device": "cerebras",
            "infer_params": {"temperature": 0.0, "max_tokens": 256},
        },
        "rag_descriptor": {
            "index_type": "file",
            "path": str(rel_path),
            "prefix": str(rel_path),
        },
    }


def _coordinator_manifest() -> dict[str, object]:
    rel_path = Path("kb") / "coordinator.json"
    return {
        "id": "coordinator.main",
        "type": "coordinator",
        "llm_spec": {
            "model": "qwen-32b",
            "device": "cerebras",
            "infer_params": {"temperature": 0.0, "max_tokens": 128},
        },
        "rag_descriptor": {
            "index_type": "file",
            "path": str(rel_path),
            "prefix": str(rel_path),
        },
    }


def test_orchestrator_handles_multi_domain_environment(tmp_path: Path) -> None:
    wiring_path = tmp_path / "wiring.yaml"
    env_path = tmp_path / "env.yaml"

    wiring_payload = {
        "agents": {
            "ris": {
                "module": "acpt.agents.ris_agent",
                "class": "RISAgent",
                "manifest": _agent_manifest("agent.ris"),
            },
            "v2i": {
                "module": "acpt.agents.v2i_agent",
                "class": "V2IAgent",
                "manifest": _agent_manifest("agent.v2i"),
            },
            "noma": {
                "module": "acpt.agents.noma_agent",
                "class": "NOMAAgent",
                "manifest": _agent_manifest("agent.noma"),
            },
        },
        "coordinator": {
            "module": "acpt.agents.coordinator_agent",
            "class": "CoordinatorAgent",
            "manifest": _coordinator_manifest(),
            "metrics": {"energy": 0.4, "fairness": 0.3, "latency": 0.3},
        },
    }

    env_payload = {
        "environment": {
            "module": "acpt.environments.multi_domain_environment",
            "class": "MultiDomainEnvironment",
            "config": {
                "delegates": [
                    {
                        "module": "acpt.environments.ris_environment",
                        "class": "RISEnvironment",
                        "agents": ["agent.ris"],
                        "seed": 5,
                        "kwargs": {"tile_count": 16, "max_steps": 6},
                    },
                    {
                        "module": "acpt.environments.noma_environment",
                        "class": "NOMAEnvironment",
                        "agents": ["agent.noma"],
                        "seed": 7,
                        "kwargs": {"max_steps": 6},
                    },
                    {
                        "module": "acpt.environments.v2i_environment",
                        "class": "V2IEnvironment",
                        "agents": ["agent.v2i"],
                        "seed": 11,
                        "kwargs": {"lane_count": 3, "max_steps": 6},
                    },
                ]
            },
        }
    }

    wiring_path.write_text(yaml.safe_dump(wiring_payload), encoding="utf-8")
    env_path.write_text(yaml.safe_dump(env_payload), encoding="utf-8")

    orchestrator = Orchestrator(str(wiring_path), str(env_path), steps=2)
    result = orchestrator.run()

    assert len(result["history"]) == 2
    for entry in result["history"]:
        observations = entry["observations"]
        reward = entry["reward"]
        assert set(observations.keys()) == {"agent.ris", "agent.noma", "agent.v2i"}
        assert set(reward.keys()) == {"agent.ris", "agent.noma", "agent.v2i"}
        plan = entry["plan"]
        assert plan["telemetry"]["selected"]["agent"] in {"agent.ris", "agent.noma", "agent.v2i"}

    assert set(result["metrics"]) == {"energy", "fairness", "latency"}
