"""Unit tests covering interface compliance and config loading."""

from pathlib import Path
from typing import Any, Dict

import pytest

from acpt.core.interfaces import AgentInterface, CoordinatorInterface
from acpt.core.runtime.registry import ManifestValidationError, Registry, RegistryError
from acpt.utils.config_loader import load_config


class DummyAgent(AgentInterface):
    """Minimal agent implementation used for interface compliance checks."""

    def __init__(self) -> None:
        self._observations: Dict[str, Any] = {}
        self._tool_calls: Dict[str, Dict[str, Any]] = {}
        self._feedback: Dict[str, Any] = {}

    def id(self) -> str:
        return "dummy-agent"

    def llm_spec(self) -> Dict[str, Any]:
        return {
            "model": "dummy-llm",
            "device": "cpu",
            "infer_params": {"max_tokens": 32, "temperature": 0.0},
        }

    def init_rag(self, kb_descriptor: Dict[str, Any]) -> None:
        self._observations["rag"] = kb_descriptor

    def capabilities(self) -> Dict[str, Any]:
        return {"supports_tools": True, "rag": True}

    def observe(self, obs: Dict[str, Any]) -> None:
        self._observations.update(obs)

    def propose(self) -> Dict[str, Any]:
        return {"proposal": self._observations.copy()}

    def commit(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "committed", "decision": decision}

    def use_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._tool_calls[tool_name] = inputs
        return {"tool": tool_name, "echo": inputs}

    def feedback(self, telemetry: Dict[str, Any]) -> None:
        self._feedback.update(telemetry)


class DummyCoordinator(CoordinatorInterface):
    """Simple coordinator implementing metric programmability."""

    def __init__(self) -> None:
        self._metrics: Any = None
        self._rag: Dict[str, Any] = {}
        self._committed: Dict[str, Any] = {}

    def id(self) -> str:
        return "dummy-coordinator"

    def init_rag(self, global_kb_descriptor: Dict[str, Any]) -> None:
        self._rag = global_kb_descriptor

    def configure_metrics(self, metrics):
        self._metrics = metrics

    def aggregate_proposals(self, proposals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "combined": proposals,
            "metrics": self._metrics,
            "rag": self._rag,
        }

    def commit_plan(self, plan: Dict[str, Any]) -> None:
        self._committed = plan


def test_agent_interface_contract():
    agent = DummyAgent()
    agent.init_rag({"kb": "memory://"})
    agent.observe({"state": "ready"})
    proposal = agent.propose()

    assert agent.id() == "dummy-agent"
    assert agent.llm_spec()["model"] == "dummy-llm"
    assert proposal["proposal"]["state"] == "ready"

    decision_result = agent.commit({"action": "deploy"})
    assert decision_result["status"] == "committed"

    tool_result = agent.use_tool("echo", {"value": 42})
    assert tool_result["echo"]["value"] == 42

    agent.feedback({"reward": 0.9})
    assert agent._feedback["reward"] == 0.9


def test_coordinator_metric_programmable():
    coordinator = DummyCoordinator()
    coordinator.init_rag({"global_kb": "memory://cluster"})
    coordinator.configure_metrics(["energy", "fairness"])

    plan = coordinator.aggregate_proposals({"dummy-agent": {"score": 1.0}})
    assert plan["metrics"] == ["energy", "fairness"]
    assert plan["combined"]["dummy-agent"]["score"] == 1.0

    coordinator.commit_plan(plan)
    assert coordinator._committed == plan


def test_registry_manifest_validation():
    registry = Registry()

    manifest = {
        "id": "ris-01",
        "type": "agent",
        "llm_spec": {"model": "qwen-32b", "device": "cerebras"},
        "capabilities": ["PhaseDesign"],
        "rag_descriptor": {"index_type": "faiss", "prefix": "kb/ris-01"},
    }

    registry.register(manifest, handler=DummyAgent())

    resolved = registry.resolve("ris-01")
    assert resolved["llm_spec"]["model"] == "qwen-32b"

    handler = registry.handler("ris-01")
    assert isinstance(handler, DummyAgent)

    with pytest.raises(ManifestValidationError):
        registry.register({"id": "broken", "type": "agent"})

    with pytest.raises(RegistryError):
        registry.handler("missing")


def test_runtime_config_loader_exposes_default_llm():
    config_path = Path(__file__).resolve().parents[3] / "config" / "runtime.yaml"
    config = load_config(str(config_path))

    assert config["protocol_version"] == "acpt-1.0"
    assert config["default_llm"]["model"] == "qwen-32b"