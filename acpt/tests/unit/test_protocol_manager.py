"""Tests for protocol manager validation and routing."""

from typing import Any, Dict

import pytest

from acpt.core.runtime import ProtocolManager, ProtocolValidationError, Registry


class EchoAgent:
    """Simple handler exposing a propose method for routing tests."""

    def propose(self, value: int) -> Dict[str, Any]:
        return {"echo": value}


@pytest.fixture()
def registry() -> Registry:
    reg = Registry()
    reg.register(
        {
            "id": "ris-01",
            "type": "agent",
            "llm_spec": {"model": "qwen-32b", "device": "cerebras"},
            "capabilities": ["PhaseDesign"],
            "rag_descriptor": {"index_type": "faiss", "prefix": "kb/ris-01"},
        },
        handler=EchoAgent(),
    )
    return reg


@pytest.fixture()
def protocol_manager(registry: Registry) -> ProtocolManager:
    return ProtocolManager(registry)


def test_send_rpc_assigns_correlation_id(protocol_manager: ProtocolManager):
    message = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "agent:ris-01.propose",
        "params": {"value": 5},
    }

    response = protocol_manager.send_rpc(message)

    assert response["jsonrpc"] == "2.0"
    assert response["result"] == {"echo": 5}
    assert "cid" in response and isinstance(response["cid"], str)


def test_send_rpc_raises_on_schema_violation(protocol_manager: ProtocolManager):
    with pytest.raises(ProtocolValidationError):
        protocol_manager.send_rpc({"method": "agent:ris-01.propose"})


def test_send_rpc_reports_routing_errors(protocol_manager: ProtocolManager):
    message = {
        "jsonrpc": "2.0",
        "method": "agent:ris-02.propose",
        "params": {"value": 1},
    }

    response = protocol_manager.send_rpc(message)
    assert "error" in response
    assert response["error"]["code"] == "registry_error"

    bad_method = {
        "jsonrpc": "2.0",
        "method": "agent:ris-01.missing",
    }
    response = protocol_manager.send_rpc(bad_method)
    assert response["error"]["code"] == "protocol_error"
    assert "missing" in response["error"]["message"]


def test_send_rpc_preserves_client_cid(protocol_manager: ProtocolManager):
    message = {
        "jsonrpc": "2.0",
        "cid": "custom-123",
        "method": "agent:ris-01.propose",
        "params": {"value": 7},
    }

    response = protocol_manager.send_rpc(message)
    assert response["cid"] == "custom-123"
    assert response["result"]["echo"] == 7