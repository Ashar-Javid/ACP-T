"""Unit tests for utils.registrar helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acpt.utils import ManifestValidationError, register_agent, validate_manifest


def _agent_manifest(agent_id: str) -> dict[str, object]:
    return {
        "id": agent_id,
        "type": "agent",
        "llm_spec": {"model": "qwen-32b", "device": "cerebras"},
        "rag_descriptor": {"index_type": "file", "prefix": "kb/demo.json"},
    }


def test_validate_manifest_accepts_valid_document() -> None:
    manifest = _agent_manifest("agent.unit")
    validated = validate_manifest(manifest)
    assert validated["id"] == "agent.unit"


def test_validate_manifest_rejects_missing_fields() -> None:
    with pytest.raises(ManifestValidationError):
        validate_manifest({"id": "broken"})


def test_register_agent_updates_wiring(tmp_path: Path) -> None:
    wiring_path = tmp_path / "wiring.yaml"
    manifest = _agent_manifest("agent.demo")

    register_agent(
        manifest,
        wiring_path,
        module="acpt.agents.demo_agent",
        class_name="DemoAgent",
    )

    wiring = yaml.safe_load(wiring_path.read_text(encoding="utf-8"))
    assert wiring["agents"]["agent.demo"]["module"] == "acpt.agents.demo_agent"
    assert wiring["agents"]["agent.demo"]["class"] == "DemoAgent"