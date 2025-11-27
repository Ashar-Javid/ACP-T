"""Smoke test for the add_agent CLI script."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import add_agent


def _manifest(agent_id: str) -> dict[str, object]:
    return {
        "id": agent_id,
        "type": "agent",
        "llm_spec": {"model": "qwen-32b", "device": "cerebras"},
        "rag_descriptor": {"index_type": "file", "prefix": "kb/demo.json"},
    }


def test_add_agent_cli_updates_wiring(tmp_path: Path) -> None:
    manifest_path = tmp_path / "agent.json"
    manifest_path.write_text(json.dumps(_manifest("agent.cli")), encoding="utf-8")

    wiring_path = tmp_path / "wiring.yaml"
    wiring_path.write_text("agents: {}\ntools: {}\nknowledge: {}\n", encoding="utf-8")

    exit_code = add_agent.main(
        [
            "--manifest",
            str(manifest_path),
            "--wiring",
            str(wiring_path),
            "--module",
            "acpt.agents.cli_agent",
            "--class-name",
            "CliAgent",
        ]
    )

    assert exit_code == 0
    wiring = yaml.safe_load(wiring_path.read_text(encoding="utf-8"))
    assert wiring["agents"]["agent.cli"]["class"] == "CliAgent"