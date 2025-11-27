"""Unit tests for the objective-driven ControllerAgent."""

from __future__ import annotations

import pytest

from acpt.agents.controller.controller_agent import ControllerAgent


def _make_controller(tmp_path, objective: str = "EE") -> ControllerAgent:
    controller = ControllerAgent(optimization_objective=objective)
    kb_path = tmp_path / "kb" / "controller.json"
    descriptor = {"index_type": "file", "path": str(kb_path)}
    controller.init_rag(descriptor)
    return controller


def _snapshot(overrides: dict | None = None) -> dict:
    base = {
        "env_state": {
            "latest": {
                "observations": {
                    "agent.a": {"energy_cost": 1.0, "power": 1.2, "SNR": 12.5},
                    "agent.b": {"energy_cost": 0.6, "power": 0.9, "SNR": 14.0},
                }
            }
        },
        "metrics": {
            "latest": {
                "metrics": {
                    "throughput": {"agent.a": 1.2, "agent.b": 2.8},
                    "fairness": 0.91,
                }
            }
        },
    }
    if overrides:
        base.update(overrides)
    return base


def test_controller_plan_energy_objective_selects_lowest_cost(tmp_path):
    controller = _make_controller(tmp_path)
    plan = controller.plan(_snapshot())

    assert plan["objective"] == "EE"
    assert plan["selected_agent"] == "agent.b"  # lowest energy cost
    assert plan["actions"]["agent.b"]["activate"] is True
    assert plan["actions"]["agent.a"]["activate"] is False
    assert controller.last_plan == plan
    assert isinstance(plan["context_hash"], str) and plan["context_hash"]


def test_controller_plan_throughput_uses_metric_route(tmp_path):
    controller = _make_controller(tmp_path)

    controller.register_tool_route(
        "agent.b",
        "predictor",
        lambda payload: {"target_rate": payload["observation"].get("SNR", 0.0) + 1.0},
    )

    plan = controller.plan(_snapshot(), optimization_objective="throughput")

    assert plan["objective"] == "THROUGHPUT"
    assert plan["selected_agent"] == "agent.b"
    setpoints = plan["actions"]["agent.b"]["setpoints"]
    assert pytest.approx(setpoints["target_rate"], rel=1e-6) == 15.0
    fallback = plan["actions"]["agent.a"]["setpoints"]
    assert "notes" in fallback


def test_controller_tool_routing_and_missing_route(tmp_path):
    controller = _make_controller(tmp_path)
    controller.register_tool_route("agent.a", "optimizer", lambda payload: {"target_power": 0.5})

    result = controller.use_tool("agent.a", "optimizer", {"initial": 1.0})
    assert result["target_power"] == 0.5

    with pytest.raises(KeyError):
        controller.use_tool("agent.unknown", "optimizer", {})

