"""Tests for the RewardAgent multi-objective pipeline."""

from __future__ import annotations

from typing import Dict

import pytest

from acpt.agents.reward import RewardAgent


@pytest.fixture
def sample_transition() -> Dict[str, object]:
    return {
        "state": {
            "observations": {
                "agent.a": {"SNR": 15.0, "energy_cost": 1.2},
                "agent.b": {"SNR": 10.0, "energy_cost": 0.8},
            }
        },
        "outcome": {
            "telemetry": {"latency": [6.0, 7.0]},
            "observations": {
                "agent.a": {"SNR": 15.0, "energy_cost": 1.2},
                "agent.b": {"SNR": 10.0, "energy_cost": 0.8},
            },
        },
    }


def test_reward_agent_scalar_output(sample_transition):
    agent = RewardAgent(weights={"energy_efficiency": 0.5, "sum_rate": 0.5})
    reward = agent.evaluate(sample_transition)

    assert isinstance(reward, float)
    # Sum-rate dominates positive contribution
    assert reward > 0.0


def test_reward_agent_vector_output(sample_transition):
    agent = RewardAgent(vector_output=True)
    rewards = agent.evaluate(sample_transition)

    assert set(rewards.keys()) == set(RewardAgent.available_rewards())
    assert rewards["outage_probability"] <= 1.0


def test_custom_reward_registration(sample_transition):
    @RewardAgent.register_reward("custom_metric")
    def _custom(agent: RewardAgent, data):
        return 42.0

    agent = RewardAgent(objectives=["custom_metric"], vector_output=False)
    try:
        reward = agent.evaluate(sample_transition)
        assert reward == 42.0
    finally:
        # cleanup to avoid leaking to other tests
        RewardAgent._REWARD_REGISTRY.pop("custom_metric", None)
