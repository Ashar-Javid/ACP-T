"""Unit tests for toy NR environment."""

from __future__ import annotations

import math
from typing import Dict

import pytest

from acpt.environments.multi_domain_environment import MultiDomainEnvironment
from acpt.environments.toy_nr_env import ToyNREnvironment


@pytest.fixture
def env_config() -> Dict[str, object]:
	return {
		"time_step": 0.25,
		"channel_models": ["rician", "rayleigh"],
		"rats": [
			{"name": "mmwave", "freq": 28.0, "base_snr": 19.0, "pathloss_exponent": 2.2},
			{"name": "sub6", "freq": 3.5, "base_snr": 13.0, "pathloss_exponent": 1.9},
		],
		"agents": {
			"ris-1": {"initial_pos": [10.0, 0.0], "initial_power": 1.5},
			"v2i-1": {"initial_pos": [0.0, 6.0], "initial_power": 1.2},
		},
	}


@pytest.fixture
def toy_env(env_config: Dict[str, object]) -> ToyNREnvironment:
	return ToyNREnvironment(env_config)


@pytest.fixture
def multi_domain_env() -> MultiDomainEnvironment:
	return MultiDomainEnvironment(
		delegates=[
			{
				"module": "acpt.environments.ris_environment",
				"class": "RISEnvironment",
				"agents": ["agent.ris"],
				"kwargs": {"tile_count": 8, "max_steps": 4},
			},
			{
				"module": "acpt.environments.noma_environment",
				"class": "NOMAEnvironment",
				"agents": ["agent.noma"],
				"kwargs": {"max_steps": 4},
			},
			{
				"module": "acpt.environments.v2i_environment",
				"class": "V2IEnvironment",
				"agents": ["agent.v2i"],
				"kwargs": {"lane_count": 2, "max_steps": 4},
			},
		]
	)


def _expected_snr(env: ToyNREnvironment, pos: Dict[str, float], power: float) -> float:
	metadata = env.metadata()
	distance = math.hypot(pos[0], pos[1]) + 1e-6
	snr_sum = 0.0
	rats = metadata.get("rats", [])
	for rat in rats:
		base = float(rat.get("base_snr", 10.0))
		exponent = float(rat.get("pathloss_exponent", 2.0))
		snr_sum += base - exponent * math.log10(1.0 + distance)
	avg_snr = snr_sum / max(len(rats), 1)
	power_gain = 10.0 * math.log10(max(power, 1e-6))
	return round(avg_snr + power_gain, 3)


def test_reset_returns_expected_snapshot(toy_env: ToyNREnvironment, env_config: Dict[str, object]) -> None:
	observations = toy_env.reset()

	assert set(observations) == set(env_config["agents"])

	for agent_id, state in observations.items():
		initial = env_config["agents"][agent_id]
		assert state["pos"] == pytest.approx(initial["initial_pos"], rel=1e-3)
		assert state["energy_cost"] == pytest.approx(0.0)
		expected_snr = _expected_snr(toy_env, initial["initial_pos"], initial["initial_power"])
		assert state["SNR"] == pytest.approx(expected_snr)


def test_step_updates_position_and_energy(toy_env: ToyNREnvironment, env_config: Dict[str, object]) -> None:
	toy_env.reset()

	actions = {
		"ris-1": {"delta_pos": [1.0, -1.0], "power": 2.0},
		"v2i-1": {"delta_pos": [0.5, 0.5]},
	}

	observations = toy_env.step(actions)

	ris_state = observations["ris-1"]
	assert ris_state["pos"] == pytest.approx([11.0, -1.0], rel=1e-3)
	assert ris_state["energy_cost"] == pytest.approx(2.0 * toy_env.metadata()["time_step"])
	assert ris_state["SNR"] == pytest.approx(_expected_snr(toy_env, [11.0, -1.0], 2.0))

	v2i_state = observations["v2i-1"]
	assert v2i_state["pos"] == pytest.approx([0.5, 6.5], rel=1e-3)
	# Power unchanged, so energy cost uses previous value.
	original_power = env_config["agents"]["v2i-1"]["initial_power"]
	assert v2i_state["energy_cost"] == pytest.approx(original_power * toy_env.metadata()["time_step"])
	assert v2i_state["SNR"] == pytest.approx(_expected_snr(toy_env, [0.5, 6.5], original_power))


def test_invalid_delta_pos_raises(toy_env: ToyNREnvironment) -> None:
	toy_env.reset()

	with pytest.raises(ValueError):
		toy_env.step({"ris-1": {"delta_pos": [0.0]}})


def test_metadata_reflects_configuration(toy_env: ToyNREnvironment, env_config: Dict[str, object]) -> None:
	info = toy_env.metadata()

	assert info["time_step"] == env_config["time_step"]
	assert info["channel_models"] == env_config["channel_models"]

	env_rats = env_config["rats"]
	assert len(info["rats"]) == len(env_rats)
	for expected, actual in zip(env_rats, info["rats"]):
		assert actual["name"] == expected["name"]
		assert actual["freq"] == expected["freq"]
		assert actual["base_snr"] == expected["base_snr"]
		assert actual["pathloss_exponent"] == expected["pathloss_exponent"]


def test_multi_domain_reset_exposes_all_agents(multi_domain_env: MultiDomainEnvironment) -> None:
	transition = multi_domain_env.reset(seed=123)

	assert set(transition.state) == {"agent.ris", "agent.noma", "agent.v2i"}
	assert transition.reward == {"agent.ris": 0.0, "agent.noma": 0.0, "agent.v2i": 0.0}
	assert transition.done is False


def test_multi_domain_step_propagates_actions(multi_domain_env: MultiDomainEnvironment) -> None:
	multi_domain_env.reset(seed=321)

	actions = {
		"agent.ris": {"ris_phase_update": {"phase": 0.1}},
		"agent.noma": {"noma_resource_plan": {"allocation": [0.7, 0.3]}},
		"agent.v2i": {"v2i_link_plan": {"power_allocation": [0.5, 0.5]}},
	}

	transition = multi_domain_env.step(actions)

	assert set(transition.state) == {"agent.ris", "agent.noma", "agent.v2i"}
	assert transition.reward.keys() == {"agent.ris", "agent.noma", "agent.v2i"}
	assert transition.done is False
