"""Deterministic toy NR environment supporting multi-RAT simulations."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .base_env import BaseEnvironment


class ToyNREnvironment(BaseEnvironment):
    """Simple large-scale fading model for testing agent interactions."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **overrides: Any) -> None:
        params: Dict[str, Any] = dict(config or {})
        if overrides:
            params.update(overrides)

        self._config = params
        self._rats: List[Dict[str, Any]] = list(params.get("rats", []))
        if not self._rats:
            self._rats = [
                {"name": "mmwave", "freq": 28.0, "base_snr": 18.0, "pathloss_exponent": 2.2},
                {"name": "sub6", "freq": 3.5, "base_snr": 12.0, "pathloss_exponent": 2.0},
            ]

        self._channel_models: List[str] = params.get("channel_models", ["rician", "rayleigh"])
        self._time_step = float(params.get("time_step", 0.1))
        agents_cfg = params.get("agents", {})
        self._agent_defaults = {
            agent_id: {
                "pos": list(agent_cfg.get("initial_pos", [0.0, 0.0])),
                "power": float(agent_cfg.get("initial_power", 1.0)),
            }
            for agent_id, agent_cfg in agents_cfg.items()
        }

        self._agents: Dict[str, Dict[str, Any]] = {}
        self._time = 0.0

    def reset(self) -> Dict[str, Dict[str, Any]]:
        self._time = 0.0
        self._agents = {agent_id: dict(state) for agent_id, state in self._agent_defaults.items()}
        return self._snapshot()

    def step(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(actions, Mapping):
            raise TypeError("Actions must be a mapping of agent_id -> action dict")

        self._time += self._time_step
        for agent_id, state in self._agents.items():
            action = actions.get(agent_id, {}) or {}
            delta = action.get("delta_pos", [0.0, 0.0])
            if len(delta) != 2:
                raise ValueError("delta_pos must contain two elements")

            state["pos"][0] += float(delta[0])
            state["pos"][1] += float(delta[1])

            if "power" in action:
                state["power"] = float(action["power"])

            state["energy_cost"] = state["power"] * self._time_step

        return self._snapshot()

    def metadata(self) -> Dict[str, Any]:
        return {
            "rats": [
                {
                    "name": rat.get("name"),
                    "freq": rat.get("freq"),
                    "base_snr": rat.get("base_snr"),
                    "pathloss_exponent": rat.get("pathloss_exponent"),
                }
                for rat in self._rats
            ],
            "channel_models": list(self._channel_models),
            "time_step": self._time_step,
        }

    def _snapshot(self) -> Dict[str, Dict[str, Any]]:
        observations: Dict[str, Dict[str, Any]] = {}
        for agent_id, state in self._agents.items():
            snr = self._compute_snr(state["pos"], state["power"])
            observations[agent_id] = {
                "SNR": snr,
                "pos": [round(state["pos"][0], 3), round(state["pos"][1], 3)],
                "energy_cost": round(float(state.get("energy_cost", 0.0)), 6),
            }
        return observations

    def _compute_snr(self, pos: Iterable[float], power: float) -> float:
        x, y = float(pos[0]), float(pos[1])
        distance = math.hypot(x, y) + 1e-6
        snr_acc = 0.0
        for rat in self._rats:
            base_snr = float(rat.get("base_snr", 10.0))
            pathloss = float(rat.get("pathloss_exponent", 2.0))
            snr = base_snr - pathloss * math.log10(1.0 + distance)
            snr_acc += snr
        avg_snr = snr_acc / max(len(self._rats), 1)
        power_gain = 10.0 * math.log10(max(power, 1e-6))
        return round(avg_snr + power_gain, 3)