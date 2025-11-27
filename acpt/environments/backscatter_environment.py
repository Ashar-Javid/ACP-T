"""Simplified NOMA + backscatter uplink environment for pipeline testing."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from acpt.core.interfaces.environment_interface import (
    EnvironmentInterface,
    NakagamiFadingModel,
    RayleighFadingModel,
    Transition,
)


class BackscatterUplinkEnvironment(EnvironmentInterface):
    """Coarse abstraction of an uplink featuring NOMA users and passive backscatter tags."""

    def __init__(
        self,
        *,
        carrier_freq_hz: float = 3.5e9,
        tx_power_dbm: float = 28.0,
        noise_floor_dbm: float = -98.0,
        user_count: int = 2,
        tag_count: int = 1,
        cell_radius_m: float = 150.0,
        max_steps: int = 120,
    ) -> None:
        super().__init__()
        self._carrier_freq_hz = float(carrier_freq_hz)
        self._tx_power_dbm = float(tx_power_dbm)
        self._noise_floor_dbm = float(noise_floor_dbm)
        self._cell_radius = float(cell_radius_m)
        self._max_steps = int(max_steps)
        self._user_count = max(int(user_count), 1)
        self._tag_count = max(int(tag_count), 0)

        self._users: List[MutableMapping[str, Any]] = []
        self._tags: List[MutableMapping[str, Any]] = []
        self._last_state: Dict[str, Any] = {}
        self._last_reward: Dict[str, float] = {"agent.noma": 0.0, "agent.backscatter": 0.0}
        self._episode_steps = 0

    def reset(self, seed: Optional[int] = None) -> Transition:  # noqa: D401
        self._reset_time(seed)
        self._episode_steps = 0
        rng = random.Random(seed)

        self._users = []
        for idx in range(self._user_count):
            distance = rng.uniform(self._cell_radius * 0.2, self._cell_radius)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            self._users.append(
                {
                    "id": f"ue_{idx}",
                    "pos": [distance * math.cos(phase), distance * math.sin(phase), 1.5],
                    "velocity": rng.uniform(-1.0, 1.0),
                    "allocation": 1.0 / self._user_count,
                }
            )

        self._tags = []
        for idx in range(self._tag_count):
            offset = rng.uniform(5.0, 30.0)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            self._tags.append(
                {
                    "id": f"tag_{idx}",
                    "pos": [offset * math.cos(angle), offset * math.sin(angle), 1.5],
                    "reflection": 0.5,
                }
            )

        for idx, _ in enumerate(self._users):
            self.register_fading_model(f"noma_uplink_{idx}", NakagamiFadingModel(m_factor=1.4, omega=1.0, seed=(seed or 0) + idx))
        for idx, _ in enumerate(self._tags):
            self.register_fading_model(f"backscatter_tag_{idx}", RayleighFadingModel(sigma=4.0, seed=(seed or 0) + 100 + idx))

        self._last_state = self._compute_state()
        self._last_reward = {"agent.noma": 0.0, "agent.backscatter": 0.0}
        return Transition(self.observe(), self.reward(), False, self._info())

    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:  # noqa: D401
        self._increment_time()
        self._episode_steps += 1

        noma_action = action_dict.get("agent.noma", {}) or {}
        allocation = self._normalize_allocation(noma_action.get("power_allocation"))
        for idx, user in enumerate(self._users):
            user["allocation"] = allocation[idx]
            self._apply_uplink_mobility(user)

        backscatter_action = action_dict.get("agent.backscatter", {}) or {}
        reflections = backscatter_action.get("reflection_profile")
        if isinstance(reflections, Sequence):
            for tag, coeff in zip(self._tags, reflections):
                tag["reflection"] = float(max(min(coeff, 1.0), 0.0))

        self._last_state = self._compute_state()
        done = self._episode_steps >= self._max_steps
        return Transition(self.observe(), self.reward(), done, self._info())

    def observe(self) -> Dict[str, Any]:  # noqa: D401
        return dict(self._last_state)

    def reward(self) -> Dict[str, float]:  # noqa: D401
        return dict(self._last_reward)

    # ------------------------------------------------------------------
    # Internal helpers

    def _apply_uplink_mobility(self, user: MutableMapping[str, Any]) -> None:
        drift = user.get("velocity", 0.0)
        user_pos = user.setdefault("pos", [0.0, 0.0, 1.5])
        user_pos[0] += drift * 0.5
        if abs(user_pos[0]) > self._cell_radius:
            user["velocity"] = -drift

    def _compute_state(self) -> Dict[str, Any]:
        noma_rows: List[Dict[str, Any]] = []
        throughput_components: List[float] = []

        for idx, user in enumerate(self._users):
            snr_state = {"SNR": self._uplink_snr(user)}
            self._apply_fading(f"noma_uplink_{idx}", snr_state)
            sinr = max(10 ** (snr_state["SNR"] / 10.0) * user["allocation"], 1e-9)
            rate = self._spectral_efficiency(sinr)
            throughput_components.append(rate)
            noma_rows.append(
                {
                    "id": user["id"],
                    "allocation": round(user["allocation"], 4),
                    "snr_db": round(snr_state["SNR"], 3),
                    "throughput_mbps": round(rate, 3),
                    "pos": [round(float(v), 3) for v in user["pos"]],
                }
            )

        tag_rows: List[Dict[str, Any]] = []
        tag_signal = 0.0
        for idx, tag in enumerate(self._tags):
            snr_state = {"SNR": self._backscatter_snr(tag)}
            self._apply_fading(f"backscatter_tag_{idx}", snr_state)
            signal = max(10 ** (snr_state["SNR"] / 10.0) * tag.get("reflection", 0.5), 1e-9)
            tag_signal += signal
            tag_rows.append(
                {
                    "id": tag["id"],
                    "reflection": round(tag.get("reflection", 0.5), 3),
                    "snr_db": round(snr_state["SNR"], 3),
                    "pos": [round(float(v), 3) for v in tag["pos"]],
                }
            )

        throughput_sum = sum(throughput_components)
        fairness = self._jains_index(throughput_components)
        noma_reward = throughput_sum * fairness
        backscatter_reward = math.log10(1.0 + tag_signal)

        self._last_reward = {
            "agent.noma": noma_reward,
            "agent.backscatter": backscatter_reward,
        }

        state = {
            "agent.noma": {
                "uplink_users": noma_rows,
                "metrics": {
                    "throughput_sum_mbps": round(throughput_sum, 3),
                    "fairness": round(fairness, 3),
                },
            },
            "agent.backscatter": {
                "tags": tag_rows,
                "aggregate_signal": round(tag_signal, 6),
            },
        }
        return state

    def _uplink_snr(self, user: Mapping[str, Any]) -> float:
        distance = self._distance(user.get("pos", [0.0, 0.0, 1.5]))
        pathloss = self._free_space_pathloss(distance)
        rx_power_dbm = self._tx_power_dbm - pathloss
        return rx_power_dbm - self._noise_floor_dbm

    def _backscatter_snr(self, tag: Mapping[str, Any]) -> float:
        distance = self._distance(tag.get("pos", [0.0, 0.0, 1.5]))
        pathloss = self._free_space_pathloss(distance) * 2.0  # double-hop penalty
        rx_power_dbm = self._tx_power_dbm - pathloss
        return rx_power_dbm - (self._noise_floor_dbm - 3.0)

    def _normalize_allocation(self, allocation: Any) -> List[float]:
        if isinstance(allocation, Sequence) and allocation:
            values = [float(max(val, 0.0)) for val in allocation[: self._user_count]]
            total = sum(values) or 1.0
            return [val / total for val in values] + [1.0 / self._user_count] * (self._user_count - len(values))
        return [1.0 / self._user_count] * self._user_count

    def _spectral_efficiency(self, sinr_linear: float) -> float:
        return (10.0 / 1.0) * math.log2(1.0 + sinr_linear)

    def _free_space_pathloss(self, distance_m: float) -> float:
        distance_m = max(distance_m, 1.0)
        freq_mhz = self._carrier_freq_hz / 1e6
        return 32.44 + 20.0 * math.log10(distance_m / 1000.0) + 20.0 * math.log10(freq_mhz)

    def _distance(self, pos: Sequence[float]) -> float:
        return math.sqrt(pos[0] ** 2 + pos[1] ** 2 + (pos[2] - 10.0) ** 2)

    def _jains_index(self, values: Sequence[float]) -> float:
        values = list(values) if values else [0.0]
        numerator = sum(values) ** 2
        denominator = len(values) * sum(v * v for v in values)
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _info(self) -> Dict[str, Any]:
        return {
            "time_index": self._time_index,
            "carrier_freq_hz": self._carrier_freq_hz,
            "user_count": self._user_count,
            "tag_count": self._tag_count,
        }
