"""NOMA environment modeling SIC decoding under composite fading.

Topology Diagram::

    [gNodeB]
       |\
       | \__ UE_far (weak channel)
       |____ UE_near (strong channel)

Mobility Layer::

    Users drift slowly; distances re-sampled with small perturbations each step.

Fading Layer::

    Hybrid Rician (near user) + Rayleigh (far user) to stress SIC ordering.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from acpt.core.interfaces.environment_interface import (
    EnvironmentInterface,
    NakagamiFadingModel,
    RicianFadingModel,
    Transition,
)


class NOMAEnvironment(EnvironmentInterface):
    """Downlink NOMA simulator with successive interference cancellation."""

    def __init__(
        self,
        *,
        tx_power_dbm: float = 33.0,
        bandwidth_hz: float = 20e6,
        noise_density_dbm_hz: float = -174.0,
        max_steps: int = 180,
        pair_count: int = 1,
    ) -> None:
        super().__init__()
        self._tx_power_dbm = float(tx_power_dbm)
        self._bandwidth_hz = float(bandwidth_hz)
        self._noise_density_dbm_hz = float(noise_density_dbm_hz)
        self._max_steps = int(max_steps)
        self._pair_count = max(int(pair_count), 1)
        self._users: List[MutableMapping[str, Any]] = []
        self._last_state: Dict[str, Any] = {}
        self._last_reward: Dict[str, float] = {"agent.noma": 0.0}
        self._episode_steps = 0

    def reset(self, seed: Optional[int] = None) -> Transition:
        self._reset_time(seed)
        self._episode_steps = 0
        self._users = self._spawn_users(seed)
        self._register_fading(seed)
        self._last_state = self._compute_observations([0.6, 0.4])
        self._last_reward = {"agent.noma": 0.0}
        return Transition(self.observe(), self.reward(), False, self._info())

    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:
        self._increment_time()
        self._episode_steps += 1
        action = action_dict.get("agent.noma", {}) or {}
        plan = action.get("noma_resource_plan", {})
        allocation = self._extract_allocation(plan)
        self._update_channels()
        self._last_state = self._compute_observations(allocation)
        done = self._episode_steps >= self._max_steps
        return Transition(self.observe(), self.reward(), done, self._info())

    def observe(self) -> Dict[str, Any]:
        return {"agent.noma": dict(self._last_state.get("agent.noma", {}))}

    def reward(self) -> Dict[str, float]:
        return dict(self._last_reward)

    # ------------------------------------------------------------------

    def _spawn_users(self, seed: Optional[int]) -> List[MutableMapping[str, Any]]:
        rng = random.Random(seed)
        users = []
        for idx in range(self._pair_count):
            far_distance = rng.uniform(90.0, 140.0)
            near_distance = rng.uniform(30.0, 60.0)
            users.append({"id": f"ue_far_{idx}", "distance": far_distance, "type": "far"})
            users.append({"id": f"ue_near_{idx}", "distance": near_distance, "type": "near"})
        return users

    def _register_fading(self, seed: Optional[int]) -> None:
        for idx, user in enumerate(self._users):
            key = f"noma_{user['id']}"
            if user["type"] == "near":
                self.register_fading_model(key, RicianFadingModel(k_factor=9.0, sigma=3.5, seed=(seed or 0) + 5 + idx))
            else:
                self.register_fading_model(key, NakagamiFadingModel(m_factor=1.2, omega=1.0, seed=(seed or 0) + 9 + idx))

    def _extract_allocation(self, plan: Mapping[str, Any]) -> List[float]:
        allocation = plan.get("allocation") if isinstance(plan, Mapping) else None
        target_length = 2 * self._pair_count
        if isinstance(allocation, Sequence) and allocation:
            values = [float(x) for x in allocation[:target_length]]
            total = sum(values) or 1.0
            return [val / total for val in values]

        power_budget = float(plan.get("power_budget", 1.0)) if isinstance(plan, Mapping) else 1.0
        pattern: List[float] = []
        for _ in range(self._pair_count):
            pattern.extend([0.6 * power_budget, 0.4 * power_budget])
        total = sum(pattern) or 1.0
        return [val / total for val in pattern]

    def _update_channels(self) -> None:
        for user in self._users:
            jitter = self._rng.uniform(-1.0, 1.0)
            user["distance"] = max(user["distance"] + jitter, 10.0)

    def _compute_observations(self, allocation: Sequence[float]) -> Dict[str, Any]:
        allocation = list(allocation) if allocation else [0.6, 0.4]
        allocation += [0.0] * (2 - len(allocation))
        sinr: List[float] = []
        rates: List[float] = []
        channel_rows: List[Dict[str, Any]] = []

        for idx, user in enumerate(self._users):
            pathloss_db = self._large_scale_loss(user["distance"])
            snr_state = {"SNR": self._tx_power_dbm - pathloss_db - self._noise_floor_dbm()}
            fading_key = f"noma_{user['id']}"
            self._apply_fading(fading_key, snr_state)
            gain_db = float(snr_state["SNR"])
            sinr_val = self._sic_sinr(idx, gain_db, allocation)
            rate_val = self._capacity(sinr_val)
            sinr.append(sinr_val)
            rates.append(rate_val)
            channel_rows.append(
                {
                    "id": user["id"],
                    "distance_m": round(user["distance"], 2),
                    "pathloss_db": round(pathloss_db, 2),
                    "post_fading_snr_db": round(gain_db, 2),
                    "sic_order": idx,
                    "sinr": round(sinr_val, 3),
                    "capacity_mbps": round(rate_val, 3),
                }
            )

        total_capacity = sum(rates)
        fairness = self._jains_index(rates)
        self._last_reward = {"agent.noma": total_capacity * fairness}
        return {"agent.noma": {"channels": channel_rows, "allocation": [round(a, 4) for a in allocation], "metrics": {"throughput_sum_mbps": round(total_capacity, 3), "jains_index": round(fairness, 3)}}}

    def _sic_sinr(self, user_index: int, gain_db: float, allocation: Sequence[float]) -> float:
        tx_power_mw = 10 ** (self._tx_power_dbm / 10.0)
        power_linear = [tx_power_mw * max(p, 0.0) for p in allocation]
        gain_linear = 10 ** (gain_db / 10.0)
        noise_linear = 10 ** (self._noise_floor_dbm() / 10.0)
        useful = power_linear[user_index] * gain_linear
        interference = sum(power_linear[user_index + 1 :]) * gain_linear
        return useful / (interference + noise_linear)

    def _capacity(self, sinr: float) -> float:
        return (self._bandwidth_hz / 1e6) * math.log2(1.0 + sinr)

    def _large_scale_loss(self, distance_m: float) -> float:
        return 32.44 + 20.0 * math.log10(max(distance_m / 1000.0, 1e-3)) + 20.0 * math.log10(2.6e9 / 1e6)

    def _noise_floor_dbm(self) -> float:
        return self._noise_density_dbm_hz + 10.0 * math.log10(self._bandwidth_hz)

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
            "bandwidth_hz": self._bandwidth_hz,
            "tx_power_dbm": self._tx_power_dbm,
            "pair_count": self._pair_count,
        }
