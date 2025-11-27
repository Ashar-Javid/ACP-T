"""Vehicle-to-infrastructure environment with explicit mobility and fading layers.

Topology Diagram::

    Vehicles ---> ---> [RSU] <--- Roadside corridor (4 lanes)

Mobility Layer::

    Linear lane-following with Gaussian velocity drift per vehicle.

Fading Layer::

    Composite pathloss + Rician fast fading per V2I link.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from acpt.core.interfaces.environment_interface import (
    EnvironmentInterface,
    MobilityModel,
    RicianFadingModel,
    Transition,
)


class _LinearMobility(MobilityModel):
    def __init__(self, base_speed: float, drift_sigma: float = 1.5, seed: Optional[int] = None) -> None:
        self._base_speed = base_speed
        self._drift_sigma = drift_sigma
        self._rng = random.Random(seed)

    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:  # noqa: D401
        velocity = state.setdefault("velocity", self._base_speed)
        velocity += self._rng.gauss(0.0, self._drift_sigma) * 0.1
        velocity = max(velocity, 0.0)
        state["velocity"] = velocity
        pos = state.setdefault("pos", [0.0, 0.0, 0.0])
        pos[0] += velocity


class V2IEnvironment(EnvironmentInterface):
    """Mobility-aware V2I simulator producing link-quality metrics for coordination agents."""

    def __init__(
        self,
        *,
        rsu_position: Sequence[float] = (0.0, 0.0, 10.0),
        lane_count: int = 4,
        lane_spacing: float = 3.5,
        max_steps: int = 200,
        noise_floor_dbm: float = -96.0,
        tx_power_dbm: float = 26.0,
    ) -> None:
        super().__init__()
        self._rsu_position = tuple(float(v) for v in rsu_position)
        self._lane_count = lane_count
        self._lane_spacing = float(lane_spacing)
        self._max_steps = int(max_steps)
        self._noise_floor_dbm = float(noise_floor_dbm)
        self._tx_power_dbm = float(tx_power_dbm)
        self._vehicles: List[MutableMapping[str, Any]] = []
        self._last_state: Dict[str, Any] = {}
        self._last_reward: Dict[str, float] = {"agent.v2i": 0.0}
        self._episode_steps = 0

    def reset(self, seed: Optional[int] = None) -> Transition:
        self._reset_time(seed)
        self._episode_steps = 0
        self._vehicles = self._spawn_vehicles(seed)
        self._register_models(seed)
        self._last_state = self._compute_observations([1.0 for _ in self._vehicles])
        self._last_reward = {"agent.v2i": 0.0}
        return Transition(self.observe(), self.reward(), False, self._info())

    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:
        self._increment_time()
        self._episode_steps += 1
        for vehicle in self._vehicles:
            self._apply_mobility(vehicle["id"], vehicle)
        action = action_dict.get("agent.v2i", {}) or {}
        plan = action.get("v2i_link_plan", {})
        allocation = self._extract_allocation(plan)
        self._last_state = self._compute_observations(allocation)
        done = self._episode_steps >= self._max_steps
        return Transition(self.observe(), self.reward(), done, self._info())

    def observe(self) -> Dict[str, Any]:
        return {"agent.v2i": dict(self._last_state.get("agent.v2i", {}))}

    def reward(self) -> Dict[str, float]:
        return dict(self._last_reward)

    # ------------------------------------------------------------------

    def _spawn_vehicles(self, seed: Optional[int]) -> List[MutableMapping[str, Any]]:
        rng = random.Random(seed)
        vehicles: List[MutableMapping[str, Any]] = []
        for lane in range(self._lane_count):
            start_x = rng.uniform(-150.0, -20.0)
            y = (lane - self._lane_count / 2.0) * self._lane_spacing
            vehicle_id = f"veh_{lane}"
            vehicles.append({"id": vehicle_id, "pos": [start_x, y, 1.5], "velocity": rng.uniform(18.0, 33.0)})
        return vehicles

    def _register_models(self, seed: Optional[int]) -> None:
        for vehicle in self._vehicles:
            vid = vehicle["id"]
            self.register_mobility_model(vid, _LinearMobility(vehicle["velocity"], seed=(seed or 0) + hash(vid) % 997))
            self.register_fading_model(
                f"v2i_{vid}",
                RicianFadingModel(k_factor=3.0, sigma=4.0, seed=(seed or 0) + hash(vid) % 4093),
            )

    def _extract_allocation(self, plan: Mapping[str, Any]) -> List[float]:
        if isinstance(plan, Mapping):
            alloc = plan.get("power_allocation")
            if isinstance(alloc, Sequence) and alloc:
                values = [float(v) for v in alloc]
                if sum(values):
                    total = sum(values)
                    return [val / total for val in values]
        return [1.0 for _ in self._vehicles]

    def _compute_observations(self, allocation: Sequence[float]) -> Dict[str, Any]:
        allocation = list(allocation)
        if len(allocation) < len(self._vehicles):
            allocation.extend([1.0] * (len(self._vehicles) - len(allocation)))
        tx_power_mw = 10 ** (self._tx_power_dbm / 10.0)

        vehicle_rows: List[Dict[str, Any]] = []
        scores: List[float] = []
        for idx, vehicle in enumerate(self._vehicles):
            alloc = max(allocation[idx], 0.0)
            power_mw = tx_power_mw * alloc
            pathloss_db = self._pathloss(self._distance3(vehicle["pos"], self._rsu_position))
            rx_power_dbm = self._tx_power_dbm + 10.0 * math.log10(max(alloc, 1e-6)) - pathloss_db
            snr_state = {"SNR": rx_power_dbm - self._noise_floor_dbm}
            self._apply_fading(f"v2i_{vehicle['id']}", snr_state)
            snr_db = float(snr_state["SNR"])
            sinr_linear = 10 ** (snr_db / 10.0)
            throughput_mbps = (10.0 * math.log10(1.0 + sinr_linear)) * 5.0
            scores.append(sinr_linear)
            vehicle_rows.append(
                {
                    "id": vehicle["id"],
                    "pos": [round(float(p), 3) for p in vehicle["pos"]],
                    "velocity_mps": round(float(vehicle.get("velocity", 0.0)), 3),
                    "pathloss_db": round(pathloss_db, 3),
                    "snr_db": round(snr_db, 3),
                    "allocated_power_mw": round(power_mw, 3),
                    "throughput_mbps": round(throughput_mbps, 3),
                }
            )

        avg_score = sum(scores) / max(len(scores), 1)
        coverage = sum(1 for s in scores if s > 5.0) / max(len(scores), 1)
        self._last_reward = {"agent.v2i": avg_score * coverage}
        return {
            "agent.v2i": {
                "vehicles": vehicle_rows,
                "link_score": round(avg_score, 3),
                "coverage_probability": round(coverage, 3),
            }
        }

    def _pathloss(self, distance: float) -> float:
        distance = max(distance, 1.0)
        freq_mhz = 5.9e9 / 1e6
        return 32.44 + 20.0 * math.log10(distance / 1000.0) + 20.0 * math.log10(freq_mhz)

    def _distance3(self, a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _info(self) -> Dict[str, Any]:
        return {
            "time_index": self._time_index,
            "vehicle_count": len(self._vehicles),
            "rsu_position": [round(v, 3) for v in self._rsu_position],
        }
