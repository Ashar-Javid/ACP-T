"""Reconfigurable intelligent surface environment with LoS and Rician fading.

Topology Diagram::

    [gNodeB]----d_bs_ris----[RIS Panel]~~~~> Users
           \___________________________/
                  Direct LoS Paths

Mobility Layer::

    UE positions wander within a corridor while the RIS stays static above street level.

Fading Layer::

    LoS component via deterministic Friis propagation + per-user Rician fading for NLoS.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from acpt.core.interfaces.environment_interface import (
    EnvironmentInterface,
    RicianFadingModel,
    Transition,
)


class RISEnvironment(EnvironmentInterface):
    """High-fidelity RIS simulator combining LoS geometry with Rician fading."""

    def __init__(
        self,
        *,
        tile_count: int = 64,
        carrier_freq_hz: float = 28e9,
        tx_power_dbm: float = 30.0,
        noise_floor_dbm: float = -94.0,
        bs_position: Sequence[float] = (0.0, 0.0, 25.0),
        ris_position: Sequence[float] = (70.0, 0.0, 18.0),
        corridor_length: float = 120.0,
        corridor_width: float = 20.0,
        max_steps: int = 240,
        user_count: int = 3,
    ) -> None:
        super().__init__()
        self._tile_count = int(tile_count)
        self._carrier_freq = float(carrier_freq_hz)
        self._wavelength = 3.0e8 / self._carrier_freq
        self._tx_power_dbm = float(tx_power_dbm)
        self._noise_floor_dbm = float(noise_floor_dbm)
        self._bs_pos = tuple(float(v) for v in bs_position)
        self._ris_pos = tuple(float(v) for v in ris_position)
        self._corridor_length = float(corridor_length)
        self._corridor_width = float(corridor_width)
        self._max_steps = int(max_steps)
        self._user_count = max(int(user_count), 1)
        self._ris_phases: List[float] = [0.0] * self._tile_count
        self._tile_spacing = 0.5 * self._wavelength
        self._users: List[MutableMapping[str, Any]] = []
        self._last_state: Dict[str, Any] = {}
        self._last_reward: Dict[str, float] = {"agent.ris": 0.0}
        self._episode_steps = 0

    def reset(self, seed: Optional[int] = None) -> Transition:
        self._reset_time(seed)
        self._episode_steps = 0
        self._ris_phases = [0.0] * self._tile_count
        self._users = self._spawn_users(seed)
        self._register_user_fading(seed)
        self._last_state = self._compute_observations()
        self._last_reward = {"agent.ris": 0.0}
        return Transition(self.observe(), self.reward(), False, self._info())

    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:
        self._increment_time()
        self._episode_steps += 1
        action = action_dict.get("agent.ris", {}) or {}
        update = action.get("ris_phase_update", {})
        projection = update.get("projection") if isinstance(update, Mapping) else None
        if isinstance(projection, Sequence) and projection:
            self._ris_phases = [float(val) for val in projection][: self._tile_count]
        else:
            phase_value = update.get("phase") if isinstance(update, Mapping) else None
            if phase_value is not None:
                self._ris_phases = [float(phase_value)] * self._tile_count

        for user in self._users:
            self._apply_user_mobility(user)

        self._last_state = self._compute_observations()
        done = self._episode_steps >= self._max_steps
        return Transition(self.observe(), self.reward(), done, self._info())

    def observe(self) -> Dict[str, Any]:
        return {"agent.ris": dict(self._last_state.get("agent.ris", {}))}

    def reward(self) -> Dict[str, float]:
        return dict(self._last_reward)

    # ------------------------------------------------------------------
    # Internal helpers

    def _spawn_users(self, seed: Optional[int]) -> List[MutableMapping[str, Any]]:
        users: List[MutableMapping[str, Any]] = []
        self._rng.seed(seed)
        for idx in range(self._user_count):
            x = self._ris_pos[0] + self._rng.uniform(-self._corridor_length / 2.0, self._corridor_length / 2.0)
            y = self._rng.uniform(-self._corridor_width / 2.0, self._corridor_width / 2.0)
            z = 1.5
            users.append({"id": f"ue_{idx}", "pos": [x, y, z], "velocity": self._rng.uniform(-1.0, 1.0)})
        return users

    def _register_user_fading(self, seed: Optional[int]) -> None:
        for idx, _ in enumerate(self._users):
            channel_id = f"ris_user_{idx}"
            self.register_fading_model(channel_id, RicianFadingModel(k_factor=7.0, sigma=1.2, seed=(seed or 0) + idx))

    def _apply_user_mobility(self, user: MutableMapping[str, Any]) -> None:
        drift = user.get("velocity", 0.0)
        user["pos"][0] += drift * 0.5
        corridor_edge = self._corridor_length / 2.0
        if user["pos"][0] > self._ris_pos[0] + corridor_edge or user["pos"][0] < self._ris_pos[0] - corridor_edge:
            user["velocity"] = -drift

    def _compute_observations(self) -> Dict[str, Any]:
        tiles = self._tile_positions()
        observations: Dict[str, Any] = {
            "agent.ris": {
                "phase_profile": [round(float(phase), 4) for phase in self._ris_phases],
                "users": [],
                "geometry": {
                    "bs": [round(v, 3) for v in self._bs_pos],
                    "ris": [round(v, 3) for v in self._ris_pos],
                },
            }
        }

        snr_acc = 0.0
        user_count = 0
        for idx, user in enumerate(self._users):
            pos = user["pos"]
            los_db = self._compute_los_snr(pos, tiles)
            fading_state = {"SNR": los_db}
            self._apply_fading(f"ris_user_{idx}", fading_state)
            effective_db = float(fading_state["SNR"])
            rician_delta = effective_db - los_db
            observations["agent.ris"]["users"].append(
                {
                    "id": user["id"],
                    "pos": [round(float(p), 3) for p in pos],
                    "los_snr_db": round(los_db, 3),
                    "rician_delta_db": round(rician_delta, 3),
                    "effective_snr_db": round(effective_db, 3),
                }
            )
            snr_acc += effective_db
            user_count += 1

        if user_count:
            self._last_reward = {"agent.ris": snr_acc / user_count}
        else:
            self._last_reward = {"agent.ris": 0.0}

        return observations

    def _compute_los_snr(self, user_pos: Sequence[float], tiles: Sequence[Sequence[float]]) -> float:
        distance_bs_ris = self._distance3(self._bs_pos, self._ris_pos)
        los_gain = 0j
        for phase, tile_pos in zip(self._ris_phases, tiles):
            d_ris_user = self._distance3(tile_pos, user_pos)
            total_distance = distance_bs_ris + d_ris_user
            phase_offset = 2.0 * math.pi * total_distance / self._wavelength
            los_gain += cmath.exp(1j * (phase - phase_offset))
        array_factor = abs(los_gain) / max(len(tiles), 1)
        pathloss_db = self._free_space_pathloss(distance_bs_ris + self._distance3(self._ris_pos, user_pos))
        rx_power_dbm = self._tx_power_dbm - pathloss_db
        snr_db = rx_power_dbm - self._noise_floor_dbm + 10.0 * math.log10(max(array_factor ** 2, 1e-9))
        return snr_db

    def _tile_positions(self) -> List[Sequence[float]]:
        tiles: List[Sequence[float]] = []
        base_x, base_y, base_z = self._ris_pos
        for idx in range(self._tile_count):
            offset = (idx - self._tile_count / 2.0) * self._tile_spacing
            tiles.append((base_x, base_y + offset, base_z))
        return tiles

    def _free_space_pathloss(self, distance_m: float) -> float:
        distance_m = max(distance_m, 1e-3)
        freq_mhz = self._carrier_freq / 1e6
        return 32.44 + 20.0 * math.log10(distance_m / 1000.0) + 20.0 * math.log10(freq_mhz)

    def _distance3(self, a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _info(self) -> Dict[str, Any]:
        return {
            "time_index": self._time_index,
            "carrier_freq_hz": self._carrier_freq,
            "tile_count": self._tile_count,
            "user_count": len(self._users),
        }
