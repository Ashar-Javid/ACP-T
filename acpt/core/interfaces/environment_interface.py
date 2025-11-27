"""Abstract environment interface shared across ACP simulators."""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


@dataclass
class Transition:
    """Standardized transition tuple returned by ACP environments."""

    state: Dict[str, Any]
    reward: Dict[str, float]
    done: bool
    info: Dict[str, Any]


class EnvironmentInterface(abc.ABC):
    """Base class defining the contract for ACP-compatible environments.

    Concrete environments must emit structured transitions following the
    ``Transition`` dataclass. Implementations may delegate reward computation to
    a dedicated reward agent; the ``reward`` mapping should reflect the values
    used by the controller/coordinator.
    """

    def __init__(self) -> None:
        self._time_index: int = 0
        self._rng = random.Random()
        self._mobility_models: Dict[str, MobilityModel] = {}
        self._fading_models: Dict[str, FadingModel] = {}

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Transition:
        """Reset the environment using *seed* and return the initial transition."""

    @abc.abstractmethod
    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:
        """Advance the environment with the provided agent actions."""

    @abc.abstractmethod
    def observe(self) -> Dict[str, Any]:
        """Return the latest state observations made available to agents."""

    @abc.abstractmethod
    def reward(self) -> Dict[str, float]:
        """Return the reward mapping for the most recent transition."""

    def _reset_time(self, seed: Optional[int] = None) -> None:
        self._time_index = 0
        if seed is not None:
            self._rng.seed(seed)

    def _increment_time(self) -> int:
        self._time_index += 1
        return self._time_index

    def register_mobility_model(self, agent_id: str, model: "MobilityModel") -> None:
        self._mobility_models[agent_id] = model

    def register_fading_model(self, channel_id: str, model: "FadingModel") -> None:
        self._fading_models[channel_id] = model

    def _apply_mobility(self, agent_id: str, state: MutableMapping[str, Any]) -> None:
        model = self._mobility_models.get(agent_id)
        if model is not None:
            model.step(self._time_index, state)

    def _apply_fading(self, channel_id: str, state: MutableMapping[str, Any]) -> None:
        model = self._fading_models.get(channel_id)
        if model is not None:
            model.step(self._time_index, state)


class MobilityModel(abc.ABC):
    """Interface for mobility models that update agent state in-place."""

    @abc.abstractmethod
    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:
        """Mutate *state* to reflect agent motion at *time_index*."""


class FadingModel(abc.ABC):
    """Interface for wireless fading models applied to channels or links."""

    @abc.abstractmethod
    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:
        """Mutate *state* to account for fading at *time_index*."""


class RandomWalkMobility(MobilityModel):
    """Simple 2D random walk for prototype environments."""

    def __init__(self, step_size: float = 0.5, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._step_size = step_size

    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:  # noqa: D401
        pos = state.setdefault("pos", [0.0, 0.0])
        dx = self._rng.uniform(-self._step_size, self._step_size)
        dy = self._rng.uniform(-self._step_size, self._step_size)
        pos[0] = float(pos[0]) + dx
        pos[1] = float(pos[1]) + dy


class RicianFadingModel(FadingModel):
    """Rician fading approximation modeling LOS-heavy propagation."""

    def __init__(self, k_factor: float = 5.0, sigma: float = 2.0, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._k = k_factor
        self._sigma = sigma

    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:  # noqa: D401
        snr = state.get("SNR", 0.0)
        los = math.sqrt(self._k / (self._k + 1.0))
        nlos = math.sqrt(1.0 / (self._k + 1.0)) * self._rng.gauss(0.0, self._sigma)
        state["SNR"] = float(snr) + los + nlos


class RayleighFadingModel(FadingModel):
    """Rayleigh fading model capturing rich-scattering NLoS channels."""

    def __init__(self, sigma: float = 6.0, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._sigma = sigma

    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:  # noqa: D401
        offset = self._rng.gauss(0.0, self._sigma)
        state["SNR"] = float(state.get("SNR", 0.0)) + offset


class NakagamiFadingModel(FadingModel):
    """Nakagami-m fading distribution for generalized multipath settings."""

    def __init__(self, m_factor: float = 1.5, omega: float = 1.0, seed: Optional[int] = None) -> None:
        if m_factor <= 0.0:
            raise ValueError("m_factor must be positive for Nakagami fading")
        if omega <= 0.0:
            raise ValueError("omega must be positive for Nakagami fading")
        self._rng = random.Random(seed)
        self._m = m_factor
        self._omega = omega

    def step(self, time_index: int, state: MutableMapping[str, Any]) -> None:  # noqa: D401
        # Sample power gain following Nakagami distribution (shape m, spread omega).
        # Using inverse transform via gamma distribution approximation.
        gamma_shape = self._m
        gamma_scale = self._omega / self._m
        gain = self._rng.gammavariate(gamma_shape, gamma_scale)
        snr = float(state.get("SNR", 0.0))
        # Convert gain to dB impact.
        snr += 10.0 * math.log10(max(gain, 1e-9))
        state["SNR"] = snr
