"""Reward computation agent supporting multi-objective metrics."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC

import math
import statistics
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from acpt.utils import get_logger


RewardFunc = Callable[["RewardAgent", Mapping[str, Any]], float]


class RewardAgent:
    """Flexible reward computation agent for ACP orchestrations."""

    _REWARD_REGISTRY: Dict[str, RewardFunc] = {}
    _DEFAULT_OBJECTIVES = [
        "energy_efficiency",
        "fairness",
        "latency",
        "sum_rate",
        "outage_probability",
    ]

    def __init__(
        self,
        objectives: Optional[Sequence[str]] = None,
        *,
        weights: Optional[Mapping[str, float]] = None,
        vector_output: bool = False,
        outage_threshold: float = 5.0,
    ) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._objectives = [str(obj) for obj in (objectives or self._DEFAULT_OBJECTIVES)]
        self._weights = {str(k): float(v) for k, v in (weights or {}).items()}
        self._vector_output = vector_output
        self._outage_threshold = float(outage_threshold)
        self._validate_objectives()

    # ------------------------------------------------------------------
    # Public API

    def evaluate(
        self,
        transition: Optional[Mapping[str, Any]] = None,
        *,
        state: Optional[Mapping[str, Any]] = None,
        action: Optional[Mapping[str, Any]] = None,
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> float | Dict[str, float]:
        """Compute rewards for the supplied transition tuple."""

        data: Dict[str, Any] = dict(transition or {})
        if state is not None:
            data["state"] = state
        if action is not None:
            data["action"] = action
        if outcome is not None:
            data["outcome"] = outcome

        values: Dict[str, float] = {}
        for objective in self._objectives:
            func = self._REWARD_REGISTRY.get(objective)
            if func is None:
                raise KeyError(f"Reward objective '{objective}' is not registered.")
            values[objective] = float(func(self, data))

        if self._vector_output:
            return values

        reward = 0.0
        for name, value in values.items():
            weight = self._weights.get(name, 1.0)
            reward += weight * value
        return reward

    @classmethod
    def register_reward(cls, name: str) -> Callable[[RewardFunc], RewardFunc]:
        """Decorator used to register a custom reward function."""

        key = str(name)

        def decorator(func: RewardFunc) -> RewardFunc:
            cls._REWARD_REGISTRY[key] = func
            return func

        return decorator

    @classmethod
    def available_rewards(cls) -> Sequence[str]:
        """Return the identifiers of all registered reward functions."""

        return tuple(sorted(cls._REWARD_REGISTRY.keys()))

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Update the aggregation weights for scalar reward output."""

        self._weights = {str(k): float(v) for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Internal helpers

    def _validate_objectives(self) -> None:
        missing = [name for name in self._objectives if name not in self._REWARD_REGISTRY]
        if missing:
            raise ValueError(f"Unknown reward objectives requested: {missing}")

    def _observations(self, data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        candidates: List[Mapping[str, Any]] = []
        outcome = data.get("outcome")
        if isinstance(outcome, Mapping):
            obs = outcome.get("observations") or outcome.get("state")
            if isinstance(obs, Mapping):
                candidates.append(obs)
        state = data.get("state")
        if isinstance(state, Mapping):
            obs = state.get("observations")
            if isinstance(obs, Mapping):
                candidates.append(obs)
        own = data.get("observations")
        if isinstance(own, Mapping):
            candidates.append(own)

        if not candidates:
            return {}
        merged: Dict[str, Dict[str, Any]] = {}
        for mapping in candidates:
            for agent_id, payload in mapping.items():
                merged.setdefault(str(agent_id), {}).update(_ensure_mapping(payload))
        return merged

    def _metrics(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        metrics = data.get("metrics")
        if isinstance(metrics, Mapping):
            return {str(k): v for k, v in metrics.items()}
        outcome = data.get("outcome")
        if isinstance(outcome, Mapping):
            telemetry = outcome.get("telemetry")
            if isinstance(telemetry, Mapping):
                return {str(k): v for k, v in telemetry.items()}
        return {}


# ---------------------------------------------------------------------------
# Default reward functions


@RewardAgent.register_reward("energy_efficiency")
def _reward_energy_efficiency(agent: RewardAgent, data: Mapping[str, Any]) -> float:
    observations = agent._observations(data)
    if not observations:
        return 0.0
    total_throughput = 0.0
    total_energy = 0.0
    for payload in observations.values():
        throughput = payload.get("throughput")
        if throughput is None:
            throughput = payload.get("SNR") or payload.get("rate")
        total_throughput += float(throughput or 0.0)
        energy = payload.get("energy_cost")
        if energy is None:
            energy = payload.get("power")
        total_energy += max(float(energy or 0.0), 1e-9)
    return total_throughput / total_energy if total_energy else 0.0


@RewardAgent.register_reward("fairness")
def _reward_fairness(agent: RewardAgent, data: Mapping[str, Any]) -> float:
    observations = agent._observations(data)
    if not observations:
        return 0.0
    values = []
    for payload in observations.values():
        metric = payload.get("throughput")
        if metric is None:
            metric = payload.get("SNR") or payload.get("rate")
        values.append(float(metric or 0.0))
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value**2 for value in values)
    return numerator / denominator if denominator else 0.0


@RewardAgent.register_reward("latency")
def _reward_latency(agent: RewardAgent, data: Mapping[str, Any]) -> float:
    metrics = agent._metrics(data)
    latencies = _extract_numeric(metrics, ["latency", "latency_ms"])
    if not latencies:
        outcome = data.get("outcome")
        if isinstance(outcome, Mapping):
            latencies = _extract_numeric(outcome, ["latency", "latency_ms"])
    if not latencies:
        return 0.0
    mean_latency = statistics.fmean(latencies)
    return -float(mean_latency)


@RewardAgent.register_reward("sum_rate")
def _reward_sum_rate(agent: RewardAgent, data: Mapping[str, Any]) -> float:
    observations = agent._observations(data)
    total = 0.0
    for payload in observations.values():
        rate = payload.get("throughput")
        if rate is None:
            rate = payload.get("SNR") or payload.get("rate")
        total += float(rate or 0.0)
    return total


@RewardAgent.register_reward("outage_probability")
def _reward_outage_probability(agent: RewardAgent, data: Mapping[str, Any]) -> float:
    observations = agent._observations(data)
    if not observations:
        return 0.0

    threshold = agent._outage_threshold
    total = 0
    outages = 0
    for payload in observations.values():
        sinr = payload.get("SINR")
        if sinr is None:
            sinr = payload.get("SNR") or payload.get("throughput")
        if sinr is None:
            continue
        total += 1
        if float(sinr) < threshold:
            outages += 1
    if total == 0:
        return 0.0
    return outages / total


# ---------------------------------------------------------------------------
# Utilities


def _ensure_mapping(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items()}
    return {"value": payload}


def _extract_numeric(container: Mapping[str, Any], keys: Sequence[str]) -> List[float]:
    values: List[float] = []
    for key in keys:
        value = container.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(float(item) for item in value if isinstance(item, (int, float)))
    return values
