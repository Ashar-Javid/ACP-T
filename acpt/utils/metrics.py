"""Metrics registry, KPI computations, and telemetry helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from .logging_utils import get_logger


MetricFunc = Callable[[Mapping[str, Any]], float]

_LOGGER = get_logger("Metrics")

_DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "ris_v2i_noma_case"
    / "results.jsonl"
)

_METRICS_REGISTRY: Dict[str, MetricFunc] = {}


def register_metric(name: str, func: MetricFunc, *, overwrite: bool = False) -> None:
    """Register *func* under *name* in the metrics registry."""

    if not callable(func):
        raise TypeError("Metric function must be callable.")

    key = str(name)
    if key in _METRICS_REGISTRY and not overwrite:
        raise ValueError(f"Metric '{key}' already registered.")

    _METRICS_REGISTRY[key] = func


def list_metrics() -> Iterable[str]:
    """Return an iterable of registered metric names."""

    return tuple(_METRICS_REGISTRY.keys())


def compute_metric(name: str, record: Mapping[str, Any]) -> float:
    """Compute a single metric by *name* from *record*."""

    try:
        func = _METRICS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Metric '{name}' is not registered.") from exc
    return float(func(record))


def compute_metrics(
    record: Mapping[str, Any],
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute a set of metrics for *record* using *metric_names* or all registered metrics."""

    names = tuple(metric_names) if metric_names is not None else tuple(_METRICS_REGISTRY.keys())
    return {name: compute_metric(name, record) for name in names}


def persist_step(
    record: Mapping[str, Any],
    *,
    metric_names: Optional[Iterable[str]] = None,
    results_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Persist *record* with computed metrics to the JSONL results log."""

    metrics = compute_metrics(record, metric_names)
    payload: Dict[str, Any] = dict(record)
    payload.setdefault("metrics", {}).update(metrics)

    path = Path(results_path) if results_path is not None else _DEFAULT_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_serializable(payload), default=_fallback_serializer) + "\n")

    _LOGGER.info("Persisted telemetry for step=%s to %s", payload.get("step"), path)
    return payload


def load_results(results_path: Optional[Path | str] = None) -> Iterable[Dict[str, Any]]:
    """Yield telemetry records from the JSONL results log."""

    path = Path(results_path) if results_path is not None else _DEFAULT_RESULTS_PATH
    if not path.exists():
        return []

    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _LOGGER.warning("Skipping malformed telemetry line: %s", exc)
    return records


def _serializable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): _serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serializable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fallback_serializer(obj: Any) -> Any:
    return str(obj)


def _energy_kpi(record: Mapping[str, Any]) -> float:
    observations = record.get("observations", {})
    return float(
        sum(float(agent.get("energy_cost", 0.0)) for agent in observations.values())
    )


def _throughput_kpi(record: Mapping[str, Any]) -> float:
    observations = record.get("observations", {})
    return float(sum(float(agent.get("SNR", 0.0)) for agent in observations.values()))


def _fairness_kpi(record: Mapping[str, Any]) -> float:
    observations = record.get("observations", {})
    values = [float(agent.get("SNR", 0.0)) for agent in observations.values()]
    if not values:
        return 1.0
    numerator = (sum(values) ** 2)
    denominator = len(values) * sum(value ** 2 for value in values)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _latency_kpi(record: Mapping[str, Any]) -> float:
    plan = record.get("plan", {})
    telemetry = plan.get("telemetry", {})
    latencies = telemetry.get("latency") or telemetry.get("latency_ms")
    if isinstance(latencies, (list, tuple, set)):
        if not latencies:
            return 0.0
        return float(sum(float(value) for value in latencies) / len(latencies))
    if isinstance(latencies, (int, float)):
        return float(latencies)
    # Fallback: distance moved encoded via observations powering simple heuristic
    observations = record.get("observations", {})
    return float(sum(math.hypot(*agent.get("pos", [0.0, 0.0])) for agent in observations.values()))


def _handoff_success_kpi(record: Mapping[str, Any]) -> float:
    plan = record.get("plan", {})
    allocations = plan.get("allocations", {})
    if allocations:
        approvals = [bool(details.get("approved")) for details in allocations.values()]
        return float(sum(approvals) / len(approvals)) if approvals else 0.0
    telemetry = plan.get("telemetry", {})
    value = telemetry.get("handoff_success") or telemetry.get("handoff_success_rate")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _ensure_defaults() -> None:
    defaults = {
        "energy": _energy_kpi,
        "throughput": _throughput_kpi,
        "fairness": _fairness_kpi,
        "latency": _latency_kpi,
        "handoff_success": _handoff_success_kpi,
    }
    for name, func in defaults.items():
        if name not in _METRICS_REGISTRY:
            _METRICS_REGISTRY[name] = func


_ensure_defaults()


__all__ = [
    "compute_metric",
    "compute_metrics",
    "list_metrics",
    "load_results",
    "persist_step",
    "register_metric",
]
