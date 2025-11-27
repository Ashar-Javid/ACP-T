"""Utility helpers for combining agent metrics into scalar utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return a normalized copy of *weights* ensuring they sum to one."""

    cleaned = {str(k): float(v) for k, v in weights.items() if float(v) > 0.0}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {str(k): 0.0 for k in weights}
    return {key: value / total for key, value in cleaned.items()}


def compute_weighted_utility(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute a weighted utility score given metric estimates and weights."""

    normal = normalize_weights(weights)
    return sum(normal.get(metric, 0.0) * float(metrics.get(metric, 0.0)) for metric in normal)


def rank_candidates(candidates: Mapping[str, Mapping[str, float]], weights: Mapping[str, float]) -> Iterable[Tuple[str, float]]:
    """Yield (candidate_id, utility) pairs ordered from highest to lowest utility."""

    scored = [
        (candidate, compute_weighted_utility(metrics, weights))
        for candidate, metrics in candidates.items()
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)
