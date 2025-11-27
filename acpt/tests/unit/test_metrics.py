"""Unit tests for metrics registry and telemetry helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acpt.utils import compute_metric, compute_metrics, persist_step, register_metric


def _sample_record() -> dict[str, object]:
    return {
        "step": 1,
        "observations": {
            "agent.alpha": {"energy_cost": 0.3, "SNR": 15.0, "pos": [1.0, 0.0]},
            "agent.beta": {"energy_cost": 0.2, "SNR": 9.0, "pos": [0.0, 2.0]},
        },
        "plan": {
            "allocations": {
                "agent.alpha": {"approved": True},
                "agent.beta": {"approved": False},
            },
            "telemetry": {"latency": [8.0, 12.0]},
        },
    }


def test_default_metrics_compute_expected_values() -> None:
    record = _sample_record()
    metrics = compute_metrics(record)

    assert pytest.approx(metrics["energy"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["throughput"], rel=1e-6) == 24.0
    assert 0.0 <= metrics["fairness"] <= 1.0
    assert pytest.approx(metrics["latency"], rel=1e-6) == 10.0
    assert pytest.approx(metrics["handoff_success"], rel=1e-6) == 0.5


def test_register_custom_metric_allows_extension() -> None:
    record = _sample_record()

    register_metric("custom_metric", lambda data: 42.0, overwrite=True)
    assert compute_metric("custom_metric", record) == 42.0


def test_persist_step_appends_jsonl(tmp_path: Path) -> None:
    record = _sample_record()
    results_path = tmp_path / "results.jsonl"

    payload = persist_step(record, results_path=results_path)

    assert results_path.exists()
    contents = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    logged = json.loads(contents[0])
    assert logged["step"] == record["step"]
    assert "metrics" in logged and "energy" in logged["metrics"]
    assert payload["metrics"]["energy"] == logged["metrics"]["energy"]