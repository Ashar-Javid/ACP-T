"""End-to-end demo runner for the RIS/V2I/NOMA orchestration example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from acpt.core.runtime import Orchestrator
from acpt.utils import load_config, load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RIS/V2I/NOMA end-to-end demo.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "params.yaml"),
        help="Path to the experiment parameters YAML file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of steps (defaults to value from params.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write aggregated KPI summary (JSON).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing telemetry log before running the demo.",
    )
    return parser.parse_args()


def _load_params(path: Path) -> Dict[str, Any]:
    params = load_config(str(path))
    if "experiment" not in params:
        raise ValueError("Params file must include an 'experiment' section.")
    return params["experiment"]


def main() -> None:
    args = parse_args()

    base = Path(__file__).resolve().parent
    params = _load_params(Path(args.config))

    steps = args.steps or int(params.get("steps", 20))
    orchestrator_cfg = params.get("orchestrator", {})
    wiring = base / orchestrator_cfg.get("wiring", "wiring.yaml")
    env_config = base / orchestrator_cfg.get("environment", "env_config.yaml")
    results_path = base / params.get("results_path", "results.jsonl")

    if args.reset and results_path.exists():
        results_path.unlink()

    orchestrator = Orchestrator(str(wiring), str(env_config), steps=steps)
    result = orchestrator.run()

    history = result.get("history", [])
    print("Orchestrator completed", len(history), "steps")
    if history:
        last_plan = history[-1]["plan"]
        print("Selected agent:", last_plan.get("telemetry", {}).get("selected", {}))

    collected = list(load_results(results_path))
    latest_metrics: Dict[str, Any] = {}
    if not collected:
        print("No telemetry records found in results.jsonl.")
    else:
        latest = collected[-1]
        metrics = latest.get("metrics", {})
        latest_metrics = metrics
        print("Latest metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    if args.output:
        summary = {
            "steps": len(history),
            "latest_metrics": latest_metrics,
            "params": params,
            "results_path": str(results_path),
        }
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Wrote summary to", args.output)


if __name__ == "__main__":
    main()
