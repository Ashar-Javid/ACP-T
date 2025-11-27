"""Command-line entry point for starting the ACP-T coordinator stack."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from acpt.core.runtime import Orchestrator, ProtocolManager, Registry
from acpt.core.runtime.context_handler import ContextHandler
from acpt.utils import load_config


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "runtime.yaml"
_DEFAULT_SCENARIO_DIR = _PROJECT_ROOT / "examples" / "ris_v2i_noma_case"
_DEFAULT_WIRING = _DEFAULT_SCENARIO_DIR / "wiring.yaml"
_DEFAULT_ENV = _DEFAULT_SCENARIO_DIR / "env_config.yaml"


@dataclass
class RuntimeSettings:
    """Resolved runtime configuration for launching the orchestrator."""

    config_path: Path
    scenario: str
    wiring_path: Path
    environment_config_path: Path
    steps: int
    task: str
    coordinator_metrics: Optional[Mapping[str, Any]]
    metrics_filter: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Boot the ACP-T orchestrator runtime.")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to the runtime configuration YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="Named scenario within the config file to launch (default: %(default)s).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric name to highlight in the summary (repeat for multiple).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate modules and configuration without running the coordinator loop.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Report runtime duration for the coordination loop.",
    )
    return parser


def _configure_logging(console: Console) -> None:
    handler = RichHandler(console=console, show_time=True, show_path=False, rich_tracebacks=True)
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _load_settings(config_path: Path, scenario: str, metrics_filter: Iterable[str] | None) -> RuntimeSettings:
    config_data = load_config(str(config_path))

    if "experiment" in config_data:
        config_data = config_data["experiment"]

    scenarios = config_data.get("scenarios") if isinstance(config_data, Mapping) else None
    if scenarios:
        if scenario not in scenarios:
            raise KeyError(f"Scenario '{scenario}' not found in configuration.")
        scenario_cfg = scenarios[scenario]
    else:
        scenario_cfg = config_data

    orchestrator_cfg = scenario_cfg.get("orchestrator", {}) if isinstance(scenario_cfg, Mapping) else {}

    wiring = orchestrator_cfg.get("wiring") or scenario_cfg.get("wiring") or config_data.get("wiring")
    environment_cfg = (
        orchestrator_cfg.get("environment")
        or scenario_cfg.get("environment")
        or config_data.get("environment")
    )
    steps = int(
        orchestrator_cfg.get("steps")
        or scenario_cfg.get("steps")
        or config_data.get("steps")
        or 5
    )
    task = (
        orchestrator_cfg.get("task")
        or scenario_cfg.get("task")
        or config_data.get("task")
        or "network_optimization"
    )
    coordinator_metrics = (
        orchestrator_cfg.get("coordinator_metrics")
        or scenario_cfg.get("coordinator_metrics")
        or config_data.get("coordinator_metrics")
    )

    base_dir = config_path.parent
    wiring_path = _resolve_path(wiring, base_dir, _DEFAULT_WIRING)
    environment_path = _resolve_path(environment_cfg, base_dir, _DEFAULT_ENV)

    metrics_tuple = tuple(metrics_filter or ())

    return RuntimeSettings(
        config_path=config_path,
        scenario=scenario,
        wiring_path=wiring_path,
        environment_config_path=environment_path,
        steps=steps,
        task=task,
        coordinator_metrics=coordinator_metrics,
        metrics_filter=metrics_tuple,
    )


def _resolve_path(candidate: Optional[str], base_dir: Path, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    path = Path(candidate)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _render_summary(console: Console, settings: RuntimeSettings, agents: Mapping[str, Any], tools: Mapping[str, Any]) -> None:
    table = Table(title="ACP-T Boot Summary", show_lines=False)
    table.add_column("Component", justify="left", no_wrap=True)
    table.add_column("Details", justify="left")

    table.add_row("Scenario", settings.scenario)
    table.add_row("Wiring", str(settings.wiring_path))
    table.add_row("Environment", str(settings.environment_config_path))
    table.add_row("Agents", ", ".join(sorted(agents.keys())) or "<none>")
    table.add_row("Tools", ", ".join(sorted(tools.keys())) or "<none>")
    console.print(table)


def _render_metrics(console: Console, metrics: Mapping[str, Any]) -> None:
    if not metrics:
        console.print("[yellow]No metrics were produced by the coordinator.[/]")
        return

    table = Table(title="Coordinator Metrics", show_lines=False)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    for name, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        table.add_row(name, formatted)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    _configure_logging(console)

    metrics_filter = args.metric or []

    try:
        settings = _load_settings(Path(args.config), args.scenario, metrics_filter)
    except Exception as exc:  # pragma: no cover - CLI surface
        console.print(f"[red]Failed to load configuration: {exc}[/]")
        return 1

    console.log(
        f"Booting scenario '{settings.scenario}' (wiring={settings.wiring_path}, environment={settings.environment_config_path})"
    )

    try:
        orchestrator = Orchestrator(
            str(settings.wiring_path),
            str(settings.environment_config_path),
            steps=settings.steps,
            task=settings.task,
            coordinator_metrics=settings.coordinator_metrics,
        )

        wiring_doc = load_config(str(settings.wiring_path))

        registry = Registry()
        protocol = ProtocolManager(registry)
        agents = orchestrator._register_agents(wiring_doc.get("agents", {}), registry)
        tools = orchestrator._register_tools(wiring_doc.get("tools", []), registry)
        coordinator, _ = orchestrator._initialise_coordinator(
            wiring_doc.get("coordinator", {}), registry, protocol
        )
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        console.print(f"[red]Boot failed while initialising components: {exc}[/]")
        return 1

    _render_summary(console, settings, agents, tools)

    if args.dry_run:
        console.log(f"Dry-run complete; coordinator instance '{coordinator.id()}' initialised.")
        return 0

    context = ContextHandler()
    context.append_message(
        "system",
        {
            "event": "boot_start",
            "scenario": settings.scenario,
            "steps": settings.steps,
        },
        ttl=300.0,
    )

    start_time = time.perf_counter() if args.profile else None
    console.log(f"Launching coordinator loop for {settings.steps} steps...")
    try:
        result = orchestrator.run()
    except Exception as exc:  # pragma: no cover - orchestrator failure path
        console.print(f"[red]Coordinator execution failed: {exc}[/]")
        return 1
    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        console.log(f"Coordinator loop finished in {elapsed:.3f}s")

    history = result.get("history", [])
    metrics_block: Dict[str, Any] = {}
    if history:
        latest = history[-1]
        context.update_env_state(("latest", "observations"), latest.get("observations", {}))
        context.record_metric(("latest", "metrics"), latest.get("metrics", {}))
        plan = latest.get("plan", {})
        telemetry = plan.get("telemetry", {})
        context.append_message(
            "coordinator",
            {
                "step": latest.get("step"),
                "selected": telemetry.get("selected"),
            },
            ttl=120.0,
        )
        for agent_id, action in plan.get("actions", {}).items():
            context.record_agent_action(agent_id, action)

        requested_metrics = settings.metrics_filter or tuple(latest.get("metrics", {}).keys())
        metrics_block = {
            name: latest.get("metrics", {}).get(name)
            for name in requested_metrics
            if name in latest.get("metrics", {}) and latest.get("metrics", {}).get(name) is not None
        }

    else:
        console.print("[yellow]Coordinator completed without producing history records.[/]")

    if not metrics_block and history:
        metrics_block = history[-1].get("metrics", {})

    if not metrics_block and result.get("metrics"):
        metrics_block = dict(result["metrics"])

    _render_metrics(console, metrics_block)

    context.record_metric(("coordinator", "weights"), result.get("metrics", {}))
    snapshot = context.snapshot().to_dict()
    console.print_json(data=snapshot)

    console.log(f"Boot sequence complete. Context version={snapshot.get('version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
