# Runtime Architecture

ACP-T stitches together agents, tools, and simulators through a lightweight runtime composed of three primary layers:

1. **Registry / Protocol Manager** — central directory responsible for tracking handlers and brokering JSON-RPC messages between coordinator and delegate agents.
2. **Coordinator Agent** — metric-aware planner that ranks agent proposals, selects an execution plan, and emits structured actions plus telemetry.
3. **Environment Layer** — simulator façade returning `Transition` records consumed by agents and persisted to telemetry streams.

## Composite Simulation Stack

```
┌──────────────────────────────────────────────────────────────┐
│                     MultiDomainEnvironment                   │
│  (delegates → RIS, NOMA, V2I)                                │
├──────────────┬────────────────────┬──────────────────────────┤
│ RISEnvironment│ NOMAEnvironment    │ V2IEnvironment          │
│  • Phase ctrl │  • SIC scheduler   │  • Mobility + fading    │
│  • LoS/Rician │  • Hybrid fading   │  • Link budgeting        │
└──────────────┴────────────────────┴──────────────────────────┘
```

- Delegates expose per-agent observations (`agent.ris`, `agent.noma`, `agent.v2i`).
- The composite wrapper aligns horizons, merges rewards, and propagates coordinator actions to the correct simulator.
- Additional delegates can be registered via configuration for future modalities (e.g., THz, aerial relays).

## Execution Flow

1. `Orchestrator.run()` loads `config/wiring.yaml` and `config/environment.yaml`, instantiates the registry, coordinator, and multi-domain environment.
2. Each step:
   - Agents observe their slice of the merged environment state.
   - Agents propose actions plus utility estimates.
   - Coordinator ranks proposals, commits the selected agent, and returns per-agent actions.
   - Multi-domain environment fans out actions to delegates and emits the next `Transition`.
   - Telemetry is persisted for offline analysis (`acpt.utils.metrics.persist_step`).
3. The loop halts when any delegate signals `done` or the step budget is exhausted.

## Telemetry Surface

- Results accumulate under `history[]` entries containing plan metadata, observation snapshots, rewards, and coordinator telemetry.
- Metrics can be extended through coordinator manifests (`metrics:` block) and by registering additional KPIs via `acpt.utils.metrics.register_metric`.
- Composite info payloads provide delegate-specific diagnostics under `Transition.info["delegates"]`.

## Configuration Knobs

- `config/environment.yaml` now defaults to the composite environment. Override delegate arguments (e.g., `tile_count`, `lane_count`) to tune fidelity versus runtime.
- Example scenarios (`examples/ris_v2i_noma_case`) inherit the same structure and can be duplicated for scenario-specific runs.

Refer to `docs/reference/environments.md` for delegate-specific parameter tables and ASCII topologies.
