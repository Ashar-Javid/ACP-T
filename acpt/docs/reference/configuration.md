# Configuration Patterns

This guide outlines how to compose ACP-T scenarios using modular agents, tools, and environments.

## Wiring Overview

- `config/wiring.yaml` — Declares active agents, tools, coordinator, and their manifests.
- `config/environment.yaml` — Selects the runtime simulator. Defaults to the multi-domain environment that fans in RIS, NOMA, V2I, and (optionally) backscatter components.
- `config/blueprints/` — Contains reusable environment templates such as `ris_two_user.yaml`, `noma_three_pair.yaml`, and `uplink_noma_backscatter.yaml`.

To run an alternate blueprint:

```powershell
python -m acpt.core.runtime.orchestrator ^
  --wiring config/wiring.yaml ^
  --environment config/blueprints/noma_three_pair.yaml
```

## Delegate Specification Schema

Each entry under `environment.config.delegates` accepts the following keys:

| Key | Type | Description |
| --- | --- | --- |
| `module` / `class` | string | Fully qualified path to the environment implementation. |
| `agents` | list[str] | Agent IDs expected by the delegate (defaults inferred from transitions). |
| `args` / `kwargs` | list / mapping | Positional and keyword arguments forwarded to the delegate constructor. |
| `seed` | int | Optional deterministic seed for the delegate. |
| `fading_models` | list[dict] | Override or extend fading channels. Supports built-ins (`rician`, `rayleigh`, `nakagami`) or custom classes. Requires `channel_id`. |
| `mobility_models` | list[dict] | Register mobility behaviours (e.g., `random_walk`). Requires `agent_id`. |

### Example: Custom Fading & Mobility

```yaml
delegates:
  - name: ris_lab
    module: acpt.environments.ris_environment
    class: RISEnvironment
    kwargs:
      user_count: 2
      tile_count: 96
    fading_models:
      - channel_id: ris_user_0
        type: nakagami
        kwargs:
          m_factor: 1.6
          omega: 1.1
    mobility_models:
      - agent_id: agent.ris_probe
        type: random_walk
        kwargs:
          step_size: 0.25
```

### Example: Loading a Custom Component

```yaml
fading_models:
  - channel_id: noma_custom
    module: research.sim.fading
    class: RiceanShadowed
    kwargs:
      k_factor: 7.5
      shadow_sigma: 2.0
```

## Agent & Tool Manifests

- Agents expose `observe`, `propose`, and `feedback` methods; manifests declare `llm_spec`, optional `rag_descriptor`, and capability metadata.
- Tools implement an `invoke(payload)` callable interface. Standardized envelopes return `{ "result": ..., "diagnostics": ... }`.

Use the registrar CLI to scaffold new entries:

```powershell
python scripts/add_agent.py --manifest manifests/custom_agent.json --scaffold
python scripts/add_tool.py --manifest manifests/snr_estimator.json --scaffold
```

## Environment Variables

The orchestrator reads environment definitions from YAML, enabling teams to vary:

- Radio parameters: `carrier_freq_hz`, `tx_power_dbm`, `noise_floor_dbm`.
- Topologies: `user_count`, `pair_count`, `lane_count`, `corridor_length`.
- Mobility types: `random_walk` or custom models via `mobility_models`.
- Channel models: select between Rician, Rayleigh, Nakagami, or plug-in classes.
- Agent identities: add `agent.uav`, `agent.pedestrian`, `agent.drone` by registering corresponding manifests and including them in delegates.

For 6G-centric defaults, the blueprints leverage mmWave (28 GHz) for RIS, sub-6 GHz for vehicular links, and wideband NOMA cells, while keeping every parameter overrideable through YAML.
```}