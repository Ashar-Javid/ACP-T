# Environment Recipes

This guide describes how to assemble multi-technology simulations with different fading models, technologies, and user counts using ACP-T's existing components.

## 1. Quick RIS Sandbox

Use the standalone RIS helper to validate small scenarios without editing YAML files.

```powershell
# Text summary only
python examples/ris_simple_demo/run.py --steps 8

# 2-D scatter view (requires matplotlib)
python examples/ris_simple_demo/run.py --steps 12 --plot scatter

# 3-D visualization without showing a GUI window (e.g., CI server)
python examples/ris_simple_demo/run.py --plot 3d --no-show
```

Adjustments:

| Flag | Description |
| --- | --- |
| `--steps` | Number of simulation steps; more steps exercise longer channels. |
| `--plot` | `text`, `scatter`, or `3d`. Scatter/3D encode effective SNR via color. |
| `--no-show` | Prevents matplotlib windows from opening. Useful for batch runs. |

To customize channel behavior directly, edit `acpt/examples/ris_simple_demo/simple_environment.py` and change constructor arguments (e.g., `tile_count`, `user_count`, `noise_floor_dbm`).

## 2. Multi-Domain Scenarios with Mixed Fading

The orchestrator demo (`acpt/examples/ris_v2i_noma_case`) wires RIS, NOMA, and V2I delegates through `MultiDomainEnvironment`. Override technology mix, fading models, and user populations via the YAML config files inside that folder.

### Step-by-step

1. Duplicate `acpt/examples/ris_v2i_noma_case/env_config.yaml` and edit the delegates.
2. For each delegate, set:
   - `module` / `class`: target environment (e.g., `acpt.environments.ris_environment.RISEnvironment`).
   - `kwargs.user_count`: number of UEs for that technology.
   - `fading_models`: list of models to register. Built-in aliases: `rician`, `rayleigh`, `nakagami`.
   - `mobility_models`: optional movement logic per agent.
3. Run the orchestrator with the new config:

```powershell
python -m acpt.examples.ris_v2i_noma_case.run_demo ^
  --config acpt/examples/ris_v2i_noma_case/params.yaml ^
  --steps 25
```

(Use PowerShell line continuation ``^`` or collapse into one command.)

Example fading override snippet:

```yaml
fading_models:
  - channel_id: ris_user_0
    type: nakagami
    kwargs:
      m_factor: 1.6
      omega: 1.1
  - channel_id: noma_user_1
    type: rayleigh
```

Add multiple entries to mix technologies in one run.

### Useful commands

| Purpose | Command |
| --- | --- |
| Preview orchestrator wiring | `python - <<'PY'
from acpt.utils import load_config
print(load_config('acpt/examples/ris_v2i_noma_case/wiring.yaml').keys())
PY` |
| Reset telemetry log | `Remove-Item examples/ris_v2i_noma_case/results.jsonl` |
| Visualize metrics | `python -m acpt.examples.ris_v2i_noma_case.run_demo --output latest_metrics.json` |

## 3. Building Your Own Delegate Mix

1. Create a new config file under `config/blueprints/` or `examples/<name>/` referencing `acpt.environments.multi_domain_environment.MultiDomainEnvironment`.
2. Supply a `delegates` array mixing technologies:

```yaml
delegates:
  - name: ris_lab
    module: acpt.environments.ris_environment
    class: RISEnvironment
    kwargs:
      user_count: 3
      tile_count: 96
    fading_models:
      - channel_id: ris_user_2
        type: rician
  - name: thz_probe
    module: acpt.environments.terahertz_env
    class: TerahertzEnvironment
    kwargs:
      user_count: 5
      fading_models:
        - channel_id: thz_link_0
          module: research.lab.thz_fading
          class: ShadowedRician
          kwargs:
            shadow_sigma: 3.0
```

3. Wire the new environment into `config/wiring.yaml` (or a copy) and run:

```powershell
python -m acpt.core.runtime.orchestrator ^
  --wiring config/wiring.custom.yaml ^
  --environment config/blueprints/custom_multi_domain.yaml ^
  --steps 15
```

## 4. Tips

- Use the shared `visualize_ris_state` helper for quick textual or plotted diagnostics inside notebooks or scripts.
- Mix built-in fading aliases with fully qualified modules when experimenting with new propagation models.
- Keep telemetry logs (`results.jsonl`) under `examples/<scenario>/` to compare runs.
- When changing user counts per technology, ensure coordinator manifests describe the additional agents so the registry can route proposals.
