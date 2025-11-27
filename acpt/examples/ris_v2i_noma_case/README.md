# RIS · V2I · NOMA Composite Scenario

This example exercises the multi-domain environment that couples the RIS, NOMA, and V2I simulators under a single orchestrator loop. The scenario wiring matches `config/wiring.yaml`, while `env_config.yaml` defines delegate-specific seeds and parameters.

## Running the Demo
```powershell
python -m acpt.examples.ris_v2i_noma_case.run_demo
```

The script loads the local wiring/environment configs, executes the coordinator for the default step budget, and streams telemetry to `results.jsonl`.

## Customising
- Update `env_config.yaml` to tweak delegate kwargs (`tile_count`, `lane_count`, `max_steps`, etc.).
- Adjust `wiring.yaml` to swap in different agents or toolchains while keeping the coordinator contract unchanged.
- Visualise outputs via `visualize_results.ipynb`; the notebook expects the JSONL artifact created by `run_demo.py`.

Refer to `docs/reference/environments.md` for detailed parameter descriptions and to `docs/architecture.md` for an overview of the composite runtime topology.
