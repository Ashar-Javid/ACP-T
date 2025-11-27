# ACP-T Plug-and-Optimize Workflow

This guide summarizes the command-line utilities added in PR-7 for registering new agents, tools, and knowledge-base entries. The goal is to make plug-and-play onboarding consistent across developer environments.

## Prerequisites
- Activate your virtual environment and install the project dependencies (`pip install -r requirements.txt`).
- Ensure `config/wiring.yaml` remains under version control; it now ships with RIS/V2I/NOMA agent registrations that can be extended via the CLI.
- Agent and tool manifests must adhere to `specs/agent_manifest_schema.json` (see `utils/registrar.validate_manifest`).

## Adding an Agent
```bash
python scripts/add_agent.py \
  --manifest manifests/ris_agent.json \
  --module acpt.agents.ris_agent \
  --class-name RISAgent
```

Key flags:
- `--scaffold` generates a skeleton agent file under `acpt/agents/` using the manifest id.
- `--force` overwrites an existing entry in `config/wiring.yaml`.
- `--agents-dir` customizes where scaffolds are written (defaults to `acpt/agents`).

The script validates the manifest, updates `config/wiring.yaml`, and (if requested) scaffolds a baseline agent implementation aligned with the standardized tool envelope.

## Adding a Tool
```bash
python scripts/add_tool.py \
  --manifest manifests/gd_solver.json \
  --module acpt.tools.solvers.gd_solver \
  --class-name GradientDescentSolver
```

Flags mirror `add_agent.py`. When `--scaffold` is provided, the CLI writes a stub into `acpt/tools/custom/` and wires it into the YAML. Use `--force` to replace existing tool registrations.

## Appending Knowledge-Base Entries
```bash
python scripts/add_kb_entry.py \
  --id kb.ris \
  --document "Latest RIS tuning notes" \
  --file notes/ris_policy.txt
```

- When no `--path` is supplied, entries land in `acpt/data/kb/<id>.json`.
- Pass `--skip-wiring` to avoid mutating `config/wiring.yaml`.
- Use `--force` to overwrite an existing knowledge descriptor.

Multiple `--document` or `--file` flags may be supplied; each document is appended to the JSON array backing the knowledge store.

## Telemetry & Metrics
- The orchestrator now streams per-step telemetry to `examples/ris_v2i_noma_case/results.jsonl` via `acpt.utils.metrics.persist_step`.
- Default KPIs include `energy`, `throughput`, `fairness`, `latency`, and `handoff_success`. Extend the registry with `register_metric("custom", func)`.
- Review stored metrics and visualize trends using `examples/ris_v2i_noma_case/visualize_results.ipynb`.

## Switching Environments
- Select the runtime simulator by editing `config/environment.yaml` (module, class, and keyword configuration).
- The default `acpt.environments.multi_domain_environment.MultiDomainEnvironment` fans in RIS, NOMA, and V2I delegates. Adjust the `delegates` list to tune per-simulator kwargs or to swap in additional modalities.
- Individual high-fidelity delegates remain available:
  - `acpt.environments.ris_environment.RISEnvironment` (LoS + Rician RIS model).
  - `acpt.environments.noma_environment.NOMAEnvironment` (SIC-aware downlink scheduling).
  - `acpt.environments.v2i_environment.V2IEnvironment` (lane-based mobility with Rician fading).
- The orchestrator auto-detects `EnvironmentInterface` implementations and unwraps `Transition` objects, so no additional glue code is required when swapping between toy and high-fidelity simulators.
- See `docs/reference/environments.md` for parameter cheatsheets and ASCII topologies.

## Programmatic API
The CLI utilities are thin wrappers over `acpt.utils.registrar`, which exposes:
- `validate_manifest()` — schema validation helper.
- `register_agent()` / `register_tool()` / `register_knowledge()` — update helpers for `config/wiring.yaml`.
- `append_kb_entry()` — persistence helper for JSON-backed knowledge bases.
- `scaffold_agent_file()` / `scaffold_tool_file()` — single-file scaffold generators.

Import these functions from `acpt.utils` to integrate registration flows into other tooling.

## Verifying Changes
1. Inspect `config/wiring.yaml` for the newly added entries.
2. Run the orchestrator demo to ensure the runtime loads the updated wiring:
   ```bash
   python -m acpt.examples.ris_v2i_noma_case.run_demo
   ```
3. Commit both the wiring updates and any scaffolded files to keep the repository in sync.

Refer to the diagrams in `docs/diagrams/`, the architecture overview in `docs/architecture.md`, and simulator details in `docs/reference/environments.md` for topology/mobility/fading overlays.

Refer to `docs/diagrams/plug_and_optimize.svg` for a visual overview of the end-to-end plug-and-optimize pipeline.
