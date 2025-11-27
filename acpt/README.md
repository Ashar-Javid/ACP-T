# ACP-T Framework for 6G Control Planes

ACP-T is a modular research framework for Adaptive Control Plane (ACP) experimentation in 6G systems. It coordinates LLM-powered agents, external tool adapters, and heterogeneous simulators through a programmable coordinator and telemetry surface.

## Key Capabilities
- Orchestrates multi-agent reasoning workflows with pluggable controller, LLM, and reward agents.
- Supports Cerebras, OpenAI, Gemini, and OpenRouter LLM providers with automatic fallback to the deterministic offline stub when API keys are absent.
- Integrates third-party tooling through adapters for MATLAB Engine, ns-3 simulations, PyTorch inference, and HTTP API gateways.
- Validates and wires manifests for agents, tools, and knowledge bases using JSON Schema-backed registrars.
- Ships with standardized tool envelopes (JSON Schema + `{result, diagnostics}`) and a composite multi-domain environment that blends toy NR, RIS, NOMA, V2I, and backscatter simulators with configurable fading models (Rician, Rayleigh, Nakagami).

## Repository Layout
- `config/` – runtime defaults, agent manifests, and environment wiring templates.
- `core/interfaces/` – abstract base classes that define agent, environment, and tool contracts.
- `core/runtime/` – registry, protocol manager, and orchestrator-facing context primitives.
- `agents/` – concrete and scaffolded agent implementations consumed by the coordinator.
- `adapters/` – integration layers for MATLAB, ns-3, PyTorch, and API gateway access.
- `environments/` – reusable simulator façades exercised by the control plane.
- `utils/` – helpers for configuration loading, logging, metrics, and registrar tooling.
- `tools/` – callable utilities that agents can invoke through the orchestrator.
- `scripts/` – CLI entry points for registering agents/tools and enriching knowledge bases.
- `examples/` – end-to-end scenario wiring and sample manifests.
- `tests/` – unit and integration suites covering registrars, adapters, and CLI flows.

## Adapter Layer
- **MATLAB Adapter** (`adapters/matlab_adapter.py`) starts or reuses MATLAB Engine sessions and proxies function calls.
- **ns-3 Adapter** (`adapters/ns3_adapter.py`) builds and runs ns-3 simulations with timeout-aware subprocess control.
- **PyTorch Adapter** (`adapters/pytorch_adapter.py`) loads `torch.nn.Module` or TorchScript artifacts, returning NumPy-compatible outputs.
- **API Gateway Adapter** (`adapters/api_gateway_adapter.py`) issues authenticated REST calls to external coordinators via a persistent session.

Optional dependencies:
- MATLAB Engine for Python must be installed and licensed for MATLAB interoperability.
- ns-3 binaries must reside on the host and be accessible to the runtime.
- PyTorch (`torch`) is required for deep-learning inference; NumPy fallbacks are available for local testing.
- `requests` powers the HTTP gateway adapter.

## Getting Started
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .[dev]  # optional: formatter + pytest
pytest
```

Adapter extras:
- `pip install matlabengine` (ships with MATLAB) for the MATLAB adapter.
- `pip install torch` for the PyTorch adapter.
- Ensure ns-3 binaries are on `PATH` for the ns-3 adapter.
- `pip install requests` for the API gateway adapter.

## LLM Provider Configuration
ACP-T inspects environment variables at runtime to pick the appropriate hosted LLM. If a key is missing, the framework keeps operating in offline stub mode so tests and demos remain deterministic.

| Provider  | Environment variable     | Default model            |
|-----------|-------------------------|--------------------------|
| Cerebras  | `CEREBRAS_API_KEY`      | `qwen-32b`               |
| OpenAI    | `OPENAI_API_KEY`        | `gpt-4o-mini`            |
| Gemini    | `GEMINI_API_KEY`        | `gemini-1.5-flash`       |
| OpenRouter| `OPENROUTER_API_KEY`    | `openrouter/openai/gpt-4o-mini` |

Set the relevant variable before launching the orchestrator or running CLI utilities, for example:

```powershell
setx CEREBRAS_API_KEY "your_cerebras_token"
```

You can override `provider`, `model`, or inference parameters in agent manifests or wiring files; the factory automatically routes requests to the right endpoint and merges context into prompt payloads.

## CLI Toolkit
- `python scripts/add_agent.py --manifest path/to/manifest.json` validates and registers an agent, scaffolding source files when requested.
- `python scripts/add_tool.py --manifest path/to/manifest.json` wires tool descriptors and optional Python stubs.
- `python scripts/add_kb_entry.py --id knowledge-id --document file.md` appends documents to JSON knowledge stores.


Scaffolds produced by the registrar CLI now mirror the concrete agent/tool implementations checked into the repository—use them as a baseline when introducing new specialisations.

## Configuration Artifacts
- Manifest schemas live under `specs/` and are consumed by the registrar validators.
- `config/wiring.yaml` declares the active agents, coordinator, and tools loaded by the orchestrator.
- `config/environment.yaml` selects the runtime environment (module, class, and keyword configuration); the default references the composite multi-domain simulator. Alternate presets live under `config/blueprints/`.
- Example-specific wiring lives under `examples/`—see `examples/ris_v2i_noma_case/` for a multi-RAT preset.

## Testing & Validation
- Run `pytest` for the unit test suite. Add integration tests in `tests/integration/` when wiring new simulators.
- Adapters emit structured logs via `utils/logging_utils.py`—enable DEBUG level to diagnose orchestration issues.

## Framework Overview

- **Agents**: Implement `observe → propose → feedback` and can attach optional RAG knowledge bases (`docs/reference/kb.md`).
- **Tools**: Expose an `invoke` method returning the standardized `{result, diagnostics}` envelope.
- **Environments**: Any implementation of `EnvironmentInterface` (or `BaseEnvironment`) can be declared in YAML. The multi-domain wrapper supports delegate-specific fading/mobility overrides.
- **Telemetry**: Per-step history persists via `acpt.utils.metrics.persist_step`, enabling notebook-based analysis and regression playback.

See `docs/reference/configuration.md` for schema details and `config/blueprints/` for ready-to-use setups (2-user RIS, tri-pair NOMA, uplink NOMA+backscatter).

## Roadmap
- Add regression notebooks and scripted demos for replaying RIS/V2I/NOMA/backscatter telemetry.
- Extend tool-envelope integration tests to cover MATLAB, ns-3, and PyTorch adapters.
- Benchmark composite environment configurations for performance envelopes across delegate counts.

## Citation

If you use ACP-T in academic work, please cite:

```
@software{acpt_framework_2025,
	title        = {ACP-T: Adaptive Control Plane Toolkit for 6G Research},
	author       = {ACP-T Contributors},
	year         = {2025},
	url          = {https://github.com/your-org/acpt-6g},
	note         = {Version 0.2.0-pre, accessed 2025-11-27}
}
```

Architecture diagrams reside in `docs/diagrams/`; additional usage notes and walkthroughs live in `docs/USAGE.md`.
