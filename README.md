# ACP-T

Adaptive Control Platform for 6G research. The project links AI-driven agents with heterogeneous network simulators (NR, Terahertz, RAT) through a plug-in protocol manager. It enables rapid prototyping of control strategies, reward shaping, and telemetry feedback loops across multiple environments.

## Key Features
- Modular adapters for Matlab, ns-3, PyTorch, and custom API gateways.
- Decoupled agent stack (control, LLM, reward) with shared context handling.
- Configuration-driven deployment through `config/*.yaml` files.
- Metric, logging, and registry utilities for experiment tracking.
- Ready for integration with physical or simulated 6G environments.

## Repository Layout
```
adapters/         Interface layers to external tools (Matlab, ns-3, PyTorch, REST)
agents/           Control, LLM, and reward agents implementing decision logic
config/           YAML configs defining agents, environments, defaults
core/             Shared abstractions (protocol manager, registries, contexts)
environments/     Environment wrappers (NR, RAT simulator, Terahertz)
utils/            Config loader, logging helpers, metrics
scripts/, specs/  Automation helpers and design artefacts
```

## Prerequisites
- Python 3.10+ (matching the `.venv` settings)
- Recommended: `pip` 23+, `virtualenv` or `venv`
- Optional tools for adapters (Matlab, ns-3) if you plan to run those integrations

## Setup
```powershell
# Clone and enter the repo
# git clone https://github.com/Ashar-Javid/ACP-T.git
cd ACP-T\acpt6g

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
- Edit `config/default.yaml` to set global flags.
- Override agent or environment parameters via `config/agent_config.yaml` and `config/environment_config.yaml`.
- Any new adapter or agent must be registered in `core/registry.py` so it can be discovered at runtime.
- See `docs/environment_recipes.md` for ready-made commands and patterns to mix RIS/NOMA/V2I environments, change fading models, and scale user populations.

## Running Experiments
```powershell
# Example: launch the default pipeline (adjust module/entrypoint as needed)
python -m acpt
```

For custom scenarios:
1. Duplicate one of the environment configs under `config/`.
2. Point the protocol manager to the new config via CLI flags or env vars (e.g., `ACP_CONFIG=...`).
3. Launch the agent stack with the same module invocation.

## Testing & Quality
```powershell
pytest
```
Add new tests beside the modules they cover (e.g., `tests/core/test_protocol_manager.py`). Ensure adapters interacting with external simulators are mocked to keep the suite self-contained.

## Contributing
1. Create a feature branch.
2. Add or update documentation/tests relevant to your change.
3. Run `pytest` and linting before opening a PR.
4. Submit the PR for review following the repository guidelines.

## Citation
If this project supports your research, please cite the upcoming IEEE Network Magazine article using the entry below:

```bibtex
@article{javid2026acpt,
	title        = {ACP-T: An Agentic Communication Protocol for Dynamic Resource Coordination in Multi-Layer Telecom Networks},
	author       = {Javid, Muhammad Ashar and others},
	journal      = {IEEE Network Magazine},
	year         = {2026},
	note         = {IEEE Communications Society | In Preparation}
}
```

## Support
- File issues or feature requests via GitHub Issues.
- For adapter-specific problems, include simulator versions and configuration snippets for reproducibility.
