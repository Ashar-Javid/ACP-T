# Environment Blueprints

These YAML presets illustrate how to configure the multi-domain environment for common research scenarios.

- `ris_two_user.yaml` — Two-user RIS corridor with custom tile count and mixed Rician/Nakagami fading.
- `noma_three_pair.yaml` — Three NOMA user-pairs with heterogeneous fading profiles.
- `uplink_noma_backscatter.yaml` — Joint NOMA uplink and backscatter tags sharing the uplink medium.

To use a blueprint:

1. Copy the YAML file next to your experiment parameters (e.g., `examples/ris_v2i_noma_case/env_config.yaml`).
2. Update the `environment` path in `params.yaml` (or pass `--environment` to custom runners) to point at the copied blueprint.
3. Run `python -m acpt.examples.ris_v2i_noma_case.run_demo --config examples/ris_v2i_noma_case/params.yaml`.

Feel free to duplicate these templates and adjust delegate `kwargs`, `fading_models`, or `mobility_models` to encode additional user types (e.g., UAV, pedestrians) or radio characteristics.
