# Environment Reference

This catalog summarises the simulator façades bundled with ACP-T. Each environment adheres to `acpt.core.interfaces.environment_interface.EnvironmentInterface` and returns `Transition` tuples containing observation, reward, terminal, and info payloads.

## MultiDomainEnvironment
- **Module**: `acpt.environments.multi_domain_environment.MultiDomainEnvironment`
- **Purpose**: Fan-in wrapper that executes multiple delegate simulators (RIS, NOMA, V2I by default) while presenting a single environment contract to the orchestrator.
- **Key Inputs**:
  - `delegates`: sequence of delegate specs (`module`, `class`, optional `agents`, `kwargs`, `seed`).
  - `fading_models`: per-delegate overrides accepting built-in `rician`, `rayleigh`, `nakagami` types or fully qualified modules.
  - `mobility_models`: optional mobility overrides (e.g., `random_walk`).
- **Observations**: Union of delegate observations keyed by agent id (`agent.ris`, `agent.noma`, `agent.v2i`).
- **Rewards**: Aggregated per-agent rewards reported by the delegates.
- **Notes**: Stops when any delegate reports `done`. Provide delegate-specific `max_steps` to equalise horizon lengths.

## RISEnvironment
- **Module**: `acpt.environments.ris_environment.RISEnvironment`
- **Scenario**: Reconfigurable intelligent surface with LoS and Rician fading.
- **Parameters**: `tile_count`, `carrier_freq_hz`, `tx_power_dbm`, `noise_floor_dbm`, `corridor_length`, `corridor_width`, `max_steps`.
- **Observations**: Phase profile, per-user SNR metrics, geometry metadata for RIS and gNodeB.
- **Rewards**: Average user SNR in dB (`agent.ris`).
- **ASCII topology**:
  ```
  [gNB]---d_bs_ris---[RIS Panel]~~~> UE Corridor
          \_____________________/
                 Direct LoS
  ```

## BackscatterUplinkEnvironment
- **Module**: `acpt.environments.backscatter_environment.BackscatterUplinkEnvironment`
- **Scenario**: Uplink where active NOMA users coexist with passive backscatter tags.
- **Parameters**: `user_count`, `tag_count`, `carrier_freq_hz`, `tx_power_dbm`, `noise_floor_dbm`, `max_steps`.
- **Observations**: Separate blocks for `agent.noma` (uplink allocations, SNR, throughput) and `agent.backscatter` (reflection coefficients, aggregate signal).
- **Rewards**: Jain-scaled throughput for NOMA users; log-amplitude signal capture for backscatter tags.
- **Notes**: Accepts fading overrides via `fading_models` in the delegate spec.
- **ASCII topology**:
  ```
  UE uplink ---> [BS]
           \     /
            \---[Passive Tag]
  ```

## NOMAEnvironment
- **Module**: `acpt.environments.noma_environment.NOMAEnvironment`
- **Scenario**: Downlink NOMA cell with SIC ordering and hybrid fading.
- **Parameters**: `tx_power_dbm`, `bandwidth_hz`, `noise_density_dbm_hz`, `max_steps`.
- **Observations**: Channel table with pathloss, post-fading SNR, SIC order, capacity estimates.
- **Rewards**: Jain's fairness scaled throughput (`agent.noma`).
- **ASCII topology**:
  ```
       [gNB]
       /  \
   UE_far UE_near
  ```

## V2IEnvironment
- **Module**: `acpt.environments.v2i_environment.V2IEnvironment`
- **Scenario**: Vehicle-to-infrastructure corridor with mobility and Rician fading.
- **Parameters**: `rsu_position`, `lane_count`, `lane_spacing`, `noise_floor_dbm`, `tx_power_dbm`, `max_steps`.
- **Observations**: Per-vehicle position, velocity, pathloss, SNR, throughput estimates.
- **Rewards**: Coverage-weighted average SINR (`agent.v2i`).
- **ASCII topology**:
  ```
  Lane1 ==> ==> [RSU] <== <== Lane4
  ```

## ToyNREnvironment
- **Module**: `acpt.environments.toy_nr_env.ToyNREnvironment`
- **Scenario**: Deterministic large-scale fading playground for rapid smoke tests.
- **Parameters**: `time_step`, `channel_models`, `rats`, per-agent `initial_pos` / `initial_power`.
- **Observations**: Per-agent SNR, 2D position, energy cost per step.
- **Rewards**: None (coordinator relies on agent-level metrics).
- **Usage**: Handy for validating orchestration wiring without heavy numerics.
