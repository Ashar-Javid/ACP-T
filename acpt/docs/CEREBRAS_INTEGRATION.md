# Cerebras Qwen-32B Integration Guide

This document captures the operational checklist for moving from the mock LLM to a production deployment on Cerebras hardware.

## Adapter Overview
- The runtime consumes the adapter via `acpt.utils.CerebrasLLMAdapter`.
- Configuration is expressed through `CerebrasConfig`: endpoint, batching, retry policy, latency budget, and `mock_mode` fallback.
- By default the adapter runs in deterministic mock mode. Set `mock_mode=False` (or pass `use_mock=False`) once Cerebras credentials are available.

## Authentication & Secrets
1. Request an API key or service credential from the Cerebras operations team.
2. Store credentials in a secure secret manager (Azure Key Vault, AWS Secrets Manager) or encrypted environment variables.
3. Inject the credential into the orchestrator process via `CEREBRAS_API_KEY` or a mounted secrets file. Never commit credentials to source control.
4. Pass the key to the adapter using `CerebrasConfig(api_key=...)`. Rotate credentials regularly per your security policy.

## Deployment & Connectivity
- Deploy the runtime on a node with low-latency connectivity to the Cerebras cluster (ideally on the same rack or over an RDMA-capable fabric).
- Validate that outbound connections to the Cerebras inference endpoint are permitted through firewalls and security groups.
- Use TLS for all API traffic. Verify certificates as part of the connection handshake.

## Batching Strategy
- Start with `batch_size=1` for deterministic debugging.
- Increase batch size once prompt latency measurements are stable; aim to keep inference within the configured `latency_budget_ms`.
- Leverage prompt templating to batch similar agent requests. The adapter exposes `allow_streaming` for partial responses when batching is not feasible.

## Latency & Cost Monitoring
- Capture per-request latency using adapter logs (`Diagnostics` returned by `CerebrasLLMAdapter.diagnostics()`).
- Emit latency and token usage metrics to your observability stack (Prometheus, Datadog, Azure Monitor).
- Define alert thresholds for latency budget breaches and retry spikes (indicative of throttling or outages).
- Periodically review Cerebras usage reports to track inference cost versus scenario coverage.

## Retries & Fallbacks
- The adapter retries transient failures up to `max_retries` with exponential backoff. Adjust based on SLA.
- If all retries fail, the orchestrator should catch the exception and either fall back to mock mode or skip the timestep.
- Toggle mock mode explicitly by invoking `adapter.configure(mock_mode=True)` if the hardware becomes unavailable during a run.

## Local Testing Flow
1. Run unit tests (`pytest acpt/tests/integration/test_llm_adapter.py`) to validate the mock pathway.
2. Execute end-to-end demos (`python examples/ris_v2i_noma_case/run_demo.py`) with default mock configuration to produce deterministic telemetry.
3. Switch to real hardware by setting `mock_mode=False` and providing real endpoint + credentials.

## Production Rollout Checklist
- [ ] Secrets stored in an approved secret manager.
- [ ] Network connectivity verified from orchestrator host to Cerebras endpoint.
- [ ] Latency/cost dashboards configured.
- [ ] Mock-mode fallback procedure documented for operations.
- [ ] Load test executed to confirm batching, retry, and monitoring strategies under volume.

Refer to `acpt/utils/llm_cerebras.py` for adapter implementation details and extend `_invoke_cerebras` with the official Cerebras SDK or REST client when integrating with live hardware.
