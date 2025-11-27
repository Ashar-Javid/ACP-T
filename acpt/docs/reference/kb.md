# Knowledge Base (KB) Guide

ACP-T ships with a lightweight retrieval-augmented generation (RAG) layer that agents can query for contextual hints, safety policies, and scenario-specific notes.

## Concepts

- **KB Descriptor** – JSON object declaring the backing store. For the default file-based implementation it includes `index_type: "file"`, plus a `path` and optional `prefix` for relative lookups.
- **KBManager** – Utility responsible for initializing, persisting, and retrieving documents. Agents obtain an instance via `init_rag(descriptor)`.
- **Documents** – Plain strings appended to the JSON array referenced by the descriptor. The registrar CLI (`scripts/add_kb_entry.py`) automates creation.

## Wiring KBs

To attach a KB to an agent or the coordinator, supply a `rag_descriptor` block inside its manifest in `config/wiring.yaml`:

```yaml
agent.ris:
  manifest:
    rag_descriptor:
      index_type: file
      path: data/kb/ris-01.json
      prefix: data/kb/ris-01.json
```

The orchestrator resolves relative paths against `config/wiring.yaml`, creating directories as required.

## Programmatic Usage

```python
from acpt.knowledge import KBManager

descriptor = {"index_type": "file", "path": "data/kb/agent.ris.json"}
kb = KBManager(descriptor)
kb.initialize(["RIS tuning heuristics", "Fallback policy: maximize fairness"])
notes = kb.retrieve("fairness", k=2)
```

## CLI Helpers

- `python scripts/add_kb_entry.py --id kb.ris --document "RIS tuning"` appends a document and updates `config/wiring.yaml` unless `--skip-wiring` is passed.
- Use multiple `--document` flags to batch-ingest snippets. For large markdown files, prefer `--file path/to/doc.md`.

## Best Practices

- Store scenario-specific telemetry insights (e.g., beam alignment hints) to inform LLM-driven agents.
- Track provenance in the document text (`[source: paper XYZ]`).
- Keep descriptors under version control; JSON arrays can be diffed alongside code.
- For non-file backends (vector DBs, Redis), implement a custom manager exposing the same `initialize`/`retrieve` interface and reference it via `module`/`class` in the descriptor.
