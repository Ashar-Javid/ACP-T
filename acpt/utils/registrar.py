"""Helper utilities for manifest validation and wiring updates."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from jsonschema import Draft7Validator, ValidationError as JSONSchemaError

from .logging_utils import get_logger


class RegistrarError(RuntimeError):
    """Raised when registrar operations fail."""


class ManifestValidationError(RegistrarError):
    """Raised when a manifest violates the published schema."""


_LOGGER = get_logger("Registrar")


def load_manifest(path: str) -> Dict[str, Any]:
    """Load a JSON manifest from *path* and return it as a dictionary."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, Mapping):
        raise ManifestValidationError("Manifest document must be a mapping.")
    return dict(data)


def validate_manifest(manifest: Mapping[str, Any], schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate *manifest* against the agent manifest schema."""

    manifest_dict = dict(manifest)
    validator = _load_validator(schema_path)
    try:
        validator.validate(manifest_dict)
    except JSONSchemaError as exc:
        raise ManifestValidationError(str(exc)) from exc
    return manifest_dict


@lru_cache(maxsize=4)
def _load_validator(schema_path: Optional[str]) -> Draft7Validator:
    schema_file = (
        Path(schema_path)
        if schema_path is not None
        else Path(__file__).resolve().parents[2] / "specs" / "agent_manifest_schema.json"
    )
    if not schema_file.exists():
        raise FileNotFoundError(f"Manifest schema not found: {schema_file}")

    with schema_file.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    return Draft7Validator(schema)


def load_wiring(path: Path) -> Dict[str, Any]:
    """Load the wiring YAML file, returning a canonical mapping."""

    if not path.exists():
        return {"agents": {}, "tools": {}, "knowledge": {}}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise RegistrarError("Wiring document must be a mapping at the top level.")

    return {
        "agents": dict(data.get("agents", {})),
        "tools": dict(data.get("tools", {})),
        "knowledge": dict(data.get("knowledge", {})),
    }


def save_wiring(path: Path, wiring: Mapping[str, Any]) -> None:
    """Persist *wiring* mapping to *path* as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "agents": wiring.get("agents", {}),
                "tools": wiring.get("tools", {}),
                "knowledge": wiring.get("knowledge", {}),
            },
            handle,
            sort_keys=False,
        )


def register_agent(
    manifest: Mapping[str, Any],
    wiring_path: Path,
    *,
    module: Optional[str] = None,
    class_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Register an agent manifest in the wiring map."""

    validated = validate_manifest(manifest)
    if validated.get("type") != "agent":
        raise ManifestValidationError("Manifest type must be 'agent'.")

    agent_id = validated["id"]
    entry: Dict[str, Any] = {"manifest": validated}
    if module:
        entry["module"] = module
    if class_name:
        entry["class"] = class_name

    wiring = load_wiring(wiring_path)
    if agent_id in wiring["agents"] and not overwrite:
        raise RegistrarError(f"Agent '{agent_id}' already exists. Use overwrite to replace.")

    wiring["agents"][agent_id] = entry
    save_wiring(wiring_path, wiring)
    _LOGGER.info("Registered agent '%s' in wiring map.", agent_id)
    return entry


def register_tool(
    manifest: Mapping[str, Any],
    wiring_path: Path,
    *,
    module: Optional[str] = None,
    class_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Register a tool manifest in the wiring map."""

    validated = validate_manifest(manifest)
    if validated.get("type") != "tool":
        raise ManifestValidationError("Manifest type must be 'tool'.")

    tool_id = validated["id"]
    entry: Dict[str, Any] = {"manifest": validated}
    if module:
        entry["module"] = module
    if class_name:
        entry["class"] = class_name

    wiring = load_wiring(wiring_path)
    if tool_id in wiring["tools"] and not overwrite:
        raise RegistrarError(f"Tool '{tool_id}' already exists. Use overwrite to replace.")

    wiring["tools"][tool_id] = entry
    save_wiring(wiring_path, wiring)
    _LOGGER.info("Registered tool '%s' in wiring map.", tool_id)
    return entry


def register_knowledge(
    knowledge_id: str,
    descriptor: Mapping[str, Any],
    wiring_path: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Register a knowledge base descriptor in the wiring map."""

    knowledge_id = str(knowledge_id)
    descriptor_dict = dict(descriptor)
    if "path" not in descriptor_dict and "prefix" not in descriptor_dict:
        raise RegistrarError("Knowledge descriptor must include 'path' or 'prefix'.")

    path_hint = descriptor_dict.get("path") or descriptor_dict.get("prefix")
    if path_hint is None:
        raise RegistrarError("Knowledge descriptor must include a storage hint.")

    if "prefix" not in descriptor_dict:
        descriptor_dict["prefix"] = path_hint
    if "index_type" not in descriptor_dict:
        descriptor_dict["index_type"] = "file"

    wiring = load_wiring(wiring_path)
    if knowledge_id in wiring["knowledge"] and not overwrite:
        raise RegistrarError(
            f"Knowledge descriptor '{knowledge_id}' already exists. Use overwrite to replace."
        )

    wiring["knowledge"][knowledge_id] = descriptor_dict
    save_wiring(wiring_path, wiring)
    _LOGGER.info("Registered knowledge descriptor '%s'.", knowledge_id)
    return descriptor_dict


def append_kb_entry(kb_path: Path, document: str) -> Path:
    """Append *document* to the JSON knowledge base located at *kb_path*."""

    kb_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[str]
    if kb_path.exists():
        with kb_path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle) or []
            except json.JSONDecodeError as exc:
                raise RegistrarError(f"Knowledge base file is corrupted: {kb_path}") from exc
        if not isinstance(data, list):
            raise RegistrarError("Knowledge base file must contain a JSON array.")
        records = [str(item) for item in data]
    else:
        records = []

    records.append(str(document))
    with kb_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    _LOGGER.info("Appended knowledge entry to %s", kb_path)
    return kb_path


def scaffold_agent_file(base_dir: Path, agent_id: str, class_name: Optional[str] = None) -> Path:
    """Create a scaffold agent file under *base_dir* for *agent_id*."""

    class_name = class_name or derive_agent_class_name(agent_id)
    module_name = derive_module_basename(agent_id)
    target = base_dir / f"{module_name}.py"
    if target.exists():
        raise RegistrarError(f"Agent scaffold already exists at {target}")

    template = f'''"""Scaffolded agent implementation for '{agent_id}'."""

from __future__ import annotations

from typing import Any, Dict

from acpt.core.interfaces import AgentInterface


class {class_name}(AgentInterface):
    """Auto-generated scaffold for agent '{agent_id}'."""

    def id(self) -> str:
        return "{agent_id}"

    def llm_spec(self) -> Dict[str, Any]:
        return {{"model": "qwen-32b", "device": "cerebras", "infer_params": {{"temperature": 0.0}}}}

    def init_rag(self, kb_descriptor: Dict[str, Any]) -> None:
        raise NotImplementedError("Implement RAG initialization for your agent.")

    def capabilities(self) -> Dict[str, Any]:
        return {{"intent": "custom_intent", "rag": False, "tools": []}}

    def observe(self, obs: Dict[str, Any]) -> None:
        raise NotImplementedError("Store incoming observations here.")

    def propose(self) -> Dict[str, Any]:
        raise NotImplementedError("Return the agent proposal payload.")

    def commit(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Handle a committed decision from the coordinator.")

    def use_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Invoke registered tools as required.")

    def feedback(self, telemetry: Dict[str, Any]) -> None:
        raise NotImplementedError("Process environment feedback for learning.")
'''
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    _LOGGER.info("Created agent scaffold at %s", target)
    return target


def scaffold_tool_file(base_dir: Path, tool_id: str, class_name: Optional[str] = None) -> Path:
    """Create a scaffold tool file under *base_dir* for *tool_id*."""

    class_name = class_name or derive_tool_class_name(tool_id)
    module_name = derive_module_basename(tool_id)
    target = base_dir / f"{module_name}.py"
    if target.exists():
        raise RegistrarError(f"Tool scaffold already exists at {target}")

    template = f'''"""Scaffolded tool implementation for '{tool_id}'."""

from __future__ import annotations

from typing import Any, Dict

from acpt.core.interfaces import ToolInterface


class {class_name}(ToolInterface):
    """Auto-generated scaffold for tool '{tool_id}'."""

    def name(self) -> str:
        return "{tool_id}"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Execute your tool logic here.")

    def metadata(self) -> Dict[str, Any]:
        return {{"type": "custom", "description": "Update with runtime metadata."}}
'''
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    _LOGGER.info("Created tool scaffold at %s", target)
    return target


def derive_agent_class_name(agent_id: str) -> str:
    """Return a best-effort class name for *agent_id*."""

    return _derive_class_name(agent_id, suffix="Agent")


def derive_tool_class_name(tool_id: str) -> str:
    """Return a best-effort class name for *tool_id*."""

    return _derive_class_name(tool_id, suffix="Tool")


def derive_module_basename(identifier: str) -> str:
    """Return a lowercase module stem derived from *identifier*."""

    tokens = re.split(r"[^A-Za-z0-9]+", identifier)
    stem = "_".join(token.lower() for token in tokens if token)
    return stem or "component"


def _derive_class_name(identifier: str, *, suffix: str) -> str:
    tokens = re.split(r"[^A-Za-z0-9]+", identifier)
    stem = "".join(token.capitalize() for token in tokens if token)
    if not stem:
        stem = "Component"
    if not stem.endswith(suffix):
        stem = f"{stem}{suffix}"
    return stem


__all__ = [
    "RegistrarError",
    "ManifestValidationError",
    "append_kb_entry",
    "derive_agent_class_name",
    "derive_module_basename",
    "derive_tool_class_name",
    "load_manifest",
    "load_wiring",
    "register_agent",
    "register_knowledge",
    "register_tool",
    "save_wiring",
    "scaffold_agent_file",
    "scaffold_tool_file",
    "validate_manifest",
]
