"""Component registry with manifest validation for ACP runtime elements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from jsonschema import Draft7Validator, ValidationError as JSONSchemaError


class RegistryError(RuntimeError):
    """Raised when registry operations fail."""


class ManifestValidationError(RegistryError):
    """Raised when a component manifest violates the schema."""


class Registry:
    """Central registry responsible for component discovery and handler resolution."""

    def __init__(self, manifest_schema_path: Optional[str] = None) -> None:
        schema_path = (
            Path(manifest_schema_path)
            if manifest_schema_path is not None
            else Path(__file__).resolve().parents[2] / "specs" / "agent_manifest_schema.json"
        )
        if not schema_path.exists():
            raise FileNotFoundError(f"Manifest schema not found: {schema_path}")

        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)

        self._validator = Draft7Validator(schema)
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Any] = {}

    def register(self, manifest: Mapping[str, Any], handler: Optional[Any] = None) -> None:
        """Register a component manifest and optional handler instance."""

        if "handler" in manifest:
            if handler is not None:
                raise RegistryError("Handler provided both in manifest and argument.")
            handler = manifest["handler"]

        manifest_data = dict(manifest)
        manifest_data.pop("handler", None)
        manifest_data.pop("factory", None)

        try:
            self._validator.validate(manifest_data)
        except JSONSchemaError as exc:  # pragma: no cover - jsonschema ensures message
            raise ManifestValidationError(str(exc)) from exc

        component_id = manifest_data["id"]

        stored_manifest = dict(manifest)
        stored_manifest.pop("handler", None)
        factory = stored_manifest.get("factory")
        if factory is not None and not callable(factory):
            raise RegistryError(f"Factory for '{component_id}' must be callable.")

        self._manifests[component_id] = stored_manifest
        if handler is not None:
            self._handlers[component_id] = handler

    def resolve(self, component_id: str) -> Mapping[str, Any]:
        """Return the manifest associated with *component_id*."""

        try:
            return self._manifests[component_id]
        except KeyError as exc:
            raise RegistryError(f"Component '{component_id}' is not registered.") from exc

    def handler(self, component_id: str) -> Any:
        """Return the concrete handler for *component_id* creating it via factory if needed."""

        if component_id in self._handlers:
            return self._handlers[component_id]

        manifest = self.resolve(component_id)
        factory = manifest.get("factory")
        if callable(factory):
            handler = factory()
            self._handlers[component_id] = handler
            return handler

        raise RegistryError(f"No handler or factory available for '{component_id}'.")

    def manifests(self) -> Iterable[Mapping[str, Any]]:
        """Return an iterable over stored manifests."""

        return tuple(self._manifests.values())

    def clear(self) -> None:
        """Clear all registered components (primarily for testing)."""

        self._manifests.clear()
        self._handlers.clear()
