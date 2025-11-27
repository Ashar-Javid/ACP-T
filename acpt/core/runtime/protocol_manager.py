"""Protocol manager implementing JSON-RPC validation and routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from jsonschema import Draft7Validator, ValidationError as JSONSchemaError

from .registry import Registry, RegistryError


class ProtocolError(RuntimeError):
    """Base error for protocol manager failures."""


class ProtocolValidationError(ProtocolError):
    """Raised when incoming messages fail schema validation."""


class RoutingError(ProtocolError):
    """Raised when routing to registered handlers fails."""


class ProtocolManager:
    """JSON-RPC aware protocol manager providing validation and routing services."""

    def __init__(self, registry: Registry, schema_path: Optional[str] = None) -> None:
        self._registry = registry
        schema_file = (
            Path(schema_path)
            if schema_path is not None
            else Path(__file__).resolve().parents[2] / "specs" / "message_schema.json"
        )
        if not schema_file.exists():
            raise FileNotFoundError(f"Message schema not found: {schema_file}")

        with schema_file.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)

        self._validator = Draft7Validator(schema)

    def send_rpc(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate and dispatch a JSON-RPC message to the appropriate handler."""

        envelope = dict(message)
        envelope.setdefault("cid", str(uuid4()))

        try:
            self._validator.validate(envelope)
        except JSONSchemaError as exc:
            raise ProtocolValidationError(str(exc)) from exc

        try:
            result = self._dispatch(envelope)
            return self._response(envelope, result=result)
        except ProtocolError as exc:
            return self._response(envelope, error={"code": "protocol_error", "message": str(exc)})
        except RegistryError as exc:
            return self._response(envelope, error={"code": "registry_error", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._response(envelope, error={"code": "runtime_error", "message": str(exc)})

    def _dispatch(self, envelope: Mapping[str, Any]) -> Any:
        method = envelope["method"]
        try:
            target_spec, action = method.split(".", 1)
            component_type, component_id = target_spec.split(":", 1)
        except ValueError as exc:
            raise RoutingError(f"Malformed method name: {method}") from exc

        manifest = self._registry.resolve(component_id)
        manifest_type = manifest.get("type") or manifest.get("kind")
        if manifest_type != component_type:
            raise RoutingError(
                f"Component '{component_id}' registered as '{manifest_type}', not '{component_type}'."
            )

        handler = self._registry.handler(component_id)
        if handler is None:
            raise RoutingError(f"No handler bound for component '{component_id}'.")

        if not hasattr(handler, action):
            raise RoutingError(f"Handler for '{component_id}' missing method '{action}'.")

        params = envelope.get("params", {})
        call = getattr(handler, action)
        if isinstance(params, Mapping):
            return call(**params)
        return call(params)

    @staticmethod
    def _response(envelope: Mapping[str, Any], *, result: Any = None, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "cid": envelope.get("cid"),
        }
        if "id" in envelope:
            response["id"] = envelope["id"]
        if error is not None:
            response["error"] = error
        else:
            response["result"] = result
        return response
