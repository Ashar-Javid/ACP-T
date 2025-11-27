"""Serialization helpers shared by coordinators, agents, and tools."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

from acpt.utils.logging_utils import get_logger

__all__ = [
    "SerializationError",
    "register_codec",
    "to_json",
    "from_json",
    "to_msgpack",
    "from_msgpack",
    "generate_checksum",
]


try:  # optional dependency
    import msgpack  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional path
    msgpack = None  # type: ignore


_LOGGER = get_logger("Serialization")


Encoder = Callable[[Any], Any]
Decoder = Callable[[Any], Any]


class SerializationError(RuntimeError):
    """Raised when serialization or deserialization fails."""


@dataclass(frozen=True)
class Codec:
    """Describe how to encode and decode a custom type."""

    tag: str
    types: Tuple[type, ...]
    encoder: Encoder
    decoder: Decoder


_CODEC_REGISTRY: Dict[str, Codec] = {}
_TYPE_TO_CODEC: List[Codec] = []
_TYPE_MARKER = "__acpt_type__"


def register_codec(tag: str, types: Iterable[type], encoder: Encoder, decoder: Decoder) -> None:
    """Register a codec for custom types and msgpack/JSON support."""

    tag = tag.strip()
    if not tag:
        raise ValueError("Codec tag must be non-empty")

    normalized_types = tuple(types)
    if not normalized_types:
        raise ValueError("Codec must target at least one type")

    existing = _CODEC_REGISTRY.get(tag)
    if existing is not None:
        _TYPE_TO_CODEC[:] = [codec for codec in _TYPE_TO_CODEC if codec.tag != tag]

    codec = Codec(tag=tag, types=normalized_types, encoder=encoder, decoder=decoder)
    _CODEC_REGISTRY[tag] = codec
    _TYPE_TO_CODEC.append(codec)
    _LOGGER.debug("Registered codec '%s' for types %s", tag, [t.__name__ for t in normalized_types])


def to_json(obj: Any, *, pretty: bool = False) -> str:
    """Return a JSON string representing *obj* with validation."""

    payload = _encode(obj)
    indent = 2 if pretty else None
    separators = (",", ": ") if pretty else (",", ":")
    text = json.dumps(payload, indent=indent, sort_keys=True, separators=separators)
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SerializationError("JSON validation failed") from exc
    return text


def from_json(data: str | bytes | bytearray) -> Any:
    """Parse JSON data produced by :func:`to_json` and decode it."""

    if isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8")
    else:
        text = data
    payload = json.loads(text)
    return _decode(payload)


def to_msgpack(obj: Any, *, use_bin_type: bool = True) -> bytes:
    """Serialize *obj* to msgpack using the registered codecs."""

    if msgpack is None:  # pragma: no cover - optional path
        raise SerializationError("msgpack is not installed. Install 'msgpack' to enable this feature.")
    payload = _encode(obj)
    try:
        return msgpack.dumps(payload, use_bin_type=use_bin_type)
    except (TypeError, ValueError) as exc:  # pragma: no cover - msgpack edge
        raise SerializationError("msgpack serialization failed") from exc


def from_msgpack(blob: bytes) -> Any:
    """Deserialize msgpack payloads produced by :func:`to_msgpack`."""

    if msgpack is None:  # pragma: no cover - optional path
        raise SerializationError("msgpack is not installed. Install 'msgpack' to enable this feature.")
    try:
        payload = msgpack.loads(blob, raw=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - msgpack edge
        raise SerializationError("msgpack deserialization failed") from exc
    return _decode(payload)


def generate_checksum(obj: Any, *, algorithm: str = "sha256") -> str:
    """Return a hexadecimal checksum suitable for replay buffers and logs."""

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise SerializationError(f"Unsupported hash algorithm '{algorithm}'") from exc

    if isinstance(obj, bytes):
        payload = obj
    elif isinstance(obj, str):
        payload = obj.encode("utf-8")
    else:
        canonical = json.dumps(_encode(obj), sort_keys=True, separators=(",", ":"))
        payload = canonical.encode("utf-8")
    hasher.update(payload)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers


def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise SerializationError("Non-finite floats are not supported in JSON serialization")
        return obj
    if isinstance(obj, Mapping):
        return {str(key): _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_encode(item) for item in obj), key=lambda item: str(item))

    for codec in _TYPE_TO_CODEC:
        if any(isinstance(obj, typ) for typ in codec.types):
            encoded = codec.encoder(obj)
            return {_TYPE_MARKER: codec.tag, "value": _encode(encoded)}

    if hasattr(obj, "__dict__"):
        return {
            "__class__": obj.__class__.__name__,
            "__module__": obj.__class__.__module__,
            "state": _encode(obj.__dict__),
        }

    return str(obj)


def _decode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_decode(item) for item in obj]
    if isinstance(obj, Mapping):
        marker = obj.get(_TYPE_MARKER)
        if marker:
            codec = _CODEC_REGISTRY.get(marker)
            if codec is None:
                raise SerializationError(f"Unknown codec marker '{marker}'")
            value = _decode(obj["value"])
            return codec.decoder(value)

        if "__class__" in obj and "state" in obj:
            return {
                "__class__": obj["__class__"],
                "__module__": obj.get("__module__"),
                "state": _decode(obj["state"]),
            }

        return {key: _decode(value) for key, value in obj.items()}
    raise SerializationError(f"Cannot decode type {type(obj)!r}")


# ---------------------------------------------------------------------------
# Default codecs


def _register_default_codecs() -> None:
    register_codec("complex", (complex,), _encode_complex, _decode_complex)

    if importlib.util.find_spec("numpy") is not None:
        import numpy as np  # type: ignore

        register_codec(
            "numpy.ndarray",
            (np.ndarray,),
            lambda value: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            },
            lambda payload: np.array(payload["data"], dtype=payload["dtype"]).reshape(payload["shape"]),
        )

    if importlib.util.find_spec("torch") is not None:
        import torch  # type: ignore

        def _encode_tensor(tensor: "torch.Tensor") -> Dict[str, Any]:
            return {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.size()),
                "requires_grad": bool(tensor.requires_grad),
                "data": tensor.detach().cpu().tolist(),
            }

        def _decode_tensor(payload: Mapping[str, Any]) -> "torch.Tensor":
            dtype_name = str(payload["dtype"]).split(".")[-1]
            dtype = getattr(torch, dtype_name)
            tensor = torch.tensor(payload["data"], dtype=dtype)
            if payload.get("requires_grad"):
                tensor.requires_grad_(True)
            return tensor.reshape(payload["shape"])

        register_codec("torch.tensor", (torch.Tensor,), _encode_tensor, _decode_tensor)


def _encode_complex(value: complex) -> Dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _decode_complex(payload: Mapping[str, Any]) -> complex:
    return complex(payload["real"], payload["imag"])


_register_default_codecs()
