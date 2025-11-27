"""Thread-safe runtime context handler used by the ACP orchestrator."""

from __future__ import annotations

import time
from collections.abc import Mapping as MappingABC, MutableMapping, Sequence
from threading import RLock
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, cast

from acpt.utils.logging_utils import get_logger

_LOGGER = get_logger("ContextHandler")


class FrozenDict(MappingABC):
    """Read-only mapping wrapper that exposes JSON-serializable data."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - thin wrapper
        return self._data[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - thin wrapper
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable copy of the underlying mapping."""

        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - deterministic repr
        return f"FrozenDict({self._data!r})"


class ContextHandler:
    """Manage runtime context for the orchestrator and downstream agents.

    The handler provides hierarchical storage with atomic updates, TTL-aware
    eviction, and JSON-serializable snapshots. Orchestrator components write
    environment, agent, metric, and message data through the update helpers.
    LLM agents obtain structured views via ``snapshot`` or ``get_view``.
    """

    _ENV_SCOPE = "env_state"
    _AGENT_SCOPE = "agent_actions"
    _METRIC_SCOPE = "metrics"
    _MESSAGE_SCOPE = "messages"

    def __init__(self) -> None:
        self._lock = RLock()
        self._version = 0
        self._store: Dict[str, Any] = {
            self._ENV_SCOPE: {},
            self._AGENT_SCOPE: {},
            self._METRIC_SCOPE: {},
            self._MESSAGE_SCOPE: [],
        }
        self._ttl_index: Dict[str, float] = {}
        self._message_counter = 0

    # ------------------------------------------------------------------
    # Update helpers

    def update_env_state(
        self,
        path: Sequence[str] | str,
        value: Any,
        *,
        ttl: Optional[float] = None,
    ) -> int:
        """Update a nested environment state value.

        Args:
            path: Hierarchical path (``["cell", "rssi"]`` or ``"cell.rssi"``).
            value: JSON-compatible payload to store under the path.
            ttl: Optional time-to-live in seconds for transient entries.

        Returns:
            Monotonic version after the update is applied.
        """

        normalized_path = _normalize_path(path)
        with self._lock:
            self._prune_expired_locked()
            _set_nested(self._store[self._ENV_SCOPE], normalized_path, value)
            self._assign_ttl(self._ENV_SCOPE, normalized_path, ttl)
            return self._bump_version_locked()

    def record_agent_action(
        self,
        agent_id: str,
        action: Mapping[str, Any],
        *,
        ttl: Optional[float] = None,
    ) -> int:
        """Record the latest action emitted by an agent."""

        agent_key = agent_id.strip()
        if not agent_key:
            raise ValueError("agent_id must be non-empty")

        with self._lock:
            self._prune_expired_locked()
            self._store[self._AGENT_SCOPE][agent_key] = dict(action)
            self._assign_ttl(self._AGENT_SCOPE, (agent_key,), ttl)
            return self._bump_version_locked()

    def record_metric(
        self,
        path: Sequence[str] | str,
        value: Any,
        *,
        ttl: Optional[float] = None,
    ) -> int:
        """Record a scalar metric under a hierarchical path."""

        normalized_path = _normalize_path(path)
        with self._lock:
            self._prune_expired_locked()
            _set_nested(self._store[self._METRIC_SCOPE], normalized_path, value)
            self._assign_ttl(self._METRIC_SCOPE, normalized_path, ttl)
            return self._bump_version_locked()

    def append_message(
        self,
        channel: str,
        payload: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> int:
        """Append a message visible to agents and tools."""

        channel_name = channel.strip()
        if not channel_name:
            raise ValueError("channel must be non-empty")

        with self._lock:
            self._prune_expired_locked()
            self._message_counter += 1
            message_key = f"msg:{self._message_counter}"
            entry = {
                "channel": channel_name,
                "payload": payload,
                "metadata": dict(metadata or {}),
                "timestamp": time.time(),
            }
            self._store[self._MESSAGE_SCOPE].append((message_key, entry))
            self._assign_ttl(self._MESSAGE_SCOPE, message_key, ttl)
            return self._bump_version_locked()

    # ------------------------------------------------------------------
    # Accessors

    def snapshot(self) -> FrozenDict:
        """Return an immutable, JSON-serializable snapshot of the context."""

        with self._lock:
            self._prune_expired_locked()
            payload = {
                "version": self._version,
                "timestamp": time.time(),
                self._ENV_SCOPE: _to_json_safe(self._store[self._ENV_SCOPE]),
                self._AGENT_SCOPE: _to_json_safe(self._store[self._AGENT_SCOPE]),
                self._METRIC_SCOPE: _to_json_safe(self._store[self._METRIC_SCOPE]),
                self._MESSAGE_SCOPE: _to_json_safe(
                    [entry for _, entry in self._store[self._MESSAGE_SCOPE]]
                ),
            }
            return FrozenDict(payload)

    def get_view(
        self,
        scope: str,
        path: Sequence[str] | str | None = None,
        *,
        default: Any = None,
    ) -> Any:
        """Return a JSON-serializable view of a section of the context."""

        if scope not in self._store:
            raise KeyError(f"Unknown scope '{scope}'")

        with self._lock:
            self._prune_expired_locked()
            data = self._store[scope]
            if scope == self._MESSAGE_SCOPE and path is not None:
                raise ValueError("Messages do not support hierarchical lookups")
            if path is None:
                if scope == self._MESSAGE_SCOPE:
                    return _to_json_safe([entry for _, entry in data])
                return _to_json_safe(data)

            normalized_path = _normalize_path(path)
            try:
                value = _get_nested(data, normalized_path)
            except KeyError:
                return default
            return _to_json_safe(value)

    @property
    def version(self) -> int:
        """Return the current version counter."""

        with self._lock:
            return self._version

    # ------------------------------------------------------------------
    # Internal helpers

    def _bump_version_locked(self) -> int:
        self._version += 1
        return self._version

    def _assign_ttl(
        self,
        scope: str,
        identifier: Sequence[str] | str | Tuple[str, ...],
        ttl: Optional[float],
    ) -> None:
        key = _make_ttl_key(scope, identifier)
        if ttl is None:
            self._ttl_index.pop(key, None)
            return
        expiry = time.monotonic() + max(ttl, 0.0)
        self._ttl_index[key] = expiry

    def _prune_expired_locked(self) -> None:
        if not self._ttl_index:
            return

        now = time.monotonic()
        expired = [item for item, deadline in self._ttl_index.items() if deadline <= now]
        if not expired:
            return

        for item in expired:
            scope, locator = item.split("::", maxsplit=1)
            if scope == self._MESSAGE_SCOPE:
                before = len(self._store[self._MESSAGE_SCOPE])
                self._store[self._MESSAGE_SCOPE] = [
                    message for message in self._store[self._MESSAGE_SCOPE] if message[0] != locator
                ]
                if len(self._store[self._MESSAGE_SCOPE]) != before:
                    _LOGGER.debug("Pruned expired message %s", locator)
            elif scope == self._AGENT_SCOPE:
                removed = self._store[self._AGENT_SCOPE].pop(locator, None)
                if removed is not None:
                    _LOGGER.debug("Pruned agent action %s", locator)
            else:
                path = tuple(part for part in locator.split("/") if part)
                if scope == self._ENV_SCOPE:
                    if _delete_nested(self._store[self._ENV_SCOPE], path):
                        _LOGGER.debug("Pruned env path %s", locator)
                elif scope == self._METRIC_SCOPE:
                    if _delete_nested(self._store[self._METRIC_SCOPE], path):
                        _LOGGER.debug("Pruned metric path %s", locator)

            self._ttl_index.pop(item, None)


def _normalize_path(path: Sequence[str] | str) -> Tuple[str, ...]:
    if isinstance(path, str):
        parts = tuple(segment for segment in path.split(".") if segment)
    else:
        parts = tuple(str(segment) for segment in path if str(segment))
    if not parts:
        raise ValueError("path must contain at least one non-empty segment")
    return parts


def _set_nested(container: MutableMapping[str, Any], path: Tuple[str, ...], value: Any) -> None:
    current: MutableMapping[str, Any] = container
    for segment in path[:-1]:
        next_node = current.get(segment)
        if not isinstance(next_node, MutableMapping):
            next_node = {}
            current[segment] = next_node
        current = cast(MutableMapping[str, Any], next_node)
    current[path[-1]] = value


def _get_nested(container: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = container
    for segment in path:
        if not isinstance(current, MappingABC) or segment not in current:
            raise KeyError(path)
        current = current[segment]
    return current


def _delete_nested(container: MutableMapping[str, Any], path: Tuple[str, ...]) -> bool:
    if not path:
        return False

    parents: list[Tuple[MutableMapping[str, Any], str]] = []
    current: MutableMapping[str, Any] = container
    for segment in path[:-1]:
        next_node = current.get(segment)
        if not isinstance(next_node, MutableMapping):
            return False
        parents.append((current, segment))
        current = cast(MutableMapping[str, Any], next_node)

    removed = current.pop(path[-1], None) is not None
    if not removed:
        return False

    for mapping, key in reversed(parents):
        child = mapping.get(key)
        if isinstance(child, dict) and not child:
            mapping.pop(key, None)
        else:
            break
    return True


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, MappingABC):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        sanitized = [_to_json_safe(item) for item in value]
        return sorted(sanitized, key=lambda item: str(item))
    return str(value)


def _make_ttl_key(scope: str, identifier: Sequence[str] | str | Tuple[str, ...]) -> str:
    if isinstance(identifier, str):
        locator = identifier
    else:
        locator = "/".join(str(part) for part in identifier)
    return f"{scope}::{locator}"
