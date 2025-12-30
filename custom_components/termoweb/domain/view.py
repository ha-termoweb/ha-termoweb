"""Read-only façade for accessing domain state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .ids import NodeType
from .state import (
    DomainState,
    DomainStateStore,
    PowerMonitorState,
    build_state_from_payload,
)


class DomainStateView:
    """Provide read-only access to domain state with legacy fallbacks."""

    def __init__(
        self,
        dev_id: str,
        store: DomainStateStore | None,
        legacy_provider: Callable[[], Mapping[str, Any] | None],
    ) -> None:
        """Initialise the view for ``dev_id`` using ``store`` when available."""

        self._dev_id = str(dev_id)
        self._store = store
        self._legacy_provider = legacy_provider

    def update_store(self, store: DomainStateStore | None) -> None:
        """Refresh the backing domain store reference."""

        self._store = store

    def _legacy_device(self) -> Mapping[str, Any] | None:
        """Return the legacy coordinator payload for this device."""

        candidate = self._legacy_provider()
        return candidate if isinstance(candidate, Mapping) else None

    def _legacy_settings_payload(
        self, node_type: str | NodeType, addr: str
    ) -> Mapping[str, Any] | None:
        """Return the legacy settings payload for ``(node_type, addr)``."""

        legacy = self._legacy_device()
        if not isinstance(legacy, Mapping):
            return None

        node_type_key = (
            node_type.value if isinstance(node_type, NodeType) else str(node_type)
        ).lower()

        settings_by_type = legacy.get("settings")
        if isinstance(settings_by_type, Mapping):
            per_type = settings_by_type.get(node_type_key)
            if isinstance(per_type, Mapping):
                payload = per_type.get(addr)
                if isinstance(payload, Mapping):
                    return payload

        node_section = legacy.get(node_type_key)
        if isinstance(node_section, Mapping):
            per_type_settings = node_section.get("settings")
            if isinstance(per_type_settings, Mapping):
                payload = per_type_settings.get(addr)
                if isinstance(payload, Mapping):
                    return payload

        return None

    def _build_state(self, node_type: str | NodeType, addr: str) -> DomainState | None:
        """Return a domain state object using the store or legacy payloads."""

        if self._store is not None:
            state = self._store.get_state(node_type, addr)
            if state is not None:
                return state

        payload = self._legacy_settings_payload(node_type, addr)
        if payload is None:
            return None

        return build_state_from_payload(node_type, payload)

    def get_heater_state(
        self, node_type: str | NodeType, addr: str
    ) -> DomainState | None:
        """Return the heater/thermostat/accumulator state for ``addr``."""

        return self._build_state(node_type, addr)

    def get_power_monitor_state(self, addr: str) -> PowerMonitorState | None:
        """Return the power monitor state for ``addr`` when present."""

        state = self._build_state(NodeType.POWER_MONITOR, addr)
        return state if isinstance(state, PowerMonitorState) else None
