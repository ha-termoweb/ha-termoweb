"""Shared websocket helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from ..api import RESTClient
from ..const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN,
    WS_NAMESPACE,
    signal_ws_data,
    signal_ws_status,
)
from ..inventory import (
    HEATER_NODE_TYPES,
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ducaheat_ws import DucaheatWSClient
    from .termoweb_ws import TermoWebWSClient

from .ws_health import WsHealthTracker

_LOGGER = logging.getLogger(__name__)


def resolve_ws_update_section(section: str | None) -> tuple[str | None, str | None]:
    """Map a websocket path segment onto the node section bucket."""

    if not section:
        return None, None

    lowered = section.lower()
    if lowered in {"status", "samples", "settings", "advanced"}:
        return lowered, None
    if lowered == "advanced_setup":
        return "advanced", "advanced_setup"
    if lowered in {"setup", "prog", "prog_temps", "capabilities"}:
        return "settings", lowered
    return "settings", lowered


def forward_ws_sample_updates(
    hass: HomeAssistant,
    entry_id: str,
    dev_id: str,
    updates: Mapping[str, Mapping[str, Any]],
    *,
    logger: logging.Logger | None = None,
    log_prefix: str = "WS",
) -> None:
    """Relay websocket heater sample updates to the energy coordinator."""

    record = hass.data.get(DOMAIN, {}).get(entry_id)
    if not isinstance(record, Mapping):
        return
    energy_coordinator = record.get("energy_coordinator")
    handler = getattr(energy_coordinator, "handle_ws_samples", None)
    if not callable(handler):
        return

    alias_map: dict[str, str] = {"htr": "htr", "acm": "acm", "pmo": "pmo"}

    def _merge_aliases(candidate: Mapping[str, str] | None) -> None:
        if not isinstance(candidate, Mapping):
            return
        for raw_type, canonical in candidate.items():
            normalized_raw = normalize_node_type(raw_type, use_default_when_falsey=True)
            normalized_canonical = normalize_node_type(
                canonical,
                use_default_when_falsey=True,
            )
            if normalized_raw:
                alias_map[normalized_raw] = normalized_canonical or canonical

    inventory: Inventory | None = None
    candidate = record.get("inventory")
    if isinstance(candidate, Inventory):
        inventory = candidate
    if inventory is None:
        coordinator = record.get("coordinator")
        candidate = getattr(coordinator, "inventory", None)
        if isinstance(candidate, Inventory):
            inventory = candidate

    if isinstance(inventory, Inventory):
        _, heater_aliases = inventory.heater_sample_address_map
        _, pmo_aliases = inventory.power_monitor_sample_address_map
        _merge_aliases(heater_aliases)
        _merge_aliases(pmo_aliases)

    normalized_updates: dict[str, dict[str, Any]] = {}
    lease_seconds: float | None = None
    for raw_type, section in updates.items():
        if not isinstance(section, Mapping):
            continue
        node_type = normalize_node_type(raw_type, use_default_when_falsey=True)
        if not node_type:
            continue
        canonical_type = alias_map.get(node_type, node_type)
        samples_section: Mapping[str, Any] | None = None
        lease_candidate: Any = None
        if "samples" in section and isinstance(section.get("samples"), Mapping):
            samples_section = section["samples"]
            lease_candidate = section.get("lease_seconds")
        else:
            samples_section = section
            lease_candidate = section.get("lease_seconds")
        if lease_candidate is not None:
            try:
                lease_value = float(lease_candidate)
            except (TypeError, ValueError):
                lease_value = None
            else:
                if lease_value > 0:
                    lease_seconds = (
                        max(lease_seconds or 0.0, lease_value)
                        if lease_seconds is not None
                        else lease_value
                    )
        if not isinstance(samples_section, Mapping):
            continue
        bucket = normalized_updates.setdefault(canonical_type, {})
        for raw_addr, payload in samples_section.items():
            if raw_addr == "lease_seconds":
                continue
            addr = normalize_node_addr(raw_addr, use_default_when_falsey=True)
            if not addr:
                continue
            bucket[addr] = payload

    normalized_updates = {
        node_type: dict(section)
        for node_type, section in normalized_updates.items()
        if section
    }
    if not normalized_updates:
        return

    active_logger = logger or _LOGGER
    try:
        handler(
            dev_id,
            normalized_updates,
            lease_seconds=lease_seconds,
        )
    except Exception:  # pragma: no cover - defensive logging
        active_logger.debug(
            "%s: forwarding heater samples failed",
            log_prefix,
            exc_info=True,
        )


def translate_path_update(
    payload: Any,
    *,
    resolve_section: Callable[[str | None], tuple[str | None, str | None]],
) -> dict[str, Any] | None:
    """Translate ``{"path": ..., "body": ...}`` websocket frames into nodes."""

    if not isinstance(payload, Mapping):
        return None
    if "nodes" in payload:
        return None
    path = payload.get("path")
    body = payload.get("body")
    if not isinstance(path, str) or body is None:
        return None

    path = path.split("?", 1)[0]
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None

    try:
        devs_idx = segments.index("devs")
    except ValueError:
        devs_idx = -1

    if devs_idx >= 0:
        relevant = segments[devs_idx + 1 :]
        node_type_idx = 1
        addr_idx = 2
        section_idx = 3
        if len(relevant) <= addr_idx:
            return None
    else:
        relevant = segments
        node_type_idx = 0
        addr_idx = 1
        section_idx = 2
        if len(relevant) <= addr_idx:
            return None

    node_type = normalize_node_type(relevant[node_type_idx])
    addr = normalize_node_addr(relevant[addr_idx])
    if not node_type or not addr:
        return None

    section = relevant[section_idx] if len(relevant) > section_idx else None
    remainder = relevant[section_idx + 1 :] if len(relevant) > section_idx + 1 else []

    target_section, nested_key = resolve_section(section)
    if target_section is None:
        return None

    payload_body: Any = body
    for segment in reversed(remainder):
        payload_body = {segment: payload_body}
    if nested_key:
        payload_body = {nested_key: payload_body}

    return {node_type: {target_section: {addr: payload_body}}}


DUCAHEAT_NAMESPACE = "/api/v2/socket_io"


@dataclass
class WSStats:
    """Track websocket frame and event stats."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


class HandshakeError(RuntimeError):
    """Raised when a websocket handshake fails."""

    def __init__(
        self,
        status: int,
        url: str,
        detail: str,
        response_snippet: str | None = None,
    ) -> None:
        super().__init__(f"handshake failed: status={status}, detail={detail}")
        self.status = status
        self.url = url
        self.detail = detail
        self.response_snippet = response_snippet if response_snippet is not None else detail


class _WsLeaseMixin:
    """Provide reconnect backoff management for websocket clients."""

    def __init__(self) -> None:
        self._payload_idle_window: float = 240.0
        self._subscription_refresh_lock = asyncio.Lock()
        self._subscription_refresh_failed = False
        self._backoff_idx = 0
        self._backoff = (5, 10, 30, 120, 300)

    def _reset_backoff(self) -> None:
        self._backoff_idx = 0

    def _next_backoff(self) -> float:
        idx = min(self._backoff_idx, len(self._backoff) - 1)
        self._backoff_idx = min(self._backoff_idx + 1, len(self._backoff) - 1)
        return self._backoff[idx]


class _WSStatusMixin:
    """Provide shared websocket status helpers."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str

    def _status_should_reset_health(self, status: str) -> bool:
        """Return True when a status should clear healthy tracking."""

        return False

    def _ws_state_bucket(self) -> dict[str, Any]:
        """Return the websocket state bucket for the current device."""

        ws_state = getattr(self, "_ws_state", None)
        if isinstance(ws_state, dict):
            return ws_state

        hass_data = getattr(self.hass, "data", None)
        if hass_data is None:
            hass_data = {}
            setattr(self.hass, "data", hass_data)  # type: ignore[attr-defined]

        domain_bucket = hass_data.setdefault(DOMAIN, {})
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        ws_bucket = entry_bucket.setdefault("ws_state", {})
        ws_state = ws_bucket.setdefault(self.dev_id, {})
        setattr(self, "_ws_state", ws_state)
        return ws_state

    def _ws_health_tracker(self) -> WsHealthTracker:
        """Return the :class:`WsHealthTracker` for this websocket client."""

        cached = getattr(self, "_ws_tracker", None)
        if isinstance(cached, WsHealthTracker):
            if (
                getattr(self, "_pending_default_cadence_hint", False)
                and hasattr(self, "_apply_payload_window_hint")
                and not getattr(self, "_suppress_default_cadence_hint", False)
            ):
                self._pending_default_cadence_hint = False
                self._apply_payload_window_hint(
                    source="cadence",
                    lease_seconds=120,
                    candidates=[30, 75, "90"],
                )
            return cached

        hass_data = getattr(self.hass, "data", None)
        if hass_data is None:
            hass_data = {}
            setattr(self.hass, "data", hass_data)  # type: ignore[attr-defined]

        domain_bucket = hass_data.setdefault(DOMAIN, {})
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        trackers = entry_bucket.setdefault("ws_trackers", {})
        tracker = trackers.get(self.dev_id)
        created = False
        if not isinstance(tracker, WsHealthTracker):
            tracker = WsHealthTracker(self.dev_id)
            trackers[self.dev_id] = tracker
            created = True
        legacy_status = getattr(self, "_status", None)
        if (
            isinstance(legacy_status, str)
            and legacy_status
            and tracker.status == "stopped"
        ):
            tracker.status = legacy_status
        legacy_since = getattr(self, "_healthy_since", None)
        if tracker.healthy_since is None and isinstance(legacy_since, (int, float)):
            tracker.healthy_since = float(legacy_since)
        legacy_payload = getattr(self, "_last_payload_at", None)
        if tracker.last_payload_at is None and isinstance(legacy_payload, (int, float)):
            tracker.last_payload_at = float(legacy_payload)
        legacy_heartbeat = getattr(self, "_last_heartbeat_at", None)
        if tracker.last_heartbeat_at is None and isinstance(
            legacy_heartbeat, (int, float)
        ):
            tracker.last_heartbeat_at = float(legacy_heartbeat)
        setattr(self, "_ws_tracker", tracker)
        should_apply_hint = created and hasattr(self, "_apply_payload_window_hint")
        if should_apply_hint and getattr(self, "_suppress_default_cadence_hint", False):
            should_apply_hint = False
        if should_apply_hint:
            self._apply_payload_window_hint(
                source="cadence",
                lease_seconds=120,
                candidates=[30, 75, "90"],
            )
        elif getattr(self, "_pending_default_cadence_hint", False) and hasattr(
            self, "_apply_payload_window_hint"
        ):
            self._pending_default_cadence_hint = False
            self._apply_payload_window_hint(
                source="cadence",
                lease_seconds=120,
                candidates=[30, 75, "90"],
            )
        return tracker

    def _notify_ws_status(
        self,
        tracker: WsHealthTracker,
        *,
        reason: str,
        health_changed: bool = False,
        payload_changed: bool = False,
    ) -> None:
        """Dispatch websocket status updates with tracker metadata."""

        dispatcher = getattr(self, "_dispatcher_mock", None)
        if dispatcher is None:
            dispatcher = async_dispatcher_send

        payload: dict[str, Any] = {
            "dev_id": self.dev_id,
            "status": tracker.status,
            "reason": reason,
        }
        if health_changed:
            payload["health_changed"] = True
        if payload_changed:
            payload["payload_changed"] = True
        payload["payload_stale"] = tracker.payload_stale

        dispatcher(self.hass, signal_ws_status(self.entry_id), payload)

    def _mark_ws_payload(
        self,
        *,
        timestamp: float | None = None,
        stale_after: float | None = None,
        reason: str = "payload",
    ) -> None:
        """Update tracker payload timestamps and emit changes if required."""

        tracker = self._ws_health_tracker()
        changed = tracker.mark_payload(timestamp=timestamp, stale_after=stale_after)
        setattr(self, "_last_payload_at", tracker.last_payload_at)
        state = self._ws_state_bucket()
        state["last_payload_at"] = tracker.last_payload_at
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale
        state["payload_stale_after"] = tracker.payload_stale_after
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )

    def _mark_ws_heartbeat(
        self,
        *,
        timestamp: float | None = None,
        reason: str = "heartbeat",
    ) -> None:
        """Record a websocket heartbeat and emit staleness changes."""

        tracker = self._ws_health_tracker()
        changed = tracker.mark_heartbeat(timestamp=timestamp)
        state = self._ws_state_bucket()
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )

    def _refresh_ws_payload_state(
        self,
        *,
        now: float | None = None,
        reason: str = "refresh",
    ) -> None:
        """Re-evaluate payload staleness and emit notifications if it changed."""

        tracker = self._ws_health_tracker()
        changed = tracker.refresh_payload_state(now=now)
        state = self._ws_state_bucket()
        state["payload_stale"] = tracker.payload_stale
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )

    def _update_status(self, status: str) -> None:
        """Publish websocket status updates to Home Assistant listeners."""

        tracker = self._ws_health_tracker()
        now = time.time()

        stats = getattr(self, "_stats", None)
        last_event_ts = getattr(stats, "last_event_ts", None) if stats else None
        last_event_at = getattr(self, "_last_event_at", None)

        healthy_since = tracker.healthy_since
        reset_health = False
        if status == "healthy" and healthy_since is None:
            candidate = last_event_at or last_event_ts or now
            healthy_since = candidate
        elif self._status_should_reset_health(status):
            healthy_since = None
            reset_health = True

        status_changed, health_changed = tracker.update_status(
            status,
            healthy_since=healthy_since,
            timestamp=now,
            reset_health=reset_health,
        )

        setattr(self, "_status", tracker.status)
        setattr(self, "_healthy_since", tracker.healthy_since)

        payload_changed = tracker.refresh_payload_state(now=now)

        state = self._ws_state_bucket()
        state["status"] = tracker.status
        state["last_event_at"] = last_event_ts or last_event_at or None
        state["healthy_since"] = tracker.healthy_since
        state["healthy_minutes"] = tracker.healthy_minutes(now=now)
        state["frames_total"] = getattr(stats, "frames_total", 0) if stats else 0
        state["events_total"] = getattr(stats, "events_total", 0) if stats else 0
        state["last_payload_at"] = tracker.last_payload_at
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale

        if (
            status_changed
            or health_changed
            or payload_changed
            or status in {"healthy", "connected"}
        ):
            self._notify_ws_status(
                tracker,
                reason="status",
                health_changed=health_changed,
                payload_changed=payload_changed,
            )


@dataclass
class NodeDispatchContext:
    """Container for shared node dispatch metadata."""

    payload: Any
    inventory: Inventory | None
    record: MutableMapping[str, Any] | None


def _prepare_nodes_dispatch(
    hass: HomeAssistant,
    *,
    entry_id: str,
    coordinator: Any,
    raw_nodes: Any,
    inventory: Inventory | None = None,
) -> NodeDispatchContext:
    """Normalise node payload data for downstream websocket dispatch."""

    record_container = hass.data.get(DOMAIN, {})
    record_raw = (
        record_container.get(entry_id) if isinstance(record_container, dict) else None
    )
    record_mapping = record_raw if isinstance(record_raw, Mapping) else None
    record_mutable: MutableMapping[str, Any] | None = (
        record_raw if isinstance(record_raw, MutableMapping) else None
    )

    inventory_container: Inventory | None = None
    if isinstance(inventory, Inventory):
        inventory_container = inventory
    elif isinstance(record_mapping, Mapping):
        candidate_inventory = record_mapping.get("inventory")
        if isinstance(candidate_inventory, Inventory):
            inventory_container = candidate_inventory

    if inventory_container is None and hasattr(coordinator, "inventory"):
        candidate_inventory = getattr(coordinator, "inventory", None)
        if isinstance(candidate_inventory, Inventory):
            inventory_container = candidate_inventory

    if isinstance(record_mutable, MutableMapping) and isinstance(
        inventory_container, Inventory
    ):
        record_mutable["inventory"] = inventory_container

    return NodeDispatchContext(
        payload=raw_nodes,
        inventory=inventory_container,
        record=record_mutable,
    )


class _WSCommon(_WSStatusMixin):
    """Shared helpers for websocket clients."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str
    _coordinator: Any

    def __init__(self, *, inventory: Inventory | None = None) -> None:
        """Initialise shared websocket state."""

        self._inventory: Inventory | None = inventory

    def _ensure_type_bucket(
        self,
        nodes_by_type: MutableMapping[str, Any],
        node_type: str,
        *,
        dev_map: MutableMapping[str, Any] | None = None,
    ) -> MutableMapping[str, Any]:
        """Return the node bucket for ``node_type`` with default sections."""

        if not isinstance(nodes_by_type, MutableMapping):
            return {}

        normalized_type = normalize_node_type(node_type)
        if not normalized_type:
            return {}

        existing = nodes_by_type.get(normalized_type)
        if isinstance(existing, MutableMapping):
            bucket: MutableMapping[str, Any] = existing
        elif isinstance(existing, Mapping):
            bucket = dict(existing)
            nodes_by_type[normalized_type] = bucket
        else:
            bucket = {}
            nodes_by_type[normalized_type] = bucket

        addrs = bucket.get("addrs")
        if isinstance(addrs, Iterable) and not isinstance(addrs, (list, str, bytes)):
            bucket["addrs"] = list(addrs)
        elif not isinstance(addrs, list):
            bucket.setdefault("addrs", [])

        for section in ("settings", "samples", "status", "advanced"):
            section_payload = bucket.get(section)
            if isinstance(section_payload, MutableMapping):
                continue
            if isinstance(section_payload, Mapping):
                bucket[section] = dict(section_payload)
            else:
                bucket[section] = {}

        if not isinstance(dev_map, MutableMapping):
            return bucket

        settings_section = dev_map.get("settings")
        if isinstance(settings_section, MutableMapping):
            settings_map: MutableMapping[str, Any] = settings_section
        elif isinstance(settings_section, Mapping):
            settings_map = dict(settings_section)
            dev_map["settings"] = settings_map
        else:
            settings_map = {}
            dev_map["settings"] = settings_map

        existing_settings = settings_map.get(normalized_type)
        if isinstance(existing_settings, MutableMapping):
            pass
        elif isinstance(existing_settings, Mapping):
            settings_map[normalized_type] = dict(existing_settings)
        else:
            settings_map[normalized_type] = {}

        nodes_section = dev_map.get("nodes_by_type")
        if isinstance(nodes_section, MutableMapping):
            nodes_section.setdefault(normalized_type, bucket)
        elif isinstance(nodes_section, Mapping):
            nodes_map = dict(nodes_section)
            nodes_map.setdefault(normalized_type, bucket)
            dev_map["nodes_by_type"] = nodes_map
        else:
            dev_map["nodes_by_type"] = {normalized_type: bucket}

        return bucket

    def _apply_heater_addresses(
        self,
        _nodes_payload: Any,
        *,
        inventory: Inventory | None = None,
        log_prefix: str = "WS",
        logger: logging.Logger | None = None,
        prefer_inventory: bool = True,
    ) -> dict[str, list[str]]:
        """Update entry and coordinator state with heater address data."""
        record_container = self.hass.data.get(DOMAIN, {})
        record_raw = (
            record_container.get(self.entry_id)
            if isinstance(record_container, dict)
            else None
        )
        record = record_raw if isinstance(record_raw, Mapping) else None
        record_mutable: MutableMapping[str, Any] | None = (
            record_raw if isinstance(record_raw, MutableMapping) else None
        )
        record_mapping = record_raw if isinstance(record_raw, Mapping) else None

        active_logger = logger or _LOGGER

        inventory_container: Inventory | None = None
        if isinstance(inventory, Inventory):
            inventory_container = inventory
        else:
            if inventory is not None:
                active_logger.debug(
                    "%s: ignoring unexpected inventory container (type=%s): %s",
                    log_prefix,
                    type(inventory).__name__,
                    inventory,
                )
            if prefer_inventory:
                if isinstance(self._inventory, Inventory):
                    inventory_container = self._inventory
                elif isinstance(record_mapping, Mapping):
                    candidate_inventory = record_mapping.get("inventory")
                    if isinstance(candidate_inventory, Inventory):
                        inventory_container = candidate_inventory

        if not isinstance(inventory_container, Inventory):
            active_logger.error(
                "%s: missing inventory for heater address update on %s",
                log_prefix,
                self.dev_id,
            )
            return {}

        try:
            inventory_heater_map, _ = inventory_container.heater_sample_address_map
        except Exception:  # pragma: no cover - defensive cache guard
            active_logger.debug(
                "%s: failed to resolve heater sample addresses from inventory",
                log_prefix,
                exc_info=True,
            )
            return {}

        try:
            inventory_power_map, _ = (
                inventory_container.power_monitor_sample_address_map
            )
        except Exception:  # pragma: no cover - defensive cache guard
            active_logger.debug(
                "%s: failed to resolve power monitor sample addresses",
                log_prefix,
                exc_info=True,
            )
            inventory_power_map = {}

        cleaned_map: dict[str, list[str]] = {
            node_type: list(addresses)
            for node_type, addresses in inventory_heater_map.items()
        }

        for node_type in HEATER_NODE_TYPES:
            cleaned_map.setdefault(node_type, [])

        pmo_addresses = inventory_power_map.get("pmo", [])
        if pmo_addresses:
            cleaned_map["pmo"] = list(pmo_addresses)

        if isinstance(record_mutable, MutableMapping):
            record_mutable["inventory"] = inventory_container
            record_mutable.pop("sample_aliases", None)
        if not isinstance(self._inventory, Inventory):
            self._inventory = inventory_container

        energy_coordinator = (
            record.get("energy_coordinator") if isinstance(record, Mapping) else None
        )

        if hasattr(energy_coordinator, "update_addresses"):
            energy_coordinator.update_addresses(inventory_container)

        return cleaned_map

    def _dispatch_nodes(self, payload: dict[str, Any]) -> None:
        raw_nodes = payload.get("nodes") if "nodes" in payload else payload
        context = _prepare_nodes_dispatch(
            self.hass,
            entry_id=self.entry_id,
            coordinator=self._coordinator,
            raw_nodes=raw_nodes,
            inventory=self._inventory,
        )
        inventory = context.inventory if isinstance(context.inventory, Inventory) else None
        if inventory is not None and not isinstance(self._inventory, Inventory):
            self._inventory = inventory

        self._apply_heater_addresses(
            raw_nodes,
            inventory=inventory,
            log_prefix="WS",
            logger=_LOGGER,
        )

        if not isinstance(inventory, Inventory):
            _LOGGER.error("WS: missing inventory for dispatch on device %s", self.dev_id)
            return

        try:
            addresses_by_type = inventory.addresses_by_type
        except Exception:  # pragma: no cover - defensive cache guard
            _LOGGER.debug(
                "WS: failed to resolve inventory addresses; using empty payload",
                exc_info=True,
            )
            addresses_by_type: dict[str, list[str]] = {}
        payload_copy = {
            "dev_id": self.dev_id,
            "node_type": None,
            "addresses_by_type": addresses_by_type,
        }
        async_dispatcher_send(self.hass, signal_ws_data(self.entry_id), payload_copy)


class WebSocketClient(_WsLeaseMixin, _WSStatusMixin):
    """Delegate to the correct backend websocket client."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
        session: aiohttp.ClientSession | None = None,
        protocol: str | None = None,
        namespace: str = WS_NAMESPACE,
        inventory: Inventory | None = None,
    ) -> None:
        _WsLeaseMixin.__init__(self)
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        self._protocol_hint = protocol
        self._namespace = namespace or WS_NAMESPACE
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._delegate: DucaheatWSClient | TermoWebWSClient | None = None
        self._brand = (
            BRAND_DUCAHEAT
            if getattr(api_client, "_is_ducaheat", False)
            else BRAND_TERMOWEB
        )
        self._inventory = inventory

    def start(self) -> asyncio.Task:
        if self._delegate is not None:
            return self._delegate.start()
        if self._brand == BRAND_DUCAHEAT:
            from .ducaheat_ws import DucaheatWSClient

            self._delegate = DucaheatWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=DUCAHEAT_NAMESPACE,
                inventory=self._inventory,
            )
        else:
            from .termoweb_ws import TermoWebWSClient

            self._delegate = TermoWebWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=self._namespace,
                inventory=self._inventory,
            )
        return self._delegate.start()

    async def stop(self) -> None:
        if self._delegate is not None:
            await self._delegate.stop()

    def is_running(self) -> bool:
        return bool(self._delegate and self._delegate.is_running())

    async def ws_url(self) -> str:
        if self._delegate and hasattr(self._delegate, "ws_url"):
            return await self._delegate.ws_url()
        return ""


__all__ = [
    "DUCAHEAT_NAMESPACE",
    "HandshakeError",
    "WSStats",
    "WebSocketClient",
    "WsHealthTracker",
    "forward_ws_sample_updates",
    "resolve_ws_update_section",
    "translate_path_update",
]

time_mod = time.monotonic


def __getattr__(name: str) -> Any:
    """Lazily expose backend websocket client implementations."""

    if name == "DucaheatWSClient":
        from .ducaheat_ws import DucaheatWSClient as _DucaheatWSClient

        return _DucaheatWSClient
    if name == "TermoWebWSClient":
        from .termoweb_ws import TermoWebWSClient as _TermoWebWSClient

        return _TermoWebWSClient
    raise AttributeError(name)
