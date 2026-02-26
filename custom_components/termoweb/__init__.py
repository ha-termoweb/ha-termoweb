"""Home Assistant entry point for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
import inspect
import logging
import time
import typing
from typing import Any

from aiohttp import ClientError
from homeassistant import config_entries as config_entries_module
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import async_call_later

from .backend import Backend, create_backend, create_rest_client
from .backend.debug import build_unknown_node_probe_requests
from .backend.rest_client import BackendAuthError, BackendRateLimitError, RESTClient
from .backend.sanitize import redact_text
from .const import (
    BRAND_DUCAHEAT as BRAND_DUCAHEAT,
    BRAND_TEVOLVE as BRAND_TEVOLVE,
    CONF_BRAND,
    DEFAULT_BRAND,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    signal_ws_status,
)
from .coordinator import EnergyStateCoordinator, StateCoordinator, build_device_metadata
from .hourly_poller import HourlySamplesPoller
from .inventory import (
    Inventory,
    build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)
from .runtime import EntryRuntime
from .services.energy_history import (
    async_import_energy_history_with_rate_limit,
    async_register_import_energy_history_service,
)
from .services.ws_debug_probe import async_register_ws_debug_probe_service
from .throttle import reset_samples_rate_limit_state
from .utils import async_get_integration_version as _async_get_integration_version

_LOGGER = logging.getLogger(__name__)

SupportsDiagnostics = getattr(config_entries_module, "SupportsDiagnostics", None)

PLATFORMS = ["button", "binary_sensor", "climate", "number", "sensor", "switch"]

reset_samples_rate_limit_state()

_SUPPORTED_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm", "pmo"})


async def _async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    nodes: Inventory | None = None,
    node_types: Iterable[str] | None = None,
    addresses: Iterable[str] | None = None,
    day_chunk_hours: int = 24,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Delegate to the energy helper with shared rate limiting and filters."""

    await async_import_energy_history_with_rate_limit(
        hass,
        entry,
        nodes=nodes,
        node_types=node_types,
        addresses=addresses,
        day_chunk_hours=day_chunk_hours,
        reset_progress=reset_progress,
        max_days=max_days,
    )


async def _async_probe_unknown_node_types(
    backend: Backend,
    dev_id: str,
    inventory: Inventory,
) -> None:
    """Log discovery probes for node types not yet supported."""

    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return

    client = getattr(backend, "client", None)
    probe_get = getattr(client, "debug_probe_get", None)
    authed_headers = getattr(client, "authed_headers", None)
    if not callable(probe_get) or not callable(authed_headers):
        return

    seen: set[tuple[str, str]] = set()
    for node in inventory.nodes:
        node_type = normalize_node_type(
            getattr(node, "type", None),
            use_default_when_falsey=True,
        )
        if not node_type or node_type in _SUPPORTED_NODE_TYPES:
            continue
        addr = normalize_node_addr(
            getattr(node, "addr", None),
            use_default_when_falsey=True,
        )
        dedupe_key = (node_type, addr)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        display_addr = addr or "<missing>"
        _LOGGER.debug("Unknown node type found: %s/%s", node_type, display_addr)

        requests = build_unknown_node_probe_requests(
            backend.brand,
            dev_id,
            node_type,
            addr,
        )
        if not requests:
            continue

        try:
            headers = await authed_headers()
        except Exception as err:  # noqa: BLE001 - best-effort logging only
            _LOGGER.debug(
                "Probe header preparation failed for %s/%s: %s",
                node_type,
                display_addr,
                redact_text(str(err)),
            )
            continue

        for path, params in requests:
            try:
                await probe_get(path, headers=headers, params=params)
            except Exception as err:  # noqa: BLE001 - best-effort logging only
                _LOGGER.debug(
                    "Probe GET %s failed for %s/%s: %s",
                    path,
                    node_type,
                    display_addr,
                    redact_text(str(err)),
                )


async def async_list_devices(client: RESTClient) -> Any:
    """Call ``list_devices`` logging auth/connection issues consistently."""

    try:
        return await client.list_devices()
    except BackendAuthError as err:
        _LOGGER.info("list_devices auth error: %s", err)
        raise
    except (TimeoutError, ClientError, BackendRateLimitError) as err:
        _LOGGER.info("list_devices connection error: %s", err)
        raise


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:  # noqa: C901
    """Set up the TermoWeb integration for a config entry."""
    username = entry.data["username"]
    password = entry.data["password"]
    base_interval = int(DEFAULT_POLL_INTERVAL)
    if "poll_interval" in entry.data or "poll_interval" in entry.options:
        new_data = dict(entry.data)
        new_options = dict(entry.options)
        new_data.pop("poll_interval", None)
        new_options.pop("poll_interval", None)
        hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options
        )
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)

    supports_diagnostics_value = (
        SupportsDiagnostics.YES if SupportsDiagnostics is not None else True
    )
    entry.supports_diagnostics = supports_diagnostics_value
    update_payload = entry.data | {"supports_diagnostics": True}
    update_method = getattr(entry, "async_update", None)
    if callable(update_method):
        update_result = update_method(update_payload)
        if inspect.isawaitable(update_result):
            await update_result
    else:
        hass.config_entries.async_update_entry(entry, data=update_payload)

    version = await _async_get_integration_version(hass)

    client = create_rest_client(hass, username, password, brand)
    backend = create_backend(brand=brand, client=client)
    try:
        devices = await async_list_devices(client)
    except BackendAuthError as err:
        raise ConfigEntryAuthFailed from err
    except (TimeoutError, ClientError, BackendRateLimitError) as err:
        raise ConfigEntryNotReady from err

    if not devices:
        _LOGGER.info("list_devices returned no devices")
        raise ConfigEntryNotReady

    dev: Mapping[str, typing.Any] | None = None
    dev_id = ""
    if isinstance(devices, list):
        for index, candidate in enumerate(devices):
            if not isinstance(candidate, Mapping):
                continue
            candidate_id = str(
                candidate.get("dev_id")
                or candidate.get("id")
                or candidate.get("serial_id")
                or ""
            ).strip()
            if candidate_id:
                dev = candidate
                dev_id = candidate_id
                break
            _LOGGER.debug(
                "Skipping device entry without identifier at index %s: %s",
                index,
                candidate,
            )
    elif isinstance(devices, Mapping):
        dev = devices
        dev_id = str(
            dev.get("dev_id") or dev.get("id") or dev.get("serial_id") or ""
        ).strip()
    else:
        _LOGGER.debug("Unexpected list_devices payload: %s", devices)

    if not dev_id:
        _LOGGER.info("list_devices returned no usable devices")
        raise ConfigEntryNotReady

    device_metadata = build_device_metadata(dev_id, dev)
    nodes = await client.get_nodes(dev_id)
    node_inventory = build_node_inventory(nodes)
    # Inventory-centric design: build and freeze the gateway/node topology once
    # during setup so every runtime component can trust the shared metadata.
    inventory = Inventory(dev_id, node_inventory)
    await _async_probe_unknown_node_types(backend, dev_id, inventory)
    if inventory.nodes:
        type_counts = Counter(node.type for node in inventory.nodes)
        summary = ", ".join(
            f"{node_type}:{count}" for node_type, count in sorted(type_counts.items())
        )
    else:
        summary = "none"
    _LOGGER.info("%s: discovered node types: %s", dev_id, summary)

    coordinator = StateCoordinator(
        hass,
        client,
        base_interval,
        dev_id,
        device_metadata,
        None,
        inventory,
        brand=brand,
    )

    energy_coordinator = EnergyStateCoordinator(
        hass,
        client,
        dev_id,
        inventory,
        state_coordinator=coordinator,
    )
    energy_coordinator.update_addresses(inventory)
    await energy_coordinator.async_config_entry_first_refresh()

    poller = HourlySamplesPoller(hass, energy_coordinator, backend, inventory)
    await poller.async_setup()

    debug_enabled = bool(entry.options.get("debug", entry.data.get("debug", False)))

    hass.data.setdefault(DOMAIN, {})
    runtime = EntryRuntime(
        backend=backend,
        client=backend.client,
        coordinator=coordinator,
        energy_coordinator=energy_coordinator,
        dev_id=dev_id,
        inventory=inventory,
        hourly_poller=poller,
        config_entry=entry,
        base_poll_interval=max(base_interval, MIN_POLL_INTERVAL),
        stretched=False,
        poll_suspended=False,
        poll_resume_unsub=None,
        ws_tasks={},
        ws_clients={},
        ws_state={},
        ws_trackers={},
        version=version,
        brand=brand,
        debug=debug_enabled,
        boost_runtime={},
        boost_temperature={},
        climate_entities={},
    )
    hass.data[DOMAIN][entry.entry_id] = runtime

    async def _async_handle_hass_stop(_event: Any) -> None:
        """Stop background activity gracefully when Home Assistant stops."""

        await _async_shutdown_entry(runtime)

    remove_stop_listener = hass.bus.async_listen_once(
        EVENT_HOMEASSISTANT_STOP, _async_handle_hass_stop
    )
    entry.async_on_unload(remove_stop_listener)

    async def _start_ws(dev_id: str) -> None:
        """Ensure a websocket client exists and is running for ``dev_id``."""
        backend: Backend = runtime.backend
        tasks = runtime.ws_tasks
        clients = runtime.ws_clients
        if dev_id in tasks and not tasks[dev_id].done():
            return
        ws_client = clients.get(dev_id)
        if not ws_client:
            ws_client = backend.create_ws_client(
                hass,
                entry_id=entry.entry_id,
                dev_id=dev_id,
                coordinator=coordinator,
                inventory=inventory,
            )
            clients[dev_id] = ws_client
        task = ws_client.start()
        tasks[dev_id] = task
        _LOGGER.info("WS: started read-only client for %s", dev_id)

    runtime.start_ws = _start_ws

    def _recalc_poll_interval() -> None:
        """Suspend REST polling when websocket trackers are healthy and fresh."""

        tasks = runtime.ws_tasks
        trackers = runtime.ws_trackers
        base_interval = runtime.base_poll_interval
        suspended = runtime.poll_suspended

        def _cancel_timer() -> None:
            handle = runtime.poll_resume_unsub
            if callable(handle):
                try:
                    handle()
                except Exception:  # noqa: BLE001 - defensive cancellation
                    _LOGGER.debug(
                        "WS: failed to cancel poll resume timer", exc_info=True
                    )
            runtime.poll_resume_unsub = None

        if not tasks:
            if suspended:
                coordinator.update_interval = timedelta(seconds=base_interval)
                runtime.poll_suspended = False
                runtime.stretched = False
                _cancel_timer()
                _LOGGER.info(
                    "WS: websocket clients idle; resuming REST polling at %ss",
                    base_interval,
                )
            return

        now = time.time()
        any_running = False
        all_healthy = True
        fresh_payload = True
        earliest_deadline: float | None = None

        for dev_id, task in tasks.items():
            if task.done():
                all_healthy = False
                fresh_payload = False
                continue
            any_running = True
            tracker = trackers.get(dev_id)
            if tracker is None:
                all_healthy = False
                fresh_payload = False
                continue
            status = getattr(tracker, "status", None)
            if status != "healthy":
                all_healthy = False
            payload_at = getattr(tracker, "last_payload_at", None)
            if payload_at is None:
                fresh_payload = False
            else:
                is_stale = getattr(tracker, "is_payload_stale", None)
                try:
                    stale = (
                        bool(is_stale(now=now))
                        if callable(is_stale)
                        else bool(getattr(tracker, "payload_stale", False))
                    )
                except TypeError:
                    stale = bool(is_stale()) if callable(is_stale) else False
                if stale:
                    fresh_payload = False
            deadline_func = getattr(tracker, "stale_deadline", None)
            deadline: float | None = None
            if callable(deadline_func):
                try:
                    deadline = deadline_func()
                except TypeError:
                    deadline = None
            if isinstance(deadline, (int, float)):
                if earliest_deadline is None or deadline < earliest_deadline:
                    earliest_deadline = deadline

        if not any_running:
            if suspended:
                coordinator.update_interval = timedelta(seconds=base_interval)
                runtime.poll_suspended = False
                runtime.stretched = False
                _cancel_timer()
                _LOGGER.info(
                    "WS: websocket trackers stopped; resuming REST polling at %ss",
                    base_interval,
                )
            return

        if all_healthy and fresh_payload:
            if not suspended:
                coordinator.update_interval = None
                runtime.poll_suspended = True
                runtime.stretched = True
                _LOGGER.info(
                    "WS: trackers healthy with fresh payloads; suspending REST polling",
                )
            if earliest_deadline is not None:
                delay = max(0.0, earliest_deadline - time.time())
                _cancel_timer()

                def _resume_callback(_now: Any) -> None:
                    runtime.poll_resume_unsub = None
                    _recalc_poll_interval()

                runtime.poll_resume_unsub = async_call_later(
                    hass, delay, _resume_callback
                )
            else:
                _cancel_timer()
            return

        if suspended:
            coordinator.update_interval = timedelta(seconds=base_interval)
            _LOGGER.info(
                "WS: tracker unhealthy or payload stale; resuming REST polling at %ss",
                base_interval,
            )
        runtime.poll_suspended = False
        runtime.stretched = False
        _cancel_timer()

    runtime.recalc_poll = _recalc_poll_interval

    def _on_ws_status(payload: dict[str, Any]) -> None:
        """Recalculate polling intervals when websocket state changes."""

        should_recalc = False
        if isinstance(payload, Mapping):
            if (
                payload.get("health_changed")
                or payload.get("payload_changed")
                or payload.get("reason") == "status"
            ):
                should_recalc = True
        else:
            should_recalc = True
        if should_recalc:
            _recalc_poll_interval()

    unsub = async_dispatcher_connect(
        hass, signal_ws_status(entry.entry_id), _on_ws_status
    )
    runtime.unsub_ws_status = unsub

    # First refresh (inventory etc.)
    await coordinator.async_config_entry_first_refresh()

    # Always-on push: start the websocket client for this device
    hass.async_create_task(_start_ws(dev_id))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await async_register_import_energy_history_service(
        hass,
        _async_import_energy_history,
    )

    await async_register_ws_debug_probe_service(hass)

    _LOGGER.info("TermoWeb setup complete (v%s)", version)
    return True


@dataclass(frozen=True, slots=True)
class _ShutdownTargets:
    """Container for shutdown handles derived from runtime storage."""

    poller: Any | None
    ws_tasks: dict[str, Any]
    ws_clients: dict[str, Any]
    unsub_ws_status: Callable[[], None] | None
    poll_resume_unsub: Callable[[], None] | None


def _collect_shutdown_targets(
    runtime: EntryRuntime,
) -> _ShutdownTargets | None:
    """Return shutdown targets or ``None`` if shutdown already ran."""

    if runtime._shutdown_complete:  # noqa: SLF001
        return None
    runtime._shutdown_complete = True  # noqa: SLF001
    ws_tasks = runtime.ws_tasks
    ws_clients = runtime.ws_clients
    return _ShutdownTargets(
        poller=runtime.hourly_poller,
        ws_tasks=ws_tasks,
        ws_clients=ws_clients,
        unsub_ws_status=runtime.unsub_ws_status,
        poll_resume_unsub=runtime.poll_resume_unsub,
    )


async def _shutdown_hourly_poller(poller: Any) -> None:
    """Stop the hourly poller when it exposes async_shutdown."""

    if not hasattr(poller, "async_shutdown"):
        return
    try:
        await poller.async_shutdown()
    except Exception:  # pragma: no cover - defensive shutdown logging
        _LOGGER.exception("Failed to stop hourly samples poller")


async def _shutdown_ws_tasks(ws_tasks: Mapping[str, typing.Any]) -> None:
    """Cancel and await websocket tasks."""

    for dev_id, task in list(ws_tasks.items()):
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception("WS task for %s raised during cancel", dev_id)
                continue
        if hasattr(task, "__await__"):
            try:
                await task  # type: ignore[func-returns-value]
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception("WS task for %s failed to cancel cleanly", dev_id)


async def _shutdown_ws_clients(ws_clients: Mapping[str, typing.Any]) -> None:
    """Stop websocket clients that expose a stop coroutine."""

    for dev_id, client in list(ws_clients.items()):
        stop = getattr(client, "stop", None)
        if not callable(stop):
            continue
        try:
            await stop()
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception("WS client for %s failed to stop", dev_id)


def _shutdown_runtime_callback(
    runtime: EntryRuntime,
    key: str,
    callback: Callable[[], None] | None,
    error_message: str,
) -> None:
    """Invoke and clear a runtime callback with error logging."""

    if callable(callback):
        try:
            callback()
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception(error_message)
    setattr(runtime, key, None)


async def _async_shutdown_entry(runtime: EntryRuntime) -> None:
    """Cancel websocket tasks and listeners for an integration record."""

    targets = _collect_shutdown_targets(runtime)
    if targets is None:
        return

    await _shutdown_hourly_poller(targets.poller)
    await _shutdown_ws_tasks(targets.ws_tasks)
    await _shutdown_ws_clients(targets.ws_clients)
    _shutdown_runtime_callback(
        runtime,
        "unsub_ws_status",
        targets.unsub_ws_status,
        "Failed to unsubscribe websocket status listener",
    )
    _shutdown_runtime_callback(
        runtime,
        "poll_resume_unsub",
        targets.poll_resume_unsub,
        "Failed to cancel suspended poll resume timer",
    )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry for TermoWeb."""
    domain_data = hass.data.get(DOMAIN)
    rec = domain_data.get(entry.entry_id) if domain_data else None
    if not rec:
        return True

    await _async_shutdown_entry(rec)

    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if ok and domain_data:
        domain_data.pop(entry.entry_id, None)

    return ok


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate a config entry; no migrations are needed yet."""
    return True


async def async_update_entry_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options updates; recompute interval if needed."""
    runtime = hass.data[DOMAIN][entry.entry_id]
    debug_enabled = bool(entry.options.get("debug", entry.data.get("debug", False)))
    if isinstance(runtime, EntryRuntime):
        runtime.debug = debug_enabled
        if callable(runtime.recalc_poll):
            runtime.recalc_poll()
