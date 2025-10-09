"""Home Assistant entry point for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Iterable, Mapping, MutableMapping
from datetime import timedelta
import importlib
import inspect
import logging
import sys
from typing import Any, Final

from aiohttp import ClientError

try:  # pragma: no cover - compatibility shim for older Home Assistant cores
    from homeassistant.config_entries import ConfigEntry, SupportsDiagnostics
except ImportError:  # pragma: no cover - tests provide stubbed config entries
    from homeassistant.config_entries import ConfigEntry  # type: ignore[misc]

    SupportsDiagnostics = None  # type: ignore[assignment]
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.dispatcher import async_dispatcher_connect

try:  # pragma: no cover - fallback for test stubs
    from homeassistant.setup import ATTR_COMPONENT
except ImportError:  # pragma: no cover - tests provide minimal attributes
    ATTR_COMPONENT = "component"
try:  # pragma: no cover - compatibility shim for event helpers
    from homeassistant.helpers.event import async_call_later
except ImportError:  # pragma: no cover - tests provide minimal helpers
    async_call_later = None  # type: ignore[assignment]

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .backend import Backend, DucaheatRESTClient, WsClientProto, create_backend
from .const import (
    BRAND_DUCAHEAT,
    CONF_BRAND,
    DEFAULT_BRAND,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    get_brand_api_base,
    get_brand_basic_auth,
    signal_ws_status,
)
from .coordinator import StateCoordinator
from .energy import (
    async_import_energy_history as _async_import_energy_history_impl,
    async_register_import_energy_history_service,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)
from .installation import InstallationSnapshot
from .inventory import build_node_inventory
from .utils import async_get_integration_version as _async_get_integration_version

try:  # pragma: no cover - fallback for test stubs
    from homeassistant.const import EVENT_COMPONENT_LOADED, EVENT_HOMEASSISTANT_STOP
except ImportError:  # pragma: no cover - tests provide minimal constants
    EVENT_COMPONENT_LOADED = "component_loaded"
    EVENT_HOMEASSISTANT_STOP = "homeassistant_stop"

DIAGNOSTICS_HELPER_KEY: Final = f"{DOMAIN}_diagnostics"

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["button", "binary_sensor", "climate", "select", "sensor"]

reset_samples_rate_limit_state()


async def _async_ensure_diagnostics_platform(hass: HomeAssistant) -> None:
    """Ensure the diagnostics platform registers once the component is ready."""

    state: MutableMapping[str, Any] = hass.data.setdefault(DIAGNOSTICS_HELPER_KEY, {})
    if state.get("registered"):
        _LOGGER.debug("Diagnostics platform already registered; skipping ensure call")
        return

    state.setdefault("registered", False)
    state.setdefault("attempts", 0)

    try:
        diagnostics = importlib.import_module("homeassistant.components.diagnostics")
    except ImportError:  # pragma: no cover - diagnostics component unavailable
        _LOGGER.debug(
            "Diagnostics integration unavailable; cannot register diagnostics platform"
        )
        return

    module_name = f"{__package__}.diagnostics"
    platform = sys.modules.get(module_name)
    if platform is None:
        platform = importlib.import_module(module_name)

    data_keys = [
        key
        for key in (
            getattr(diagnostics, "_DIAGNOSTICS_DATA", None),
            getattr(diagnostics, "DIAGNOSTICS_DATA", None),
            getattr(diagnostics, "DATA_DIAGNOSTICS", None),
            getattr(diagnostics, "DOMAIN", None),
        )
        if key is not None
    ]

    def _finalise_registration() -> None:
        """Record registration and clear pending listeners or timers."""

        state["registered"] = True
        listener = state.pop("listener", None)
        if callable(listener):
            try:
                listener()
            except Exception:  # pragma: no cover - defensive cleanup
                _LOGGER.debug(
                    "Failed to unsubscribe diagnostics listener", exc_info=True
                )
        cancel_retry = state.pop("cancel_retry", None)
        if callable(cancel_retry):
            try:
                cancel_retry()
            except Exception:  # pragma: no cover - defensive cleanup
                _LOGGER.debug("Failed to cancel diagnostics retry timer", exc_info=True)

    def _schedule_retry() -> None:
        """Schedule a retry once diagnostics data becomes available."""

        if state.get("registered"):  # pragma: no cover - defensive guard
            return

        if state.get("listener") is None:
            listen = getattr(hass.bus, "async_listen", None)
            if not callable(listen):
                listen = getattr(hass.bus, "async_listen_once", None)

            if callable(listen):
                target_domain = getattr(diagnostics, "DOMAIN", "diagnostics")

                async def _handle_component_loaded(event: Any) -> None:
                    component = (
                        getattr(event, "data", {}).get(ATTR_COMPONENT)
                        if event
                        else None
                    )
                    if component != target_domain:
                        return
                    await _attempt_register("component_loaded")

                state["listener"] = listen(
                    EVENT_COMPONENT_LOADED, _handle_component_loaded
                )
                _LOGGER.debug(
                    "Diagnostics registration listener installed for %s", target_domain
                )

        if state.get("cancel_retry") is None and async_call_later is not None:
            loop = getattr(hass, "loop", None)

            def _timer_callback(_: Any) -> None:  # pragma: no cover - runtime timer
                state["cancel_retry"] = None
                coroutine = _attempt_register("timer")
                if inspect.isawaitable(coroutine):
                    if loop is not None:
                        loop.create_task(coroutine)
                    else:  # pragma: no cover - fallback path for missing loop
                        asyncio.create_task(coroutine)

            cancel = async_call_later(hass, timedelta(seconds=5), _timer_callback)
            state["cancel_retry"] = cancel
            _LOGGER.debug("Diagnostics retry timer scheduled")

    async def _attempt_register(reason: str) -> None:
        """Attempt to register the diagnostics platform and log outcomes."""

        if state.get("registered"):  # pragma: no cover - defensive guard
            _LOGGER.debug(  # pragma: no cover - defensive guard
                "Skipping diagnostics registration attempt triggered by %s; already registered",
                reason,
            )
            return

        state["attempts"] += 1
        attempt_no = state["attempts"]
        _LOGGER.debug(
            "Diagnostics registration attempt %s triggered by %s", attempt_no, reason
        )

        register_async = getattr(
            diagnostics, "async_register_diagnostics_platform", None
        )
        if callable(register_async):
            try:
                result = register_async(hass, DOMAIN, platform)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception(
                    "Diagnostics registration attempt %s failed via async helper",
                    attempt_no,
                )
                return

            _LOGGER.debug(
                "Diagnostics registration attempt %s succeeded via async helper",
                attempt_no,
            )
            _finalise_registration()
            return

        register_sync = getattr(diagnostics, "_register_diagnostics_platform", None)
        if not callable(register_sync):
            _LOGGER.debug(
                "Diagnostics registration attempt %s blocked; register helper missing",
                attempt_no,
            )
            return

        diagnostics_data: Any | None = None
        used_key: Any | None = None
        for key in data_keys:
            diagnostics_data = hass.data.get(key)
            if diagnostics_data is not None:
                used_key = key
                break

        if diagnostics_data is None:
            _LOGGER.debug(
                "Diagnostics registration attempt %s deferred; diagnostics data missing",
                attempt_no,
            )
            _schedule_retry()
            return

        platforms = getattr(diagnostics_data, "platforms", None)
        if platforms is None:
            platforms = {}
        elif not isinstance(platforms, MutableMapping):
            if isinstance(platforms, Mapping):
                platforms = dict(platforms)
            else:
                _LOGGER.debug(
                    "Diagnostics registration attempt %s deferred; invalid platforms container",
                    attempt_no,
                )
                _schedule_retry()
                return

        if getattr(diagnostics_data, "platforms", None) is not platforms:
            try:
                setattr(diagnostics_data, "platforms", platforms)
            except Exception:  # pragma: no cover - compatibility shim
                try:
                    object.__setattr__(diagnostics_data, "platforms", platforms)
                except Exception:
                    _LOGGER.debug(
                        "Diagnostics registration attempt %s deferred; unable to update platforms",
                        attempt_no,
                    )
                    _schedule_retry()
                    return

        if DOMAIN in platforms:
            _LOGGER.debug(
                "Diagnostics registration attempt %s skipped; platform already registered",
                attempt_no,
            )
            _finalise_registration()
            return

        try:
            register_sync(hass, DOMAIN, platform)
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception(
                "Diagnostics registration attempt %s failed via sync helper",
                attempt_no,
            )
            return

        _LOGGER.debug(
            "Diagnostics registration attempt %s succeeded via sync helper (key=%s)",
            attempt_no,
            used_key,
        )
        _finalise_registration()

    await _attempt_register("initial")


def create_rest_client(
    hass: HomeAssistant, username: str, password: str, brand: str
) -> RESTClient:
    """Return a REST client configured for the selected brand."""

    session = aiohttp_client.async_get_clientsession(hass)
    api_base = get_brand_api_base(brand)
    basic_auth = get_brand_basic_auth(brand)
    client_cls = DucaheatRESTClient if brand == BRAND_DUCAHEAT else RESTClient
    return client_cls(
        session,
        username,
        password,
        api_base=api_base,
        basic_auth_b64=basic_auth,
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


async def _async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    nodes: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    *,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Delegate to the energy helper with shared rate limiting."""

    rate_state = default_samples_rate_limit_state()
    await _async_import_energy_history_impl(
        hass,
        entry,
        nodes,
        reset_progress=reset_progress,
        max_days=max_days,
        rate_limit=rate_state,
    )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:  # noqa: C901
    """Set up the TermoWeb integration for a config entry."""
    username = entry.data["username"]
    password = entry.data["password"]
    base_interval = int(
        entry.options.get(
            "poll_interval", entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
    )
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)

    await _async_ensure_diagnostics_platform(hass)

    if SupportsDiagnostics is not None and hasattr(entry, "supports_diagnostics"):
        entry.supports_diagnostics = SupportsDiagnostics.YES

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

    dev: Mapping[str, Any] | None = None
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

    dev = dev or {}
    nodes = await client.get_nodes(dev_id)
    node_inventory = build_node_inventory(nodes)
    snapshot = InstallationSnapshot(
        dev_id=dev_id,
        raw_nodes=nodes,
        node_inventory=node_inventory,
    )

    if node_inventory:
        type_counts = Counter(node.type for node in node_inventory)
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
        dev,
        nodes,
        node_inventory,
    )

    debug_enabled = bool(entry.options.get("debug", entry.data.get("debug", False)))

    hass.data.setdefault(DOMAIN, {})
    await _async_ensure_diagnostics_platform(hass)
    hass.data[DOMAIN][entry.entry_id] = data = {
        "backend": backend,
        "client": backend.client,
        "coordinator": coordinator,
        "dev_id": dev_id,
        "snapshot": snapshot,
        "node_inventory": list(snapshot.inventory),
        "config_entry": entry,
        "base_poll_interval": max(base_interval, MIN_POLL_INTERVAL),
        "stretched": False,
        "ws_tasks": {},  # dev_id -> asyncio.Task
        "ws_clients": {},  # dev_id -> WS clients
        "ws_state": {},  # dev_id -> status attrs
        "version": version,
        "brand": brand,
        "debug": debug_enabled,
        "boost_runtime": {},
    }

    async def _async_handle_hass_stop(_event: Any) -> None:
        """Stop background activity gracefully when Home Assistant stops."""

        await _async_shutdown_entry(data)

    remove_stop_listener = hass.bus.async_listen_once(
        EVENT_HOMEASSISTANT_STOP, _async_handle_hass_stop
    )
    entry.async_on_unload(remove_stop_listener)

    async def _start_ws(dev_id: str) -> None:
        """Ensure a websocket client exists and is running for ``dev_id``."""
        backend: Backend = data["backend"]
        tasks: dict[str, asyncio.Task] = data["ws_tasks"]
        clients: dict[str, WsClientProto] = data["ws_clients"]
        if dev_id in tasks and not tasks[dev_id].done():
            return
        ws_client = clients.get(dev_id)
        if not ws_client:
            ws_client = backend.create_ws_client(
                hass,
                entry_id=entry.entry_id,
                dev_id=dev_id,
                coordinator=coordinator,
            )
            clients[dev_id] = ws_client
        task = ws_client.start()
        tasks[dev_id] = task
        _LOGGER.info("WS: started read-only client for %s", dev_id)

    def _recalc_poll_interval() -> None:
        """Stretch polling if all running WS clients report healthy; else restore."""
        stretched = data["stretched"]
        tasks: dict[str, asyncio.Task] = data["ws_tasks"]
        state: dict[str, dict[str, Any]] = data["ws_state"]

        if not tasks:
            if stretched:
                coordinator.update_interval = timedelta(
                    seconds=data["base_poll_interval"]
                )
                data["stretched"] = False
            return

        all_healthy = True
        for dev_id, task in tasks.items():
            if task.done():
                all_healthy = False
                break
            s = state.get(dev_id) or {}
            if s.get("status") != "healthy":
                all_healthy = False
                break

        if all_healthy and not stretched:
            coordinator.update_interval = timedelta(seconds=STRETCHED_POLL_INTERVAL)
            data["stretched"] = True
            _LOGGER.info(
                "WS: healthy for ≥5m; stretching REST polling to %ss",
                STRETCHED_POLL_INTERVAL,
            )
        elif (not all_healthy) and stretched:
            coordinator.update_interval = timedelta(seconds=data["base_poll_interval"])
            data["stretched"] = False
            _LOGGER.info(
                "WS: no longer healthy; restoring REST polling to %ss",
                data["base_poll_interval"],
            )

    data["recalc_poll"] = _recalc_poll_interval

    def _on_ws_status(_payload: dict) -> None:
        """Recalculate polling intervals when websocket status changes."""
        _recalc_poll_interval()

    unsub = async_dispatcher_connect(
        hass, signal_ws_status(entry.entry_id), _on_ws_status
    )
    data["unsub_ws_status"] = unsub

    # First refresh (inventory etc.)
    await coordinator.async_config_entry_first_refresh()

    # Always-on push: start for all current devices
    for dev_id in coordinator.data or {}:
        hass.async_create_task(_start_ws(dev_id))

    # Start for any devices discovered later
    def _on_coordinator_updated() -> None:
        """Start websocket clients for newly discovered devices."""
        for dev_id in coordinator.data or {}:
            if dev_id not in data["ws_tasks"]:
                hass.async_create_task(_start_ws(dev_id))

    coordinator.async_add_listener(_on_coordinator_updated)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await async_register_import_energy_history_service(
        hass,
        _async_import_energy_history,
    )

    await async_register_ws_debug_probe_service(hass)

    _LOGGER.info("TermoWeb setup complete (v%s)", version)
    return True


async def async_register_ws_debug_probe_service(hass: HomeAssistant) -> None:
    """Register the ws_debug_probe debug helper service."""

    if hass.services.has_service(DOMAIN, "ws_debug_probe"):
        return

    async def _async_ws_debug_probe(call: ServiceCall) -> None:
        """Emit a websocket dev_data probe for debugging."""

        entry_filter = call.data.get("entry_id")
        dev_filter = call.data.get("dev_id")
        domain_records = hass.data.get(DOMAIN, {})
        if not isinstance(domain_records, Mapping):
            _LOGGER.debug("ws_debug_probe: integration data unavailable")
            return

        entries: list[tuple[str, Mapping[str, Any]]] = []
        if entry_filter:
            record = domain_records.get(entry_filter)
            if isinstance(record, Mapping):
                entries.append((entry_filter, record))
        else:
            entries = [
                (entry_id, rec)
                for entry_id, rec in domain_records.items()
                if isinstance(rec, Mapping)
            ]

        if not entries:
            _LOGGER.debug("ws_debug_probe: no matching config entries")
            return

        tasks: list[Awaitable[Any]] = []
        for entry_id, rec in entries:
            if not rec.get("debug"):
                _LOGGER.debug(
                    "ws_debug_probe: debug helpers disabled for entry %s",
                    entry_id,
                )
                continue
            clients = rec.get("ws_clients")
            if not isinstance(clients, Mapping) or not clients:
                _LOGGER.debug(
                    "ws_debug_probe: no websocket clients for entry %s",
                    entry_id,
                )
                continue
            if dev_filter:
                target_dev_ids: Iterable[str] = [str(dev_filter)]
            else:
                target_dev_ids = [str(dev) for dev in clients]
            for dev_id in target_dev_ids:
                client = clients.get(dev_id)
                if client is None:
                    _LOGGER.debug(
                        "ws_debug_probe: websocket client missing for %s/%s",
                        entry_id,
                        dev_id,
                    )
                    continue
                probe = getattr(client, "debug_probe", None)
                if probe is None:
                    _LOGGER.debug(
                        "ws_debug_probe: client %s/%s has no debug_probe",
                        entry_id,
                        dev_id,
                    )
                    continue
                try:
                    result = probe()
                except TypeError as err:
                    _LOGGER.debug(
                        "ws_debug_probe: client %s/%s probe invocation failed: %s",
                        entry_id,
                        dev_id,
                        err,
                    )
                    continue
                if asyncio.iscoroutine(result):
                    tasks.append(result)
                else:
                    _LOGGER.debug(
                        "ws_debug_probe: client %s/%s returned non-awaitable probe",
                        entry_id,
                        dev_id,
                    )

        if not tasks:
            _LOGGER.debug("ws_debug_probe: no matching websocket clients to probe")
            return

        _LOGGER.debug("ws_debug_probe: awaiting %d websocket probe(s)", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, Exception):
                _LOGGER.debug("ws_debug_probe: probe raised %s", res)

    hass.services.async_register(
        DOMAIN,
        "ws_debug_probe",
        _async_ws_debug_probe,
    )


async def _async_shutdown_entry(rec: MutableMapping[str, Any]) -> None:
    """Cancel websocket tasks and listeners for an integration record."""

    if not isinstance(rec, MutableMapping):
        return

    if rec.get("_shutdown_complete"):
        return

    rec["_shutdown_complete"] = True

    ws_tasks = rec.get("ws_tasks")
    if isinstance(ws_tasks, Mapping):
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

    ws_clients = rec.get("ws_clients")
    if isinstance(ws_clients, Mapping):
        for dev_id, client in list(ws_clients.items()):
            stop = getattr(client, "stop", None)
            if not callable(stop):
                continue
            try:
                await stop()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception("WS client for %s failed to stop", dev_id)

    unsub = rec.get("unsub_ws_status")
    if callable(unsub):
        try:
            unsub()
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception("Failed to unsubscribe websocket status listener")
        rec["unsub_ws_status"] = None


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

    if ok and domain_data is not None and not domain_data:
        diag_state = hass.data.pop(DIAGNOSTICS_HELPER_KEY, None)
        if isinstance(diag_state, MutableMapping):
            listener = diag_state.pop("listener", None)
            if callable(listener):
                try:
                    listener()
                except Exception:  # pragma: no cover - defensive cleanup
                    _LOGGER.debug(
                        "Failed to remove diagnostics listener on unload",
                        exc_info=True,
                    )

    return ok


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate a config entry; no migrations are needed yet."""
    return True


async def async_update_entry_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options updates; recompute interval if needed."""
    rec = hass.data[DOMAIN][entry.entry_id]
    rec["debug"] = bool(entry.options.get("debug", entry.data.get("debug", False)))
    rec["recalc_poll"]()
