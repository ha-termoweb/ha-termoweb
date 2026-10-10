"""Home Assistant entry point for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import functools
import logging
import time
import typing
from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.typing import ConfigType

from .backend import (
    Backend,
    backend_capabilities,
    create_backend,
    create_radio_client,
    create_rest_client,
)
from .backend.radio import RadioLinkError
from .backend.radio.discovery import LISTEN_PLACEHOLDER_NET
from .backend.radio_client import RadioError
from .backend.radio_power import PowerManager
from .backend.rest_client import BackendAuthError, BackendRateLimitError, RESTClient
from .const import (
    BRAND_DUCAHEAT as BRAND_DUCAHEAT,
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
    BRAND_TEVOLVE as BRAND_TEVOLVE,
    CONF_BRAND,
    CONF_DEVICE,
    CONF_DIALECT,
    CONF_HOST,
    CONF_NETWORK_ID,
    CONF_NODES,
    CONF_PORT,
    CONF_RADIO_DEVICE_ID,
    CONF_RADIO_POWER,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    RADIO_BRANDS,
    RADIO_TYPE_NANOCUL,
    signal_ws_status,
)
from .coordinator import (
    DeviceMetadata,
    EnergyStateCoordinator,
    StateCoordinator,
    build_device_metadata,
)
from .domain.ids import NodeType
from .energy import energy_import_store
from .identifiers import build_cloud_unique_id
from .inventory import (
    Inventory,
    build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)
from .runtime import EntryRuntime, TermoWebConfigEntry
from .services.energy_history import async_register_import_energy_history_service
from .services.radio_capture import async_register_radio_capture_service
from .services.radio_pairing import async_register_radio_pairing_services
from .services.radio_survey import async_register_radio_survey_service
from .throttle import reset_samples_rate_limit_state
from .utils import async_get_integration_version as _async_get_integration_version

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["button", "binary_sensor", "climate", "number", "sensor"]
LOCK_PLATFORMS = ["lock"]
MONITOR_PLATFORMS = ["binary_sensor", "sensor"]  # gateway online + frames heard

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

_SETUP_ERRORS = (
    TimeoutError,
    ClientError,
    BackendRateLimitError,
    RadioLinkError,
    RadioError,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the integration's services once; they look up loaded entries."""
    await async_register_import_energy_history_service(hass)
    await async_register_radio_survey_service(hass)
    await async_register_radio_capture_service(hass)
    await async_register_radio_pairing_services(hass)
    return True


def _platforms_for_brand(brand: str) -> list[str]:
    """Return entity platforms enabled for the configured brand's backend."""

    capabilities = backend_capabilities(brand)
    if capabilities.frame_monitor:
        return list(MONITOR_PLATFORMS)
    if capabilities.lock:
        return [*PLATFORMS, *LOCK_PLATFORMS]
    return list(PLATFORMS)


reset_samples_rate_limit_state()


def _log_unknown_node_types(inventory: Inventory) -> None:
    """Log node types the integration does not support yet."""

    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return

    seen: set[tuple[str, str]] = set()
    for node in inventory.nodes:
        node_type = normalize_node_type(
            getattr(node, "type", None),
            use_default_when_falsey=True,
        )
        if not node_type or NodeType.coerce(node_type) is not None:
            continue
        addr = normalize_node_addr(
            getattr(node, "addr", None),
            use_default_when_falsey=True,
        )
        if (node_type, addr) in seen:
            continue
        seen.add((node_type, addr))
        _LOGGER.debug("Unknown node type found: %s/%s", node_type, addr or "<missing>")


async def async_list_devices(client: RESTClient) -> Any:
    """Call ``list_devices`` logging auth/connection issues consistently."""

    try:
        return await client.list_devices()
    except BackendAuthError as err:
        _LOGGER.info("list_devices auth error: %s", err)
        raise
    except (
        TimeoutError,
        ClientError,
        BackendRateLimitError,
        RadioLinkError,
        RadioError,
    ) as err:
        _LOGGER.info("list_devices connection error: %s", err)
        raise


def _create_monitor_client(data: Mapping[str, Any]) -> Any:
    """Return the listen-only radio client of a monitor entry (dialect A to start)."""

    if data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        return create_radio_client(
            data[CONF_DEVICE],
            0,
            "A",
            [],
            LISTEN_PLACEHOLDER_NET,
            serial_url=data[CONF_DEVICE],
            device_id=data.get(CONF_RADIO_DEVICE_ID),
            listen_only=True,
        )
    return create_radio_client(
        data[CONF_HOST],
        int(data[CONF_PORT]),
        "A",
        [],
        LISTEN_PLACEHOLDER_NET,
        listen_only=True,
    )


def _create_client(hass: HomeAssistant, entry: ConfigEntry, brand: str) -> Any:
    """Return the cloud REST client, or the radio gateway client for radio entries."""

    data = entry.data
    if brand == BRAND_RADIO_MONITOR:
        return _create_monitor_client(data)
    if brand == BRAND_RADIO:

        def _save_power(settings: dict[str, Any]) -> None:
            """Store the power manager's settings in the entry options."""

            hass.config_entries.async_update_entry(
                entry, options={**entry.options, CONF_RADIO_POWER: settings}
            )

        power = PowerManager(lambda: entry.options.get(CONF_RADIO_POWER), _save_power)
        if data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
            return create_radio_client(
                data[CONF_DEVICE],
                0,
                data[CONF_DIALECT],
                data.get(CONF_NODES, []),
                bytes.fromhex(data[CONF_NETWORK_ID]),
                power=power,
                serial_url=data[CONF_DEVICE],
                device_id=data.get(CONF_RADIO_DEVICE_ID),
            )
        return create_radio_client(
            data[CONF_HOST],
            int(data[CONF_PORT]),
            data[CONF_DIALECT],
            data.get(CONF_NODES, []),
            bytes.fromhex(data[CONF_NETWORK_ID]),
            power=power,
        )
    return create_rest_client(hass, data["username"], data["password"], brand)


async def async_setup_entry(  # noqa: C901
    hass: HomeAssistant, entry: TermoWebConfigEntry
) -> bool:
    """Set up the TermoWeb integration for a config entry."""
    base_interval = int(DEFAULT_POLL_INTERVAL)
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)

    version = await _async_get_integration_version(hass)

    client = _create_client(hass, entry, brand)
    if brand in RADIO_BRANDS:
        # Release the gateway or serial port if setup fails after connecting;
        # a retry opens a new client, and the gateway serves one at a time.
        entry.async_on_unload(client.async_close)
    backend = create_backend(brand=brand, client=client)
    try:
        devices = await async_list_devices(client)
    except BackendAuthError as err:
        raise ConfigEntryAuthFailed from err
    except _SETUP_ERRORS as err:
        raise ConfigEntryNotReady from err

    if not devices:
        _LOGGER.info("list_devices returned no devices")
        raise ConfigEntryNotReady

    dev: Mapping[str, typing.Any] | None = None
    dev_id = ""
    if isinstance(devices, list):
        usable: list[tuple[str, Mapping[str, typing.Any]]] = []
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
                usable.append((candidate_id, candidate))
                continue
            _LOGGER.debug(
                "Skipping device entry without identifier at index %s: %s",
                index,
                candidate,
            )
        if usable:
            dev_id, dev = usable[0]
        if len(usable) > 1:
            _LOGGER.warning(
                "This account has %d gateways; only the first (%s) is set up. "
                "Ignored: %s",
                len(usable),
                dev.get("name") or dev_id,
                ", ".join(
                    str(ignored.get("name") or ignored_id)
                    for ignored_id, ignored in usable[1:]
                ),
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

    geo_data = None
    try:
        geo_data = await client.get_geo_data(dev_id)
    except Exception:  # noqa: BLE001 - best-effort, must not break setup
        _LOGGER.debug("Failed to fetch geo_data for %s", dev_id)
    if geo_data is not None:
        device_metadata = DeviceMetadata(
            dev_id=device_metadata.dev_id,
            name=device_metadata.name,
            model=device_metadata.model,
            serial_id=device_metadata.serial_id,
            fw_version=device_metadata.fw_version,
            geo_data=geo_data,
        )

    try:
        nodes = await client.get_nodes(dev_id)
    except BackendAuthError as err:
        raise ConfigEntryAuthFailed from err
    except _SETUP_ERRORS as err:
        raise ConfigEntryNotReady from err
    node_inventory = build_node_inventory(nodes)
    # Inventory-centric design: build and freeze the gateway/node topology once
    # during setup so every runtime component can trust the shared metadata.
    inventory = Inventory(dev_id, node_inventory)
    _log_unknown_node_types(inventory)
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
        inventory,
        brand=brand,
        entry_id=entry.entry_id,
    )

    energy_coordinator = EnergyStateCoordinator(
        hass,
        client,
        dev_id,
        inventory,
        state_coordinator=coordinator,
    )
    await energy_coordinator.async_config_entry_first_refresh()
    entry.async_on_unload(energy_coordinator.async_start_hourly_poll())

    runtime = EntryRuntime(
        backend=backend,
        client=backend.client,
        coordinator=coordinator,
        energy_coordinator=energy_coordinator,
        dev_id=dev_id,
        inventory=inventory,
        config_entry=entry,
        base_poll_interval=max(base_interval, MIN_POLL_INTERVAL),
        poll_suspended=False,
        poll_resume_unsub=None,
        ws_tasks={},
        ws_clients={},
        ws_state={},
        ws_trackers={},
        version=version,
        brand=brand,
        boost_runtime={},
        boost_temperature={},
    )
    entry.runtime_data = runtime
    # Runs on unload (after the platforms) and when setup fails from here on,
    # so a ConfigEntryNotReady retry leaves no websocket or listener behind.
    entry.async_on_unload(functools.partial(_async_shutdown_entry, runtime))

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
                coordinator.resume_polling(base_interval)
                runtime.poll_suspended = False
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
                coordinator.resume_polling(base_interval)
                runtime.poll_suspended = False
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
            coordinator.resume_polling(base_interval)
            _LOGGER.info(
                "WS: tracker unhealthy or payload stale; resuming REST polling at %ss",
                base_interval,
            )
        runtime.poll_suspended = False
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
    entry.async_create_background_task(
        hass, _start_ws(dev_id), f"{DOMAIN}-start-ws-{entry.entry_id}"
    )

    platforms = _platforms_for_brand(brand)
    await hass.config_entries.async_forward_entry_setups(entry, platforms)

    _LOGGER.info("TermoWeb setup complete (v%s)", version)
    return True


@dataclass(frozen=True, slots=True)
class _ShutdownTargets:
    """Container for shutdown handles derived from runtime storage."""

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
        ws_tasks=ws_tasks,
        ws_clients=ws_clients,
        unsub_ws_status=runtime.unsub_ws_status,
        poll_resume_unsub=runtime.poll_resume_unsub,
    )


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


async def async_unload_entry(hass: HomeAssistant, entry: TermoWebConfigEntry) -> bool:
    """Unload the platforms; the runtime shuts down afterwards via async_on_unload."""
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)
    return await hass.config_entries.async_unload_platforms(
        entry, _platforms_for_brand(brand)
    )


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the entry's stored energy import progress."""
    await energy_import_store(hass, entry.entry_id).async_remove()


def _migrate_to_1_2(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Case-fold the cloud account unique ID and drop the legacy poll interval."""

    data = {k: v for k, v in entry.data.items() if k != "poll_interval"}
    options = {k: v for k, v in entry.options.items() if k != "poll_interval"}
    unique_id = entry.unique_id
    username = data.get("username")
    if isinstance(username, str) and data.get(CONF_BRAND) not in RADIO_BRANDS:
        folded = build_cloud_unique_id(data.get(CONF_BRAND, DEFAULT_BRAND), username)
        duplicate = hass.config_entries.async_entry_for_domain_unique_id(DOMAIN, folded)
        if duplicate is None or duplicate.entry_id == entry.entry_id:
            unique_id = folded
        else:
            _LOGGER.warning(
                "Entry '%s' is the same account as entry '%s'; delete one of them",
                entry.title,
                duplicate.title,
            )
    hass.config_entries.async_update_entry(
        entry, data=data, options=options, unique_id=unique_id, minor_version=2
    )


def _migrate_to_1_3(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Drop the ``supports_diagnostics`` key that old releases stored."""

    data = {k: v for k, v in entry.data.items() if k != "supports_diagnostics"}
    hass.config_entries.async_update_entry(entry, data=data, minor_version=3)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate a config entry to the current version."""
    if entry.version > 1:
        return False  # downgrade from a newer release
    if entry.minor_version < 2:
        _migrate_to_1_2(hass, entry)
    if entry.minor_version < 3:
        _migrate_to_1_3(hass, entry)
    return True
