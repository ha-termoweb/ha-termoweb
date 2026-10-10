"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import typing
from typing import Any, Final

from homeassistant.const import UnitOfTemperature
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from custom_components.termoweb.backend.sanitize import redact_text
from custom_components.termoweb.boost import (
    ALLOWED_BOOST_MINUTES_SET,
    coerce_boost_minutes,
    supports_boost,
)
from custom_components.termoweb.coerce import as_bool, as_float
from custom_components.termoweb.domain.state import (
    AccumulatorState,
    DomainState,
    HeaterState,
    ThermostatState,
)
from custom_components.termoweb.identifiers import build_heater_unique_id
from custom_components.termoweb.inventory import (
    Inventory,
    Node,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import EntryRuntime, require_runtime
from custom_components.termoweb.utils import build_node_device_info

_LOGGER = logging.getLogger(__name__)

SettingsResolver = Callable[[], DomainState | None]


DEFAULT_BOOST_DURATION: Final = 60
DEFAULT_BOOST_TEMPERATURE: Final = 20.0  # degrees Celsius
# Setpoint range the devices accept, in degrees Celsius.
SETPOINT_MIN_C: Final = 5.0
SETPOINT_MAX_C: Final = 30.0
# Seconds to wait for the WebSocket echo of a write before reading the node by REST.
WS_ECHO_FALLBACK_REFRESH: Final = 4.0


@dataclass(frozen=True, slots=True)
class BoostButtonMetadata:
    """Metadata describing an accumulator boost helper button."""

    minutes: int | None
    unique_suffix: str
    label: str
    icon: str
    action: str = "start"


def _build_boost_button_metadata() -> tuple[BoostButtonMetadata, ...]:
    """Return the configured metadata describing boost helper buttons."""

    return (
        BoostButtonMetadata(
            None,
            "start",
            "Start boost",
            "mdi:flash-outline",
            action="start",
        ),
        BoostButtonMetadata(
            None,
            "cancel",
            "Cancel boost",
            "mdi:flash-off",
            action="cancel",
        ),
    )


BOOST_BUTTON_METADATA: Final[tuple[BoostButtonMetadata, ...]] = (
    _build_boost_button_metadata()
)


@dataclass(frozen=True, slots=True)
class HeaterPlatformDetails:
    """Immutable heater platform metadata resolved from inventory."""

    inventory: Inventory
    default_name_simple: Callable[[str], str]

    @property
    def addrs_by_type(self) -> dict[str, list[str]]:
        """Return heater addresses grouped by node type."""

        forward_map, _ = self.inventory.heater_address_map
        return forward_map

    def iter_metadata(self) -> Iterator[tuple[str, Node, str, str]]:
        """Yield heater metadata derived from the inventory."""

        yield from self.inventory.iter_heater_platform_metadata(
            self.default_name_simple,
        )


def resolve_boost_runtime_minutes(
    state: DomainState | None,
    *,
    default: int = DEFAULT_BOOST_DURATION,
) -> int:
    """Return the device's default boost duration, or ``default`` when unknown."""

    minutes = coerce_boost_minutes(getattr(state, "boost_time", None))
    return minutes if minutes in ALLOWED_BOOST_MINUTES_SET else default


def ws_echo_expected(coordinator: Any) -> bool:
    """Return True when a healthy WebSocket is expected to echo a write."""

    connection = coordinator.domain_view.get_gateway_connection_state()
    return (
        str(connection.status or "").lower() in {"connected", "healthy"}
        and not connection.payload_stale
        and not connection.idle_restart_pending
    )


class NodeRefreshFallback:
    """Refresh one node by REST when the WebSocket echo of a write may not arrive."""

    def __init__(self, entity: Any, node_type: str, addr: str) -> None:
        """Bind the fallback to ``entity`` and its node."""

        self._entity = entity
        self._node = (node_type, addr)
        self._task: asyncio.Task[None] | None = None

    def schedule(self) -> None:
        """Schedule a single-node refresh unless the WebSocket will echo the write."""

        self.cancel()
        coordinator = self._entity.coordinator
        if ws_echo_expected(coordinator):
            _LOGGER.debug("Skipping refresh fallback node=%s: ws healthy", self._node)
            return
        self._task = self._entity.hass.async_create_background_task(
            self._run(coordinator),
            f"termoweb-refresh-fallback-{self._node[0]}-{self._node[1]}",
        )

    def cancel(self) -> None:
        """Cancel a pending fallback refresh."""

        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _run(self, coordinator: Any) -> None:
        """Refresh the node after the WebSocket echo delay."""

        await asyncio.sleep(WS_ECHO_FALLBACK_REFRESH)
        try:
            await coordinator.async_refresh_heater(self._node)
        except Exception as err:  # noqa: BLE001 - a failed refresh must not crash
            _LOGGER.error(
                "Refresh fallback failed node=%s: %s", self._node, redact_text(str(err))
            )


async def async_backend_write(description: str, write: Awaitable[Any]) -> Any:
    """Await a backend write, raising HomeAssistantError when it fails."""

    try:
        return await write
    except HomeAssistantError:
        raise
    except Exception as err:  # any backend failure fails the call
        raise HomeAssistantError(f"{description} failed: {err}") from err


def resolve_state_units(state: DomainState | None) -> str:
    """Return the temperature units (``C``/``F``) reported by ``state``."""

    units_value = getattr(state, "units", None) if state is not None else None
    units = (units_value or "C").upper()
    return "C" if units not in {"C", "F"} else units


def to_device_temperature(celsius: float, units: str) -> float:
    """Return a Celsius temperature expressed in device units ``C``/``F``."""

    return celsius * 9 / 5 + 32 if units == "F" else celsius


def resolve_acm_boost_setpoint(state: DomainState | None) -> float | None:
    """Return the device boost temperature, falling back to the setpoint."""

    boost_temp = as_float(getattr(state, "boost_temp", None))
    if boost_temp is None:
        boost_temp = as_float(getattr(state, "stemp", None))
    return boost_temp


async def async_start_acm_boost(
    backend: Any,
    dev_id: str,
    addr: str,
    state: DomainState | None,
    *,
    minutes: int,
) -> None:
    """Start an accumulator boost for ``minutes`` at the device boost setpoint."""

    setpoint = resolve_acm_boost_setpoint(state)
    if setpoint is None:
        raise ValueError(f"no boost temperature or setpoint known for acm {addr}")
    await backend.set_acm_boost_state(
        dev_id,
        addr,
        boost=True,
        boost_time=minutes,
        stemp=setpoint,
        units=resolve_state_units(state),
    )


async def async_cancel_acm_boost(backend: Any, dev_id: str, addr: str) -> None:
    """Cancel an active accumulator boost."""

    await backend.set_acm_boost_state(dev_id, addr, boost=False)


def iter_boostable_heater_nodes(
    details: HeaterPlatformDetails,
    *,
    accumulators_only: bool = False,
) -> Iterator[tuple[str, Node, str, str]]:
    """Yield heater nodes that expose boost functionality."""

    for node_type, node, addr_str, base_name in details.iter_metadata():
        if accumulators_only and node_type != "acm":
            continue
        if supports_boost(node):
            yield node_type, node, addr_str, base_name


@dataclass(slots=True)
class BoostState:
    """Derived boost metadata for a heater node."""

    active: bool | None
    minutes_remaining: int | None
    end_datetime: datetime | None
    end_iso: str | None
    end_label: str | None


def _derive_boost_state(
    source: Any,
    coordinator: Any,
) -> BoostState:
    """Return derived boost metadata for ``source`` using ``coordinator``."""

    def _get_field(field: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(field)
        return getattr(source, field, None)

    def _parse_iso_timestamp(value: str) -> datetime | None:
        """Parse an ISO timestamp string defensively."""

        return dt_util.parse_datetime(value)

    boost_active = as_bool(_get_field("boost_active"))
    if boost_active is None:
        mode = _get_field("mode")
        if isinstance(mode, str):
            boost_active = mode.strip().lower() == "boost"
        else:
            boost_active = False

    boost_day: Any = _get_field("boost_end_day")
    boost_minute: Any = _get_field("boost_end_min")

    boost_end_dt: datetime | None = None
    derived_dt = _get_field("boost_end_datetime")
    if isinstance(derived_dt, datetime):
        boost_end_dt = derived_dt
    elif isinstance(derived_dt, str):
        boost_end_dt = _parse_iso_timestamp(derived_dt)

    boost_minutes: int | None = coerce_boost_minutes(_get_field("boost_minutes_delta"))
    resolver = getattr(coordinator, "resolve_boost_end", None)
    # The coordinator stores the derived end time and minutes as a pair, so
    # resolve both from the raw day/minute fields only when neither is cached.
    if (
        callable(resolver)
        and boost_day is not None
        and boost_minute is not None
        and boost_end_dt is None
        and boost_minutes is None
    ):
        try:
            boost_end_dt, boost_minutes = resolver(boost_day, boost_minute)
        except Exception:  # noqa: BLE001 - defensive
            boost_end_dt = None
            boost_minutes = None

    if boost_minutes is None:
        boost_minutes = coerce_boost_minutes(_get_field("boost_remaining"))

    if boost_minutes is None and boost_end_dt is not None:
        delta_seconds = (boost_end_dt - dt_util.now()).total_seconds()
        boost_minutes = int(max(0.0, delta_seconds) // 60)

    if boost_minutes is not None and boost_minutes <= 0:
        boost_minutes = None

    if boost_end_dt is None and boost_minutes is not None:
        boost_end_dt = dt_util.now() + timedelta(minutes=boost_minutes)
    boost_end_iso = boost_end_dt.isoformat() if boost_end_dt is not None else None

    placeholder_iso = boost_end_iso.strip() if isinstance(boost_end_iso, str) else None
    placeholder_detected = False
    if (boost_end_dt is not None and boost_end_dt.year <= 1971) or (
        placeholder_iso and placeholder_iso.startswith("1970-")
    ):
        placeholder_detected = True

    if placeholder_detected:
        boost_end_dt = None
        boost_end_iso = None

    end_label: str | None = None
    if boost_active is False and boost_end_dt is None and boost_end_iso is None:
        end_label = "Never"

    return BoostState(
        active=boost_active,
        minutes_remaining=boost_minutes,
        end_datetime=boost_end_dt,
        end_iso=boost_end_iso,
        end_label=end_label,
    )


def derive_boost_state(
    settings: Mapping[str, typing.Any] | None, coordinator: Any
) -> BoostState:
    """Return derived boost metadata for mapping ``settings``."""

    source = settings if isinstance(settings, Mapping) else {}
    return _derive_boost_state(source, coordinator)


def derive_boost_state_from_domain(
    state: DomainState | None, coordinator: Any
) -> BoostState:
    """Return derived boost metadata for typed ``state``."""

    return _derive_boost_state(state or {}, coordinator)


# ruff: enable=C901


def log_skipped_nodes(
    platform_name: str,
    inventory: Inventory | HeaterPlatformDetails,
    *,
    logger: logging.Logger | None = None,
    skipped_types: Iterable[str] = ("pmo",),
) -> None:
    """Log skipped TermoWeb nodes for a given platform."""

    log = logger or _LOGGER
    if isinstance(inventory, HeaterPlatformDetails):
        inventory = inventory.inventory
    addresses_by_type = inventory.addresses_by_type

    for node_type in skipped_types:
        addresses = addresses_by_type.get(node_type, [])
        if not addresses:
            continue
        log.debug(
            "Skipping TermoWeb %s nodes for %s platform: %s",
            node_type,
            platform_name,
            ", ".join(sorted(addresses)),
        )


def heater_platform_details_for_entry(
    runtime: EntryRuntime,
    *,
    default_name_simple: Callable[[str], str],
) -> HeaterPlatformDetails:
    """Return heater platform metadata derived from ``runtime``."""

    return HeaterPlatformDetails(
        inventory=runtime.inventory,
        default_name_simple=default_name_simple,
    )


def boostable_accumulator_details_for_entry(
    runtime: EntryRuntime,
    *,
    default_name_simple: Callable[[str], str],
    platform_name: str,
    logger: logging.Logger | None = None,
    accumulators_only: bool = True,
) -> tuple[HeaterPlatformDetails, list[tuple[str, str, str]]]:
    """Return boostable accumulator metadata for a config entry."""

    details = heater_platform_details_for_entry(
        runtime,
        default_name_simple=default_name_simple,
    )

    metadata: list[tuple[str, str, str]] = [
        (node_type, addr_str, base_name)
        for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
            details,
            accumulators_only=accumulators_only,
        )
    ]

    log_skipped_nodes(platform_name, details, logger=logger)

    return details, metadata


def build_settings_resolver(
    coordinator: Any,
    dev_id: str,
    node_type: str,
    addr: str,
) -> SettingsResolver:
    """Return callable resolving the domain state for a node."""

    def _resolver() -> DomainState | None:
        return coordinator.domain_view.get_heater_state(node_type, addr)

    return _resolver


class HeaterNodeBase(CoordinatorEntity):
    """Base entity implementing common TermoWeb heater behaviour."""

    def __init__(
        self,
        coordinator: Any,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str | None,
        unique_id: str | None = None,
        *,
        device_name: str | None = None,
        node_type: str | None = None,
        inventory: Inventory,
    ) -> None:
        """Initialise a heater entity tied to a TermoWeb device."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = normalize_node_addr(addr)
        if name is not None:
            self._attr_name = name
        resolved_type = (
            normalize_node_type(
                node_type,
                default="htr",
                use_default_when_falsey=True,
            )
            or "htr"
        )
        self._node_type = resolved_type
        self._attr_unique_id = unique_id or build_heater_unique_id(
            dev_id, resolved_type, self._addr
        )
        self._device_name = device_name or name
        self._inventory = inventory

    @property
    def should_poll(self) -> bool:
        """Home Assistant should not poll heater entities."""
        return False

    @property
    def available(self) -> bool:
        """Return True when the last update succeeded and the node is in inventory."""
        return super().available and self._device_available()

    def _device_available(self) -> bool:
        """Return True when the immutable inventory exposes this node."""

        return self._inventory.has_node(self._node_type, self._addr)

    def _heater_state(self) -> DomainState | None:
        """Return the current cached domain state."""

        return self.coordinator.domain_view.get_heater_state(
            self._node_type, self._addr
        )

    def heater_state(self) -> HeaterState | AccumulatorState | ThermostatState | None:
        """Return the typed heater state for this entity."""

        state = self._heater_state()
        if isinstance(state, (HeaterState, AccumulatorState, ThermostatState)):
            return state
        return None

    def accumulator_state(self) -> AccumulatorState | None:
        """Return the accumulator state when present."""

        state = self.heater_state()
        return state if isinstance(state, AccumulatorState) else None

    def boost_state(self) -> BoostState:
        """Return derived boost metadata for this heater."""

        return derive_boost_state_from_domain(self.heater_state(), self.coordinator)

    def _client(self) -> Any:
        """Return the backend used for write operations."""
        try:
            return require_runtime(self.hass, self._entry_id).backend
        except LookupError:
            return None

    def _units(self) -> str:
        """Return the configured temperature units for this heater."""
        return resolve_state_units(self.heater_state())

    def _temperature_unit(self) -> UnitOfTemperature:
        """Return the HA temperature unit matching the device's units."""
        if self._units() == "F":
            return UnitOfTemperature.FAHRENHEIT
        return UnitOfTemperature.CELSIUS

    def _setpoint_range(self) -> tuple[float, float]:
        """Return the (min, max) setpoint in the device's units."""
        units = self._units()
        return (
            to_device_temperature(SETPOINT_MIN_C, units),
            to_device_temperature(SETPOINT_MAX_C, units),
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the heater."""
        return build_node_device_info(
            self.hass,
            self._entry_id,
            self._dev_id,
            self._addr,
            name=self._device_name,
            node_type=self._node_type,
        )
