"""Radio gateway and nanoCUL setup: probing, discovery and config flow steps."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import functools
import logging
from pathlib import Path
import re
from typing import Any

from homeassistant import config_entries
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigEntryState,
)
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo
import voluptuous as vol

from .backend import BackendCapabilities, backend_capabilities
from .backend.radio import DIALECT_A, DIALECTS, RadioLink, RadioLinkError
from .backend.radio.discovery import (
    DISCOVERY_DIALECTS,
    SCAN_ADDRESSES,
    NetworkSighting,
    discover_network,
    probe_heaters,
    survey_network,
)
from .backend.radio.link import DEFAULT_PORT as RADIO_DEFAULT_PORT
from .backend.radio.pairing import PAIRING_DIALECTS, PairedHeater, pair_new_network
from .backend.radio.serial_link import serial_device_id, serial_opener
from .backend.radio.survey import RawBurst, SurveyReport, analyse
from .backend.radio_client import dev_id_from_mac
from .const import (
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
    CONF_BRAND,
    CONF_DEVICE,
    CONF_DIALECT,
    CONF_HOST,
    CONF_NETWORK_ID,
    CONF_NODES,
    CONF_PORT,
    CONF_RADIO_DEVICE_ID,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DOMAIN,
    RADIO_GATEWAY_LABEL,
    RADIO_TYPE_ESP32,
    RADIO_TYPE_NANOCUL,
)
from .radio_pairing import async_site_network_id, radio_node

_LOGGER = logging.getLogger(__name__)


def entry_capabilities(entry: ConfigEntry) -> BackendCapabilities:
    """Return the optional features of the backend serving ``entry``."""
    return backend_capabilities(entry.data.get(CONF_BRAND, DEFAULT_BRAND))


def _is_mac_id(dev_id: str | None) -> bool:
    """Return True when ``dev_id`` came from a radio MAC (so it names the hardware)."""
    return dev_id is not None and dev_id_from_mac(dev_id) == dev_id


DIALECT_AUTO = "auto"
CONF_RESCAN = "rescan"
MANUAL_DEVICE = "manual"  # nanoCUL port choice: type a path or URL instead
ADVANCED = "advanced"  # collapsed form section: port and dialect
SERIAL_BY_ID_DIR = "/dev/serial/by-id"
NANOCUL_LABEL = "nanoCUL"
MONITOR_LABEL = "Radio monitor"  # title of a listen-only entry
_CAPABLE = "dialect_capable"  # flow state: firmware switches dialects at runtime
_GATEWAY_ID = "gateway_id"  # flow state: the gateway's dev_id, seeds the network id
_PAIRING = "pairing"  # flow state: the user chose to pair new heaters


@dataclass(frozen=True)
class SerialPort:
    """A serial port offered for a nanoCUL stick."""

    device: str  # stable /dev/serial/by-id path when there is one
    description: str
    serial_number: str | None
    manufacturer: str | None = None
    product: str | None = None
    usb: bool = False  # a USB serial adapter (not a built-in UART)


def _by_id_path(device: str) -> str:
    """Return the /dev/serial/by-id link for ``device``, else ``device`` itself."""
    try:
        links = sorted(Path(SERIAL_BY_ID_DIR).iterdir())
    except OSError:
        return device
    real = Path(device).resolve()
    for link in links:
        if link.resolve() == real:
            return str(link)
    return device


def list_serial_ports() -> list[SerialPort]:
    """Return the host's serial ports (blocking: run in the executor)."""
    from serial.tools import list_ports  # noqa: PLC0415 - pyserial, USB sticks only

    return [
        SerialPort(
            _by_id_path(port.device),
            port.description or port.device,
            port.serial_number,
            port.manufacturer,
            port.product,
            port.vid is not None,
        )
        for port in list_ports.comports()
    ]


# "nanoCUL", "nano_cul", "CUL868" or "culfw" in a port's name or USB strings.
_CUL_NAME = re.compile(r"(?<![a-z])(?:nano[ _-]?)?cul(?:fw)?(?![a-z])", re.IGNORECASE)


def likely_nanocul_port(ports: Sequence[SerialPort], used: set[str]) -> str | None:
    """Return the port to pre-select for a nanoCUL, or None when unsure.

    A port whose name or USB strings say CUL wins; otherwise the only USB
    serial port that no TermoWeb entry uses. Nothing is opened to decide.
    """
    free = [port for port in ports if port.device not in used]
    for port in free:
        text = " ".join(
            filter(
                None, (port.device, port.description, port.manufacturer, port.product)
            )
        )
        if _CUL_NAME.search(text):
            return port.device
    usb = [port.device for port in free if port.usb]
    return usb[0] if len(usb) == 1 else None


class RadioSetupError(Exception):
    """Discovery failed; ``reason`` is the config flow error key."""

    def __init__(self, reason: str) -> None:
        """Store the error key."""
        super().__init__(reason)
        self.reason = reason


class UnknownDialectError(RadioSetupError):
    """Bursts were heard but no known dialect decoded; carries the survey report."""

    def __init__(self, report: SurveyReport) -> None:
        """Store the report for the flow to save."""
        super().__init__("unknown_dialect")
        self.report = report


AnalyseSurvey = Callable[[Sequence[RawBurst]], Awaitable[SurveyReport]]


async def _analyse_inline(bursts: Sequence[RawBurst]) -> SurveyReport:
    """Analyse survey bursts on the calling loop (tests and fallbacks)."""
    return analyse(bursts)


def _dialect_field(defaults: Mapping[str, Any]) -> dict[Any, Any]:
    """Return the optional radio dialect field (auto by default)."""
    return {
        vol.Optional(
            CONF_DIALECT, default=defaults.get(CONF_DIALECT, DIALECT_AUTO)
        ): vol.In([DIALECT_AUTO, *DIALECTS]),
    }


def _advanced(fields: dict[Any, Any]) -> dict[Any, Any]:
    """Return ``fields`` as a collapsed "advanced" form section."""
    return {
        vol.Required(ADVANCED, default={}): section(
            vol.Schema(fields), {"collapsed": True}
        )
    }


def _flatten(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return form input with the advanced section's fields moved to the top."""
    flat = {key: value for key, value in user_input.items() if key != ADVANCED}
    return {**flat, **user_input.get(ADVANCED, {})}


def _radio_schema(defaults: Mapping[str, Any]) -> vol.Schema:
    """Build the radio gateway form: the address, port and dialect tucked away."""
    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=defaults.get(CONF_HOST, "")): str,
            **_advanced(
                {
                    vol.Optional(
                        CONF_PORT, default=defaults.get(CONF_PORT, RADIO_DEFAULT_PORT)
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
                    **_dialect_field(defaults),
                }
            ),
        }
    )


def _reconfigure_radio_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the radio reconfigure schema: address plus an optional re-scan."""
    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=defaults.get(CONF_HOST, "")): str,
            vol.Required(
                CONF_PORT, default=defaults.get(CONF_PORT, RADIO_DEFAULT_PORT)
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
            vol.Optional(CONF_RESCAN, default=False): bool,
        }
    )


def _nanocul_schema(
    ports: list[SerialPort], defaults: Mapping[str, Any], likely: str | None = None
) -> vol.Schema:
    """Build the nanoCUL port choice: detected ports plus a manual entry."""
    choices = {port.device: f"{port.description} ({port.device})" for port in ports}
    choices[MANUAL_DEVICE] = "Enter the port path or URL myself"
    default = defaults.get(CONF_DEVICE)
    if default not in choices:
        default = likely if likely in choices else None
    if default is None and len(choices) == 1:
        default = MANUAL_DEVICE  # no port detected: typing it is the only choice
    device = (
        vol.Required(CONF_DEVICE)
        if default is None
        else vol.Required(CONF_DEVICE, default=default)
    )
    return vol.Schema({device: vol.In(choices), **_advanced(_dialect_field(defaults))})


def _nanocul_manual_schema(defaults: Mapping[str, Any]) -> vol.Schema:
    """Build the manual nanoCUL form: a device path or pyserial URL."""
    device = defaults.get(CONF_DEVICE)
    return vol.Schema(
        {
            vol.Required(
                CONF_DEVICE,
                default="" if device in (None, MANUAL_DEVICE) else device,
            ): str,
            **_advanced(_dialect_field(defaults)),
        }
    )


def _reconfigure_nanocul_schema(defaults: Mapping[str, Any]) -> vol.Schema:
    """Build the nanoCUL reconfigure schema: port plus an optional re-scan."""
    return vol.Schema(
        {
            vol.Required(CONF_DEVICE, default=defaults.get(CONF_DEVICE, "")): str,
            vol.Optional(CONF_RESCAN, default=False): bool,
        }
    )


async def probe_gateway(host: str, port: int) -> str:
    """Connect to the radio gateway without transmitting; return its ``dev_id``."""
    link = RadioLink(
        host,
        port,
        DIALECTS["A"],
        network_id=b"\x00\x00",
        auto_ack=False,
    )
    info = await link.connect()
    await link.close()
    dev_id = dev_id_from_mac(info.mac)
    if dev_id is None:
        raise RadioSetupError("no_gateway_mac")
    return dev_id


async def probe_nanocul(device: str, usb_serial: str | None = None) -> tuple[str, bool]:
    """Connect to a nanoCUL without transmitting; return ``(dev_id, dialect_capable)``.

    Stock termoweb_rx firmware reports neither a MAC nor runtime dialects: its
    id then comes from the USB serial number (or the path), and only dialect A
    is possible.
    """
    link = RadioLink(
        device,
        0,
        DIALECT_A,
        network_id=b"\x00\x00",
        auto_ack=False,
        open_connection=serial_opener(device),
    )
    info = await link.connect()
    await link.close()
    dev_id = dev_id_from_mac(info.mac) or serial_device_id(device, usb_serial)
    return dev_id, info.dialect is not None


def radio_link_factory(radio: Mapping[str, Any]) -> Any:
    """Return the RadioLink factory for a flow's radio: TCP, or the nanoCUL's port."""
    if radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        return functools.partial(
            RadioLink, open_connection=serial_opener(radio[CONF_DEVICE])
        )
    return RadioLink


def radio_address(radio: Mapping[str, Any]) -> tuple[str, int]:
    """Return the (host, port) label RadioLink uses for a flow's radio."""
    if radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        return radio[CONF_DEVICE], 0
    return radio[CONF_HOST], radio[CONF_PORT]


async def survey_sighting(
    host: str,
    port: int,
    *,
    link_factory: Any = RadioLink,
    analyse_survey: AnalyseSurvey = _analyse_inline,
) -> NetworkSighting | None:
    """Raw-survey after silent discovery; raise UnknownDialectError for new dialects.

    Returns the network a known dialect was heard on, or None when the air was
    silent or the gateway firmware has no survey.
    """
    bursts = await survey_network(host, port, link_factory=link_factory)
    if bursts is None:
        return None
    report = await analyse_survey(bursts)
    _LOGGER.info("Radio survey verdict: %s", report.verdict)
    if report.verdict == "silent":
        return None
    if report.verdict == "known" and report.dialect in DIALECTS:
        known = DIALECTS[report.dialect]
        if report.network_ids:
            return NetworkSighting(known, bytes.fromhex(report.network_ids[0]))
        if known.network_id is not None:
            return NetworkSighting(known, known.network_id)
        return None
    raise UnknownDialectError(report)


async def find_network(
    host: str,
    port: int,
    dialect: str,
    network_id: bytes | None,
    *,
    link_factory: Any = RadioLink,
    dialect_capable: bool = True,
    analyse_survey: AnalyseSurvey = _analyse_inline,
) -> NetworkSighting:
    """Return the heater network: the known one, else learned by listening.

    When listening hears nothing in a known dialect, a raw survey tells a
    silent installation (``no_traffic``) from one that speaks an unknown
    dialect (UnknownDialectError).
    """
    if dialect != DIALECT_AUTO and (
        network_id is not None or DIALECTS[dialect].network_id is not None
    ):
        known = DIALECTS[dialect]
        return NetworkSighting(known, network_id or known.network_id)
    if dialect != DIALECT_AUTO:
        dialects: tuple[Any, ...] = (DIALECTS[dialect],)
    elif dialect_capable:
        dialects = DISCOVERY_DIALECTS
    else:
        dialects = (DIALECT_A,)  # stock nanoCUL firmware: dialect A only
    sighting = await discover_network(
        host,
        port,
        dialects=dialects,
        link_factory=link_factory,
    )
    if sighting is None:
        sighting = await survey_sighting(
            host,
            port,
            link_factory=link_factory,
            analyse_survey=analyse_survey,
        )
    if sighting is None:
        raise RadioSetupError("no_traffic")
    return sighting


async def discover_radio(
    host: str,
    port: int,
    dialect: str,
    network_id: bytes | None,
    *,
    link_factory: Any = RadioLink,
    dialect_capable: bool = True,
    analyse_survey: AnalyseSurvey = _analyse_inline,
) -> tuple[NetworkSighting, dict[int, Any]]:
    """Learn the network (unless given), then confirm heaters by address."""
    sighting = await find_network(
        host,
        port,
        dialect,
        network_id,
        link_factory=link_factory,
        dialect_capable=dialect_capable,
        analyse_survey=analyse_survey,
    )
    heaters = await probe_heaters(
        host,
        port,
        sighting.dialect,
        sighting.network_id,
        set(SCAN_ADDRESSES) | sighting.sources,
        link_factory=link_factory,
    )
    if not heaters:
        raise RadioSetupError("no_heaters")
    return sighting, heaters


async def pair_radio(
    host: str,
    port: int,
    dialect: str,
    network_id: bytes,
    *,
    link_factory: Any = RadioLink,
    dialect_capable: bool = True,
) -> tuple[NetworkSighting, dict[int, Any]]:
    """Pair new heaters into ``network_id``; return them like discover_radio does."""
    if dialect != DIALECT_AUTO:
        dialects: tuple[Any, ...] = (DIALECTS[dialect],)
    elif dialect_capable:
        dialects = PAIRING_DIALECTS
    else:
        dialects = (DIALECT_A,)  # stock nanoCUL firmware: dialect A only
    paired_dialect, paired = await pair_new_network(
        host, port, network_id, dialects=dialects, link_factory=link_factory
    )
    if paired_dialect is None or not paired:
        raise RadioSetupError("no_heaters_paired")
    heaters = {heater.node_id: heater for heater in paired}
    return NetworkSighting(paired_dialect, network_id, frozenset(heaters)), heaters


def _radio_connection(radio: Mapping[str, Any]) -> dict[str, Any]:
    """Return the entry data that says how to reach the gateway or stick."""
    if radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        return {
            CONF_RADIO_TYPE: RADIO_TYPE_NANOCUL,
            CONF_DEVICE: radio[CONF_DEVICE],
            CONF_RADIO_DEVICE_ID: radio[CONF_RADIO_DEVICE_ID],
        }
    return {
        CONF_RADIO_TYPE: RADIO_TYPE_ESP32,
        CONF_HOST: radio[CONF_HOST],
        CONF_PORT: radio[CONF_PORT],
    }


def radio_monitor_entry_data(radio: Mapping[str, Any]) -> dict[str, Any]:
    """Return entry data for a listen-only entry: no dialect, network or heaters."""
    return {
        CONF_BRAND: BRAND_RADIO_MONITOR,
        **_radio_connection(radio),
    }


def radio_new_entry_data(radio: Mapping[str, Any]) -> dict[str, Any]:
    """Return entry data for a gateway whose heaters are found after setup.

    A chosen dialect is kept; its network id only when the dialect has a fixed
    one (A). Without ``network_id`` the network is still to be learned.
    """
    data: dict[str, Any] = {
        CONF_BRAND: BRAND_RADIO,
        **_radio_connection(radio),
        CONF_NODES: [],
    }
    dialect = radio.get(CONF_DIALECT, DIALECT_AUTO)
    if dialect != DIALECT_AUTO:
        data[CONF_DIALECT] = dialect
        if (network_id := DIALECTS[dialect].network_id) is not None:
            data[CONF_NETWORK_ID] = network_id.hex().upper()
    return data


def radio_entry_data(
    radio: Mapping[str, Any], sighting: NetworkSighting, heaters: dict[int, Any]
) -> dict[str, Any]:
    """Return config entry data for a discovered radio installation."""
    return {
        CONF_BRAND: BRAND_RADIO,
        **_radio_connection(radio),
        CONF_DIALECT: sighting.dialect.name,
        CONF_NETWORK_ID: sighting.network_id.hex().upper(),
        CONF_NODES: [
            radio_node(
                addr, heater.serial if isinstance(heater, PairedHeater) else None
            )
            for addr, heater in sorted(heaters.items())
        ],
    }


def _entry_title(data: Mapping[str, Any]) -> str:
    """Return a new radio entry's title: the stick's port or the gateway's address."""
    if data[CONF_RADIO_TYPE] == RADIO_TYPE_NANOCUL:
        return f"{NANOCUL_LABEL} ({data[CONF_DEVICE]})"
    return f"{RADIO_GATEWAY_LABEL} ({data[CONF_HOST]})"


def _probe_error(err: Exception) -> str:
    """Return the form error key for a failed gateway probe."""
    return err.reason if isinstance(err, RadioSetupError) else "cannot_connect_radio"


class RadioConfigFlow(config_entries.ConfigFlow):
    """Radio gateway and nanoCUL steps of the config flow (registered by a subclass)."""

    def __init__(self) -> None:
        """Initialise radio discovery state."""
        super().__init__()
        self._radio: dict[str, Any] = {}
        self._radio_task: asyncio.Task[Any] | None = None
        self._radio_result: dict[str, Any] | None = None
        self._radio_error: str | None = None

    async def async_step_radio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect the radio gateway address, then discover the heater network.

        The fallback for networks without mDNS: normally the gateway is found
        by zeroconf discovery and no address is typed at all.
        """
        if user_input is None:
            return self.async_show_form(
                step_id="radio", data_schema=_radio_schema(self._radio)
            )
        self._radio = {
            CONF_PORT: RADIO_DEFAULT_PORT,
            **_flatten(user_input),
            CONF_RADIO_TYPE: RADIO_TYPE_ESP32,
        }
        self._radio[CONF_HOST] = str(self._radio[CONF_HOST]).strip()
        try:
            dev_id = await probe_gateway(self._radio[CONF_HOST], self._radio[CONF_PORT])
        except (RadioLinkError, RadioSetupError) as err:
            return self.async_show_form(
                step_id="radio",
                data_schema=_radio_schema(self._radio),
                errors={"base": _probe_error(err)},
            )
        await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
        self._abort_if_unique_id_configured()
        self._radio[_GATEWAY_ID] = dev_id
        return await self.async_step_radio_method()

    async def async_step_zeroconf(
        self, discovery_info: ZeroconfServiceInfo
    ) -> FlowResult:
        """Handle an ESP32 radio gateway found by its mDNS advertisement.

        The gateway's id comes from the TXT ``mac`` record (or a probe when it
        is missing). A known gateway at a new address gets its entry updated.
        """
        if discovery_info.ip_address.version != 4:
            return self.async_abort(reason="not_ipv4_address")
        host = str(discovery_info.ip_address)
        port = discovery_info.port or RADIO_DEFAULT_PORT
        _LOGGER.debug(
            "Radio gateway advertised at %s:%s: %s",
            host,
            port,
            discovery_info.properties,
        )
        dev_id = dev_id_from_mac(discovery_info.properties.get("mac"))
        if dev_id is None:
            try:
                dev_id = await probe_gateway(host, port)
            except RadioLinkError:
                return self.async_abort(reason="cannot_connect_radio")
            except RadioSetupError as err:
                return self.async_abort(reason=err.reason)
        await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
        self._update_monitor_address(dev_id, host, port)
        self._abort_if_unique_id_configured(updates={CONF_HOST: host, CONF_PORT: port})
        self._radio = {
            CONF_HOST: host,
            CONF_PORT: port,
            CONF_RADIO_TYPE: RADIO_TYPE_ESP32,
            _GATEWAY_ID: dev_id,
        }
        self.context["title_placeholders"] = {"name": f"{RADIO_GATEWAY_LABEL} ({host})"}
        return await self.async_step_zeroconf_confirm()

    def _update_monitor_address(self, dev_id: str, host: str, port: int) -> None:
        """Move a listen-only entry of this gateway to its new address."""
        entry = self.hass.config_entries.async_entry_for_domain_unique_id(
            DOMAIN, f"{BRAND_RADIO_MONITOR}:{dev_id}"
        )
        if (
            entry is not None
            and self.hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_HOST: host, CONF_PORT: port}
            )
            and entry.state is ConfigEntryState.LOADED
        ):
            self.hass.config_entries.async_schedule_reload(entry.entry_id)

    async def async_step_zeroconf_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Ask before setting up a discovered gateway, then check it answers."""
        placeholders = {"host": self._radio[CONF_HOST]}
        if user_input is None:
            self._set_confirm_only()
            return self.async_show_form(
                step_id="zeroconf_confirm", description_placeholders=placeholders
            )
        try:
            dev_id = await probe_gateway(self._radio[CONF_HOST], self._radio[CONF_PORT])
        except (RadioLinkError, RadioSetupError) as err:
            return self.async_show_form(
                step_id="zeroconf_confirm",
                description_placeholders=placeholders,
                errors={"base": _probe_error(err)},
            )
        if dev_id != self._radio[_GATEWAY_ID]:
            await self.async_set_unique_id(
                f"{BRAND_RADIO}:{dev_id}", raise_on_progress=False
            )
            self._abort_if_unique_id_configured()
            self._radio[_GATEWAY_ID] = dev_id
        return await self.async_step_radio_method()

    async def async_step_radio_method(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Offer to add the gateway now (heaters found later), to pair, or to listen."""
        return self.async_show_menu(
            step_id="radio_method",
            menu_options=["radio_discover", "radio_pair", "radio_monitor"],
        )

    async def async_step_radio_monitor(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create a listen-only entry: it records traffic and never transmits.

        Its own unique id lets a normal entry for the same gateway be added
        later; only one of them can hold the port at a time.
        """
        dev_id = self._radio[_GATEWAY_ID]
        await self.async_set_unique_id(f"{BRAND_RADIO_MONITOR}:{dev_id}")
        self._abort_if_unique_id_configured()
        data = radio_monitor_entry_data(self._radio)
        where = data[CONF_DEVICE] if self._nanocul_flow() else data[CONF_HOST]
        return self.async_create_entry(title=f"{MONITOR_LABEL} ({where})", data=data)

    async def async_step_radio_pair(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Explain pairing mode; start pairing when the user submits."""
        if user_input is None:
            return self.async_show_form(
                step_id="radio_pair", data_schema=vol.Schema({})
            )
        self._radio[_PAIRING] = True
        return await self.async_step_radio_pair_run()

    async def async_step_radio_pair_run(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Pair heaters into the site network while showing progress."""
        if self._radio_task is None:
            self._radio_result = None
            self._radio_error = None
            network_id = await async_site_network_id(
                self.hass, str(self._radio.get(_GATEWAY_ID))
            )
            host, port = radio_address(self._radio)
            self._radio_task = self.hass.async_create_task(
                pair_radio(
                    host,
                    port,
                    self._radio.get(CONF_DIALECT, DIALECT_AUTO),
                    network_id,
                    link_factory=radio_link_factory(self._radio),
                    dialect_capable=self._radio.get(_CAPABLE, True),
                )
            )
        if not self._radio_task.done():
            return self.async_show_progress(
                step_id="radio_pair_run",
                progress_action="radio_pair",
                progress_task=self._radio_task,
            )
        return await self._radio_task_done()

    async def async_step_radio_discover(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create the entry at once; its heaters are found in the background."""
        data = radio_new_entry_data(self._radio)
        return self.async_create_entry(title=_entry_title(data), data=data)

    async def async_step_radio_rescan(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Scan the entry's known network for heaters while showing progress."""
        if self._radio_task is None:
            self._radio_result = None
            self._radio_error = None
            host, port = radio_address(self._radio)
            self._radio_task = self.hass.async_create_task(
                discover_radio(
                    host,
                    port,
                    self._radio[CONF_DIALECT],
                    bytes.fromhex(self._radio[CONF_NETWORK_ID]),
                    link_factory=radio_link_factory(self._radio),
                )
            )
        if not self._radio_task.done():
            return self.async_show_progress(
                step_id="radio_rescan",
                progress_action="radio_rescan",
                progress_task=self._radio_task,
            )
        return await self._radio_task_done()

    async def _radio_task_done(self) -> FlowResult:
        """Collect the finished scan or pairing task and move to radio_finish."""
        task, self._radio_task = self._radio_task, None
        assert task is not None  # only called once the task finished
        try:
            sighting, heaters = task.result()
        except RadioSetupError as err:
            self._radio_error = err.reason
        except RadioLinkError:
            self._radio_error = self._connect_error()
        except Exception:
            _LOGGER.exception("Unexpected error during radio setup")
            self._radio_error = "unknown"
        else:
            self._radio_result = radio_entry_data(self._radio, sighting, heaters)
        return self.async_show_progress_done(next_step_id="radio_finish")

    async def async_step_radio_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create (or update) the radio entry, or return to the form with the error."""
        if self._radio_result is None:
            step_id, schema = self._radio_form()
            return self.async_show_form(
                step_id=step_id,
                data_schema=schema,
                errors={"base": self._radio_error or "unknown"},
            )
        data = self._radio_result
        entry = self._reconfigure_entry()
        if entry is not None:
            return self.async_update_reload_and_abort(entry, data=data)
        return self.async_create_entry(title=_entry_title(data), data=data)

    def _nanocul_flow(self) -> bool:
        """Return True while the flow sets up (or reconfigures) a nanoCUL."""
        return self._radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL

    def _connect_error(self) -> str:
        """Return the flow error key for a radio that cannot be reached."""
        return (
            "cannot_connect_nanocul" if self._nanocul_flow() else "cannot_connect_radio"
        )

    def _radio_form(self) -> tuple[str, vol.Schema]:
        """Return the step id and schema to show again after a failed pairing or rescan."""
        if self._radio.get(_PAIRING):
            return "radio_pair", vol.Schema({})
        if self._nanocul_flow():
            return "reconfigure_nanocul", _reconfigure_nanocul_schema(self._radio)
        return "reconfigure_radio", _reconfigure_radio_schema(self._radio)

    async def async_step_nanocul(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Choose the nanoCUL's serial port (or a manual path), then discover.

        The most likely port is pre-selected without opening any port.
        """
        ports = await self.hass.async_add_executor_job(list_serial_ports)
        if user_input is None:
            likely = likely_nanocul_port(ports, self._used_serial_ports())
            return self.async_show_form(
                step_id="nanocul",
                data_schema=_nanocul_schema(ports, self._radio, likely),
            )
        user_input = _flatten(user_input)
        if user_input[CONF_DEVICE] == MANUAL_DEVICE:
            self._radio = {**user_input, CONF_RADIO_TYPE: RADIO_TYPE_NANOCUL}
            return await self.async_step_nanocul_manual()
        usb_serial = next(
            (p.serial_number for p in ports if p.device == user_input[CONF_DEVICE]),
            None,
        )
        return await self._start_nanocul(
            "nanocul", user_input, usb_serial, _nanocul_schema(ports, user_input)
        )

    async def async_step_nanocul_manual(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Type the nanoCUL's port path or pyserial URL, then discover."""
        if user_input is None:
            return self.async_show_form(
                step_id="nanocul_manual",
                data_schema=_nanocul_manual_schema(self._radio),
            )
        user_input = _flatten(user_input)
        return await self._start_nanocul(
            "nanocul_manual", user_input, None, _nanocul_manual_schema(user_input)
        )

    def _used_serial_ports(self) -> set[str]:
        """Return the serial ports that TermoWeb entries already use."""
        return {
            str(entry.data[CONF_DEVICE])
            for entry in self.hass.config_entries.async_entries(DOMAIN)
            if entry.data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL
            and CONF_DEVICE in entry.data
        }

    async def _start_nanocul(
        self,
        step_id: str,
        user_input: dict[str, Any],
        usb_serial: str | None,
        schema: vol.Schema,
    ) -> FlowResult:
        """Probe the stick, check its firmware can speak the dialect, discover."""
        device = str(user_input[CONF_DEVICE]).strip()
        self._radio = {
            **user_input,
            CONF_DEVICE: device,
            CONF_RADIO_TYPE: RADIO_TYPE_NANOCUL,
        }
        errors: dict[str, str] = {}
        try:
            dev_id, capable = await probe_nanocul(device, usb_serial)
        except RadioLinkError:
            errors["base"] = "cannot_connect_nanocul"
        else:
            wants_b = user_input.get(CONF_DIALECT, DIALECT_AUTO) not in (
                DIALECT_AUTO,
                "A",
            )
            if wants_b and not capable:
                errors["base"] = "dialect_unsupported_firmware"
        if errors:
            return self.async_show_form(
                step_id=step_id, data_schema=schema, errors=errors
            )
        await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
        self._abort_if_unique_id_configured()
        self._radio.update(
            {
                CONF_RADIO_DEVICE_ID: dev_id,
                _CAPABLE: capable,
                _GATEWAY_ID: dev_id,
            }
        )
        return await self.async_step_radio_method()

    async def async_step_reconfigure_nanocul(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change the nanoCUL's port; optionally scan for heaters again."""
        entry = self._get_reconfigure_entry()
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure_nanocul",
                data_schema=_reconfigure_nanocul_schema(dict(entry.data)),
            )
        device = str(user_input[CONF_DEVICE]).strip()
        self._radio = {**entry.data, CONF_DEVICE: device}
        try:
            dev_id, capable = await probe_nanocul(device)
        except RadioLinkError:
            return self.async_show_form(
                step_id="reconfigure_nanocul",
                data_schema=_reconfigure_nanocul_schema(self._radio),
                errors={"base": "cannot_connect_nanocul"},
            )
        dialect = entry.data.get(CONF_DIALECT, DIALECT_AUTO)
        if dialect not in (DIALECT_AUTO, "A") and not capable:
            return self.async_show_form(
                step_id="reconfigure_nanocul",
                data_schema=_reconfigure_nanocul_schema(self._radio),
                errors={"base": "dialect_unsupported_firmware"},
            )
        # A stick without a MAC is known only by its port, so only two MAC ids
        # can prove that the user plugged in a different stick.
        if _is_mac_id(dev_id) and _is_mac_id(entry.data.get(CONF_RADIO_DEVICE_ID)):
            await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
            self._abort_if_unique_id_mismatch()
        if user_input.get(CONF_RESCAN) and CONF_NETWORK_ID in entry.data:
            return await self.async_step_radio_rescan()
        return self.async_update_reload_and_abort(
            entry, data_updates={CONF_DEVICE: device}
        )

    def _reconfigure_entry(self) -> ConfigEntry | None:
        """Return the entry being reconfigured, or None during initial setup."""
        if self.source != SOURCE_RECONFIGURE:
            return None
        return self._get_reconfigure_entry()

    async def async_step_reconfigure_radio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change the gateway address; optionally scan for heaters again."""
        entry = self._get_reconfigure_entry()
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure_radio",
                data_schema=_reconfigure_radio_schema(dict(entry.data)),
            )
        host, port = user_input[CONF_HOST], user_input[CONF_PORT]
        self._radio = {**entry.data, CONF_HOST: host, CONF_PORT: port}
        try:
            dev_id = await probe_gateway(host, port)
        except (RadioLinkError, RadioSetupError) as err:
            reason = (
                err.reason
                if isinstance(err, RadioSetupError)
                else "cannot_connect_radio"
            )
            return self.async_show_form(
                step_id="reconfigure_radio",
                data_schema=_reconfigure_radio_schema(self._radio),
                errors={"base": reason},
            )
        await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
        self._abort_if_unique_id_mismatch()
        if user_input.get(CONF_RESCAN) and CONF_NETWORK_ID in entry.data:
            return await self.async_step_radio_rescan()
        return self.async_update_reload_and_abort(
            entry, data_updates={CONF_HOST: host, CONF_PORT: port}
        )
