"""Config flow handlers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import functools
import logging
from pathlib import Path
from typing import Any

from aiohttp import ClientError
from homeassistant import config_entries
from homeassistant.config_entries import SOURCE_RECONFIGURE, ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
import voluptuous as vol

from . import async_list_devices
from .backend import BackendCapabilities, backend_capabilities, create_rest_client
from .backend.radio import DIALECT_A, DIALECTS, RadioLink, RadioLinkError
from .backend.radio.discovery import (
    DISCOVERY_DIALECTS,
    SCAN_ADDRESSES,
    SURVEY_S,
    NetworkSighting,
    discover_network,
    probe_heaters,
    survey_network,
)
from .backend.radio.link import DEFAULT_PORT as RADIO_DEFAULT_PORT
from .backend.radio.pairing import (
    IDLE_STOP_S,
    PAIR_WINDOW_S,
    PAIRING_DIALECTS,
    PairingError,
    pair_new_network,
)
from .backend.radio.serial_link import serial_device_id, serial_opener
from .backend.radio.survey import RawBurst, SurveyReport, analyse
from .backend.radio_client import dev_id_from_mac
from .backend.radio_power import KEY_RATED_POWER
from .backend.rest_client import BackendAuthError, BackendRateLimitError
from .const import (
    BRAND_LABELS,
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
    CONF_RADIO_POWER,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DOMAIN,
    RADIO_GATEWAY_LABEL,
    RADIO_TYPE_ESP32,
    RADIO_TYPE_NANOCUL,
    get_brand_label,
)
from .identifiers import build_cloud_unique_id
from .radio_pairing import add_nodes, async_site_network_id, radio_node
from .radio_rehome import RehomeError, async_rehome, rehome_summary
from .radio_survey import ISSUE_URL, async_analyse, async_save_report, report_payload
from .runtime import require_runtime
from .utils import async_get_integration_version

_LOGGER = logging.getLogger(__name__)


async def _get_version(hass: HomeAssistant) -> str:
    """Read integration version from manifest (DRY)."""
    return await async_get_integration_version(hass)


def _login_schema(
    default_user: str = "",
    default_brand: str = DEFAULT_BRAND,
) -> vol.Schema:
    """Build the login form schema with provided defaults."""
    return vol.Schema(
        {
            vol.Required(
                CONF_BRAND,
                default=default_brand
                if default_brand in BRAND_LABELS
                else DEFAULT_BRAND,
            ): vol.In(BRAND_LABELS),
            vol.Required("username", default=default_user): str,
            vol.Required("password"): str,
        }
    )


async def _validate_login(
    hass: HomeAssistant, username: str, password: str, brand: str
) -> None:
    """Ensure the provided credentials authenticate successfully."""
    client = create_rest_client(hass, username, password, brand)
    await async_list_devices(client)


async def _login_error(
    hass: HomeAssistant, username: str, password: str, brand: str
) -> str | None:
    """Return the flow error key for a failed login, or None when it succeeds."""
    try:
        await _validate_login(hass, username, password, brand)
    except BackendAuthError:
        return "invalid_auth"
    except BackendRateLimitError:
        return "rate_limited"
    except ClientError, TimeoutError:
        return "cannot_connect"
    except Exception:
        _LOGGER.exception("Unexpected error while logging in")
        return "unknown"
    return None


def _capabilities(entry: ConfigEntry) -> BackendCapabilities:
    """Return the optional features of the backend serving ``entry``."""
    return backend_capabilities(entry.data.get(CONF_BRAND, DEFAULT_BRAND))


def _radio_key(data: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return what identifies the radio connection: the serial port or host:port."""
    if data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        return (RADIO_TYPE_NANOCUL, str(data.get(CONF_DEVICE, "")).strip())
    return (
        RADIO_TYPE_ESP32,
        str(data.get(CONF_HOST, "")).strip().casefold(),
        int(data.get(CONF_PORT, 0)),
    )


def _radio_in_use(
    hass: HomeAssistant, radio: Mapping[str, Any], exclude_entry_id: str | None = None
) -> bool:
    """Return True when another enabled entry already talks to this gateway or stick.

    The ESP32 bridge drops its current client when a new one connects, and a
    serial port has one reader, so probing such a radio breaks the running entry.
    """
    wanted = _radio_key(radio)
    return any(
        entry.entry_id != exclude_entry_id
        and entry.disabled_by is None
        and _capabilities(entry).local_radio
        and _radio_key(entry.data) == wanted
        for entry in hass.config_entries.async_entries(DOMAIN)
    )


def _is_mac_id(dev_id: str | None) -> bool:
    """Return True when ``dev_id`` came from a radio MAC (so it names the hardware)."""
    return dev_id is not None and dev_id_from_mac(dev_id) == dev_id


DIALECT_AUTO = "auto"
RATED_POWER_FIELD = "rated_power_"  # + radio address, in the radio options form
MAX_RATED_POWER_W = 10000
CONF_RESCAN = "rescan"
MANUAL_DEVICE = "manual"  # nanoCUL port choice: type a path or URL instead
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
        )
        for port in list_ports.comports()
    ]


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


def _radio_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the radio gateway form schema with provided defaults."""
    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=defaults.get(CONF_HOST, "")): str,
            vol.Required(
                CONF_PORT, default=defaults.get(CONF_PORT, RADIO_DEFAULT_PORT)
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
            vol.Optional(
                CONF_DIALECT, default=defaults.get(CONF_DIALECT, DIALECT_AUTO)
            ): vol.In([DIALECT_AUTO, *DIALECTS]),
            vol.Optional(
                CONF_NETWORK_ID, default=defaults.get(CONF_NETWORK_ID, "")
            ): str,
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


def _dialect_fields(defaults: Mapping[str, Any]) -> dict[Any, Any]:
    """Return the optional dialect and network id fields shared by radio forms."""
    return {
        vol.Optional(
            CONF_DIALECT, default=defaults.get(CONF_DIALECT, DIALECT_AUTO)
        ): vol.In([DIALECT_AUTO, *DIALECTS]),
        vol.Optional(CONF_NETWORK_ID, default=defaults.get(CONF_NETWORK_ID, "")): str,
    }


def _nanocul_schema(ports: list[SerialPort], defaults: Mapping[str, Any]) -> vol.Schema:
    """Build the nanoCUL port choice: detected ports plus a manual entry."""
    choices = {port.device: f"{port.description} ({port.device})" for port in ports}
    choices[MANUAL_DEVICE] = "Enter the port path or URL myself"
    default = defaults.get(CONF_DEVICE)
    if default not in choices:
        default = next(iter(choices))
    return vol.Schema(
        {
            vol.Required(CONF_DEVICE, default=default): vol.In(choices),
            **_dialect_fields(defaults),
        }
    )


def _nanocul_manual_schema(defaults: Mapping[str, Any]) -> vol.Schema:
    """Build the manual nanoCUL form: a device path or pyserial URL."""
    device = defaults.get(CONF_DEVICE)
    return vol.Schema(
        {
            vol.Required(
                CONF_DEVICE,
                default="" if device in (None, MANUAL_DEVICE) else device,
            ): str,
            **_dialect_fields(defaults),
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


def parse_network_id(text: str | None) -> bytes | None:
    """Return 4 hex digits as two bytes, None when blank; raise ValueError otherwise."""
    cleaned = (text or "").replace(" ", "").replace(":", "").strip()
    if not cleaned:
        return None
    if len(cleaned) != 4:
        raise ValueError("network id must be 4 hex digits")
    return bytes.fromhex(cleaned)


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
    """Learn the network (unless given), then confirm heaters by address.

    When listening hears nothing in a known dialect, a raw survey tells a
    silent installation from one that speaks an unknown dialect.
    """
    if dialect != DIALECT_AUTO and (
        network_id is not None or DIALECTS[dialect].network_id is not None
    ):
        known = DIALECTS[dialect]
        sighting = NetworkSighting(known, network_id or known.network_id)
    else:
        if dialect != DIALECT_AUTO:
            dialects: tuple[Any, ...] = (DIALECTS[dialect],)
        elif dialect_capable:
            dialects = DISCOVERY_DIALECTS
        else:
            dialects = (DIALECT_A,)  # stock nanoCUL firmware: dialect A only
        sighting = await discover_network(
            host, port, dialects=dialects, link_factory=link_factory
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


def radio_entry_data(
    radio: Mapping[str, Any], sighting: NetworkSighting, heaters: dict[int, Any]
) -> dict[str, Any]:
    """Return config entry data for a discovered radio installation."""
    return {
        CONF_BRAND: BRAND_RADIO,
        **_radio_connection(radio),
        CONF_DIALECT: sighting.dialect.name,
        CONF_NETWORK_ID: sighting.network_id.hex().upper(),
        CONF_NODES: [radio_node(addr) for addr in sorted(heaters)],
    }


class TermoWebConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup and (optional) reconfigure without use_push."""

    VERSION = 1
    MINOR_VERSION = 5

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> TermoWebOptionsFlow:
        """Return the options flow handler for this config entry."""
        return TermoWebOptionsFlow(config_entry)

    @classmethod
    @callback
    def async_supports_options_flow(cls, config_entry: ConfigEntry) -> bool:
        """Offer options only to backends that have any (the radio gateway)."""
        return _capabilities(config_entry).options_flow

    def __init__(self) -> None:
        """Initialise radio discovery state."""
        super().__init__()
        self._radio: dict[str, Any] = {}
        self._radio_task: asyncio.Task[Any] | None = None
        self._radio_result: dict[str, Any] | None = None
        self._radio_error: str | None = None
        self._radio_placeholders: dict[str, str] = {}

    async def _handle_login_workflow(
        self,
        *,
        step_id: str,
        user_input: dict[str, Any] | None,
        defaults: dict[str, Any],
        version: str,
    ) -> tuple[FlowResult | None, dict[str, Any]]:
        """Handle shared login form validation and error handling."""

        default_user = (defaults.get("username") or "").strip()
        default_brand = defaults.get(CONF_BRAND, DEFAULT_BRAND)
        if default_brand not in BRAND_LABELS:
            default_brand = DEFAULT_BRAND

        if user_input is None:
            schema = _login_schema(
                default_user=default_user,
                default_brand=default_brand,
            )
            return (
                self.async_show_form(
                    step_id=step_id,
                    data_schema=schema,
                    description_placeholders={"version": version},
                ),
                {},
            )

        username = (user_input.get("username") or default_user).strip()
        password = user_input.get("password") or ""
        brand_in = user_input.get(CONF_BRAND, default_brand)
        brand = brand_in if brand_in in BRAND_LABELS else DEFAULT_BRAND

        errors: dict[str, str] = {}
        if error := await _login_error(self.hass, username, password, brand):
            errors["base"] = error

        if errors:
            schema = _login_schema(
                default_user=username or default_user,
                default_brand=brand,
            )
            return (
                self.async_show_form(
                    step_id=step_id,
                    data_schema=schema,
                    errors=errors,
                    description_placeholders={"version": version},
                ),
                {},
            )

        data = {
            "username": username,
            "password": password,
            CONF_BRAND: brand,
        }
        return None, data

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Offer the cloud account or the local radio gateway."""
        return self.async_show_menu(
            step_id="user", menu_options=["cloud", "radio", "nanocul"]
        )

    async def async_step_cloud(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect credentials and create the config entry."""
        ver = await _get_version(self.hass)
        _LOGGER.info("TermoWeb config flow started (v%s)", ver)

        result, data = await self._handle_login_workflow(
            step_id="cloud",
            user_input=user_input,
            defaults={
                "username": "",
                CONF_BRAND: DEFAULT_BRAND,
            },
            version=ver,
        )

        if result is not None:
            return result

        username = data["username"]
        brand = data[CONF_BRAND]

        await self.async_set_unique_id(build_cloud_unique_id(brand, username))
        self._abort_if_unique_id_configured()

        title = f"{get_brand_label(brand)} ({username})"
        return self.async_create_entry(title=title, data=data)

    async def async_step_radio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Collect the radio gateway address, then discover the heater network."""
        if user_input is None:
            return self.async_show_form(
                step_id="radio", data_schema=_radio_schema(self._radio)
            )
        self._radio = {**user_input, CONF_RADIO_TYPE: RADIO_TYPE_ESP32}
        if _radio_in_use(self.hass, self._radio):
            return self.async_abort(reason="already_in_use")
        errors: dict[str, str] = {}
        try:
            network_id = parse_network_id(user_input.get(CONF_NETWORK_ID))
        except ValueError:
            errors[CONF_NETWORK_ID] = "invalid_network_id"
            network_id = None
        if not errors:
            try:
                dev_id = await probe_gateway(
                    user_input[CONF_HOST], user_input[CONF_PORT]
                )
            except RadioLinkError:
                errors["base"] = "cannot_connect_radio"
            except RadioSetupError as err:
                errors["base"] = err.reason
        if errors:
            return self.async_show_form(
                step_id="radio", data_schema=_radio_schema(self._radio), errors=errors
            )
        await self.async_set_unique_id(f"{BRAND_RADIO}:{dev_id}")
        self._abort_if_unique_id_configured()
        self._radio["network_bytes"] = network_id
        self._radio[_GATEWAY_ID] = dev_id
        return await self.async_step_radio_method()

    async def async_step_radio_method(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Offer to find paired heaters, to pair new ones, or to only listen."""
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
            self._radio_placeholders = {}
            network_id = self._radio.get("network_bytes")
            if network_id is None:
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
        """Listen for heater traffic and scan heaters while showing progress."""
        if self._radio_task is None:
            self._radio_result = None
            self._radio_error = None
            self._radio_placeholders = {}
            host, port = radio_address(self._radio)
            self._radio_task = self.hass.async_create_task(
                discover_radio(
                    host,
                    port,
                    self._radio.get(CONF_DIALECT, DIALECT_AUTO),
                    self._radio.get("network_bytes"),
                    link_factory=radio_link_factory(self._radio),
                    dialect_capable=self._radio.get(_CAPABLE, True),
                    analyse_survey=functools.partial(async_analyse, self.hass),
                )
            )
        if not self._radio_task.done():
            return self.async_show_progress(
                step_id="radio_discover",
                progress_action="radio_discover",
                progress_task=self._radio_task,
            )
        return await self._radio_task_done()

    async def _radio_task_done(self) -> FlowResult:
        """Collect the finished discovery or pairing task and move to radio_finish."""
        task, self._radio_task = self._radio_task, None
        assert task is not None  # only called once the task finished
        try:
            sighting, heaters = task.result()
        except UnknownDialectError as err:
            self._radio_error = err.reason
            await self._save_survey_report(err.report)
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

    async def _save_survey_report(self, report: SurveyReport) -> None:
        """Save an unknown-dialect survey report and point the error text at it."""
        payload = report_payload(
            report,
            seconds=SURVEY_S,
            gateway=None,
            radio_type=self._radio.get(CONF_RADIO_TYPE),
            dialect=self._radio.get(CONF_DIALECT, DIALECT_AUTO),
            version=await _get_version(self.hass),
        )
        path = await async_save_report(self.hass, "setup", payload)
        self._radio_placeholders = {
            "report": path or "(the report could not be saved, see the log)",
            "issue_url": ISSUE_URL,
        }

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
                description_placeholders=self._radio_placeholders or None,
            )
        data = self._radio_result
        entry = self._reconfigure_entry()
        if entry is not None:
            return self.async_update_reload_and_abort(entry, data=data)
        if data[CONF_RADIO_TYPE] == RADIO_TYPE_NANOCUL:
            title = f"{NANOCUL_LABEL} ({data[CONF_DEVICE]})"
        else:
            title = f"{RADIO_GATEWAY_LABEL} ({data[CONF_HOST]})"
        return self.async_create_entry(title=title, data=data)

    def _nanocul_flow(self) -> bool:
        """Return True while the flow sets up (or reconfigures) a nanoCUL."""
        return self._radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL

    def _connect_error(self) -> str:
        """Return the flow error key for a radio that cannot be reached."""
        return (
            "cannot_connect_nanocul" if self._nanocul_flow() else "cannot_connect_radio"
        )

    def _radio_form(self) -> tuple[str, vol.Schema]:
        """Return the step id and schema to show again after a failed discovery."""
        if self._radio.get(_PAIRING):
            return "radio_pair", vol.Schema({})
        if self._reconfigure_entry() is not None:
            if self._nanocul_flow():
                return "reconfigure_nanocul", _reconfigure_nanocul_schema(self._radio)
            return "reconfigure_radio", _reconfigure_radio_schema(self._radio)
        if self._nanocul_flow():
            return "nanocul_manual", _nanocul_manual_schema(self._radio)
        return "radio", _radio_schema(self._radio)

    async def async_step_nanocul(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Choose the nanoCUL's serial port (or a manual path), then discover."""
        ports = await self.hass.async_add_executor_job(list_serial_ports)
        if user_input is None:
            return self.async_show_form(
                step_id="nanocul", data_schema=_nanocul_schema(ports, self._radio)
            )
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
        return await self._start_nanocul(
            "nanocul_manual", user_input, None, _nanocul_manual_schema(user_input)
        )

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
        if _radio_in_use(self.hass, self._radio):
            return self.async_abort(reason="already_in_use")
        errors: dict[str, str] = {}
        network_id = None
        try:
            network_id = parse_network_id(user_input.get(CONF_NETWORK_ID))
        except ValueError:
            errors[CONF_NETWORK_ID] = "invalid_network_id"
        if not errors:
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
                "network_bytes": network_id,
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
        if _radio_in_use(self.hass, self._radio, entry.entry_id):
            return self.async_abort(reason="already_in_use")
        try:
            dev_id, capable = await probe_nanocul(device)
        except RadioLinkError:
            return self.async_show_form(
                step_id="reconfigure_nanocul",
                data_schema=_reconfigure_nanocul_schema(self._radio),
                errors={"base": "cannot_connect_nanocul"},
            )
        if entry.data.get(CONF_DIALECT) != "A" and not capable:
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
        if user_input.get(CONF_RESCAN):
            self._radio["network_bytes"] = bytes.fromhex(entry.data[CONF_NETWORK_ID])
            self._radio[_CAPABLE] = capable
            return await self.async_step_radio_discover()
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
        if _radio_in_use(self.hass, self._radio, entry.entry_id):
            return self.async_abort(reason="already_in_use")
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
        if user_input.get(CONF_RESCAN):
            self._radio["network_bytes"] = bytes.fromhex(entry.data[CONF_NETWORK_ID])
            return await self.async_step_radio_discover()
        return self.async_update_reload_and_abort(
            entry, data_updates={CONF_HOST: host, CONF_PORT: port}
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change the password (or radio address) of the same account or device."""
        entry = self._get_reconfigure_entry()
        capabilities = _capabilities(entry)
        if capabilities.frame_monitor:
            return self.async_abort(reason="monitor_reconfigure")
        if capabilities.local_radio:
            if entry.data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
                return await self.async_step_reconfigure_nanocul()
            return await self.async_step_reconfigure_radio()

        if user_input is not None:
            await self.async_set_unique_id(
                build_cloud_unique_id(
                    user_input.get(CONF_BRAND, DEFAULT_BRAND),
                    user_input.get("username") or "",
                )
            )
            self._abort_if_unique_id_mismatch()

        result, data = await self._handle_login_workflow(
            step_id="reconfigure",
            user_input=user_input,
            defaults={
                "username": entry.data.get("username") or "",
                CONF_BRAND: entry.data.get(CONF_BRAND, DEFAULT_BRAND),
            },
            version=await _get_version(self.hass),
        )
        if result is not None:
            return result
        return self.async_update_reload_and_abort(entry, data_updates=data)

    async def async_step_reauth(self, entry_data: Mapping[str, Any]) -> FlowResult:
        """Start reauthentication after the cloud rejected the stored password."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Ask for the account's new password, then reload the entry with it."""
        entry = self._get_reauth_entry()
        username = entry.data["username"]
        brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)
        errors: dict[str, str] = {}
        if user_input is not None:
            password = user_input["password"]
            if error := await _login_error(self.hass, username, password, brand):
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={"password": password}
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema({vol.Required("password"): str}),
            errors=errors or None,
            description_placeholders={
                "username": username,
                "brand": get_brand_label(brand),
            },
        )


class TermoWebOptionsFlow(config_entries.OptionsFlow):
    """Options flow for radio gateway entries: heater power, pairing, rehoming."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Store the entry being configured."""
        self.entry = entry
        self._pair_task: asyncio.Task[Any] | None = None
        self._pair_error: str | None = None
        self._paired: list[int] = []
        self._rehome_task: asyncio.Task[Any] | None = None
        self._rehome_summary = ""

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Let the user choose settings, pairing or rehoming."""
        return self.async_show_menu(
            step_id="init", menu_options=["settings", "pair_heaters", "rehome"]
        )

    async def async_step_settings(self, user_input: dict[str, Any] | None = None):
        """Show or save the heater power form of a radio entry."""
        radio_addrs = [
            str(node.get("addr")) for node in self.entry.data.get(CONF_NODES, [])
        ]
        power = dict(self.entry.options.get(CONF_RADIO_POWER) or {})
        if user_input is not None:
            # Keep option keys this form does not own (e.g. energy-import progress).
            data: dict[str, Any] = dict(self.entry.options)
            if radio_addrs:
                data[CONF_RADIO_POWER] = {
                    **power,
                    KEY_RATED_POWER: {
                        addr: int(user_input.get(f"{RATED_POWER_FIELD}{addr}", 0))
                        for addr in radio_addrs
                    },
                }
            return self.async_create_entry(title="", data=data)

        fields: dict[Any, Any] = {}
        rated = power.get(KEY_RATED_POWER) or {}
        for addr in radio_addrs:
            fields[
                vol.Optional(
                    f"{RATED_POWER_FIELD}{addr}", default=int(rated.get(addr) or 0)
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=0, max=MAX_RATED_POWER_W))
        ver = await _get_version(self.hass)
        heaters = (
            "Heater power in watts (0 = unknown), used by the power limit: "
            + ", ".join(f"{RATED_POWER_FIELD}{a} = heater {a}" for a in radio_addrs)
            if radio_addrs
            else ""
        )
        return self.async_show_form(
            step_id="settings",
            data_schema=vol.Schema(fields),
            description_placeholders={"version": ver, "heaters": heaters},
        )

    async def async_step_pair_heaters(self, user_input: dict[str, Any] | None = None):
        """Explain pairing mode; start pairing into this entry's network on submit."""
        if user_input is None:
            errors = {"base": self._pair_error} if self._pair_error else {}
            return self.async_show_form(
                step_id="pair_heaters", data_schema=vol.Schema({}), errors=errors
            )
        return await self.async_step_pair_run()

    async def async_step_pair_run(self, user_input: dict[str, Any] | None = None):
        """Pair heaters with the running gateway connection while showing progress."""
        if self._pair_task is None:
            try:
                client = require_runtime(self.hass, self.entry.entry_id).client
            except LookupError:
                return self.async_abort(reason="not_loaded")
            self._pair_task = self.hass.async_create_task(
                client.async_pair(PAIR_WINDOW_S, idle_stop_s=IDLE_STOP_S)
            )
        if not self._pair_task.done():
            return self.async_show_progress(
                step_id="pair_run",
                progress_action="pair_heaters",
                progress_task=self._pair_task,
            )
        task, self._pair_task = self._pair_task, None
        self._pair_error = None
        try:
            paired = task.result()
        except PairingError as err:
            paired = err.paired
            if not paired:
                self._pair_error = "no_free_address"
        except RadioLinkError:
            paired, self._pair_error = [], "cannot_connect_radio"
        except Exception:
            _LOGGER.exception("Unexpected error while pairing radio heaters")
            paired, self._pair_error = [], "unknown"
        self._paired = [heater.node_id for heater in paired]
        if not self._paired and self._pair_error is None:
            self._pair_error = "no_heaters_paired"
        return self.async_show_progress_done(next_step_id="pair_done")

    async def async_step_pair_done(self, user_input: dict[str, Any] | None = None):
        """Add the paired heaters and reload, or show the pairing form with the error."""
        if not self._paired:
            return await self.async_step_pair_heaters()
        add_nodes(self.hass, self.entry, self._paired)
        return self.async_create_entry(title="", data=dict(self.entry.options))

    async def async_step_rehome(self, user_input: dict[str, Any] | None = None):
        """Explain the network move; start it when the user submits."""
        if user_input is None:
            heaters = ", ".join(
                str(node.get("name") or node.get("addr"))
                for node in self.entry.data.get(CONF_NODES, [])
            )
            return self.async_show_form(
                step_id="rehome",
                data_schema=vol.Schema({}),
                description_placeholders={"heaters": heaters},
            )
        return await self.async_step_rehome_run()

    async def async_step_rehome_run(self, user_input: dict[str, Any] | None = None):
        """Reset, re-pair and restore the heaters while showing progress."""
        if self._rehome_task is None:
            try:
                runtime = require_runtime(self.hass, self.entry.entry_id)
            except LookupError:
                return self.async_abort(reason="not_loaded")
            self._rehome_task = self.hass.async_create_task(
                async_rehome(self.hass, runtime, PAIR_WINDOW_S)
            )
        if not self._rehome_task.done():
            return self.async_show_progress(
                step_id="rehome_run",
                progress_action="rehome",
                progress_task=self._rehome_task,
            )
        task, self._rehome_task = self._rehome_task, None
        try:
            self._rehome_summary = rehome_summary(task.result())
        except RehomeError as err:
            self._rehome_summary = str(err)
        except Exception:
            _LOGGER.exception("Unexpected error while moving the radio heaters")
            self._rehome_summary = (
                "Unexpected error. See the Home Assistant log. Heaters that were "
                "reset keep their saved settings for the Radio pair action."
            )
        return self.async_show_progress_done(next_step_id="rehome_done")

    async def async_step_rehome_done(self, user_input: dict[str, Any] | None = None):
        """Show what the move did; close the options when the user confirms."""
        if user_input is None:
            return self.async_show_form(
                step_id="rehome_done",
                data_schema=vol.Schema({}),
                description_placeholders={"summary": self._rehome_summary},
            )
        return self.async_create_entry(title="", data=dict(self.entry.options))
