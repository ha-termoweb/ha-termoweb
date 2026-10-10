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
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
import voluptuous as vol

from . import async_list_devices, create_rest_client
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
    BRAND_DUCAHEAT,
    BRAND_LABELS,
    BRAND_RADIO,
    BRAND_TERMOWEB,
    BRAND_TEVOLVE,
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


DIALECT_AUTO = "auto"
RATED_POWER_FIELD = "rated_power_"  # + radio address, in the radio options form
MAX_RATED_POWER_W = 10000
CONF_RESCAN = "rescan"
MANUAL_DEVICE = "manual"  # nanoCUL port choice: type a path or URL instead
SERIAL_BY_ID_DIR = "/dev/serial/by-id"
NANOCUL_LABEL = "nanoCUL"
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


def radio_entry_data(
    radio: Mapping[str, Any], sighting: NetworkSighting, heaters: dict[int, Any]
) -> dict[str, Any]:
    """Return config entry data for a discovered radio installation."""
    if radio.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        connection: dict[str, Any] = {
            CONF_RADIO_TYPE: RADIO_TYPE_NANOCUL,
            CONF_DEVICE: radio[CONF_DEVICE],
            CONF_RADIO_DEVICE_ID: radio[CONF_RADIO_DEVICE_ID],
        }
    else:
        connection = {
            CONF_RADIO_TYPE: RADIO_TYPE_ESP32,
            CONF_HOST: radio[CONF_HOST],
            CONF_PORT: radio[CONF_PORT],
        }
    return {
        CONF_BRAND: BRAND_RADIO,
        **connection,
        CONF_DIALECT: sighting.dialect.name,
        CONF_NETWORK_ID: sighting.network_id.hex().upper(),
        CONF_NODES: [radio_node(addr) for addr in sorted(heaters)],
        "supports_diagnostics": True,
    }


class TermoWebConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup and (optional) reconfigure without use_push."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> TermoWebOptionsFlow:
        """Return the options flow handler for this config entry."""
        return TermoWebOptionsFlow(config_entry)

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
        try:
            await _validate_login(self.hass, username, password, brand)
        except BackendAuthError:
            errors["base"] = "invalid_auth"
        except BackendRateLimitError:
            errors["base"] = "rate_limited"
        except ClientError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error during %s step", step_id)
            errors["base"] = "unknown"

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
        data["supports_diagnostics"] = True
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

        unique_id = username if brand == BRAND_TERMOWEB else f"{brand}:{username}"
        await self.async_set_unique_id(unique_id)
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
        """Offer to find heaters that are already paired, or to pair new ones."""
        return self.async_show_menu(
            step_id="radio_method", menu_options=["radio_discover", "radio_pair"]
        )

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
            self.hass.config_entries.async_update_entry(entry, data=data)
            return self.async_abort(reason="reconfigure_successful")
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
        entry = self._reconfigure_entry()
        if entry is None:
            return self.async_abort(reason="no_config_entry")
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure_nanocul",
                data_schema=_reconfigure_nanocul_schema(dict(entry.data)),
            )
        device = str(user_input[CONF_DEVICE]).strip()
        self._radio = {**entry.data, CONF_DEVICE: device}
        try:
            _dev_id, capable = await probe_nanocul(device)
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
        if user_input.get(CONF_RESCAN):
            self._radio["network_bytes"] = bytes.fromhex(entry.data[CONF_NETWORK_ID])
            self._radio[_CAPABLE] = capable
            return await self.async_step_radio_discover()
        self.hass.config_entries.async_update_entry(
            entry, data={**entry.data, CONF_DEVICE: device}
        )
        return self.async_abort(reason="reconfigure_successful")

    def _reconfigure_entry(self) -> ConfigEntry | None:
        """Return the entry being reconfigured, or None during initial setup."""
        entry_id = self.context.get("entry_id")
        if self.context.get("source") != "reconfigure" or not entry_id:
            return None
        return self.hass.config_entries.async_get_entry(entry_id)

    async def async_step_reconfigure_radio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change the gateway address; optionally scan for heaters again."""
        entry = self._reconfigure_entry()
        if entry is None:
            return self.async_abort(reason="no_config_entry")
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure_radio",
                data_schema=_reconfigure_radio_schema(dict(entry.data)),
            )
        host, port = user_input[CONF_HOST], user_input[CONF_PORT]
        self._radio = {**entry.data, CONF_HOST: host, CONF_PORT: port}
        try:
            await probe_gateway(host, port)
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
        if user_input.get(CONF_RESCAN):
            self._radio["network_bytes"] = bytes.fromhex(entry.data[CONF_NETWORK_ID])
            return await self.async_step_radio_discover()
        self.hass.config_entries.async_update_entry(
            entry, data={**entry.data, CONF_HOST: host, CONF_PORT: port}
        )
        return self.async_abort(reason="reconfigure_successful")

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Reconfigure username/password (no use_push)."""
        entry_id = self.context.get("entry_id")
        entry: ConfigEntry | None = (
            self.hass.config_entries.async_get_entry(entry_id) if entry_id else None
        )
        if entry is None:
            return self.async_abort(reason="no_config_entry")
        if entry.data.get(CONF_BRAND) == BRAND_RADIO:
            if entry.data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
                return await self.async_step_reconfigure_nanocul()
            return await self.async_step_reconfigure_radio()

        ver = await _get_version(self.hass)

        current_user = entry.data.get("username") or entry.data.get("email") or ""
        current_brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)

        result, data = await self._handle_login_workflow(
            step_id="reconfigure",
            user_input=user_input,
            defaults={
                "username": current_user,
                CONF_BRAND: current_brand,
            },
            version=ver,
        )

        if result is not None:
            return result

        username = data["username"]
        password = data["password"]
        brand = data[CONF_BRAND]

        new_data = dict(entry.data)
        new_data.update(
            {
                "username": username,
                "password": password,
                CONF_BRAND: brand,
            }
        )
        new_data.pop("poll_interval", None)
        new_options = dict(entry.options)
        new_options.pop("poll_interval", None)

        self.hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options
        )
        return self.async_abort(reason="reconfigure_successful")


class TermoWebOptionsFlow(config_entries.OptionsFlow):
    """Options flow: debug and heater power; radio entries can also pair heaters."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Store the entry being configured."""
        self.entry = entry
        self._pair_task: asyncio.Task[Any] | None = None
        self._pair_error: str | None = None
        self._paired: list[int] = []
        self._rehome_task: asyncio.Task[Any] | None = None
        self._rehome_summary = ""

    def _is_radio(self) -> bool:
        """Return True for a radio gateway or nanoCUL entry."""
        return self.entry.data.get(CONF_BRAND) == BRAND_RADIO

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Radio entries choose settings or pairing; others go to the settings form."""
        if self._is_radio() and user_input is None:
            return self.async_show_menu(
                step_id="init", menu_options=["settings", "pair_heaters", "rehome"]
            )
        return await self._settings_step("init", user_input)

    async def async_step_settings(self, user_input: dict[str, Any] | None = None):
        """Show or save the settings form of a radio entry."""
        return await self._settings_step("settings", user_input)

    async def _settings_step(self, step_id: str, user_input: dict[str, Any] | None):
        """Show or process the options form: debug, plus heater power for radio."""
        radio_addrs = (
            [str(node.get("addr")) for node in self.entry.data.get(CONF_NODES, [])]
            if self._is_radio()
            else []
        )
        power = dict(self.entry.options.get(CONF_RADIO_POWER) or {})
        if user_input is not None:
            data: dict[str, Any] = {"debug": bool(user_input.get("debug", False))}
            if radio_addrs:
                data = {**self.entry.options, **data}
                data[CONF_RADIO_POWER] = {
                    **power,
                    KEY_RATED_POWER: {
                        addr: int(user_input.get(f"{RATED_POWER_FIELD}{addr}", 0))
                        for addr in radio_addrs
                    },
                }
            return self.async_create_entry(title="", data=data)

        debug_default = bool(
            self.entry.options.get("debug", self.entry.data.get("debug", False))
        )
        fields: dict[Any, Any] = {vol.Optional("debug", default=debug_default): bool}
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
            step_id=step_id,
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
