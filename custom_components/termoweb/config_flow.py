"""Config flow handlers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import ClientError
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
import voluptuous as vol

from . import async_list_devices, create_rest_client
from .backend.radio import DIALECTS, RadioLink, RadioLinkError
from .backend.radio.discovery import (
    DISCOVERY_DIALECTS,
    SCAN_ADDRESSES,
    NetworkSighting,
    discover_network,
    probe_heaters,
)
from .backend.radio.link import DEFAULT_PORT as RADIO_DEFAULT_PORT
from .backend.radio_client import dev_id_from_mac
from .backend.rest_client import BackendAuthError, BackendRateLimitError
from .const import (
    BRAND_DUCAHEAT,
    BRAND_LABELS,
    BRAND_RADIO,
    BRAND_TERMOWEB,
    BRAND_TEVOLVE,
    CONF_BRAND,
    CONF_DIALECT,
    CONF_HOST,
    CONF_NETWORK_ID,
    CONF_NODES,
    CONF_PORT,
    DEFAULT_BRAND,
    DOMAIN,
    RADIO_GATEWAY_LABEL,
    get_brand_label,
)
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
CONF_RESCAN = "rescan"


class RadioSetupError(Exception):
    """Discovery failed; ``reason`` is the config flow error key."""

    def __init__(self, reason: str) -> None:
        """Store the error key."""
        super().__init__(reason)
        self.reason = reason


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


async def discover_radio(
    host: str, port: int, dialect: str, network_id: bytes | None
) -> tuple[NetworkSighting, dict[int, Any]]:
    """Learn the network (unless given), then confirm heaters by address."""
    if dialect != DIALECT_AUTO and (
        network_id is not None or DIALECTS[dialect].network_id is not None
    ):
        known = DIALECTS[dialect]
        sighting = NetworkSighting(known, network_id or known.network_id)
    else:
        dialects = (
            DISCOVERY_DIALECTS if dialect == DIALECT_AUTO else (DIALECTS[dialect],)
        )
        sighting = await discover_network(host, port, dialects=dialects)
        if sighting is None:
            raise RadioSetupError("no_traffic")
    heaters = await probe_heaters(
        host,
        port,
        sighting.dialect,
        sighting.network_id,
        set(SCAN_ADDRESSES) | sighting.sources,
    )
    if not heaters:
        raise RadioSetupError("no_heaters")
    return sighting, heaters


def radio_entry_data(
    host: str, port: int, sighting: NetworkSighting, heaters: dict[int, Any]
) -> dict[str, Any]:
    """Return config entry data for a discovered radio installation."""
    return {
        CONF_BRAND: BRAND_RADIO,
        CONF_HOST: host,
        CONF_PORT: port,
        CONF_DIALECT: sighting.dialect.name,
        CONF_NETWORK_ID: sighting.network_id.hex().upper(),
        CONF_NODES: [
            {"type": "htr", "addr": str(addr), "name": f"Heater {addr}"}
            for addr in sorted(heaters)
        ],
        "supports_diagnostics": True,
    }


class TermoWebConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup and (optional) reconfigure without use_push."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise radio discovery state."""
        super().__init__()
        self._radio: dict[str, Any] = {}
        self._radio_task: asyncio.Task[Any] | None = None
        self._radio_result: dict[str, Any] | None = None
        self._radio_error: str | None = None

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
        return self.async_show_menu(step_id="user", menu_options=["cloud", "radio"])

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
        self._radio = dict(user_input)
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
        return await self.async_step_radio_discover()

    async def async_step_radio_discover(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Listen for heater traffic and scan heaters while showing progress."""
        if self._radio_task is None:
            self._radio_result = None
            self._radio_error = None
            self._radio_task = self.hass.async_create_task(
                discover_radio(
                    self._radio[CONF_HOST],
                    self._radio[CONF_PORT],
                    self._radio.get(CONF_DIALECT, DIALECT_AUTO),
                    self._radio.get("network_bytes"),
                )
            )
        if not self._radio_task.done():
            return self.async_show_progress(
                step_id="radio_discover",
                progress_action="radio_discover",
                progress_task=self._radio_task,
            )
        task, self._radio_task = self._radio_task, None
        try:
            sighting, heaters = task.result()
        except RadioSetupError as err:
            self._radio_error = err.reason
        except RadioLinkError:
            self._radio_error = "cannot_connect_radio"
        except Exception:
            _LOGGER.exception("Unexpected error during radio discovery")
            self._radio_error = "unknown"
        else:
            self._radio_result = radio_entry_data(
                self._radio[CONF_HOST], self._radio[CONF_PORT], sighting, heaters
            )
        return self.async_show_progress_done(next_step_id="radio_finish")

    async def async_step_radio_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Create (or update) the radio entry, or return to the form with the error."""
        if self._radio_result is None:
            step_id = "reconfigure_radio" if self._reconfigure_entry() else "radio"
            schema = (
                _reconfigure_radio_schema(self._radio)
                if step_id == "reconfigure_radio"
                else _radio_schema(self._radio)
            )
            return self.async_show_form(
                step_id=step_id,
                data_schema=schema,
                errors={"base": self._radio_error or "unknown"},
            )
        data = self._radio_result
        entry = self._reconfigure_entry()
        if entry is not None:
            self.hass.config_entries.async_update_entry(entry, data=data)
            return self.async_abort(reason="reconfigure_successful")
        title = f"{RADIO_GATEWAY_LABEL} ({data[CONF_HOST]})"
        return self.async_create_entry(title=title, data=data)

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
    """Options flow to toggle debug logging."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Store the entry being configured."""
        self.entry = entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Show or process the debug options form."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={"debug": bool(user_input.get("debug", False))},
            )

        debug_default = bool(
            self.entry.options.get("debug", self.entry.data.get("debug", False))
        )
        schema = vol.Schema({vol.Optional("debug", default=debug_default): bool})
        ver = await _get_version(self.hass)
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            description_placeholders={"version": ver},
        )


async def async_get_options_flow(config_entry: ConfigEntry):
    """Return the options flow handler for this config entry."""
    return TermoWebOptionsFlow(config_entry)
