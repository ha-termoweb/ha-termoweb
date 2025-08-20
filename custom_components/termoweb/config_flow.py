from __future__ import annotations

import logging
from typing import Any

from aiohttp import ClientError
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import aiohttp_client
from homeassistant.loader import async_get_integration
import voluptuous as vol

from .api import TermoWebAuthError, TermoWebClient, TermoWebRateLimitError
from .const import DEFAULT_POLL_INTERVAL, DOMAIN, MAX_POLL_INTERVAL, MIN_POLL_INTERVAL

_LOGGER = logging.getLogger(__name__)


async def _get_version(hass: HomeAssistant) -> str:
    """Read integration version from manifest (DRY)."""
    integ = await async_get_integration(hass, DOMAIN)
    return integ.version or "unknown"


def _poll_schema(default_poll: int) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                "poll_interval",
                default=max(MIN_POLL_INTERVAL, int(default_poll)),
            ): vol.All(int, vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL)),
        }
    )


def _login_schema(default_user: str = "", default_poll: int = DEFAULT_POLL_INTERVAL) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required("username", default=default_user): str,
            vol.Required("password"): str,
            vol.Required(
                "poll_interval",
                default=max(MIN_POLL_INTERVAL, int(default_poll)),
            ): vol.All(int, vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL)),
        }
    )


async def _validate_login(hass: HomeAssistant, username: str, password: str) -> None:
    session = aiohttp_client.async_get_clientsession(hass)
    client = TermoWebClient(session, username, password)
    await client.list_devices()


class TermoWebConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup and (optional) reconfigure without use_push."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        ver = await _get_version(self.hass)
        _LOGGER.info("TermoWeb config flow started (v%s)", ver)

        if user_input is None:
            schema = _login_schema(default_user="", default_poll=DEFAULT_POLL_INTERVAL)
            return self.async_show_form(step_id="user", data_schema=schema, description_placeholders={"version": ver})

        username = (user_input.get("username") or "").strip()
        password = user_input.get("password") or ""
        poll_interval = int(user_input.get("poll_interval", DEFAULT_POLL_INTERVAL))

        errors: dict[str, str] = {}
        try:
            await _validate_login(self.hass, username, password)
        except TermoWebAuthError:
            errors["base"] = "invalid_auth"
        except TermoWebRateLimitError:
            errors["base"] = "rate_limited"
        except ClientError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error in login")
            errors["base"] = "unknown"

        if errors:
            schema = _login_schema(default_user=username, default_poll=poll_interval)
            return self.async_show_form(step_id="user", data_schema=schema, errors=errors, description_placeholders={"version": ver})

        await self.async_set_unique_id(username)
        self._abort_if_unique_id_configured()

        data = {"username": username, "password": password, "poll_interval": poll_interval}
        title = f"Termoweb ({username})"
        return self.async_create_entry(title=title, data=data)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Reconfigure username/password and poll interval (no use_push)."""
        entry_id = self.context.get("entry_id")
        entry: ConfigEntry | None = self.hass.config_entries.async_get_entry(entry_id) if entry_id else None
        if entry is None:
            return self.async_abort(reason="no_config_entry")

        ver = await _get_version(self.hass)

        current_user = entry.data.get("username") or entry.data.get("email") or ""
        current_poll = int(entry.options.get("poll_interval", entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)))

        if user_input is None:
            schema = vol.Schema(
                {
                    vol.Required("username", default=current_user): str,
                    vol.Required("password"): str,
                    vol.Required(
                        "poll_interval",
                        default=max(MIN_POLL_INTERVAL, int(current_poll)),
                    ): vol.All(int, vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL)),
                }
            )
            return self.async_show_form(step_id="reconfigure", data_schema=schema, description_placeholders={"version": ver})

        username = (user_input.get("username") or "").strip()
        password = user_input.get("password") or ""
        poll_interval = int(user_input.get("poll_interval", current_poll))

        errors: dict[str, str] = {}
        try:
            await _validate_login(self.hass, username, password)
        except TermoWebAuthError:
            errors["base"] = "invalid_auth"
        except TermoWebRateLimitError:
            errors["base"] = "rate_limited"
        except ClientError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error during reconfigure")
            errors["base"] = "unknown"

        if errors:
            schema = vol.Schema(
                {
                    vol.Required("username", default=username or current_user): str,
                    vol.Required("password"): str,
                    vol.Required(
                        "poll_interval",
                        default=max(MIN_POLL_INTERVAL, int(poll_interval)),
                    ): vol.All(int, vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL)),
                }
            )
            return self.async_show_form(step_id="reconfigure", data_schema=schema, errors=errors, description_placeholders={"version": ver})

        new_data = dict(entry.data)
        new_data.update({"username": username, "password": password, "poll_interval": poll_interval})
        new_options = dict(entry.options)
        new_options["poll_interval"] = poll_interval

        self.hass.config_entries.async_update_entry(entry, data=new_data, options=new_options)
        return self.async_abort(reason="reconfigure_successful")


class TermoWebOptionsFlow(config_entries.OptionsFlow):
    """Options flow to tweak poll interval only."""

    def __init__(self, entry: ConfigEntry) -> None:
        self.entry = entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        from .const import (  # local import to avoid circulars
            DEFAULT_POLL_INTERVAL,
            MAX_POLL_INTERVAL,
            MIN_POLL_INTERVAL,
        )
        current_poll = int(self.entry.options.get("poll_interval", self.entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)))
        schema = vol.Schema(
            {
                vol.Required(
                    "poll_interval",
                    default=max(MIN_POLL_INTERVAL, int(current_poll)),
                ): vol.All(int, vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL)),
            }
        )
        ver = await _get_version(self.hass)
        return self.async_show_form(step_id="init", data_schema=schema, description_placeholders={"version": ver})


async def async_get_options_flow(config_entry: ConfigEntry):
    return TermoWebOptionsFlow(config_entry)
