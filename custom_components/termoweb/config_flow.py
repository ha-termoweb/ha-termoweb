"""Config flow handlers for the TermoWeb integration."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import ClientError
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
import voluptuous as vol

from . import async_list_devices, create_rest_client
from .api import BackendAuthError, BackendRateLimitError
from .const import (
    BRAND_DUCAHEAT,
    BRAND_LABELS,
    BRAND_TERMOWEB,
    CONF_BRAND,
    DEFAULT_BRAND,
    DOMAIN,
    get_brand_label,
)
from .utils import async_get_integration_version

__all__ = ["BRAND_DUCAHEAT"]

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
                default=default_brand if default_brand in BRAND_LABELS else DEFAULT_BRAND,
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


class TermoWebConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup and (optional) reconfigure without use_push."""

    VERSION = 1

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

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Collect credentials and create the config entry."""
        ver = await _get_version(self.hass)
        _LOGGER.info("TermoWeb config flow started (v%s)", ver)

        result, data = await self._handle_login_workflow(
            step_id="user",
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

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Reconfigure username/password (no use_push)."""
        entry_id = self.context.get("entry_id")
        entry: ConfigEntry | None = self.hass.config_entries.async_get_entry(entry_id) if entry_id else None
        if entry is None:
            return self.async_abort(reason="no_config_entry")

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

        self.hass.config_entries.async_update_entry(entry, data=new_data, options=new_options)
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
        return self.async_show_form(step_id="init", data_schema=schema, description_placeholders={"version": ver})


async def async_get_options_flow(config_entry: ConfigEntry):
    """Return the options flow handler for this config entry."""
    return TermoWebOptionsFlow(config_entry)
