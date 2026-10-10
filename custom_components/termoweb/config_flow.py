"""Config flow handlers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
import voluptuous as vol

from . import async_list_devices
from .backend import create_rest_client
from .backend.rest_client import BackendAuthError, BackendRateLimitError
from .config_flow_radio import RadioConfigFlow, entry_capabilities
from .const import (
    BRAND_LABELS,
    CONF_BRAND,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DOMAIN,
    RADIO_TYPE_NANOCUL,
    get_brand_label,
)
from .identifiers import build_cloud_unique_id
from .options_flow import TermoWebOptionsFlow
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


class TermoWebConfigFlow(RadioConfigFlow, domain=DOMAIN):
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
        return entry_capabilities(config_entry).options_flow

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

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Change the password (or radio address) of the same account or device."""
        entry = self._get_reconfigure_entry()
        capabilities = entry_capabilities(entry)
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
