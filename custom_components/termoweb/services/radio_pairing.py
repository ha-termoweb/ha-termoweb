"""Services that factory-reset radio heaters and pair them (again)."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import voluptuous as vol

from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio.pairing import PAIR_WINDOW_S, PairingError
from custom_components.termoweb.backend.radio_client import (
    RadioClient,
    RadioError,
    RadioUnsupportedError,
    radio_addr,
)
from custom_components.termoweb.const import CONF_NODES, DOMAIN
from custom_components.termoweb.radio_pairing import (
    add_nodes,
    heater_snapshot,
    saved_snapshot,
    store_snapshot,
    with_manual_target,
)
from custom_components.termoweb.radio_rehome import (
    RehomeError,
    async_rehome,
    rehome_summary,
)
from custom_components.termoweb.runtime import EntryRuntime, require_runtime

_LOGGER = logging.getLogger(__name__)

SERVICE_RADIO_FACTORY_RESET = "radio_factory_reset"
SERVICE_RADIO_PAIR = "radio_pair"
SERVICE_RADIO_REHOME = "radio_rehome"
MIN_PAIR_S = 30
MAX_PAIR_S = 600
_HEATER = vol.All(vol.Coerce(int), vol.Range(min=1, max=254))
RADIO_FACTORY_RESET_SCHEMA = vol.Schema(
    {vol.Required("entry_id"): str, vol.Required("heater"): _HEATER}
)
RADIO_PAIR_SCHEMA = vol.Schema(
    {
        vol.Required("entry_id"): str,
        vol.Optional("heater"): _HEATER,
        vol.Optional("restore", default=True): bool,
        vol.Optional("timeout", default=int(PAIR_WINDOW_S)): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_PAIR_S, max=MAX_PAIR_S)
        ),
    }
)


RADIO_REHOME_SCHEMA = vol.Schema(
    {
        vol.Required("entry_id"): str,
        vol.Optional("timeout", default=int(PAIR_WINDOW_S)): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_PAIR_S, max=MAX_PAIR_S)
        ),
    }
)


def _radio_runtime(
    hass: HomeAssistant, entry_id: str
) -> tuple[EntryRuntime, RadioClient]:
    """Return the loaded radio entry's runtime and client, or raise for the user."""

    try:
        runtime = require_runtime(hass, entry_id)
    except LookupError as err:
        raise ServiceValidationError(
            f"No loaded TermoWeb entry with id {entry_id}"
        ) from err
    client = runtime.client
    if not isinstance(client, RadioClient):
        raise ServiceValidationError(
            f"TermoWeb entry {entry_id} does not use a radio gateway"
        )
    if client.listen_only:
        raise ServiceValidationError(
            f"TermoWeb entry {entry_id} is listen-only and never transmits"
        )
    return runtime, client


def _require_node(runtime: EntryRuntime, addr: int) -> None:
    """Raise unless heater ``addr`` is one of the entry's stored nodes."""

    known = set()
    for node in runtime.config_entry.data.get(CONF_NODES, []):
        try:
            known.add(radio_addr(node.get("addr")))
        except ValueError:
            continue
    if addr not in known:
        raise ServiceValidationError(
            f"Heater {addr} is not part of this radio installation"
        )


async def async_register_radio_pairing_services(hass: HomeAssistant) -> None:
    """Register the radio_factory_reset, radio_pair and radio_rehome services once."""

    if hass.services.has_service(DOMAIN, SERVICE_RADIO_PAIR):
        return

    async def _async_factory_reset(call: ServiceCall) -> dict[str, Any]:
        """Save the heater's settings, then factory-reset it (it leaves the network)."""

        entry_id, addr = call.data["entry_id"], int(call.data["heater"])
        runtime, client = _radio_runtime(hass, entry_id)
        _require_node(runtime, addr)
        snapshot = with_manual_target(heater_snapshot(runtime, addr), client, addr)
        try:
            await client.async_factory_reset(addr)
        except RadioUnsupportedError as err:
            raise ServiceValidationError(str(err)) from err
        except (RadioError, RadioLinkError) as err:
            raise HomeAssistantError(f"Factory reset failed: {err}") from err
        if snapshot is not None:
            store_snapshot(hass, runtime.config_entry, addr, snapshot)
        return {"heater": addr, "settings_saved": snapshot is not None}

    async def _async_pair(call: ServiceCall) -> dict[str, Any]:
        """Pair one heater: to its old address (and restore it), or as a new heater."""

        entry_id = call.data["entry_id"]
        timeout = int(call.data.get("timeout", PAIR_WINDOW_S))
        runtime, client = _radio_runtime(hass, entry_id)
        entry = runtime.config_entry
        addr = call.data.get("heater")
        snapshot = None
        if addr is not None:
            addr = int(addr)
            _require_node(runtime, addr)
            snapshot = saved_snapshot(entry, addr) or with_manual_target(
                heater_snapshot(runtime, addr), client, addr
            )
        _LOGGER.info("Radio pairing for %s: waiting up to %d s", entry_id, timeout)
        try:
            paired = await client.async_pair(timeout, wanted_id=addr, max_heaters=1)
        except (PairingError, RadioError, RadioLinkError) as err:
            raise HomeAssistantError(f"Radio pairing failed: {err}") from err
        if not paired:
            raise HomeAssistantError(
                f"No heater was paired within {timeout} s. Put the heater into "
                "pairing mode and try again."
            )
        node_id = paired[0].node_id
        if addr is None:
            add_nodes(hass, entry, [node_id])
            return {"heater": node_id, "added": True, "restored": False}
        restored = False
        if call.data.get("restore", True) and snapshot is not None:
            try:
                await client.async_restore(addr, **snapshot)
            except (RadioError, RadioLinkError, ValueError) as err:
                raise HomeAssistantError(
                    f"Heater {addr} is paired, but restoring its settings failed: {err}"
                ) from err
            store_snapshot(hass, entry, addr, None)
            restored = True
        await runtime.coordinator.async_refresh_heater(("htr", str(addr)))
        return {"heater": addr, "added": False, "restored": restored}

    async def _async_rehome(call: ServiceCall) -> dict[str, Any]:
        """Move every heater onto this installation's own network and restore it."""

        runtime, _client = _radio_runtime(hass, call.data["entry_id"])
        timeout = int(call.data.get("timeout", PAIR_WINDOW_S))
        try:
            result = await async_rehome(hass, runtime, timeout)
        except RehomeError as err:
            raise HomeAssistantError(str(err)) from err
        return {**result, "summary": rehome_summary(result)}

    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_REHOME,
        _async_rehome,
        schema=RADIO_REHOME_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_FACTORY_RESET,
        _async_factory_reset,
        schema=RADIO_FACTORY_RESET_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_PAIR,
        _async_pair,
        schema=RADIO_PAIR_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
