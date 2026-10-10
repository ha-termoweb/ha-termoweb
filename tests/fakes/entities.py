"""Heater nodes served by the fake cloud, for climate and lock entity tests."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import (
    climate as climate_module,
    entity as entity_module,
)
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import CONF_BRAND, DOMAIN


class FakeNodes:
    """Nodes and their settings behind the fake cloud; writes are recorded.

    ``nodes`` maps ``(node type, addr)`` to ``(name or None, settings)``.
    Writes update the served settings like the cloud would, so the refresh
    that follows a write (or a later poll) reads them back.
    """

    def __init__(
        self,
        cloud: Any,
        nodes: dict[tuple[str, str], tuple[str | None, dict[str, Any]]],
    ) -> None:
        """Serve ``nodes`` through ``cloud`` and create the write mocks."""
        self.cloud = cloud
        self.settings = {key: settings for key, (_name, settings) in nodes.items()}
        payload = []
        for (node_type, addr), (name, _settings) in nodes.items():
            node: dict[str, Any] = {"type": node_type, "addr": int(addr)}
            if name is not None:
                node["name"] = name
            payload.append(node)
        cloud.get_nodes.return_value = {"nodes": payload}
        cloud.get_node_settings.side_effect = self._get
        self.set_settings = AsyncMock(side_effect=self._set)
        self.set_boost = AsyncMock(side_effect=self._boost)
        self.set_extra_options = AsyncMock(side_effect=self._extra_options)
        self.set_lock = AsyncMock(side_effect=self._lock)

    async def _get(self, _dev_id: str, node: tuple[str, Any]) -> dict[str, Any]:
        """Return a copy of the node's current settings."""
        return dict(self.settings[(node[0], str(node[1]))])

    async def _set(self, _dev_id: str, node: tuple[str, Any], **kwargs: Any) -> None:
        """Apply a settings write to the served settings."""
        current = self.settings[(node[0], str(node[1]))]
        for key in ("mode", "stemp", "prog", "ptemp"):
            if kwargs.get(key) is not None:
                current[key] = kwargs[key]

    async def _boost(
        self, _dev_id: str, addr: Any, *, boost: bool, boost_time: Any = None, **_: Any
    ) -> None:
        """Start or stop a boost on an accumulator."""
        current = self.settings[("acm", str(addr))]
        current["boost_active"] = boost
        current["boost_remaining"] = boost_time if boost else None
        current["mode"] = "boost" if boost else "auto"

    async def _extra_options(
        self, _dev_id: str, addr: Any, *, boost_time: Any, boost_temp: Any
    ) -> None:
        """Store an accumulator's default boost length and temperature."""
        current = self.settings[("acm", str(addr))]
        if boost_time is not None:
            current["boost_time"] = boost_time
        if boost_temp is not None:
            current["boost_temp"] = f"{boost_temp:.1f}"

    async def _lock(self, _dev_id: str, node: tuple[str, Any], *, lock: bool) -> None:
        """Engage or release a node's child lock."""
        self.settings[(node[0], str(node[1]))]["lock"] = lock

    @contextmanager
    def patched(self, *, ducaheat: bool = False) -> Generator[FakeNodes]:
        """Patch the REST client writes (and Ducaheat reads) and drop write delays."""
        client = DucaheatRESTClient if ducaheat else RESTClient
        with ExitStack() as stack:
            if ducaheat:
                cloud = self.cloud
                stack.enter_context(
                    patch.object(client, "get_node_settings", cloud.get_node_settings)
                )
                stack.enter_context(
                    patch.object(client, "get_node_samples", cloud.get_node_samples)
                )
                stack.enter_context(
                    patch.object(
                        DucaheatBackend,
                        "create_ws_client",
                        lambda _self, hass, *a, **kw: cloud.create_ws_client(
                            hass, *a, **kw
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(client, "set_node_lock", self.set_lock)
                )
            for name, mock in (
                ("set_node_settings", self.set_settings),
                ("set_acm_boost_state", self.set_boost),
                ("set_acm_extra_options", self.set_extra_options),
            ):
                stack.enter_context(patch.object(client, name, mock))
            stack.enter_context(patch.object(climate_module, "_WRITE_DEBOUNCE", 0))
            stack.enter_context(
                patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0)
            )
            yield self


async def setup_entry(
    hass: HomeAssistant, entry: MockConfigEntry, *, brand: str | None = None
) -> MockConfigEntry:
    """Add ``entry`` (optionally switched to ``brand``) to hass and set it up."""
    if brand is not None:
        entry = MockConfigEntry(
            domain=DOMAIN,
            title=entry.title,
            unique_id=entry.unique_id,
            data={**entry.data, CONF_BRAND: brand},
        )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def settle(hass: HomeAssistant) -> None:
    """Let background writes and follow-up refreshes run."""
    # The idle fake websocket is a background task too, so
    # wait_background_tasks would never return; yield a few times instead.
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def call(
    hass: HomeAssistant, domain: str, service: str, entity_id: str, **data: Any
) -> None:
    """Call a service on one entity and wait for the follow-up work."""
    await hass.services.async_call(
        domain, service, {ATTR_ENTITY_ID: entity_id, **data}, blocking=True
    )
    await settle(hass)
