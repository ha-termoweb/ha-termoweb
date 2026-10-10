"""No INFO+ log line carries the raw gateway id during setup and a WS session."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.sanitize import redact_text
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient

from .conftest import DEV_ID, FakeCloud

LOGGER = "custom_components.termoweb"


def _ws_client(entry: MockConfigEntry, hass: HomeAssistant) -> TermoWebWSClient:
    """Return a real (not started) TermoWeb websocket client bound to the entry."""
    runtime = entry.runtime_data
    return TermoWebWSClient(
        hass,
        entry_id=entry.entry_id,
        dev_id=DEV_ID,
        api_client=runtime.client,
        coordinator=runtime.coordinator,
        inventory=runtime.inventory,
    )


async def test_no_raw_dev_id_at_info_or_above(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setup, a WS session with unknown-node frames and unload never log the dev_id."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()

    client = _ws_client(config_entry, hass)
    unknown: dict[str, Any] = {"nodes": {"htr": {"settings": {"9": {"mode": "auto"}}}}}
    for _ in range(3):
        client._apply_nodes_payload(unknown, merge=True, event="update")  # noqa: SLF001

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    ours = [r for r in caplog.records if r.name.startswith(LOGGER)]
    assert ours, "expected the integration to log at INFO"
    leaks = [r.getMessage() for r in ours if DEV_ID in r.getMessage()]
    assert not leaks
    unknown_lines = [r for r in ours if "unknown node" in r.getMessage()]
    assert not unknown_lines, "unknown-node frames must not log at INFO or above"


async def test_unknown_node_logged_once_at_debug(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated frames for a node outside the inventory log a single DEBUG line."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    client = _ws_client(config_entry, hass)
    unknown: dict[str, Any] = {"nodes": {"htr": {"settings": {"9": {"mode": "auto"}}}}}
    for _ in range(3):
        client._apply_nodes_payload(unknown, merge=True, event="update")  # noqa: SLF001

    lines = [r for r in caplog.records if "unknown node" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.DEBUG]


def test_redact_text_masks_gateway_id_in_urls() -> None:
    """REST URLs and messages never expose the gateway id."""
    text = f"GET https://x.example/api/v2/devs/{DEV_ID}/htr/1/settings -> 500"
    redacted = redact_text(text)
    assert DEV_ID not in redacted
    assert "/devs/012345...cdef/htr/1/settings" in redacted
