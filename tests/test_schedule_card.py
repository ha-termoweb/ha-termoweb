"""Tests for serving the schedule card and its dashboard resource (schedule_card.py)."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

from homeassistant.components.lovelace.const import LOVELACE_DATA
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.schedule_card import (
    CARD_PATH,
    CARD_URL,
    async_remove_schedule_card,
)

VERSION = json.loads((CARD_PATH.parents[1] / "manifest.json").read_text())["version"]
VERSIONED_URL = f"{CARD_URL}?v={VERSION}"


async def _setup_frontend(hass: HomeAssistant, lovelace: dict | None = None) -> None:
    """Set up http and lovelace, which the card registration needs."""
    assert await async_setup_component(hass, "http", {})
    assert await async_setup_component(hass, "lovelace", {"lovelace": lovelace or {}})


def _resource_urls(hass: HomeAssistant) -> list[str]:
    """Return the dashboard resource URLs."""
    return [item["url"] for item in hass.data[LOVELACE_DATA].resources.async_items()]


async def test_setup_serves_card_and_adds_resource(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Setup serves the bundled file and adds one versioned module resource."""
    await _setup_frontend(hass)
    assert await async_setup_component(hass, DOMAIN, {})

    items = hass.data[LOVELACE_DATA].resources.async_items()
    assert [(i["url"], i["type"]) for i in items] == [(VERSIONED_URL, "module")]
    client = await hass_client()
    resp = await client.get(VERSIONED_URL)
    assert resp.status == 200
    assert await resp.text() == CARD_PATH.read_text(encoding="utf-8")


async def test_setup_replaces_old_and_duplicate_resources(hass: HomeAssistant) -> None:
    """An outdated version is updated; manual copies and duplicates are removed."""
    await _setup_frontend(hass)
    resources = hass.data[LOVELACE_DATA].resources
    for url in (
        f"{CARD_URL}?v=2.3.1",
        "/local/termoweb/termoweb_schedule_card.js",
        CARD_URL,
        "/local/other-card.js",
    ):
        await resources.async_create_item({"res_type": "module", "url": url})

    assert await async_setup_component(hass, DOMAIN, {})

    assert sorted(_resource_urls(hass)) == ["/local/other-card.js", VERSIONED_URL]


async def test_setup_keeps_current_resource(hass: HomeAssistant) -> None:
    """A resource that already has the current version is left alone."""
    await _setup_frontend(hass)
    resources = hass.data[LOVELACE_DATA].resources
    item = await resources.async_create_item(
        {"res_type": "module", "url": VERSIONED_URL}
    )

    assert await async_setup_component(hass, DOMAIN, {})

    assert resources.async_items() == [item]


async def test_yaml_resources_are_not_changed(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """In YAML resource mode the file is served and the user is told the URL."""
    await _setup_frontend(hass, {"resource_mode": "yaml"})
    caplog.set_level(logging.INFO, logger="custom_components.termoweb")

    assert await async_setup_component(hass, DOMAIN, {})

    assert _resource_urls(hass) == []
    assert f"add {CARD_URL} as a module resource" in caplog.text
    client = await hass_client()
    assert (await client.get(CARD_URL)).status == 200


async def test_setup_without_http_skips_card(hass: HomeAssistant) -> None:
    """Without the http component there is nothing to serve; setup still works."""
    assert await async_setup_component(hass, DOMAIN, {})

    assert hass.http is None
    assert hass.services.has_service(DOMAIN, "import_energy_history")


async def test_setup_without_lovelace_only_serves_card(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> None:
    """Without dashboards the file is still served and nothing else happens."""
    assert await async_setup_component(hass, "http", {})
    assert await async_setup_component(hass, DOMAIN, {})

    assert LOVELACE_DATA not in hass.data
    client = await hass_client()
    assert (await client.get(CARD_URL)).status == 200


async def test_card_failure_does_not_block_setup(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing card registration is logged; the services still register."""
    with patch(
        "custom_components.termoweb.async_register_schedule_card",
        side_effect=RuntimeError("boom"),
    ):
        assert await async_setup_component(hass, DOMAIN, {})

    assert "Could not register the schedule card" in caplog.text
    assert hass.services.has_service(DOMAIN, "import_energy_history")


async def test_removing_last_entry_removes_resource(hass: HomeAssistant) -> None:
    """The resource stays while an entry remains and goes with the last one."""
    await _setup_frontend(hass)
    first = MockConfigEntry(domain=DOMAIN, entry_id="first")
    second = MockConfigEntry(domain=DOMAIN, entry_id="second")
    first.add_to_hass(hass)
    second.add_to_hass(hass)
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.data[LOVELACE_DATA].resources.async_create_item(
        {"res_type": "module", "url": "/local/other-card.js"}
    )

    await hass.config_entries.async_remove(first.entry_id)
    assert VERSIONED_URL in _resource_urls(hass)

    await hass.config_entries.async_remove(second.entry_id)
    assert _resource_urls(hass) == ["/local/other-card.js"]


async def test_remove_card_ignores_yaml_and_missing_lovelace(
    hass: HomeAssistant,
) -> None:
    """Removal does nothing without lovelace or in YAML resource mode."""
    await async_remove_schedule_card(hass)

    await _setup_frontend(hass, {"resource_mode": "yaml"})
    await async_remove_schedule_card(hass)

    assert _resource_urls(hass) == []
