"""Serve the bundled schedule card and add it to the dashboard resources.

The card file ships in ``assets/``. Home Assistant serves it at
``/termoweb/termoweb_schedule_card.js``, and the integration adds that URL to
the dashboard resources, with the integration version as a cache-busting query.
The file is served without long cache headers, so YAML-mode users, whose
resource URL has no version, also get the new card after an update.
A resource left from the old manual install (a copy under ``/config/www``)
is replaced, so browsers do not load an outdated copy. Dashboards in YAML
resource mode cannot be changed from code; the user adds the URL there.
"""

from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.components.http import StaticPathConfig
from homeassistant.components.lovelace.const import LOVELACE_DATA, MODE_STORAGE
from homeassistant.core import HomeAssistant
from homeassistant.loader import async_get_integration

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

CARD_FILENAME = "termoweb_schedule_card.js"
CARD_PATH = Path(__file__).parent / "assets" / CARD_FILENAME
CARD_URL = f"/{DOMAIN}/{CARD_FILENAME}"


async def async_register_schedule_card(hass: HomeAssistant) -> None:
    """Serve the card and make sure exactly one dashboard resource loads it."""

    if hass.http is None:
        return
    await hass.http.async_register_static_paths(
        [StaticPathConfig(CARD_URL, str(CARD_PATH), cache_headers=False)]
    )
    lovelace = hass.data.get(LOVELACE_DATA)
    if lovelace is None:
        return
    if lovelace.resource_mode != MODE_STORAGE:
        _LOGGER.info(
            "Dashboard resources are in YAML mode; add %s as a module resource",
            CARD_URL,
        )
        return
    integration = await async_get_integration(hass, DOMAIN)
    url = f"{CARD_URL}?v={integration.version}"
    resources = lovelace.resources
    await resources.async_get_info()  # loads the stored resources
    current = None
    for item in list(resources.async_items()):
        path = item["url"].split("?", 1)[0]
        if path == CARD_URL and current is None:
            current = item
        elif path.endswith(f"/{CARD_FILENAME}"):
            _LOGGER.info("Replacing old schedule card resource %s", item["url"])
            await resources.async_delete_item(item["id"])
    if current is None:
        await resources.async_create_item({"res_type": "module", "url": url})
        _LOGGER.info("Added schedule card resource %s", url)
    elif current["url"] != url:
        await resources.async_update_item(
            current["id"], {"res_type": "module", "url": url}
        )
        _LOGGER.info("Updated schedule card resource to %s", url)


async def async_remove_schedule_card(hass: HomeAssistant) -> None:
    """Remove the card's dashboard resource when the last entry is deleted."""

    lovelace = hass.data.get(LOVELACE_DATA)
    if lovelace is None or lovelace.resource_mode != MODE_STORAGE:
        return
    resources = lovelace.resources
    await resources.async_get_info()
    for item in list(resources.async_items()):
        if item["url"].split("?", 1)[0] == CARD_URL:
            await resources.async_delete_item(item["id"])
