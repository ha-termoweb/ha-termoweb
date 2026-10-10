"""Account and radio-entry helpers for the sensor, binary sensor and button tests."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend import factory
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from tests.fakes.radio_link import FakeRadioLink

BASE_SETTINGS = {"mode": "auto", "state": "off", "stemp": "20.0", "units": "C"}


def serve_nodes(
    cloud: Any,
    nodes: list[dict[str, Any]],
    settings: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Serve ``nodes`` from the fake cloud; return the mutable per-node settings."""
    cloud.get_nodes.return_value = {"nodes": nodes}
    served = {
        (node["type"], str(node["addr"])): {
            **BASE_SETTINGS,
            **(settings or {}).get((node["type"], str(node["addr"])), {}),
        }
        for node in nodes
    }

    async def _get(_dev_id: str, node: tuple[str, Any]) -> dict[str, Any]:
        return dict(served[(node[0], str(node[1]))])

    cloud.get_node_settings.side_effect = _get
    return served


async def setup_entry(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add ``entry`` to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


def entity_ids(hass: HomeAssistant, entry: MockConfigEntry, domain: str) -> set[str]:
    """Return the entity ids of ``domain`` registered for ``entry``."""
    return {
        e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        if e.domain == domain
    }


@contextmanager
def fake_radio_links() -> Generator[list[FakeRadioLink]]:
    """Build real RadioClients whose gateway connections are ``FakeRadioLink``s."""
    links: list[FakeRadioLink] = []
    real_create = factory.create_radio_client

    def _create(*args: Any, **kwargs: Any) -> RadioClient:
        client = real_create(*args, **kwargs)

        def _link(*a: Any, **kw: Any) -> FakeRadioLink:
            links.append(FakeRadioLink(*a, **kw))
            return links[-1]

        client._link_factory = _link  # noqa: SLF001 - the transport boundary
        client.reply_timeout = 0.01  # unanswered polls must not cost real seconds
        return client

    with patch.object(factory, "create_radio_client", _create):
        yield links


def radio_entry(
    brand: str, nodes: list[dict[str, Any]] | None = None
) -> MockConfigEntry:
    """Return a radio gateway (``radio``) or listen-only (``radio_monitor``) entry."""
    data: dict[str, Any] = {"brand": brand, "host": "10.0.0.5", "port": 2323}
    if nodes is not None:
        data.update(dialect="B", network_id="1234", nodes=nodes)
    return MockConfigEntry(
        domain=DOMAIN, minor_version=2, unique_id=f"{brand}:test", data=data
    )
