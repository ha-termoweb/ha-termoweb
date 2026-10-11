"""Radio heaters are named and identified by serial number, not radio address."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import radio_identity
from custom_components.termoweb.backend.radio.pairing import PairedHeater
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.nodes import build_node_inventory
from custom_components.termoweb.radio_discovery import HeaterFinder
from tests.fakes.radio_link import IDENTITY_RATED, MAC, NET, STATUS_SHORT, FakeRadioLink
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry

DEV = MAC.replace(":", "").lower()
SERIAL = "X123456"  # the synthetic ASCII serial in IDENTITY_RATED
CLIMATE_BY_ADDR = f"{DOMAIN}:{DEV}:htr:6:climate"
CLIMATE_BY_SERIAL = f"{DOMAIN}:{DEV}:htr:{SERIAL}:climate"
ADDRESS_BY_SERIAL = f"{DOMAIN}:{DEV}:htr:{SERIAL}:radio_address"


def _script(identity: bool = True) -> Any:
    """Return a heater script; ``answers["identity"]`` turns the 5A reply on/off."""
    answers = {"identity": identity}

    def _heater(link: FakeRadioLink) -> None:
        link.reply(0xB8, STATUS_SHORT)
        if answers["identity"]:
            link.reply(0x5A, IDENTITY_RATED)

    _heater.answers = answers  # type: ignore[attr-defined]
    return _heater


def _entities(hass: HomeAssistant, entry: MockConfigEntry) -> dict[str, str]:
    """Return ``{unique_id: entity_id}`` of the entry's registered entities."""
    return {
        e.unique_id: e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    }


def _heater_devices(hass: HomeAssistant, entry: MockConfigEntry) -> dict[str, set]:
    """Return ``{device id: identifiers}`` of the entry's heater devices."""
    return {
        device.id: set(device.identifiers)
        for device in dr.async_entries_for_config_entry(
            dr.async_get(hass), entry.entry_id
        )
        if device.model == "Heater"
    }


async def _reload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Reload the entry and let it settle."""
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Unload the entry so no radio task outlives the test."""
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_setup_reads_a_missing_serial_once_and_names_the_heater(
    hass: HomeAssistant,
) -> None:
    """The serial is read before the inventory, stored, and names the heater."""
    entry = radio_entry("radio", [{"type": "htr", "addr": "6", "name": "Heater 6"}])
    with fake_radio_links(_script()) as links:
        await setup_entry(hass, entry)

        assert entry.data["nodes"] == [
            {"type": "htr", "addr": "6", "name": f"Heater {SERIAL}", "serial": SERIAL}
        ]
        assert links[0].payloads()[0] == b"\x5a"
        assert links[0].payloads().count(b"\x5a") == 1
        device = dr.async_get(hass).async_get_device_by_identifier(
            (DOMAIN, DEV, SERIAL), entry.entry_id
        )
        assert device is not None and device.name == f"Heater {SERIAL}"
        assert _entities(hass, entry)[CLIMATE_BY_SERIAL] == "climate.heater_x123456"
        assert CLIMATE_BY_ADDR not in _entities(hass, entry)

        await _reload(hass, entry)  # the serial is stored: not read again at setup
        assert links[-1].payloads()[0] != b"\x5a"
        await _unload(hass, entry)


async def test_unanswered_identity_keeps_the_address_until_the_next_setup(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent heater keeps address IDs; once it answers they move to its serial."""
    caplog.set_level(logging.INFO)
    entry = radio_entry("radio", [{"type": "htr", "addr": "6", "name": "Heater 6"}])
    script = _script(identity=False)
    with fake_radio_links(script):
        await setup_entry(hass, entry)
        assert entry.data["nodes"] == [{"type": "htr", "addr": "6", "name": "Heater 6"}]
        assert "Heater 6 did not report its serial number" in caplog.text
        before = _entities(hass, entry)
        assert before[CLIMATE_BY_ADDR] == "climate.heater_6"
        [device_id] = _heater_devices(hass, entry)
        ent_reg = er.async_get(hass)
        ent_reg.async_update_entity("climate.heater_6", name="Lounge")  # customised

        script.answers["identity"] = True
        await _reload(hass, entry)

        assert entry.data["nodes"][0]["serial"] == SERIAL
        after = _entities(hass, entry)
        assert after[CLIMATE_BY_SERIAL] == "climate.heater_6"  # entity_id kept
        assert CLIMATE_BY_ADDR not in after
        assert sorted(after.values()) == sorted(before.values())  # no duplicates
        assert ent_reg.async_get("climate.heater_6").name == "Lounge"
        assert _heater_devices(hass, entry) == {device_id: {(DOMAIN, DEV, SERIAL)}}
        await _unload(hass, entry)


async def test_repaired_heater_keeps_its_entities_and_shows_its_new_address(
    hass: HomeAssistant,
) -> None:
    """Pairing a known serial again moves its node; devices and entities stay."""
    node = {"type": "htr", "addr": "6", "name": f"Heater {SERIAL}", "serial": SERIAL}
    entry = radio_entry("radio", [node])
    with fake_radio_links(_script()) as links:
        await setup_entry(hass, entry)
        ent_reg = er.async_get(hass)
        address = ent_reg.async_get(_entities(hass, entry)[ADDRESS_BY_SERIAL])
        assert address.entity_id == "sensor.heater_x123456_radio_address"
        assert address.disabled_by is er.RegistryEntryDisabler.INTEGRATION
        assert address.entity_category is EntityCategory.DIAGNOSTIC
        ent_reg.async_update_entity(address.entity_id, disabled_by=None)
        await _reload(hass, entry)
        assert hass.states.get(address.entity_id).state == "6"
        entities, devices = _entities(hass, entry), _heater_devices(hass, entry)
        assert entities[CLIMATE_BY_SERIAL] == "climate.heater_x123456"
        [device_id] = devices
        assert dr.async_get(hass).async_get(device_id).serial_number == SERIAL

        repaired = PairedHeater(2, SimpleNamespace(), serial=SERIAL)
        # Automatic pairing hands a re-paired heater to the heater finder.
        await HeaterFinder(hass, entry, None).async_add([], paired=[repaired])
        await hass.async_block_till_done()  # the scheduled reload

        assert entry.data["nodes"] == [{**node, "addr": "2"}]
        assert _entities(hass, entry) == entities
        assert _heater_devices(hass, entry) == devices
        assert any(dst == 2 for dst, _payload in links[-1].sent)
        assert hass.states.get(address.entity_id).state == "2"  # the new address
        await _unload(hass, entry)


def _seed(
    hass: HomeAssistant, entry_id: str, key: str, domain: str = "climate"
) -> tuple[str, str]:
    """Register a heater device and climate entity keyed by ``key``."""
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry_id, identifiers={(DOMAIN, DEV, key)}
    )
    entity = er.async_get(hass).async_get_or_create(
        domain,
        DOMAIN,
        f"{DOMAIN}:{DEV}:htr:{key}:climate",
        config_entry=hass.config_entries.async_get_entry(entry_id),
        device_id=device.id,
    )
    return device.id, entity.entity_id


async def test_rekey_skips_collisions_and_other_nodes(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Taken targets are skipped with a warning; heater 60 is not heater 6."""
    entry = MockConfigEntry(domain=DOMAIN, data={"brand": "radio"})
    entry.add_to_hass(hass)
    old = _seed(hass, entry.entry_id, "6")
    taken = _seed(hass, entry.entry_id, SERIAL)
    other = _seed(hass, entry.entry_id, "60")

    radio_identity.async_rekey_node(hass, entry.entry_id, DEV, "htr", "6", SERIAL)
    radio_identity.async_rekey_node(hass, entry.entry_id, DEV, "htr", "6", "6")

    ent_reg, dev_reg = er.async_get(hass), dr.async_get(hass)
    for (device_id, entity_id), key in ((old, "6"), (taken, SERIAL), (other, "60")):
        assert dev_reg.async_get(device_id).identifiers == {(DOMAIN, DEV, key)}
        assert ent_reg.async_get(entity_id).unique_id.endswith(f":{key}:climate")
    assert "already has a device" in caplog.text
    assert "already has its unique ID" in caplog.text
    assert SERIAL not in caplog.text  # serials are masked in logs


async def test_cloud_inventory_is_never_rekeyed(hass: HomeAssistant) -> None:
    """Nodes without a serial (every cloud node) keep their address IDs."""
    entry = MockConfigEntry(domain=DOMAIN, data={"brand": "termoweb"})
    entry.add_to_hass(hass)
    device_id, entity_id = _seed(hass, entry.entry_id, "6")
    inventory = Inventory(DEV, build_node_inventory([{"type": "htr", "addr": "6"}]))

    radio_identity.async_rekey_inventory(hass, entry.entry_id, inventory)

    assert er.async_get(hass).async_get(entity_id).unique_id == CLIMATE_BY_ADDR
    assert dr.async_get(hass).async_get(device_id).identifiers == {(DOMAIN, DEV, "6")}


async def test_learn_serials_skips_other_clients_and_bad_addresses(
    hass: HomeAssistant,
) -> None:
    """Only a transmitting radio client is asked; invalid addresses are skipped."""
    nodes = [{"type": "htr", "addr": "bogus"}]
    entry = MockConfigEntry(domain=DOMAIN, data={"brand": "radio", "nodes": nodes})
    entry.add_to_hass(hass)
    await radio_identity.async_learn_serials(hass, entry, object())  # cloud client

    client = RadioClient("10.0.0.5", 2323, "B", nodes, network_id=NET)
    await radio_identity.async_learn_serials(hass, entry, client)  # never connects
    assert entry.data["nodes"] == nodes
    assert not client.connected
