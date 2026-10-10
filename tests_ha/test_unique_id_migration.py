"""Entity unique IDs: one scheme, and a registry migration from older formats."""

from __future__ import annotations

import re

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import async_migrate_entry, config_flow
from custom_components.termoweb.const import (
    BRAND_RADIO_MONITOR,
    BRAND_TERMOWEB,
    CONF_BRAND,
    DOMAIN,
)
from custom_components.termoweb.domain.state import GeoData
from custom_components.termoweb.identifiers import (
    build_gateway_entity_unique_id,
    build_heater_unique_id,
    build_installation_entity_unique_id,
    is_canonical_unique_id,
    migrate_unique_id,
)

from .conftest import DEV_ID, PASSWORD, USERNAME, FakeCloud

MINOR_VERSION = config_flow.TermoWebConfigFlow.MINOR_VERSION
NODES = {
    "nodes": [
        {"type": "htr", "addr": 1, "name": "Heater"},
        {"type": "acm", "addr": 2, "name": "Storage"},
        {"type": "pmo", "addr": 3, "name": "Meter"},
        {"type": "thm", "addr": 4, "name": "Thermostat"},
    ]
}
GEO = GeoData(country="X", state="Y", city="Z", tz_code="Europe/X", zip="0")
P = f"{DOMAIN}:{DEV_ID}"

# Every unique ID format an entity of this integration has had on main, by
# entity domain: (domain, old unique ID, unique ID after the migration). The
# first block is what setup created before this change; the second block is
# formats from older releases (or unreleased radio builds).
CHANGED = [
    ("binary_sensor", f"{DEV_ID}_online", f"{P}:online"),
    ("sensor", f"{P}:energy_total", f"{P}:site:energy_total"),
    ("number", f"{P}:power_limit", f"{P}:site:power_limit"),
    ("sensor", f"{P}:acm:2:boost:end", f"{P}:acm:2:boost_end"),
    (
        "sensor",
        f"{P}:acm:2:boost:minutes_remaining",
        f"{P}:acm:2:boost_minutes_remaining",
    ),
]
HISTORICAL = [
    ("sensor", f"{P}:installation:info", f"{P}:site:info"),
    ("climate", f"{P}:htr:1", f"{P}:htr:1:climate"),
    ("sensor", f"{DEV_ID}:1:energy", f"{P}:htr:1:energy"),
    ("sensor", f"{P}:radio_frames", f"{P}:frames_heard"),
]
# Formats that were already canonical: the migration must leave them alone.
UNCHANGED = [
    ("binary_sensor", f"{P}:acm:2:boost_active"),
    ("button", f"{P}:acm:2:boost_cancel"),
    ("button", f"{P}:acm:2:boost_start"),
    ("button", f"{P}:acm:2:flash_display"),
    ("button", f"{P}:htr:1:flash_display"),
    ("button", f"{P}:refresh"),
    ("climate", f"{P}:acm:2:climate"),
    ("climate", f"{P}:htr:1:climate"),
    ("climate", f"{P}:thm:4:climate"),
    ("lock", f"{P}:htr:1:child_lock"),
    ("number", f"{P}:acm:2:boost_duration"),
    ("number", f"{P}:acm:2:boost_temperature"),
    ("number", f"{P}:acm:2:priority"),
    ("number", f"{P}:htr:1:priority"),
    ("number", f"{P}:thm:4:priority"),
    ("sensor", f"{P}:acm:2:charging"),
    ("sensor", f"{P}:acm:2:current_charge_per"),
    ("sensor", f"{P}:acm:2:energy"),
    ("sensor", f"{P}:acm:2:power"),
    ("sensor", f"{P}:acm:2:target_charge_per"),
    ("sensor", f"{P}:acm:2:temp"),
    ("sensor", f"{P}:htr:1:energy"),
    ("sensor", f"{P}:htr:1:power"),
    ("sensor", f"{P}:htr:1:temp"),
    ("sensor", f"{P}:pmo:3:energy"),
    ("sensor", f"{P}:pmo:3:power"),
    ("sensor", f"{P}:site:info"),
    ("sensor", f"{P}:thm:4:battery"),
    ("sensor", f"{P}:thm:4:temp"),
    ("sensor", f"{P}:frames_heard"),
]


def _entry(minor_version: int, brand: str = BRAND_TERMOWEB) -> MockConfigEntry:
    """Return a cloud entry at ``minor_version``."""
    return MockConfigEntry(
        domain=DOMAIN,
        minor_version=minor_version,
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: brand},
    )


def _register(
    hass: HomeAssistant, entry: MockConfigEntry, domain: str, unique_id: str
) -> er.RegistryEntry:
    """Register an entity the way an older release would have, with a custom id."""
    registry = er.async_get(hass)
    object_id = "old_" + re.sub(r"[^a-z0-9]+", "_", unique_id.lower()).strip("_")
    entity = registry.async_get_or_create(
        domain,
        DOMAIN,
        unique_id,
        suggested_object_id=object_id,
        config_entry=entry,
    )
    # A user customisation that must survive the migration.
    return registry.async_update_entity(entity.entity_id, name=f"Custom {object_id}")


@pytest.fixture
def full_cloud(cloud: FakeCloud) -> FakeCloud:
    """Serve every node type, a power limit and a location."""
    cloud.get_nodes.return_value = NODES
    cloud.get_power_limit.return_value = 3000
    cloud.get_geo_data.return_value = GEO
    return cloud


@pytest.mark.parametrize(("domain", "old", "new"), CHANGED + HISTORICAL)
def test_every_old_format_maps_to_a_canonical_id(
    domain: str, old: str, new: str
) -> None:
    """Each older format has a canonical replacement, and replacements are final."""
    assert migrate_unique_id(domain, old) == new
    assert is_canonical_unique_id(new)
    assert migrate_unique_id(domain, new) is None


@pytest.mark.parametrize(("domain", "unique_id"), UNCHANGED)
def test_canonical_ids_are_not_migrated(domain: str, unique_id: str) -> None:
    """IDs already in the scheme are left alone."""
    assert is_canonical_unique_id(unique_id)
    assert migrate_unique_id(domain, unique_id) is None


async def test_setup_migrates_every_old_unique_id(
    hass: HomeAssistant, full_cloud: FakeCloud
) -> None:
    """Old entities keep entity_id and customisations; no duplicate is created."""
    entry = _entry(minor_version=4)
    entry.add_to_hass(hass)
    old = {
        unique_id: _register(hass, entry, domain, unique_id)
        for domain, unique_id, _new in CHANGED + HISTORICAL[:3]
    }
    targets = {new for _domain, _old, new in CHANGED + HISTORICAL}
    kept = {
        unique_id: _register(hass, entry, domain, unique_id)
        for domain, unique_id in UNCHANGED
        if domain != "lock" and unique_id not in targets
    }
    before = len(er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id))

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    registry = er.async_get(hass)
    assert entry.minor_version == MINOR_VERSION
    for domain, old_id, new_id in CHANGED + HISTORICAL[:3]:
        entity = registry.async_get(old[old_id].entity_id)
        assert entity is not None, old_id
        assert entity.unique_id == new_id
        assert entity.name == old[old_id].name
        assert registry.async_get_entity_id(domain, DOMAIN, old_id) is None
    for unique_id, registered in kept.items():
        entity = registry.async_get(registered.entity_id)
        assert entity is not None
        assert entity.unique_id == unique_id
        assert entity.name == registered.name
    entries = er.async_entries_for_config_entry(registry, entry.entry_id)
    # Setup created nothing new: every entity it has was already registered.
    assert len(entries) == before
    assert len({(e.domain, e.unique_id) for e in entries}) == len(entries)
    assert all(is_canonical_unique_id(e.unique_id) for e in entries)


async def test_migration_skips_ids_that_already_exist(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """When the new ID is taken, the old entity is left as is with a warning."""
    entry = _entry(minor_version=4)
    entry.add_to_hass(hass)
    old = _register(hass, entry, "binary_sensor", f"{DEV_ID}_online")
    new = _register(hass, entry, "binary_sensor", f"{P}:online")

    assert await async_migrate_entry(hass, entry)

    registry = er.async_get(hass)
    assert registry.async_get(old.entity_id).unique_id == f"{DEV_ID}_online"
    assert registry.async_get(new.entity_id).unique_id == f"{P}:online"
    assert entry.minor_version == MINOR_VERSION
    assert (
        f"Not migrating {old.entity_id} to unique ID {P}:online: "
        f"{new.entity_id} already has it" in caplog.text
    )


async def test_migration_covers_radio_entries(hass: HomeAssistant) -> None:
    """Radio monitor entities created by earlier builds move to the scheme too."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        minor_version=4,
        unique_id=f"{BRAND_RADIO_MONITOR}:{DEV_ID}",
        data={CONF_BRAND: BRAND_RADIO_MONITOR, "host": "10.0.0.5", "port": 2323},
    )
    entry.add_to_hass(hass)
    frames = _register(hass, entry, "sensor", f"{P}:radio_frames")
    online = _register(hass, entry, "binary_sensor", f"{DEV_ID}_online")

    assert await async_migrate_entry(hass, entry)

    registry = er.async_get(hass)
    assert registry.async_get(frames.entity_id).unique_id == f"{P}:frames_heard"
    assert registry.async_get(online.entity_id).unique_id == f"{P}:online"


async def test_fresh_setup_ids_follow_the_scheme(
    hass: HomeAssistant, full_cloud: FakeCloud
) -> None:
    """Every entity's unique ID is canonical and names the device it belongs to."""
    entry = _entry(minor_version=MINOR_VERSION)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    devices = dr.async_get(hass)
    entities = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert len(entities) >= 30
    for entity in entities:
        assert is_canonical_unique_id(entity.unique_id), entity.unique_id
        assert migrate_unique_id(entity.domain, entity.unique_id) is None
        device = devices.async_get(entity.device_id)
        assert device is not None
        (identifier,) = device.identifiers
        scope = {
            (DOMAIN, DEV_ID): P,
            (DOMAIN, DEV_ID, "site"): f"{P}:site",
        }.get(identifier)
        if scope is None:  # a node: (domain, dev, addr) or (domain, dev, "pmo", addr)
            addr = identifier[-1]
            scope = rf"{P}:[a-z]+:{addr}"
        assert re.fullmatch(rf"{scope}:[a-z0-9_]+", entity.unique_id), (
            entity.unique_id,
            identifier,
        )


@pytest.mark.parametrize("key", ["boost:end", "Boost", "", "a__b", ":"])
def test_builders_reject_keys_outside_the_scheme(key: str) -> None:
    """A key must be one snake_case token, so no builder can mint an odd ID."""
    with pytest.raises(ValueError, match="snake_case"):
        build_gateway_entity_unique_id(DEV_ID, key or "-")
    with pytest.raises(ValueError, match="snake_case"):
        build_installation_entity_unique_id(DEV_ID, key or "-")
    if key:
        with pytest.raises(ValueError, match="snake_case"):
            build_heater_unique_id(DEV_ID, "htr", "1", suffix=key)


def test_builders_need_a_gateway_id() -> None:
    """Gateway and site IDs cannot be built without the gateway id."""
    with pytest.raises(ValueError, match="dev_id"):
        build_gateway_entity_unique_id("", "online")
    with pytest.raises(ValueError, match="dev_id"):
        build_installation_entity_unique_id("  ", "info")
