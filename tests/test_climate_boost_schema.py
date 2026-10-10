"""Boost-minutes schemas evaluated with Home Assistant's real validation engine.

``conftest._install_stubs`` replaces ``voluptuous`` with a minimal stub, so the
schema objects that ``climate.async_setup_entry`` registers are stub instances.
Home Assistant aliases ``voluptuous`` to probatio's shim; these tests load that
shim (temporarily swapping it into ``sys.modules``), rebuild the registered
schema tree with the real validators and validate against it, so real
``Range``/``In``/``Coerce`` semantics decide what is accepted.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock

from probatio.compat import install_as_voluptuous
import pytest

from conftest import (
    FakeCoordinator,
    _install_stubs,
    build_coordinator_device_state,
    build_entry_runtime,
)

_install_stubs()

from custom_components.termoweb.boost import ALLOWED_BOOST_MINUTES
from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_platform as entity_platform_module
from homeassistant.helpers.entity_platform import EntityPlatform


def _load_real_voluptuous() -> types.ModuleType:
    """Load HA's voluptuous (probatio's shim) without disturbing the stub."""

    def _owned(name: str) -> bool:
        return name == "voluptuous" or name.startswith("voluptuous.")

    saved = {name: mod for name, mod in sys.modules.items() if _owned(name)}
    for name in saved:
        del sys.modules[name]
    try:
        install_as_voluptuous()
        real = sys.modules["voluptuous"]
    finally:
        for name in [name for name in sys.modules if _owned(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
    assert hasattr(real, "MultipleInvalid"), "expected the real voluptuous package"
    return real


def _to_real(validator: Any, real: types.ModuleType) -> Any:
    """Rebuild a stub voluptuous validator tree with real voluptuous objects."""

    kind = type(validator).__name__
    if kind == "All":
        return real.All(*(_to_real(item, real) for item in validator.validators))
    if kind == "Coerce":
        return real.Coerce(validator.func)
    if kind == "In":
        return real.In(validator.container)
    if kind == "Range":
        return real.Range(min=validator.min, max=validator.max)
    if kind == "Optional":
        return real.Optional(validator.schema)
    if kind == "Required":
        return real.Required(validator.schema)
    if callable(validator):
        return validator
    raise AssertionError(f"unsupported validator in schema: {validator!r}")


async def _registered_schemas() -> dict[str, dict[Any, Any]]:
    """Run climate setup for one accumulator and return registered schemas."""

    _install_stubs()
    platform = EntityPlatform()
    entity_platform_module._set_current_platform(platform)
    FakeCoordinator.instances.clear()

    hass = HomeAssistant()
    dev_id = "dev-schema"
    nodes = {"nodes": [{"type": "acm", "addr": "1", "name": "Acc"}]}
    inventory = Inventory(dev_id, build_node_inventory(nodes))
    record = FakeCoordinator._normalise_device_record(
        build_coordinator_device_state(nodes=nodes, settings={"acm": {"1": {}}})
    )
    coordinator = FakeCoordinator(
        hass,
        client=AsyncMock(),
        dev_id=dev_id,
        dev=record,
        nodes=None,
        inventory=inventory,
        data={dev_id: record},
    )
    build_entry_runtime(
        hass=hass,
        entry_id="entry-schema",
        dev_id=dev_id,
        coordinator=coordinator,
        client=AsyncMock(),
        inventory=inventory,
    )
    await climate_module.async_setup_entry(
        hass, types.SimpleNamespace(entry_id="entry-schema"), lambda _ents: None
    )
    return {name: schema for name, schema, _ in platform.registered}


@pytest.fixture(scope="module")
def real_vol() -> types.ModuleType:
    """Return the real voluptuous module."""

    return _load_real_voluptuous()


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["start_boost", "set_acm_preset"])
@pytest.mark.parametrize(
    ("minutes", "accepted"),
    [
        (60, True),
        (180, True),
        (600, True),
        ("300", True),
        (30, False),
        (125, False),
        (0, False),
        (660, False),
    ],
)
async def test_boost_minutes_schema_matches_allowed_minutes(
    real_vol: types.ModuleType, service: str, minutes: Any, accepted: bool
) -> None:
    """Both boost schemas accept exactly ALLOWED_BOOST_MINUTES (real voluptuous)."""

    schemas = await _registered_schemas()
    schema = real_vol.Schema(
        {
            _to_real(key, real_vol): _to_real(value, real_vol)
            for key, value in schemas[service].items()
        }
    )

    if accepted:
        assert int(minutes) in ALLOWED_BOOST_MINUTES
        assert schema({"minutes": minutes}) == {"minutes": int(minutes)}
    else:
        with pytest.raises(real_vol.Invalid):
            schema({"minutes": minutes})


@pytest.mark.asyncio
async def test_boost_minutes_schemas_share_one_validator() -> None:
    """start_boost and set_acm_preset use the same minutes validator."""

    schemas = await _registered_schemas()

    def _minutes(schema: dict[Any, Any]) -> Any:
        return next(v for k, v in schema.items() if k.schema == "minutes")

    assert _minutes(schemas["start_boost"]) is _minutes(schemas["set_acm_preset"])
    assert _minutes(schemas["start_boost"]) is climate_module.BOOST_MINUTES_VALIDATOR
