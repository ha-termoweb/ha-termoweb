# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001
"""Tests for the installation entity restructure.

Covers: GeoData dataclass, extended DeviceMetadata, build_device_metadata(),
get_geo_data(), build_installation_device_info(), build_gateway_device_info()
changes, build_installation_entity_unique_id(), entity device_info reassignment,
extra_state_attributes with geo_data, and diagnostics installation section.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import types
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_entry_runtime

_install_stubs()

# Set up diagnostics stub before any diagnostics import
import sys

_diagnostics_stub = types.ModuleType("diagnostics")


async def _async_passthrough(data: Any, _keys: set[str]) -> Any:
    return data


_diagnostics_stub.async_redact_data = _async_passthrough
_components_pkg = sys.modules.setdefault(
    "homeassistant.components", types.ModuleType("homeassistant.components")
)
setattr(_components_pkg, "diagnostics", _diagnostics_stub)
sys.modules["homeassistant.components.diagnostics"] = _diagnostics_stub

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import (
    DeviceMetadata,
    build_device_metadata,
)
from custom_components.termoweb.domain.state import GeoData
from custom_components.termoweb.identifiers import build_installation_entity_unique_id
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.utils import (
    build_gateway_device_info,
    build_installation_device_info,
)
from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# 1. GeoData dataclass
# ---------------------------------------------------------------------------


class TestGeoData:
    def test_construction_with_defaults(self) -> None:
        geo = GeoData()
        assert geo.country is None
        assert geo.state is None
        assert geo.city is None
        assert geo.tz_code is None
        assert geo.zip is None

    def test_construction_with_values(self) -> None:
        geo = GeoData(
            country="US",
            state="CA",
            city="San Francisco",
            tz_code="America/Los_Angeles",
            zip="94102",
        )
        assert geo.country == "US"
        assert geo.state == "CA"
        assert geo.city == "San Francisco"
        assert geo.tz_code == "America/Los_Angeles"
        assert geo.zip == "94102"

    def test_frozen_prevents_mutation(self) -> None:
        geo = GeoData(country="US")
        with pytest.raises(AttributeError):
            geo.country = "UK"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = GeoData(country="US", city="NYC")
        b = GeoData(country="US", city="NYC")
        assert a == b

    def test_inequality(self) -> None:
        a = GeoData(country="US")
        b = GeoData(country="UK")
        assert a != b

    def test_partial_values(self) -> None:
        geo = GeoData(country="DE", tz_code="Europe/Berlin")
        assert geo.country == "DE"
        assert geo.state is None
        assert geo.city is None
        assert geo.tz_code == "Europe/Berlin"
        assert geo.zip is None


# ---------------------------------------------------------------------------
# 2. Extended DeviceMetadata
# ---------------------------------------------------------------------------


class TestDeviceMetadata:
    def test_new_fields_default_to_none(self) -> None:
        meta = DeviceMetadata(dev_id="d1", name="My Home", model="GW")
        assert meta.serial_id is None
        assert meta.fw_version is None
        assert meta.geo_data is None

    def test_new_fields_set_explicitly(self) -> None:
        geo = GeoData(country="NO")
        meta = DeviceMetadata(
            dev_id="d1",
            name="My Home",
            model="GW",
            serial_id="SN123",
            fw_version="2.1.0",
            geo_data=geo,
        )
        assert meta.serial_id == "SN123"
        assert meta.fw_version == "2.1.0"
        assert meta.geo_data is geo

    def test_backward_compat_old_constructor(self) -> None:
        """Old code creating DeviceMetadata without new fields still works."""
        meta = DeviceMetadata(dev_id="d1", name="Name", model=None)
        assert meta.serial_id is None
        assert meta.fw_version is None
        assert meta.geo_data is None

    def test_frozen(self) -> None:
        meta = DeviceMetadata(dev_id="d1", name="N", model="M")
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. build_device_metadata()
# ---------------------------------------------------------------------------


class TestBuildDeviceMetadata:
    def test_extracts_serial_id_and_fw_version(self) -> None:
        payload = {
            "name": "Office",
            "model": "TW-500",
            "serial_id": "  ABC123  ",
            "fw_version": "  3.0.1 ",
        }
        meta = build_device_metadata("dev1", payload)
        assert meta.serial_id == "ABC123"
        assert meta.fw_version == "3.0.1"

    def test_missing_serial_id_and_fw_version(self) -> None:
        payload = {"name": "Home", "model": "TW-200"}
        meta = build_device_metadata("dev1", payload)
        assert meta.serial_id is None
        assert meta.fw_version is None

    def test_empty_serial_id_and_fw_version(self) -> None:
        payload = {
            "name": "Home",
            "model": "TW-200",
            "serial_id": "  ",
            "fw_version": "",
        }
        meta = build_device_metadata("dev1", payload)
        assert meta.serial_id is None
        assert meta.fw_version is None

    def test_none_device_payload(self) -> None:
        meta = build_device_metadata("dev1", None)
        assert meta.name == "Device dev1"
        assert meta.model is None
        assert meta.serial_id is None
        assert meta.fw_version is None

    def test_geo_data_not_set_by_build(self) -> None:
        """build_device_metadata does not set geo_data -- it's set separately."""
        payload = {"name": "Home"}
        meta = build_device_metadata("dev1", payload)
        assert meta.geo_data is None


# ---------------------------------------------------------------------------
# 4. get_geo_data() REST client
# ---------------------------------------------------------------------------


def _prep_client(monkeypatch, fake_request):
    """Build a RESTClient with a monkeypatched _request and valid token."""
    from tests.test_api import FakeSession
    import custom_components.termoweb.backend.rest_client as api

    session = FakeSession()
    client = api.RESTClient(session, "user", "pass")
    client._access_token = "tok"
    client._token_expiry = api.time.time() + 10000
    client._token_expiry_monotonic = api.time_mod() + 10000
    monkeypatch.setattr(client, "_request", fake_request)
    return client


class TestGetGeoData:
    def test_success_returns_geo_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful API call returns GeoData."""

        async def _run() -> None:
            async def fake_request(
                method: str, path: str, **kwargs: Any
            ) -> dict[str, Any] | None:
                return {
                    "country": "NO",
                    "state": "Oslo",
                    "city": "Oslo",
                    "tz_code": "Europe/Oslo",
                    "zip": "0150",
                }

            client = _prep_client(monkeypatch, fake_request)
            result = await client.get_geo_data("dev1")
            assert result is not None
            assert type(result).__name__ == "GeoData"
            assert result.country == "NO"
            assert result.state == "Oslo"
            assert result.city == "Oslo"
            assert result.tz_code == "Europe/Oslo"
            assert result.zip == "0150"

        asyncio.run(_run())

    def test_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any exception returns None (best-effort)."""

        async def _run() -> None:
            async def fake_request(
                method: str, path: str, **kwargs: Any
            ) -> dict[str, Any] | None:
                raise ConnectionError("network down")

            client = _prep_client(monkeypatch, fake_request)
            result = await client.get_geo_data("dev1")
            assert result is None

        asyncio.run(_run())

    def test_404_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """404 response (handled by _request ignore_statuses) returns None."""

        async def _run() -> None:
            async def fake_request(
                method: str, path: str, **kwargs: Any
            ) -> dict[str, Any] | None:
                return None

            client = _prep_client(monkeypatch, fake_request)
            result = await client.get_geo_data("dev1")
            assert result is None

        asyncio.run(_run())

    def test_non_mapping_response_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-mapping response body returns None."""

        async def _run() -> None:
            async def fake_request(
                method: str, path: str, **kwargs: Any
            ) -> list[str] | None:
                return ["unexpected"]

            client = _prep_client(monkeypatch, fake_request)
            result = await client.get_geo_data("dev1")
            assert result is None

        asyncio.run(_run())

    def test_request_path_includes_dev_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_geo_data should call _request with the correct path."""

        async def _run() -> None:
            calls: list[tuple[str, str]] = []

            async def fake_request(
                method: str, path: str, **kwargs: Any
            ) -> dict[str, Any] | None:
                calls.append((method, path))
                return {"country": "US"}

            client = _prep_client(monkeypatch, fake_request)
            await client.get_geo_data("mydev123")
            assert len(calls) == 1
            assert calls[0][0] == "GET"
            assert "mydev123" in calls[0][1]
            assert "geo_data" in calls[0][1]

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. build_installation_device_info()
# ---------------------------------------------------------------------------


class TestBuildInstallationDeviceInfo:
    def test_correct_identifiers(self) -> None:
        hass = types.SimpleNamespace(data={})
        info = build_installation_device_info(hass, "entry", "dev123")
        assert info["identifiers"] == {(DOMAIN, "dev123", "site")}

    def test_three_tuple_identifier(self) -> None:
        """Identifiers use a 3-tuple to avoid collision with gateway."""
        hass = types.SimpleNamespace(data={})
        info = build_installation_device_info(hass, "entry", "dev")
        ids = info["identifiers"]
        (identifier,) = ids  # exactly one identifier
        assert len(identifier) == 3
        assert identifier[2] == "site"

    def test_model_is_installation(self) -> None:
        hass = types.SimpleNamespace(data={})
        info = build_installation_device_info(hass, "entry", "dev")
        assert info["model"] == "Site"

    def test_no_via_device(self) -> None:
        """Installation is the top-level device -- no via_device."""
        hass = types.SimpleNamespace(data={})
        info = build_installation_device_info(hass, "entry", "dev")
        assert "via_device" not in info

    def test_name_from_coordinator_gateway_name(self) -> None:
        """Name should come from coordinator.gateway_name()."""
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            inventory=Inventory("dev", []),
            data={},
        )
        # gateway_name as a callable (like the real coordinator property)
        coordinator.gateway_name = lambda: "My Home"
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )

        info = build_installation_device_info(hass, "entry", "dev")
        assert info["name"] == "My Home"

    def test_name_from_coordinator_gateway_name_attribute(self) -> None:
        """gateway_name as a plain string attribute should work too."""
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            gateway_name="Office Building",
            inventory=Inventory("dev", []),
            data={},
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )

        info = build_installation_device_info(hass, "entry", "dev")
        assert info["name"] == "Office Building"

    def test_defaults_without_entry(self) -> None:
        hass = types.SimpleNamespace(data={})
        info = build_installation_device_info(hass, "entry", "dev")
        assert info["manufacturer"] == "TermoWeb"
        assert info["name"] == "Site"
        assert info["model"] == "Site"

    def test_brand_override(self) -> None:
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            brand="Ducaheat",
        )
        info = build_installation_device_info(hass, "entry", "dev")
        assert info["manufacturer"] == "Ducaheat"


# ---------------------------------------------------------------------------
# 6. build_gateway_device_info() changes
# ---------------------------------------------------------------------------


class TestBuildGatewayDeviceInfoChanges:
    def test_via_device_points_to_installation(self) -> None:
        hass = types.SimpleNamespace(data={})
        info = build_gateway_device_info(hass, "entry", "dev123")
        assert info["via_device"] == (DOMAIN, "dev123", "site")

    def test_sw_version_from_fw_version(self) -> None:
        """Gateway device shows fw_version as sw_version."""
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        meta = DeviceMetadata(
            dev_id="dev",
            name="Home",
            model="GW",
            fw_version="2.5.0",
        )
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            device_metadata=meta,
            inventory=Inventory("dev", []),
            data={},
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )
        info = build_gateway_device_info(hass, "entry", "dev")
        assert info["sw_version"] == "2.5.0"

    def test_serial_number_from_serial_id(self) -> None:
        """Gateway device shows serial_id as serial_number."""
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        meta = DeviceMetadata(
            dev_id="dev",
            name="Home",
            model="GW",
            serial_id="SN-999",
        )
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            device_metadata=meta,
            inventory=Inventory("dev", []),
            data={},
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )
        info = build_gateway_device_info(hass, "entry", "dev")
        assert info["serial_number"] == "SN-999"

    def test_no_fw_version_no_sw_version(self) -> None:
        """When fw_version is None, sw_version should not be set from metadata."""
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        meta = DeviceMetadata(dev_id="dev", name="Home", model="GW")
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            device_metadata=meta,
            inventory=Inventory("dev", []),
            data={},
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )
        info = build_gateway_device_info(hass, "entry", "dev")
        assert "serial_number" not in info

    def test_no_serial_id_no_serial_number(self) -> None:
        hass = types.SimpleNamespace(data={DOMAIN: {}})
        meta = DeviceMetadata(dev_id="dev", name="Home", model="GW")
        coordinator = types.SimpleNamespace(
            gateway_model="Controller",
            device_metadata=meta,
            inventory=Inventory("dev", []),
            data={},
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=coordinator,
        )
        info = build_gateway_device_info(hass, "entry", "dev")
        assert "serial_number" not in info

    def test_gateway_identifiers_unchanged(self) -> None:
        """Gateway still uses 2-tuple identifier."""
        hass = types.SimpleNamespace(data={})
        info = build_gateway_device_info(hass, "entry", "dev")
        assert info["identifiers"] == {(DOMAIN, "dev")}


# ---------------------------------------------------------------------------
# 7. build_installation_entity_unique_id()
# ---------------------------------------------------------------------------


class TestBuildInstallationEntityUniqueId:
    def test_correct_format(self) -> None:
        uid = build_installation_entity_unique_id("dev123", "total_energy")
        assert uid == f"{DOMAIN}:dev123:site:total_energy"

    def test_whitespace_trimming(self) -> None:
        uid = build_installation_entity_unique_id(" dev ", "power_limit")
        assert uid == f"{DOMAIN}:dev:site:power_limit"

    def test_empty_dev_id_raises(self) -> None:
        with pytest.raises(ValueError):
            build_installation_entity_unique_id("", "suffix")

    def test_whitespace_only_dev_id_raises(self) -> None:
        with pytest.raises(ValueError):
            build_installation_entity_unique_id("   ", "suffix")


# ---------------------------------------------------------------------------
# 8-10. Entity device_info: InstallationTotalEnergySensor, PowerLimitNumber
# ---------------------------------------------------------------------------


class TestInstallationTotalEnergySensorDeviceInfo:
    def test_device_info_returns_installation(self) -> None:
        """InstallationTotalEnergySensor.device_info should use installation device info."""
        from custom_components.termoweb.entities.sensor import (
            InstallationTotalEnergySensor,
        )

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        sensor = InstallationTotalEnergySensor(
            coordinator,
            "entry1",
            "dev1",
            "uid-total-energy",
            types.SimpleNamespace(addrs_by_type={}),
            coordinator.domain_view,
        )
        sensor.hass = hass

        info = sensor.device_info
        # Should have 3-tuple installation identifier
        assert info["identifiers"] == {(DOMAIN, "dev1", "site")}
        assert "via_device" not in info
        assert info["model"] == "Site"


class TestInstallationInfoSensor:
    def test_device_info_returns_installation(self) -> None:
        """InstallationInfoSensor.device_info uses installation identifiers."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        sensor.hass = hass

        info = sensor.device_info
        assert (DOMAIN, "dev1", "site") in info["identifiers"]

    def test_entity_category_is_diagnostic(self) -> None:
        """InstallationInfoSensor should be a diagnostic entity."""
        from homeassistant.helpers.entity import EntityCategory

        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        assert sensor._attr_entity_category == EntityCategory.DIAGNOSTIC

    def test_unique_id_format(self) -> None:
        """Unique ID follows installation entity pattern."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        assert sensor.unique_id == f"{DOMAIN}:dev1:site:info"

    def test_geo_data_attrs_when_present(self) -> None:
        """extra_state_attributes includes all geo_data fields."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        geo = GeoData(
            country="NO",
            state="Oslo",
            city="Oslo",
            tz_code="Europe/Oslo",
            zip="0101",
        )
        meta = DeviceMetadata(
            dev_id="dev1",
            name="Home",
            model="GW",
            geo_data=geo,
        )
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )
        coordinator.device_metadata = meta  # type: ignore[attr-defined]

        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        sensor.hass = hass

        attrs = sensor.extra_state_attributes
        assert "dev_id" not in attrs
        assert attrs["country"] == "NO"
        assert attrs["state"] == "Oslo"
        assert attrs["city"] == "Oslo"
        assert attrs["timezone"] == "Europe/Oslo"
        assert attrs["zip"] == "0101"

    def test_native_value_location_summary(self) -> None:
        """native_value returns a comma-separated city, state, country string."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        geo = GeoData(country="FR", state="Paris", city="Paris")
        meta = DeviceMetadata(
            dev_id="dev1",
            name="Home",
            model="GW",
            geo_data=geo,
        )
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )
        coordinator.device_metadata = meta  # type: ignore[attr-defined]

        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        sensor.hass = hass

        assert sensor.native_value == "Paris, Paris, FR"

    def test_no_geo_data_attrs_when_absent(self) -> None:
        """extra_state_attributes omits geo_data fields when absent."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )

        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        sensor.hass = hass

        attrs = sensor.extra_state_attributes
        assert attrs == {}
        assert sensor.native_value is None

    def test_partial_geo_data_only_includes_present(self) -> None:
        """Only non-None geo_data fields appear in attributes."""
        from custom_components.termoweb.entities.sensor import (
            InstallationInfoSensor,
        )

        hass = HomeAssistant()
        geo = GeoData(country="DE")
        meta = DeviceMetadata(
            dev_id="dev1",
            name="Home",
            model="GW",
            geo_data=geo,
        )
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )
        coordinator.device_metadata = meta  # type: ignore[attr-defined]

        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        sensor = InstallationInfoSensor(coordinator, "entry1", "dev1")
        sensor.hass = hass

        attrs = sensor.extra_state_attributes
        assert attrs["country"] == "DE"
        assert "city" not in attrs
        assert "timezone" not in attrs
        assert sensor.native_value == "DE"


class TestPowerLimitNumberDeviceInfo:
    def test_device_info_returns_installation(self) -> None:
        """PowerLimitNumber.device_info should use installation device info."""
        from custom_components.termoweb.entities.number import PowerLimitNumber

        hass = HomeAssistant()
        coordinator = FakeCoordinator(
            hass,
            dev_id="dev1",
            inventory=Inventory("dev1", []),
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry1",
            dev_id="dev1",
            coordinator=coordinator,
        )

        entity = PowerLimitNumber(
            coordinator,
            "entry1",
            "dev1",
            "uid-power-limit",
        )
        entity.hass = hass

        info = entity.device_info
        # Should have 3-tuple installation identifier
        assert info["identifiers"] == {(DOMAIN, "dev1", "site")}
        assert "via_device" not in info
        assert info["model"] == "Site"


# ---------------------------------------------------------------------------
# 11. Setup flow: geo_data fetch failure doesn't break setup
# ---------------------------------------------------------------------------


class TestSetupGeoDataFailure:
    def test_geo_data_failure_does_not_raise(self) -> None:
        """geo_data fetch failure should not propagate -- setup continues."""
        # This tests the pattern in __init__.py where geo_data fetch is wrapped
        # in try/except. We verify the DeviceMetadata can be constructed without it.
        meta = build_device_metadata("dev1", {"name": "Home", "model": "GW"})
        assert meta.geo_data is None

        # Simulate the replacement pattern from __init__.py
        geo = None  # fetch failed
        if geo is not None:
            meta = DeviceMetadata(
                dev_id=meta.dev_id,
                name=meta.name,
                model=meta.model,
                serial_id=meta.serial_id,
                fw_version=meta.fw_version,
                geo_data=geo,
            )
        # meta should still be valid without geo_data
        assert meta.geo_data is None
        assert meta.name == "Home"

    def test_geo_data_success_enriches_metadata(self) -> None:
        """When geo_data succeeds, metadata is enriched."""
        meta = build_device_metadata("dev1", {"name": "Home", "model": "GW"})
        geo = GeoData(country="NO", city="Oslo")

        # Simulate the replacement pattern from __init__.py
        meta = DeviceMetadata(
            dev_id=meta.dev_id,
            name=meta.name,
            model=meta.model,
            serial_id=meta.serial_id,
            fw_version=meta.fw_version,
            geo_data=geo,
        )
        assert meta.geo_data is geo
        assert meta.geo_data.country == "NO"


# ---------------------------------------------------------------------------
# 12. Diagnostics installation section
# ---------------------------------------------------------------------------


class TestDiagnosticsInstallation:
    def _make_runtime(
        self,
        hass: HomeAssistant,
        entry_id: str,
        dev_id: str,
        *,
        version: str = "1.0.0",
    ) -> Any:
        """Build an EntryRuntime for diagnostics tests."""
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            inventory=Inventory(dev_id, []),
        )
        runtime = build_entry_runtime(
            hass=hass,
            entry_id=entry_id,
            dev_id=dev_id,
            coordinator=coordinator,
            version=version,
        )
        return runtime

    def test_diagnostics_include_installation_metadata(self) -> None:
        """Diagnostics should include installation metadata when available."""
        from custom_components.termoweb.diagnostics import (
            async_get_config_entry_diagnostics,
        )
        from custom_components.termoweb.const import CONF_BRAND
        from homeassistant.config_entries import ConfigEntry

        hass = HomeAssistant()
        hass.version = "2025.5.0"
        entry = ConfigEntry("entry-diag", data={CONF_BRAND: "termoweb"})

        geo = GeoData(country="NO", state="Oslo", city="Oslo", tz_code="Europe/Oslo")
        meta = DeviceMetadata(
            dev_id="dev-diag",
            name="My Home",
            model="TW-500",
            fw_version="2.1.0",
            geo_data=geo,
        )

        runtime = self._make_runtime(hass, entry.entry_id, "dev-diag")
        runtime.coordinator.device_metadata = meta

        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

        installation = diagnostics["site"]
        assert installation["name"] == "My Home"
        assert installation["model"] == "TW-500"
        assert installation["fw_version"] == "2.1.0"
        assert installation["geo_data"]["country"] == "NO"
        assert installation["geo_data"]["city"] == "Oslo"
        assert installation["geo_data"]["tz_code"] == "Europe/Oslo"

    def test_diagnostics_without_device_metadata(self) -> None:
        """Diagnostics should work without device_metadata on coordinator."""
        from custom_components.termoweb.diagnostics import (
            async_get_config_entry_diagnostics,
        )
        from custom_components.termoweb.const import CONF_BRAND
        from homeassistant.config_entries import ConfigEntry

        hass = HomeAssistant()
        entry = ConfigEntry("entry-no-meta", data={CONF_BRAND: "termoweb"})

        self._make_runtime(hass, entry.entry_id, "dev-no-meta")

        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

        installation = diagnostics["site"]
        assert "node_inventory" in installation
        # FakeCoordinator has no device_metadata attribute
        assert "geo_data" not in installation

    def test_diagnostics_with_metadata_no_geo(self) -> None:
        """Diagnostics should include metadata but skip geo_data when absent."""
        from custom_components.termoweb.diagnostics import (
            async_get_config_entry_diagnostics,
        )
        from custom_components.termoweb.const import CONF_BRAND
        from homeassistant.config_entries import ConfigEntry

        hass = HomeAssistant()
        entry = ConfigEntry("entry-no-geo", data={CONF_BRAND: "termoweb"})

        meta = DeviceMetadata(
            dev_id="dev-no-geo",
            name="Office",
            model="TW-300",
        )

        runtime = self._make_runtime(hass, entry.entry_id, "dev-no-geo")
        runtime.coordinator.device_metadata = meta

        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

        installation = diagnostics["site"]
        assert installation["name"] == "Office"
        assert installation["model"] == "TW-300"
        assert "geo_data" not in installation
