# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001
"""Tests for the installation-wide power limit feature."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb.backend import termoweb_ws as ws_module
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import DOMAIN, BRAND_DUCAHEAT, BRAND_TERMOWEB
from custom_components.termoweb.entities import number as entities_number_module
from custom_components.termoweb.identifiers import build_gateway_entity_unique_id
from custom_components.termoweb.runtime import EntryRuntime, require_runtime
from homeassistant.core import HomeAssistant

import custom_components.termoweb.number as number_module

PowerLimitNumber = entities_number_module.PowerLimitNumber
async_setup_entry = number_module.async_setup_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyREST:
    """Minimal REST client stub reused across tests."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer token"}
        self._ensure_token = AsyncMock()
        self._is_ducaheat = False
        self._access_token = "token"

    async def authed_headers(self) -> dict[str, str]:
        return self._headers

    async def refresh_token(self) -> None:
        self._access_token = None
        await self._ensure_token()


def _make_hass() -> HomeAssistant:
    """Return a minimal hass object usable in tests."""
    hass = HomeAssistant()
    hass.data = {}
    return hass


def _make_power_limit_entity(
    *,
    hass: HomeAssistant | None = None,
    entry_id: str = "entry-pl",
    dev_id: str = "dev-pl",
    power_limit: int | None = 5000,
) -> PowerLimitNumber:
    """Create a PowerLimitNumber wired to a runtime with the given power limit."""
    if hass is None:
        hass = _make_hass()
    coordinator = FakeCoordinator(hass, dev_id=dev_id)
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id=dev_id,
        coordinator=coordinator,
    )
    runtime.power_limit = power_limit
    unique_id = build_gateway_entity_unique_id(dev_id, "power_limit")
    entity = PowerLimitNumber(coordinator, entry_id, dev_id, unique_id)
    entity.hass = hass
    return entity


# ===========================================================================
# REST Client Tests
# ===========================================================================


class TestRESTClientGetPowerLimit:
    """Tests for RESTClient.get_power_limit."""

    @pytest.mark.asyncio
    async def test_get_power_limit_returns_int(self) -> None:
        """Mock REST response {"power_limit": "5000"}, verify returns 5000."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value={"power_limit": "5000"})
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        result = await client.get_power_limit("dev123")
        assert result == 5000
        client._request.assert_awaited_once()
        call_args = client._request.call_args
        assert call_args[0][0] == "GET"
        assert "/htr_system/power_limit" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_power_limit_zero(self) -> None:
        """Mock response {"power_limit": "0"}, verify returns 0."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value={"power_limit": "0"})
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        result = await client.get_power_limit("dev123")
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_power_limit_none_response(self) -> None:
        """Mock None response, verify returns None."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value=None)
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        result = await client.get_power_limit("dev123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_power_limit_empty_dict(self) -> None:
        """Mock empty dict response, verify returns None."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value={})
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        result = await client.get_power_limit("dev123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_power_limit_non_numeric(self) -> None:
        """Mock {"power_limit": "abc"}, verify returns None."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value={"power_limit": "abc"})
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        result = await client.get_power_limit("dev123")
        assert result is None


class TestRESTClientSetPowerLimit:
    """Tests for RESTClient.set_power_limit."""

    @pytest.mark.asyncio
    async def test_set_power_limit_sends_correct_payload(self) -> None:
        """Verify POST with {"power_limit": "5000"}."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value=None)
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        await client.set_power_limit("dev123", power_limit=5000)

        client._request.assert_awaited_once()
        call_args = client._request.call_args
        assert call_args[0][0] == "POST"
        assert "/htr_system/power_limit" in call_args[0][1]
        assert call_args[1]["json"] == {"power_limit": "5000"}

    @pytest.mark.asyncio
    async def test_set_power_limit_zero(self) -> None:
        """Verify POST with {"power_limit": "0"} for zero limit."""
        client = RESTClient.__new__(RESTClient)
        client._request = AsyncMock(return_value=None)
        client._ensure_token = AsyncMock(return_value="token")
        client._access_token = "token"
        client._token_expiry_monotonic = float("inf")
        client._api_base = "https://api.termoweb.net"
        client._user_agent = "test"
        client._requested_with = None
        client._is_ducaheat = False
        client._lock = asyncio.Lock()

        await client.set_power_limit("dev123", power_limit=0)

        call_args = client._request.call_args
        assert call_args[1]["json"] == {"power_limit": "0"}


# ===========================================================================
# Entity Tests
# ===========================================================================


class TestPowerLimitEntityNativeValue:
    """Tests for PowerLimitNumber.native_value."""

    def test_native_value_returns_power_limit(self) -> None:
        entity = _make_power_limit_entity(power_limit=5000)
        assert entity.native_value == 5000

    def test_native_value_returns_none_when_not_set(self) -> None:
        entity = _make_power_limit_entity(power_limit=None)
        assert entity.native_value is None

    def test_native_value_returns_zero(self) -> None:
        entity = _make_power_limit_entity(power_limit=0)
        assert entity.native_value == 0


class TestPowerLimitEntityAvailable:
    """Tests for PowerLimitNumber.available."""

    def test_available_true_when_power_limit_set(self) -> None:
        entity = _make_power_limit_entity(power_limit=5000)
        assert entity.available is True

    def test_available_true_when_power_limit_zero(self) -> None:
        entity = _make_power_limit_entity(power_limit=0)
        assert entity.available is True

    def test_available_false_when_power_limit_none(self) -> None:
        entity = _make_power_limit_entity(power_limit=None)
        assert entity.available is False

    def test_available_false_when_runtime_missing(self) -> None:
        """Entity returns unavailable when runtime lookup fails."""
        hass = _make_hass()
        coordinator = FakeCoordinator(hass, dev_id="dev-pl")
        unique_id = build_gateway_entity_unique_id("dev-pl", "power_limit")
        entity = PowerLimitNumber(coordinator, "missing-entry", "dev-pl", unique_id)
        entity.hass = hass
        # No runtime installed for "missing-entry"
        assert entity.available is False


class TestPowerLimitEntitySetValue:
    """Tests for PowerLimitNumber.async_set_native_value."""

    @pytest.mark.asyncio
    async def test_set_value_calls_client(self) -> None:
        hass = _make_hass()
        coordinator = FakeCoordinator(hass, dev_id="dev-pl")
        mock_client = SimpleNamespace(
            set_power_limit=AsyncMock(),
        )
        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry-pl",
            dev_id="dev-pl",
            coordinator=coordinator,
            client=mock_client,
        )
        runtime.power_limit = 3000

        unique_id = build_gateway_entity_unique_id("dev-pl", "power_limit")
        entity = PowerLimitNumber(coordinator, "entry-pl", "dev-pl", unique_id)
        entity.hass = hass

        await entity.async_set_native_value(5000.0)

        mock_client.set_power_limit.assert_awaited_once_with("dev-pl", power_limit=5000)
        assert runtime.power_limit == 5000

    @pytest.mark.asyncio
    async def test_set_value_truncates_float(self) -> None:
        """Verify float values are truncated to int."""
        hass = _make_hass()
        coordinator = FakeCoordinator(hass, dev_id="dev-pl")
        mock_client = SimpleNamespace(
            set_power_limit=AsyncMock(),
        )
        build_entry_runtime(
            hass=hass,
            entry_id="entry-pl",
            dev_id="dev-pl",
            coordinator=coordinator,
            client=mock_client,
        )

        unique_id = build_gateway_entity_unique_id("dev-pl", "power_limit")
        entity = PowerLimitNumber(coordinator, "entry-pl", "dev-pl", unique_id)
        entity.hass = hass

        await entity.async_set_native_value(4999.7)

        mock_client.set_power_limit.assert_awaited_once_with("dev-pl", power_limit=4999)


class TestPowerLimitEntityDeviceInfo:
    """Tests for PowerLimitNumber.device_info."""

    def test_device_info_returns_installation_info(self) -> None:
        entity = _make_power_limit_entity()
        info = entity.device_info
        assert info is not None
        identifiers = info.get("identifiers")
        assert identifiers is not None
        assert (DOMAIN, "dev-pl", "site") in identifiers


class TestPowerLimitEntityAttributes:
    """Tests for PowerLimitNumber class attributes."""

    def test_entity_category_is_config(self) -> None:
        from homeassistant.helpers.entity import EntityCategory

        entity = _make_power_limit_entity()
        assert entity._attr_entity_category == EntityCategory.CONFIG

    def test_translation_key(self) -> None:
        entity = _make_power_limit_entity()
        assert entity._attr_translation_key == "installation_power_limit"

    def test_unit_of_measurement(self) -> None:
        from homeassistant.const import UnitOfPower

        entity = _make_power_limit_entity()
        assert entity._attr_native_unit_of_measurement == UnitOfPower.WATT

    def test_min_max_step(self) -> None:
        entity = _make_power_limit_entity()
        assert entity._attr_native_min_value == 0
        assert entity._attr_native_max_value == 60000
        assert entity._attr_native_step == 100

    def test_unique_id(self) -> None:
        entity = _make_power_limit_entity()
        assert "power_limit" in entity._attr_unique_id


# ===========================================================================
# Brand Gating Tests
# ===========================================================================


class TestBrandGating:
    """Verify PowerLimitNumber is created only for TermoWeb brands."""

    @pytest.mark.asyncio
    async def test_power_limit_created_for_termoweb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hass = _make_hass()
        entry_id = "entry-brand-tw"
        dev_id = "dev-brand-tw"

        coordinator = FakeCoordinator(hass, dev_id=dev_id)
        build_entry_runtime(
            hass=hass,
            entry_id=entry_id,
            dev_id=dev_id,
            coordinator=coordinator,
            brand=BRAND_TERMOWEB,
        )

        # Patch the accumulator details to return empty (no accumulators)
        monkeypatch.setattr(
            entities_number_module,
            "boostable_accumulator_details_for_entry",
            lambda *_args, **_kwargs: (
                SimpleNamespace(
                    inventory=None,
                    default_name_simple=lambda addr: f"Heater {addr}",
                    iter_metadata=lambda: iter([]),
                ),
                [],
            ),
        )
        monkeypatch.setattr(
            entities_number_module,
            "heater_platform_details_for_entry",
            lambda *_args, **_kwargs: SimpleNamespace(
                inventory=None,
                iter_metadata=lambda: iter([]),
            ),
        )

        created: list[list] = []

        def fake_add(entities: list) -> None:
            created.append(entities)

        await entities_number_module.async_setup_entry(
            hass,
            SimpleNamespace(entry_id=entry_id),
            fake_add,
        )

        assert created, "async_add_entities should be called"
        all_entities = created[0]
        power_limit_entities = [
            e for e in all_entities if isinstance(e, PowerLimitNumber)
        ]
        assert len(power_limit_entities) == 1

    @pytest.mark.asyncio
    async def test_power_limit_not_created_for_ducaheat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hass = _make_hass()
        entry_id = "entry-brand-dh"
        dev_id = "dev-brand-dh"

        coordinator = FakeCoordinator(hass, dev_id=dev_id)
        build_entry_runtime(
            hass=hass,
            entry_id=entry_id,
            dev_id=dev_id,
            coordinator=coordinator,
            brand=BRAND_DUCAHEAT,
        )

        monkeypatch.setattr(
            entities_number_module,
            "boostable_accumulator_details_for_entry",
            lambda *_args, **_kwargs: (
                SimpleNamespace(
                    inventory=None,
                    default_name_simple=lambda addr: f"Heater {addr}",
                    iter_metadata=lambda: iter([]),
                ),
                [],
            ),
        )
        monkeypatch.setattr(
            entities_number_module,
            "heater_platform_details_for_entry",
            lambda *_args, **_kwargs: SimpleNamespace(
                inventory=None,
                iter_metadata=lambda: iter([]),
            ),
        )

        created: list[list] = []

        def fake_add(entities: list) -> None:
            created.append(entities)

        await entities_number_module.async_setup_entry(
            hass,
            SimpleNamespace(entry_id=entry_id),
            fake_add,
        )

        if created:
            all_entities = created[0]
            power_limit_entities = [
                e for e in all_entities if isinstance(e, PowerLimitNumber)
            ]
            assert len(power_limit_entities) == 0


# ===========================================================================
# WebSocket Tests
# ===========================================================================


class TestWSPowerLimitUpdate:
    """Tests for WebSocket power_limit handling."""

    def _make_ws_client(
        self, monkeypatch: pytest.MonkeyPatch, *, entry_id: str = "entry"
    ) -> tuple[ws_module.TermoWebWSClient, HomeAssistant, Any]:
        """Create a TermoWebWSClient wired for testing."""
        hass = _make_hass()
        hass.loop = SimpleNamespace(
            call_soon_threadsafe=lambda cb, *args: cb(*args),
            is_running=lambda: False,
        )
        coordinator = SimpleNamespace(
            data={},
            update_nodes=MagicMock(),
            async_set_updated_data=MagicMock(),
        )

        monkeypatch.setattr(
            ws_module.TermoWebWSClient, "_install_write_hook", lambda self: None
        )

        client = ws_module.TermoWebWSClient(
            hass,
            entry_id=entry_id,
            dev_id="device",
            api_client=DummyREST(),
            coordinator=coordinator,
            session=SimpleNamespace(closed=False),
        )
        return client, hass, coordinator

    def test_handle_power_limit_update_sets_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct call to _handle_power_limit_update stores value on runtime."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)

        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )

        client._handle_power_limit_update({"power_limit": "4200"})

        assert runtime.power_limit == 4200

    def test_handle_power_limit_update_non_mapping_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-mapping body is silently ignored."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )

        # Should not raise
        client._handle_power_limit_update("not a mapping")
        client._handle_power_limit_update(None)
        client._handle_power_limit_update(42)

    def test_handle_power_limit_update_missing_key_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Body without power_limit key is ignored."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )
        runtime.power_limit = 1000

        client._handle_power_limit_update({"other_key": "value"})
        assert runtime.power_limit == 1000

    def test_handle_power_limit_update_non_numeric_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-numeric power_limit value is ignored."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )
        runtime.power_limit = 1000

        client._handle_power_limit_update({"power_limit": "abc"})
        assert runtime.power_limit == 1000

    def test_ws_legacy_data_batch_routes_power_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate 'data' event with htr_system/power_limit path."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )

        batch = [
            {
                "path": "/api/v2/devs/device/htr_system/power_limit",
                "body": {"power_limit": "7500"},
            }
        ]

        client._handle_legacy_data_batch(batch)

        assert runtime.power_limit == 7500

    def test_ws_apply_nodes_payload_routes_power_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate update event with htr_system path that fails translate_path."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )

        from custom_components.termoweb.inventory import Inventory

        client._inventory = Inventory("device", [])

        payload = {
            "path": "/api/v2/devs/device/htr_system/power_limit",
            "body": {"power_limit": "9000"},
        }

        client._apply_nodes_payload(payload, merge=True, event="update")

        assert runtime.power_limit == 9000

    def test_handle_power_limit_triggers_coordinator_refresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify coordinator.async_set_updated_data is called after update."""
        client, hass, coordinator = self._make_ws_client(monkeypatch)
        build_entry_runtime(
            hass=hass,
            entry_id="entry",
            dev_id="device",
            coordinator=coordinator,
        )

        client._handle_power_limit_update({"power_limit": "3000"})

        coordinator.async_set_updated_data.assert_called_once()


# ===========================================================================
# Coordinator Tests
# ===========================================================================


class TestCoordinatorPowerLimitPolling:
    """Tests for StateCoordinator power_limit polling behavior."""

    @pytest.mark.asyncio
    async def test_coordinator_polls_power_limit_termoweb(self) -> None:
        """Verify get_power_limit is called for TermoWeb brand."""
        from custom_components.termoweb.coordinator import StateCoordinator
        from custom_components.termoweb.inventory import Inventory

        hass = _make_hass()
        mock_client = MagicMock(spec=RESTClient)
        mock_client.get_node_settings = AsyncMock(return_value={})
        mock_client.get_power_limit = AsyncMock(return_value=6000)
        mock_client.get_rtc_time = AsyncMock(return_value={})

        inventory = Inventory("dev-coord", [])

        coord = StateCoordinator(
            hass,
            mock_client,
            base_interval=30,
            dev_id="dev-coord",
            device=None,
            nodes=None,
            inventory=inventory,
            brand=BRAND_TERMOWEB,
            entry_id="entry-coord",
        )

        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry-coord",
            dev_id="dev-coord",
            coordinator=coord,
            client=mock_client,
            inventory=inventory,
        )

        await coord._async_update_data()

        mock_client.get_power_limit.assert_awaited()
        assert runtime.power_limit == 6000

    @pytest.mark.asyncio
    async def test_coordinator_skips_power_limit_ducaheat(self) -> None:
        """Verify get_power_limit is NOT called for Ducaheat brand."""
        from custom_components.termoweb.coordinator import StateCoordinator
        from custom_components.termoweb.inventory import Inventory

        hass = _make_hass()
        mock_client = MagicMock(spec=RESTClient)
        mock_client.get_node_settings = AsyncMock(return_value={})
        mock_client.get_power_limit = AsyncMock(return_value=6000)
        mock_client.get_rtc_time = AsyncMock(return_value={})

        inventory = Inventory("dev-coord-dh", [])

        coord = StateCoordinator(
            hass,
            mock_client,
            base_interval=30,
            dev_id="dev-coord-dh",
            device=None,
            nodes=None,
            inventory=inventory,
            brand=BRAND_DUCAHEAT,
            entry_id="entry-coord-dh",
        )

        build_entry_runtime(
            hass=hass,
            entry_id="entry-coord-dh",
            dev_id="dev-coord-dh",
            coordinator=coord,
            client=mock_client,
            inventory=inventory,
            brand=BRAND_DUCAHEAT,
        )

        await coord._async_update_data()

        mock_client.get_power_limit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_coordinator_handles_power_limit_failure(self) -> None:
        """Verify coordinator continues when get_power_limit raises."""
        from custom_components.termoweb.coordinator import StateCoordinator
        from custom_components.termoweb.inventory import Inventory

        hass = _make_hass()
        mock_client = MagicMock(spec=RESTClient)
        mock_client.get_node_settings = AsyncMock(return_value={})
        mock_client.get_power_limit = AsyncMock(side_effect=RuntimeError("network"))
        mock_client.get_rtc_time = AsyncMock(return_value={})

        inventory = Inventory("dev-coord-err", [])

        coord = StateCoordinator(
            hass,
            mock_client,
            base_interval=30,
            dev_id="dev-coord-err",
            device=None,
            nodes=None,
            inventory=inventory,
            brand=BRAND_TERMOWEB,
            entry_id="entry-coord-err",
        )

        runtime = build_entry_runtime(
            hass=hass,
            entry_id="entry-coord-err",
            dev_id="dev-coord-err",
            coordinator=coord,
            client=mock_client,
            inventory=inventory,
        )
        runtime.power_limit = 1000

        # Should not raise despite get_power_limit failing
        result = await coord._async_update_data()
        assert isinstance(result, dict)
        # power_limit stays at its previous value since the update failed
        assert runtime.power_limit == 1000


# ===========================================================================
# Identifiers Tests
# ===========================================================================


class TestBuildGatewayEntityUniqueId:
    """Tests for the build_gateway_entity_unique_id helper."""

    def test_returns_expected_format(self) -> None:
        uid = build_gateway_entity_unique_id("mydev", "power_limit")
        assert uid == f"{DOMAIN}:mydev:power_limit"

    def test_raises_on_empty_dev_id(self) -> None:
        with pytest.raises(ValueError, match="dev_id must be provided"):
            build_gateway_entity_unique_id("", "power_limit")

    def test_none_dev_id_produces_valid_id(self) -> None:
        """normalize_node_addr coerces None to 'None' string."""
        uid = build_gateway_entity_unique_id(None, "power_limit")
        assert "power_limit" in uid


# ===========================================================================
# Runtime Tests
# ===========================================================================


class TestRuntimePowerLimitField:
    """Tests for the power_limit field on EntryRuntime."""

    def test_power_limit_defaults_to_none(self) -> None:
        runtime = build_entry_runtime()
        assert runtime.power_limit is None

    def test_power_limit_can_be_set(self) -> None:
        runtime = build_entry_runtime()
        runtime.power_limit = 5000
        assert runtime.power_limit == 5000

    def test_power_limit_can_be_reset_to_none(self) -> None:
        runtime = build_entry_runtime()
        runtime.power_limit = 5000
        runtime.power_limit = None
        assert runtime.power_limit is None
