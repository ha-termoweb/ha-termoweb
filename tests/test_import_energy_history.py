from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone, timedelta
import importlib
import itertools
import logging
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.termoweb import utils as utils_module

from conftest import _install_stubs

_install_stubs()


async def _load_module(
    monkeypatch: pytest.MonkeyPatch, *, legacy: bool = False, load_coordinator: bool = False
):
    _install_stubs()

    stats_module = types.ModuleType("homeassistant.components.recorder.statistics")
    last_stats = AsyncMock(return_value={})
    get_period = AsyncMock(return_value={})
    delete_stats = AsyncMock()
    stats_module.async_get_last_statistics = last_stats
    stats_module.async_get_statistics_during_period = get_period
    stats_module.async_delete_statistics = delete_stats

    if legacy:
        import_stats = Mock()
        update_meta = None
        stats_module.async_add_external_statistics = import_stats
    else:
        import_stats = Mock()
        update_meta = Mock()
        stats_module.async_import_statistics = import_stats
        stats_module.async_update_statistics_metadata = update_meta
        stats_module.async_add_external_statistics = Mock()

    sys.modules.setdefault(
        "homeassistant.components.recorder",
        types.ModuleType("homeassistant.components.recorder"),
    )
    sys.modules["homeassistant.components.recorder.statistics"] = stats_module

    entity_registry_module = importlib.import_module("homeassistant.helpers.entity_registry")

    class EntityEntry:
        def __init__(
            self,
            entity_id: str,
            name: str,
            *,
            platform: str | None = None,
            unique_id: str | None = None,
            config_entry_id: str | None = None,
        ) -> None:
            self.entity_id = entity_id
            self.original_name = name
            self.platform = platform
            self.unique_id = unique_id
            self.config_entry_id = config_entry_id

    class EntityRegistry:
        def __init__(self) -> None:
            self._by_uid: dict[tuple[str, str, str], str] = {}
            self._by_entity: dict[str, EntityEntry] = {}

        def add(
            self,
            entity_id: str,
            domain: str,
            platform: str,
            unique_id: str,
            name: str,
            config_entry_id: str | None = None,
        ) -> None:
            self._by_uid[(domain, platform, unique_id)] = entity_id
            self._by_entity[entity_id] = EntityEntry(
                entity_id,
                name,
                platform=platform,
                unique_id=unique_id,
                config_entry_id=config_entry_id,
            )

        def async_get_entity_id(
            self, domain: str, platform: str, unique_id: str
        ) -> str | None:
            return self._by_uid.get((domain, platform, unique_id))

        def async_get(self, entity_id: str) -> EntityEntry | None:
            return self._by_entity.get(entity_id)

    ent_reg = EntityRegistry()
    entity_registry_module.EntityEntry = EntityEntry
    entity_registry_module.EntityRegistry = EntityRegistry
    entity_registry_module.async_get = lambda hass: ent_reg
    entity_registry_module.async_get_registry = entity_registry_module.async_get

    helpers_module = importlib.import_module("homeassistant.helpers")
    helpers_module.entity_registry = entity_registry_module

    api_module = importlib.import_module("custom_components.termoweb.api")

    class _FakeRESTClient:
        def __init__(self, session, username, password, **kwargs: Any) -> None:
            self._session = session
            self._username = username
            self._password = password
            self._api_base = kwargs.get("api_base")
            self._basic_auth_b64 = kwargs.get("basic_auth_b64")

        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev", "name": "Device"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {"nodes": [{"type": "htr", "addr": "A"}]}

        async def get_htr_settings(self, dev_id: str, addr: str) -> dict[str, Any]:
            return {}

        async def get_node_settings(
            self, dev_id: str, node: tuple[str, str | int]
        ) -> dict[str, Any]:
            return {}

        async def get_node_samples(
            self,
            dev_id: str,
            node: tuple[str, str | int],
            start: int | None = None,
            stop: int | None = None,
        ) -> list[dict[str, Any]]:
            return []

        async def get_htr_samples(
            self,
            dev_id: str,
            addr: str,
            start: int | None = None,
            stop: int | None = None,
        ) -> list[dict[str, Any]]:
            return await self.get_node_samples(dev_id, ("htr", addr), start, stop)

    monkeypatch.setattr(api_module, "RESTClient", _FakeRESTClient)

    ws_module = importlib.import_module("custom_components.termoweb.ws_client")

    class _FakeWSClient:
        def __init__(self, hass: Any, dev_id: str, *args: Any, **kwargs: Any) -> None:
            self.hass = hass
            self.dev_id = dev_id

        def start(self) -> asyncio.Task:
            return asyncio.create_task(asyncio.sleep(0))

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(ws_module, "WebSocket09Client", _FakeWSClient)

    if load_coordinator:
        importlib.reload(importlib.import_module("custom_components.termoweb.coordinator"))
        importlib.reload(importlib.import_module("custom_components.termoweb.sensor"))
    else:
        importlib.reload(importlib.import_module("custom_components.termoweb.coordinator"))

    const_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.const")
    )
    init_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.__init__")
    )
    monkeypatch.setattr(init_module, "WebSocket09Client", _FakeWSClient)

    ConfigEntry = importlib.import_module("homeassistant.config_entries").ConfigEntry
    HomeAssistant = importlib.import_module("homeassistant.core").HomeAssistant

    return (
        init_module,
        const_module,
        import_stats,
        update_meta,
        last_stats,
        get_period,
        delete_stats,
        ConfigEntry,
        HomeAssistant,
        ent_reg,
    )


def _inventory_for(
    mod: Any, nodes: dict[str, list[str]] | list[str]
) -> list[Any]:
    if isinstance(nodes, dict):
        payload_nodes = [
            {"addr": addr, "type": node_type}
            for node_type, addrs in nodes.items()
            for addr in addrs
        ]
    else:
        payload_nodes = [{"addr": addr, "type": "htr"} for addr in nodes]
    return mod.build_node_inventory({"nodes": payload_nodes})


def test_store_statistics_prefers_internal_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            _const,
            import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            _ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        stats_module = sys.modules["homeassistant.components.recorder.statistics"]
        external = Mock()
        stats_module.async_add_external_statistics = external

        metadata = {"statistic_id": "sensor.test_energy", "source": "recorder"}
        stats = [{"sum": 1.23}]

        mod._store_statistics(hass, metadata, stats)

        import_stats.assert_called_once_with(hass, metadata, stats)
        external.assert_not_called()

    asyncio.run(_run())


def test_store_statistics_external_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            _const,
            import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            _ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        stats_module = sys.modules["homeassistant.components.recorder.statistics"]
        external = Mock()
        stats_module.async_add_external_statistics = external
        if hasattr(stats_module, "async_import_statistics"):
            delattr(stats_module, "async_import_statistics")

        metadata = {
            "statistic_id": "sensor.test_energy",
            "source": "recorder",
            "name": "Test Energy",
            "unit_of_measurement": "kWh",
        }
        stats = [{"sum": 2.34}]

        mod._store_statistics(hass, metadata, stats)

        import_stats.assert_not_called()
        external.assert_called_once()
        args, _kwargs = external.call_args
        assert args[0] is hass
        ext_meta = args[1]
        assert ext_meta["statistic_id"] == "sensor:test_energy"
        assert ext_meta["source"] == "sensor"
        assert metadata["statistic_id"] == "sensor.test_energy"
        assert args[2] is stats

    asyncio.run(_run())


def test_async_import_energy_history_missing_record(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        update_entry = Mock()
        hass.config_entries = types.SimpleNamespace(async_update_entry=update_entry)

        entry = ConfigEntry(
            "1",
            options={
                "sentinel": True,
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {"A": 1},
            },
        )
        original_options = copy.deepcopy(entry.options)

        caplog.set_level(logging.DEBUG, logger=mod.__name__)

        await mod._async_import_energy_history(hass, entry)

        update_entry.assert_not_called()
        assert entry.options == original_options

    asyncio.run(_run())

    assert "no record found for energy import" in caplog.text


def test_async_import_energy_history_rebuilds_missing_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.config_entries = types.SimpleNamespace(async_update_entry=Mock())

        entry = ConfigEntry(
            "1",
            options={mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
        )

        nodes_payload = {"nodes": [{"addr": "A", "type": "htr"}]}
        client = types.SimpleNamespace()
        hass.data = {
            const.DOMAIN: {
                entry.entry_id: {
                    "client": client,
                    "dev_id": "dev",
                    "nodes": nodes_payload,
                }
            }
        }

        await mod._async_import_energy_history(hass, entry)

        inventory = hass.data[const.DOMAIN][entry.entry_id]["node_inventory"]
        assert [node.addr for node in inventory] == ["A"]

    asyncio.run(_run())


def test_async_import_energy_history_already_imported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        update_entry = Mock()
        hass.config_entries = types.SimpleNamespace(async_update_entry=update_entry)

        entry = ConfigEntry(
            "1",
            options={
                mod.OPTION_ENERGY_HISTORY_IMPORTED: True,
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {"A": 123},
            },
        )

        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock()
        hass.data = {
            const.DOMAIN: {
                entry.entry_id: {
                    "client": client,
                    "dev_id": "dev",
                    "node_inventory": _inventory_for(mod, ["A"]),
                    "config_entry": entry,
                }
            }
        }

        caplog.set_level(logging.DEBUG, logger=mod.__name__)

        await mod._async_import_energy_history(hass, entry)

        client.get_node_samples.assert_not_awaited()
        update_entry.assert_not_called()
        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {"A": 123}

    asyncio.run(_run())

    assert "energy history already imported" in caplog.text


def test_async_import_energy_history_waits_between_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry(
            "1",
            options={mod.OPTION_MAX_HISTORY_RETRIEVED: 1},
        )

        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[])
        hass.data = {
            const.DOMAIN: {
                entry.entry_id: {
                    "client": client,
                    "dev_id": "dev",
                    "node_inventory": _inventory_for(mod, ["A"]),
                    "config_entry": entry,
                }
            }
        }

        monkeypatch.setattr(mod, "_LAST_SAMPLES_QUERY", 10.0)

        monotonic_values = iter([10.4, 11.4, 11.4])

        def fake_monotonic() -> float:
            try:
                return next(monotonic_values)
            except StopIteration:
                return 11.4

        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(monotonic=fake_monotonic, time=lambda: 0.0),
        )

        sleep_calls: list[float] = []

        class AsyncioProxy:
            CancelledError = asyncio.CancelledError
            Lock = asyncio.Lock

            def __getattr__(self, name: str):
                return getattr(asyncio, name)

            async def sleep(self, delay: float) -> None:
                sleep_calls.append(delay)

        monkeypatch.setattr(mod, "asyncio", AsyncioProxy())

        fake_now = 3 * 86_400
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(hass, entry)

        assert sleep_calls
        assert sleep_calls[0] == pytest.approx(0.6, rel=1e-3)
        assert client.get_node_samples.await_count == 1

    asyncio.run(_run())


def test_async_import_energy_history_skips_invalid_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        _updates: list[dict[str, dict[str, int]]] = []

        def record_update(entry, *, options):
            _updates.append(copy.deepcopy(options))
            entry.options.update(options)

        hass = HomeAssistant()
        hass.config_entries = types.SimpleNamespace(async_update_entry=record_update)

        entry = ConfigEntry(
            "1",
            options={mod.OPTION_MAX_HISTORY_RETRIEVED: 1},
        )

        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [],
                [
                    {"t": 200_000, "counter": "1000"},
                    {"t": 201_000, "counter": "1100"},
                ],
                [
                    {"t": 202_000, "counter": "abc"},
                    {"t": 203_000, "counter": "1200"},
                ],
                [
                    {"t": 204_000, "counter": "1300"},
                    {"t": 205_000, "counter": "1300"},
                ],
            ]
        )

        hass.data = {
            const.DOMAIN: {
                entry.entry_id: {
                    "client": client,
                    "dev_id": "dev",
                    "node_inventory": _inventory_for(mod, ["A", "B", "C", "D"]),
                    "config_entry": entry,
                }
            }
        }

        uid_c = f"{const.DOMAIN}:dev:htr:C:energy"
        ent_reg.add("sensor.dev_C_energy", "sensor", const.DOMAIN, uid_c, "C energy")
        uid_d = f"{const.DOMAIN}:dev:htr:D:energy"
        ent_reg.add("sensor.dev_D_energy", "sensor", const.DOMAIN, uid_d, "D energy")

        store = Mock()
        monkeypatch.setattr(mod, "_store_statistics", store)

        monkeypatch.setattr(mod, "_LAST_SAMPLES_QUERY", 0.0)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 5 * 86_400),
        )
        fake_now = 5 * 86_400
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(hass, entry)

        assert client.get_node_samples.await_count == 4
        store.assert_not_called()

    asyncio.run(_run())

def test_import_energy_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry("1", options={"max_history_retrieved": 2})
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [
                    {"t": 345_600, "counter": "3.0"},
                    {"t": 259_200, "counter": "2.0"},
                ],
                [{"t": 172_800, "counter": "1.0"}],
            ]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }
        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        fake_now = 4 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        assert client.get_node_samples.await_count >= 2
        first_call = client.get_node_samples.await_args_list[0][0]
        second_call = client.get_node_samples.await_args_list[1][0]
        assert first_call == ("dev", ("htr", "A"), 259_199, 345_599)
        assert second_call == ("dev", ("htr", "A"), 172_799, 259_199)

        get_period.assert_awaited_once()
        delete_stats.assert_not_awaited()
        last_stats.assert_called_once()
        assert captured["meta"]["statistic_id"] == "sensor.dev_A_energy"
        stats_list = captured["stats"]
        assert [s["sum"] for s in stats_list] == [
            pytest.approx(0.001),
            pytest.approx(0.002),
        ]
        assert [s["state"] for s in stats_list] == [
            pytest.approx(0.002),
            pytest.approx(0.003),
        ]
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {
            "htr:A": 172_799
        }
        assert entry.options["max_history_retrieved"] == 2

    asyncio.run(_run())


def test_import_energy_history_with_existing_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry("1", options={"max_history_retrieved": 2})
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [
                    {"t": 345_600, "counter": "3.0"},
                    {"t": 259_200, "counter": "2.0"},
                ],
                [{"t": 172_800, "counter": "1.0"}],
            ]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }
        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        fake_now = 4 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        start_prev = FakeDateTime.fromtimestamp(86_400, timezone.utc)
        get_period.return_value = {
            "sensor.dev_A_energy": [
                {"start": start_prev, "sum": 1.0, "state": 0.0},
            ]
        }

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        get_period.assert_awaited_once()
        delete_stats.assert_not_awaited()
        last_stats.assert_not_called()
        stats_list = captured["stats"]
        assert [s["sum"] for s in stats_list] == [
            pytest.approx(1.001),
            pytest.approx(1.002),
            pytest.approx(1.003),
        ]
        assert [s["state"] for s in stats_list] == [
            pytest.approx(0.001),
            pytest.approx(0.002),
            pytest.approx(0.003),
        ]

    asyncio.run(_run())


def test_import_energy_history_clears_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry("1", options={"max_history_retrieved": 2})
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [
                    {"t": 345_600, "counter": "3.0"},
                    {"t": 259_200, "counter": "2.0"},
                ],
                [{"t": 172_800, "counter": "1.0"}],
            ]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }
        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        fake_now = 4 * 86_400
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(time=lambda: fake_now, monotonic=lambda: fake_now),
        )
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        before_start = FakeDateTime.fromtimestamp(86_400, timezone.utc)
        overlap_start = FakeDateTime.fromtimestamp(259_200, timezone.utc)
        get_period.return_value = {
            "sensor.dev_A_energy": [
                {"start": before_start, "sum": 0.5, "state": 0.0},
                {"start": overlap_start, "sum": 0.6, "state": 0.002},
            ]
        }

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        get_period.assert_awaited_once()
        last_stats.assert_not_called()
        delete_stats.assert_awaited_once()
        del_args, del_kwargs = delete_stats.await_args
        assert del_args[0] is hass
        assert del_args[1] == ["sensor.dev_A_energy"]
        assert "start_time" in del_kwargs and "end_time" in del_kwargs
        assert del_kwargs["start_time"] <= overlap_start <= del_kwargs["end_time"]

        stats_list = captured["stats"]
        assert [s["sum"] for s in stats_list] == [
            pytest.approx(0.501),
            pytest.approx(0.502),
            pytest.approx(0.503),
        ]
        assert [s["state"] for s in stats_list] == [
            pytest.approx(0.001),
            pytest.approx(0.002),
            pytest.approx(0.003),
        ]
        starts = [s["start"] for s in stats_list]
        assert len(starts) == len(set(starts))

    asyncio.run(_run())


def test_import_energy_history_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            add_stats,
            _,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch, legacy=True)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry("1", options={"max_history_retrieved": 1})
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[
                {"t": 172_800, "counter": "2.0"},
                {"t": 86_400, "counter": "1.0"},
            ]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }
        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        fake_now = 2 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        get_period.assert_awaited_once()
        delete_stats.assert_not_awaited()
        last_stats.assert_called_once()
        meta = captured["meta"]
        stats = captured["stats"]
        assert meta["statistic_id"] == "sensor.dev_A_energy"
        assert meta["source"] == "recorder"
        assert stats[0]["sum"] == pytest.approx(0.001)
        assert stats[0]["state"] == pytest.approx(0.002)

    asyncio.run(_run())


def test_import_history_uses_last_stats_and_clears_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry(
            "import", options={mod.OPTION_MAX_HISTORY_RETRIEVED: 1}
        )

        client = types.SimpleNamespace()
        sample_list = [
            {"t": 345_600, "counter": "1000"},
            {"t": 349_200, "counter": "1250"},
            {"t": 352_800, "counter": "1500"},
        ]
        client.get_node_samples = AsyncMock(return_value=sample_list)

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }

        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        fake_now = 5 * 86_400

        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(time=lambda: fake_now, monotonic=lambda: 10.0),
        )

        stats_module = sys.modules["homeassistant.components.recorder.statistics"]
        stats_module.async_get_statistics_during_period = None

        import_start_dt = mod.datetime.fromtimestamp(
            sample_list[0]["t"], timezone.utc
        ).replace(minute=0, second=0, microsecond=0)
        before_start = import_start_dt - timedelta(hours=1)

        first_last = {
            "sensor.dev_A_energy": [
                {"start": before_start, "state": "1.5", "sum": "3.0"}
            ]
        }
        second_last = {"sensor.dev_A_energy": [{"start": import_start_dt}]}

        stats_module.async_get_last_statistics = AsyncMock(
            side_effect=[first_last, second_last]
        )

        stats_module.async_delete_statistics = AsyncMock(
            side_effect=[TypeError(), None]
        )

        captured: dict[str, Any] = {}

        def capture_stats(_hass, metadata, stats):
            captured.update(meta=metadata, stats=stats)

        monkeypatch.setattr(mod, "_store_statistics", capture_stats)

        await mod._async_import_energy_history(hass, entry)

        assert stats_module.async_get_last_statistics.await_count == 2
        assert stats_module.async_delete_statistics.await_count == 2
        first_call, second_call = stats_module.async_delete_statistics.await_args_list
        assert first_call.kwargs["start_time"] == import_start_dt
        assert "end_time" in first_call.kwargs
        assert second_call.args == (hass, ["sensor.dev_A_energy"])

        meta = captured["meta"]
        stats = captured["stats"]
        assert meta["statistic_id"] == "sensor.dev_A_energy"
        assert stats
        assert stats[0]["sum"] >= 0.0

    asyncio.run(_run())


def test_import_energy_history_reset_and_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry(
            "1",
            options={
                mod.OPTION_ENERGY_HISTORY_IMPORTED: True,
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {"A": 0, "B": 0},
                "max_history_retrieved": 1,
            },
        )
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 86_400, "counter": "1.0"}]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A", "B"]),
            "config_entry": entry,
        }
        uidA = f"{const.DOMAIN}:dev:htr:A:energy"
        uidB = f"{const.DOMAIN}:dev:htr:B:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uidA, "A energy")
        ent_reg.add("sensor.dev_B_energy", "sensor", const.DOMAIN, uidB, "B energy")

        fake_now = 2 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(hass, entry, ["A"], reset_progress=True)

        client.get_node_samples.assert_awaited_once_with(
            "dev", ("htr", "A"), 86_399, 172_799
        )
        last_stats.assert_called_once()
        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"htr:A": 86_399, "B": 0}
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True

    asyncio.run(_run())


def test_import_energy_history_reset_all_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}

        updates: list[dict] = []

        def update_entry(entry, *, options):
            updates.append(copy.deepcopy(options))
            entry.options.clear()
            entry.options.update(options)

        hass.config_entries = types.SimpleNamespace(
            async_update_entry=update_entry,
        )

        entry = ConfigEntry(
            "1",
            options={
                mod.OPTION_ENERGY_HISTORY_IMPORTED: True,
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {"A": 1, "B": 2},
                mod.OPTION_MAX_HISTORY_RETRIEVED: 1,
            },
        )

        call_log: list[tuple[str, str, str, int, int]] = []

        async def sample_side_effect(
            dev_id: str, node: tuple[str, str], start: int, stop: int
        ):
            node_type, addr = node
            call_log.append((dev_id, node_type, addr, start, stop))
            base = 0.0 if addr == "A" else 100.0
            return [
                {"t": start + 1, "counter": str(base + (start // 3600) + 1)},
                {"t": stop + 1, "counter": str(base + (stop // 3600) + 1.5)},
            ]

        client = types.SimpleNamespace(
            get_node_samples=AsyncMock(side_effect=sample_side_effect)
        )

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A", "B"]),
            "config_entry": entry,
        }

        uid_a = f"{const.DOMAIN}:dev:htr:A:energy"
        uid_b = f"{const.DOMAIN}:dev:htr:B:energy"
        ent_reg.add(
            "sensor.dev_A_energy",
            "sensor",
            const.DOMAIN,
            uid_a,
            "A energy",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.dev_B_energy",
            "sensor",
            const.DOMAIN,
            uid_b,
            "B energy",
            config_entry_id=entry.entry_id,
        )

        fake_now = 5 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )
        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(
            hass,
            entry,
            None,
            reset_progress=True,
            max_days=3,
        )

        assert client.get_node_samples.await_count == 6
        assert call_log == [
            ("dev", "htr", "A", 345_599, 431_999),
            ("dev", "htr", "A", 259_199, 345_599),
            ("dev", "htr", "A", 172_799, 259_199),
            ("dev", "htr", "B", 345_599, 431_999),
            ("dev", "htr", "B", 259_199, 345_599),
            ("dev", "htr", "B", 172_799, 259_199),
        ]
        assert len(updates) == client.get_node_samples.await_count + 2
        assert all(
            mod.OPTION_ENERGY_HISTORY_PROGRESS in update for update in updates
        )
        assert mod.OPTION_ENERGY_HISTORY_IMPORTED not in updates[0]
        assert all(
            mod.OPTION_ENERGY_HISTORY_IMPORTED not in update
            for update in updates[1:-1]
        )
        assert updates[0][mod.OPTION_ENERGY_HISTORY_PROGRESS] == {}
        final_update = updates[-1]
        assert final_update[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {
            "htr:A": 172_799,
            "htr:B": 172_799,
        }
        assert final_update[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True

        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"htr:A": 172_799, "htr:B": 172_799}
        assert entry.options[mod.OPTION_MAX_HISTORY_RETRIEVED] == 1
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert last_stats.await_count == 2

    asyncio.run(_run())


def test_import_energy_history_requested_map_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry(
            "req-map",
            options={
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {},
                "max_history_retrieved": 1,
            },
        )

        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [
                    {"t": 86_400, "counter": 1_000},
                    {"t": 92_400, "counter": 2_000},
                ],
                [
                    {"t": 86_400, "counter": 3_000},
                    {"t": 92_400, "counter": 5_000},
                ],
            ]
        )

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": [],
            "config_entry": entry,
        }

        helper_calls: list[dict[str, Any]] = []
        normalize_calls: list[Any] = []

        def fake_addresses_by_node_type(nodes, *, known_types=None):
            helper_calls.append(
                {
                    "nodes": list(nodes),
                    "known_types": None if known_types is None else set(known_types),
                }
            )
            assert known_types == mod.HEATER_NODE_TYPES
            return (
                {"htr": ["A"], "acm": ["B"], "pmo": ["ignored"]},
                set(),
            )

        monkeypatch.setattr(
            utils_module, "addresses_by_node_type", fake_addresses_by_node_type
        )

        original_normalize = mod.normalize_heater_addresses

        def fake_normalize(addrs: Any) -> tuple[dict[str, list[str]], dict[str, str]]:
            normalize_calls.append(addrs)
            return original_normalize(addrs)

        monkeypatch.setattr(mod, "normalize_heater_addresses", fake_normalize)

        uid_a = f"{const.DOMAIN}:dev:htr:A:energy"
        uid_b_legacy = f"{const.DOMAIN}:dev:htr:B:energy"
        ent_reg.add(
            "sensor.dev_A_energy",
            "sensor",
            const.DOMAIN,
            uid_a,
            "A energy",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.dev_B_energy",
            "sensor",
            const.DOMAIN,
            uid_b_legacy,
            "B energy",
            config_entry_id=entry.entry_id,
        )

        monotonic_counter = itertools.count(start=0.0, step=1.0)
        fake_now = 3 * 86_400

        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                monotonic=lambda: next(monotonic_counter),
                time=lambda: fake_now,
            ),
        )
        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(
            hass,
            entry,
            {
                "htr": ["A", "B", "B"],
                "acm": "B",
                "": ["ignored"],
                "thm": ["Z"],
                "pmo": [],
            },
        )

        assert client.get_node_samples.await_count >= 2
        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert set(progress) >= {"htr:A", "acm:B"}
        assert helper_calls
        assert helper_calls[0]["known_types"] == mod.HEATER_NODE_TYPES
        assert normalize_calls

    asyncio.run(_run())


def test_import_energy_history_ignores_unavailable_requested_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}

        def update_entry(entry: Any, *, options: dict[str, Any]) -> None:
            entry.options.clear()
            entry.options.update(options)

        hass.config_entries = types.SimpleNamespace(async_update_entry=update_entry)

        entry = ConfigEntry(
            "1",
            options={mod.OPTION_MAX_HISTORY_RETRIEVED: 0},
        )

        client = types.SimpleNamespace(
            get_node_samples=AsyncMock(return_value=[]),
        )

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, ["A"]),
            "config_entry": entry,
        }

        await mod._async_import_energy_history(
            hass,
            entry,
            {"acm": ["99"]},
        )

        client.get_node_samples.assert_not_called()

    asyncio.run(_run())


def test_import_energy_history_resets_requested_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options)
        )

        entry = ConfigEntry(
            "reset-map",
            options={
                mod.OPTION_ENERGY_HISTORY_PROGRESS: {
                    "htr:X": 123,
                    "X": 456,
                }
            },
        )

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": types.SimpleNamespace(
                get_node_samples=AsyncMock(return_value=[])
            ),
            "dev_id": "dev",
            "node_inventory": [],
            "config_entry": entry,
        }

        helper_calls: list[dict[str, Any]] = []

        def fake_addresses_by_node_type(nodes, *, known_types=None):
            helper_calls.append(
                {
                    "nodes": list(nodes),
                    "known_types": None if known_types is None else set(known_types),
                }
            )
            assert known_types == mod.HEATER_NODE_TYPES
            return ({"pmo": ["X"]}, set())

        monkeypatch.setattr(
            utils_module, "addresses_by_node_type", fake_addresses_by_node_type
        )

        fake_now = 5 * 86_400

        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return super().fromtimestamp(fake_now, tz)

        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                monotonic=lambda: 0.0,
                time=lambda: fake_now,
            ),
        )
        monkeypatch.setattr(mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(
            hass,
            entry,
            {"htr": ["X", ""], "": ["ignored"]},
            reset_progress=True,
        )

        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {}
        assert helper_calls
        assert helper_calls[0]["known_types"] == mod.HEATER_NODE_TYPES

    asyncio.run(_run())


def test_energy_polling_matches_import(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            _mod,
            _const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            _ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch, load_coordinator=True)

        coord_mod = importlib.import_module("custom_components.termoweb.coordinator")
        sensor_mod = importlib.import_module("custom_components.termoweb.sensor")

        hass = HomeAssistant()
        client = types.SimpleNamespace(
            get_node_samples=AsyncMock(
                side_effect=[
                    [{"t": 9_940, "counter": "1000"}],
                    [{"t": 13_540, "counter": "1300"}],
                ]
            )
        )
        client.get_htr_samples = client.get_node_samples

        coordinator = coord_mod.EnergyStateCoordinator(
            hass,
            client,
            "dev",
            ["A"],
        )

        times = iter([10_000.0, 13_600.0])
        monkeypatch.setattr(
            coord_mod,
            "time",
            types.SimpleNamespace(
                time=lambda: next(times, 13_600.0),
                monotonic=lambda: 0.0,
            ),
        )

        data1 = await coordinator._async_update_data()
        assert data1["dev"]["htr"]["energy"]["A"] == pytest.approx(1.0)
        assert data1["dev"]["htr"]["power"] == {}

        data2 = await coordinator._async_update_data()
        assert data2["dev"]["htr"]["energy"]["A"] == pytest.approx(1.3)
        assert data2["dev"]["htr"]["power"]["A"] == pytest.approx(300.0)

        coordinator.data = data2

        energy_entity = sensor_mod.HeaterEnergyTotalSensor(
            coordinator,
            "entry",
            "dev",
            "A",
            "Heater A",
            "uid",
            "Device A",
        )
        total_entity = sensor_mod.InstallationTotalEnergySensor(
            coordinator,
            "entry",
            "dev",
            "Total",
            "total_uid",
        )

        assert energy_entity.native_value == pytest.approx(1.3)
        assert total_entity.native_value == pytest.approx(1.3)

        await asyncio.sleep(0)

    asyncio.run(_run())


def test_setup_defers_import_until_started(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
            get_period,
            delete_stats,
            ConfigEntry,
            HomeAssistant,
            _,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        tasks: list[asyncio.Task] = []

        def async_create_task(coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        listeners: list[tuple[str, object]] = []

        class Bus:
            def async_listen_once(self, event, cb):
                listeners.append((event, cb))

        hass.bus = Bus()
        hass.async_create_task = async_create_task
        hass.is_running = False
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options),
            async_forward_entry_setups=AsyncMock(return_value=None),
        )

        class Services:
            def __init__(self):
                self._svcs = {}

            def has_service(self, domain, service):
                return service in self._svcs.get(domain, set())

            def async_register(self, domain, service, func):
                self._svcs.setdefault(domain, set()).add(service)

        hass.services = Services()

        entry = ConfigEntry("1", data={"username": "u", "password": "p"})
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": object(),
            "dev_id": "dev",
            "node_inventory": _inventory_for(mod, {"htr": ["A"], "acm": ["B"]}),
            "config_entry": entry,
        }

        import_mock = AsyncMock()
        monkeypatch.setattr(mod, "_async_import_energy_history", import_mock)

        assert await mod.async_setup_entry(hass, entry) is True
        import_mock.assert_not_called()

        assert len(listeners) == 1
        event, cb = listeners[0]
        assert event == mod.EVENT_HOMEASSISTANT_STARTED

        hass.async_create_task(cb(None))
        if tasks:
            await asyncio.gather(*tasks)

        import_mock.assert_awaited_once_with(hass, entry)

    asyncio.run(_run())


def test_service_dispatches_import_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}

        tasks: list[asyncio.Task] = []

        def async_create_task(coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        hass.async_create_task = async_create_task
        hass.is_running = True
        hass.bus = types.SimpleNamespace(async_listen_once=lambda event, cb: None)

        class Services:
            def __init__(self) -> None:
                self._svcs: dict[str, dict[str, object]] = {}

            def has_service(self, domain, service):
                return service in self._svcs.get(domain, {})

            def async_register(self, domain, service, func):
                self._svcs.setdefault(domain, {})[service] = func

            def get(self, domain, service):
                return self._svcs[domain][service]

        hass.services = Services()

        entry = ConfigEntry(
            "entry-1",
            data={"username": "u", "password": "p"},
            options={mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
        )

        entries = {entry.entry_id: entry}

        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda e, *, options: e.options.update(options),
            async_forward_entry_setups=AsyncMock(return_value=None),
            async_get_entry=lambda entry_id: entries.get(entry_id),
        )

        import_mock = AsyncMock()
        monkeypatch.setattr(mod, "_async_import_energy_history", import_mock)

        assert await mod.async_setup_entry(hass, entry) is True

        rec = hass.data[const.DOMAIN][entry.entry_id]
        rec["node_inventory"] = _inventory_for(mod, {"htr": ["A"], "acm": ["B"]})

        uid_a = f"{const.DOMAIN}:dev:htr:A:energy"
        uid_b = f"{const.DOMAIN}:dev:acm:B:energy"
        ent_reg.add(
            "sensor.dev_A_energy",
            "sensor",
            const.DOMAIN,
            uid_a,
            "A energy",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.dev_B_energy",
            "sensor",
            const.DOMAIN,
            uid_b,
            "B energy",
            config_entry_id=entry.entry_id,
        )

        service = hass.services.get(const.DOMAIN, "import_energy_history")

        import_mock.reset_mock()
        call = types.SimpleNamespace(
            data={
                "entity_id": ["sensor.dev_A_energy", "sensor.dev_B_energy"],
                "reset_progress": True,
                "max_history_retrieval": 5,
            }
        )
        await service(call)

        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert args[0] is hass
        assert args[1] is entry
        assert args[2] == {"htr": ["A"], "acm": ["B"]}
        assert kwargs == {"reset_progress": True, "max_days": 5}

        import_mock.reset_mock()
        call_all = types.SimpleNamespace(
            data={"max_history_retrieval": 2, "reset_progress": False}
        )
        await service(call_all)

        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert args[0] is hass
        assert args[1] is entry
        assert args[2] == {"htr": ["A"], "acm": ["B"]}
        assert kwargs == {"reset_progress": False, "max_days": 2}

        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(_run())


def test_service_filters_invalid_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}

        tasks: list[asyncio.Task] = []

        def async_create_task(coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        hass.async_create_task = async_create_task
        hass.is_running = True
        hass.bus = types.SimpleNamespace(async_listen_once=lambda event, cb: None)

        class Services:
            def __init__(self) -> None:
                self._svcs: dict[str, dict[str, object]] = {}

            def has_service(self, domain, service):
                return service in self._svcs.get(domain, {})

            def async_register(self, domain, service, func):
                self._svcs.setdefault(domain, {})[service] = func

            def get(self, domain, service):
                return self._svcs[domain][service]

        hass.services = Services()

        entry = ConfigEntry(
            "entry-valid",
            data={"username": "u", "password": "p"},
            options={mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
        )

        entries = {entry.entry_id: entry}

        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda e, *, options: e.options.update(options),
            async_forward_entry_setups=AsyncMock(return_value=None),
            async_get_entry=lambda entry_id: entries.get(entry_id),
        )

        import_mock = AsyncMock()
        monkeypatch.setattr(mod, "_async_import_energy_history", import_mock)

        assert await mod.async_setup_entry(hass, entry) is True

        rec = hass.data[const.DOMAIN][entry.entry_id]
        rec["node_inventory"] = _inventory_for(mod, {"htr": ["A"], "acm": ["B"]})

        ent_reg.add(
            "sensor.bad_prefix",
            "sensor",
            const.DOMAIN,
            "other:bad",
            "Bad prefix",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.bad_split",
            "sensor",
            const.DOMAIN,
            f"{const.DOMAIN}:invalid",
            "Bad split",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.bad_metric",
            "sensor",
            const.DOMAIN,
            f"{const.DOMAIN}:dev:htr:A:power",
            "Bad metric",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.bad_prefix_split",
            "sensor",
            const.DOMAIN,
            f"{const.DOMAIN}:dev:energy",
            "Bad prefix split",
            config_entry_id=entry.entry_id,
        )
        ent_reg.add(
            "sensor.no_entry",
            "sensor",
            const.DOMAIN,
            f"{const.DOMAIN}:dev:htr:D:energy",
            "Missing entry",
            config_entry_id="missing",
        )
        ent_reg.add(
            "sensor.no_config",
            "sensor",
            const.DOMAIN,
            f"{const.DOMAIN}:dev:htr:C:energy",
            "No config",
        )

        class DummySet(set):
            def add(self, value):
                return None

        monkeypatch.setattr(mod, "set", lambda: DummySet(), raising=False)

        service = hass.services.get(const.DOMAIN, "import_energy_history")

        call = types.SimpleNamespace(
            data={
                "entity_id": [
                    "sensor.bad_prefix",
                    "sensor.bad_split",
                    "sensor.bad_metric",
                    "sensor.bad_prefix_split",
                    "sensor.no_entry",
                    "sensor.no_config",
                ]
            }
        )

        await service(call)

        import_mock.assert_not_called()

        import_mock.reset_mock()

        hass.data[const.DOMAIN]["other"] = {"config_entry": None}

        call_all = types.SimpleNamespace(data={})
        await service(call_all)

        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert args[0] is hass
        assert args[1] is entry

        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(_run())


def test_refresh_fallback_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        climate_mod = importlib.import_module("custom_components.termoweb.climate")
        coordinator = types.SimpleNamespace(async_request_refresh=AsyncMock())
        heater = climate_mod.HeaterClimateEntity(coordinator, "1", "dev", "A", "Heater A")
        heater.hass = HomeAssistant()
        heater._schedule_refresh_fallback()
        task = heater._refresh_fallback
        assert task is not None
        await asyncio.sleep(0)
        await heater.async_will_remove_from_hass()
        await asyncio.sleep(0)
        assert task.cancelled()
        assert heater._refresh_fallback is None

    asyncio.run(_run())


def test_async_unload_entry_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}

        entry = ConfigEntry("entry-1")

        task_cancelled = False
        task_finalized = False
        client_stopped = False
        unsub_called = False
        recalc_state = {"value": False}

        async def ws_job() -> None:
            nonlocal task_cancelled, task_finalized
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                task_cancelled = True
                raise
            finally:
                task_finalized = True

        task = asyncio.create_task(ws_job())
        await asyncio.sleep(0)

        class DummyClient:
            async def stop(self) -> None:
                nonlocal client_stopped
                client_stopped = True

        def unsub() -> None:
            nonlocal unsub_called
            unsub_called = True

        hass.data[const.DOMAIN][entry.entry_id] = {
            "config_entry": entry,
            "ws_tasks": {"dev": task},
            "ws_clients": {"dev": DummyClient()},
            "unsub_ws_status": unsub,
            "recalc_poll": (
                lambda state=recalc_state: state.__setitem__(
                    "value", not state["value"]
                )
            ),
        }

        recalc_poll = hass.data[const.DOMAIN][entry.entry_id]["recalc_poll"]

        unload_platforms = AsyncMock(return_value=True)
        hass.config_entries = types.SimpleNamespace(
            async_unload_platforms=unload_platforms,
        )

        result = await mod.async_unload_entry(hass, entry)

        assert result is True
        assert task.cancelled()
        assert task_finalized
        assert task_cancelled
        assert client_stopped
        assert unsub_called
        unload_platforms.assert_awaited_once_with(entry, mod.PLATFORMS)
        assert entry.entry_id not in hass.data[const.DOMAIN]

        hass.data[const.DOMAIN][entry.entry_id] = {"recalc_poll": recalc_poll}
        await mod.async_update_entry_options(hass, entry)
        assert recalc_state["value"] is True

    asyncio.run(_run())
