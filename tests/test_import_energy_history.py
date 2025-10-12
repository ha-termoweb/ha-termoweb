from __future__ import annotations

import asyncio
import builtins
import copy
from datetime import datetime, timezone, timedelta
import importlib
import inspect
import itertools
import logging
import sys
import time
import types
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

from custom_components.termoweb import (
    heater as heater_module,
    identifiers as identifiers_module,
    inventory as inventory_module,
)
from custom_components.termoweb import throttle as throttle_module
from custom_components.termoweb.inventory import HEATER_NODE_TYPES

from conftest import _install_stubs

_install_stubs()


def _setup_last_statistics_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sync_helper: Any | None = None,
    async_helper: Any | None = None,
):
    """Prepare recorder statistics modules for _get_last_statistics_compat tests."""

    stats_module = types.ModuleType("homeassistant.components.recorder.statistics")
    if sync_helper is not None:
        stats_module.get_last_statistics = sync_helper  # type: ignore[attr-defined]
    if async_helper is not None:
        stats_module.async_get_last_statistics = async_helper  # type: ignore[attr-defined]

    recorder_module = types.ModuleType("homeassistant.components.recorder")

    class _RecorderInstance:
        async def async_add_executor_job(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    recorder_module.get_instance = (  # type: ignore[attr-defined]
        lambda _hass: _RecorderInstance()
    )
    recorder_module.statistics = stats_module  # type: ignore[attr-defined]

    components_module = types.ModuleType("homeassistant.components")
    components_module.recorder = recorder_module  # type: ignore[attr-defined]

    homeassistant_module = sys.modules.setdefault(
        "homeassistant", types.ModuleType("homeassistant")
    )
    monkeypatch.setattr(
        homeassistant_module,
        "components",
        components_module,
        raising=False,
    )

    monkeypatch.setitem(sys.modules, "homeassistant.components", components_module)
    monkeypatch.setitem(sys.modules, "homeassistant.components.recorder", recorder_module)
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder.statistics",
        stats_module,
    )

    energy_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.energy")
    )
    setattr(energy_module, "_RECORDER_IMPORTS", None)

    return energy_module, types.SimpleNamespace(), stats_module


def test_resolve_statistics_helpers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback import path should expose async helpers when attribute is missing."""

    energy_module = importlib.import_module("custom_components.termoweb.energy")
    reset_cache = getattr(energy_module, "_reset_integration_dependencies_cache", None)
    if reset_cache is not None:
        reset_cache()

    stats_module = types.ModuleType("homeassistant.components.recorder.statistics")

    def _async_helper(*_args: Any, **_kwargs: Any) -> None:
        return None

    stats_module.async_helper = _async_helper  # type: ignore[attr-defined]

    recorder_module = types.ModuleType("homeassistant.components.recorder")

    def _get_instance(_hass: Any) -> Any:
        return types.SimpleNamespace(async_add_executor_job=None)

    recorder_module.get_instance = _get_instance  # type: ignore[attr-defined]
    recorder_module.statistics = stats_module  # type: ignore[attr-defined]

    components_module = types.ModuleType("homeassistant.components")
    components_module.recorder = recorder_module  # type: ignore[attr-defined]

    homeassistant_module = sys.modules.setdefault(
        "homeassistant", types.ModuleType("homeassistant")
    )
    homeassistant_module.components = components_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "homeassistant.components", components_module)
    monkeypatch.setitem(
        sys.modules, "homeassistant.components.recorder", recorder_module
    )
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder.statistics",
        stats_module,
    )

    import_count = {"value": 0}
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals_dict: Any | None = None,
        locals_dict: Any | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        if (
            name == "homeassistant.components.recorder"
            and fromlist
            and "statistics" in fromlist
        ):
            if import_count["value"] == 0:
                import_count["value"] += 1
                raise ImportError("missing statistics")
        return real_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    helpers = energy_module._resolve_statistics_helpers(
        types.SimpleNamespace(),
        sync_name="missing_sync",
        async_name="async_helper",
    )

    assert helpers.sync is None
    assert helpers.async_fn is _async_helper


def test_resolve_statistics_helpers_targets_recorder_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorder helpers should retain executor and target selection."""

    energy_module = importlib.import_module("custom_components.termoweb.energy")
    reset_cache = getattr(energy_module, "_reset_integration_dependencies_cache", None)
    if reset_cache is not None:
        reset_cache()

    stats_module = types.ModuleType("homeassistant.components.recorder.statistics")

    def _sync_helper(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _async_helper(*_args: Any, **_kwargs: Any) -> None:
        return None

    stats_module.sync_helper = _sync_helper  # type: ignore[attr-defined]
    stats_module.async_helper = _async_helper  # type: ignore[attr-defined]

    recorder_instance = types.SimpleNamespace(
        async_add_executor_job=AsyncMock(name="async_add_executor_job")
    )

    recorder_module = types.ModuleType("homeassistant.components.recorder")

    def _get_instance(hass: Any) -> Any:
        return recorder_instance

    recorder_module.get_instance = _get_instance  # type: ignore[attr-defined]
    recorder_module.statistics = stats_module  # type: ignore[attr-defined]

    components_module = types.ModuleType("homeassistant.components")
    components_module.recorder = recorder_module  # type: ignore[attr-defined]

    homeassistant_module = sys.modules.setdefault(
        "homeassistant", types.ModuleType("homeassistant")
    )
    homeassistant_module.components = components_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "homeassistant.components", components_module)
    monkeypatch.setitem(
        sys.modules, "homeassistant.components.recorder", recorder_module
    )
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder.statistics",
        stats_module,
    )

    setattr(energy_module, "_RECORDER_IMPORTS", None)

    hass = types.SimpleNamespace()

    helpers_hass = energy_module._resolve_statistics_helpers(
        hass,
        sync_name="sync_helper",
        async_name="async_helper",
        sync_uses_instance=False,
    )

    assert helpers_hass.executor is recorder_instance.async_add_executor_job
    assert helpers_hass.sync_target is hass
    assert helpers_hass.sync is _sync_helper
    assert helpers_hass.async_fn is _async_helper

    helpers_instance = energy_module._resolve_statistics_helpers(
        hass,
        sync_name="sync_helper",
        async_name="async_helper",
        sync_uses_instance=True,
    )

    assert helpers_instance.executor is recorder_instance.async_add_executor_job
    assert helpers_instance.sync_target is recorder_instance
    assert helpers_instance.sync is _sync_helper
    assert helpers_instance.async_fn is _async_helper


@pytest.mark.asyncio
async def test_get_last_statistics_compat_uses_modern_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modern sync helpers should accept the start_time placeholder."""

    calls: dict[str, Any] = {}

    def _modern_helper(
        hass_obj: Any,
        number_of_stats: int,
        statistic_id: str,
        types: set[str],
        start_time: datetime | None,
    ) -> dict[str, list[Any]]:
        calls["args"] = (
            hass_obj,
            number_of_stats,
            statistic_id,
            types,
            start_time,
        )
        return {statistic_id: []}

    energy_module, hass, _ = _setup_last_statistics_environment(
        monkeypatch, sync_helper=_modern_helper
    )

    result = await energy_module._get_last_statistics_compat(
        hass,
        3,
        "sensor.test",
        types={"state"},
    )

    assert result == {"sensor.test": []}
    assert calls["args"] == (hass, 3, "sensor.test", {"state"}, None)


@pytest.mark.asyncio
async def test_get_last_statistics_compat_handles_legacy_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy sync helpers without start_time should still be supported."""

    calls = {"count": 0, "args": None}

    def _legacy_helper(
        hass_obj: Any, number_of_stats: int, statistic_id: str, types: set[str]
    ) -> dict[str, list[Any]]:
        calls["count"] += 1
        calls["args"] = (hass_obj, number_of_stats, statistic_id, types)
        return {statistic_id: []}

    energy_module, hass, _ = _setup_last_statistics_environment(
        monkeypatch, sync_helper=_legacy_helper
    )

    result = await energy_module._get_last_statistics_compat(
        hass,
        2,
        "sensor.legacy",
        types={"sum"},
    )

    assert result == {"sensor.legacy": []}
    assert calls["count"] == 1
    assert calls["args"] == (hass, 2, "sensor.legacy", {"sum"})


@pytest.mark.asyncio
async def test_get_last_statistics_compat_async_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async helpers should receive keyword-only compatibility arguments."""

    async_helper = AsyncMock(return_value={"sensor.async": []})

    energy_module, hass, _ = _setup_last_statistics_environment(
        monkeypatch, async_helper=async_helper
    )

    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = await energy_module._get_last_statistics_compat(
        hass,
        1,
        "sensor.async",
        types={"state"},
        start_time=start_time,
    )

    assert result == {"sensor.async": []}
    async_helper.assert_awaited_once()
    await_call = async_helper.await_args
    assert await_call.args == (hass, 1, ["sensor.async"])
    assert await_call.kwargs["types"] == {"state"}
    assert await_call.kwargs["start_time"] is start_time


async def _load_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy: bool = False,
    load_coordinator: bool = False,
    patch_compat: bool = True,
    new_api_only: bool = False,
):
    _install_stubs()

    stats_module = types.ModuleType("homeassistant.components.recorder.statistics")
    last_stats = AsyncMock(return_value={})
    get_period = AsyncMock(return_value={})
    delete_stats = AsyncMock()
    stats_module.async_get_last_statistics = last_stats
    stats_module.async_get_statistics_during_period = get_period
    stats_module.async_delete_statistics = delete_stats

    sync_last_stats = Mock(return_value={})
    sync_get_period = Mock(return_value={})
    sync_clear_stats = Mock()
    stats_module.get_last_statistics = sync_last_stats
    stats_module.statistics_during_period = sync_get_period
    stats_module.clear_statistics = sync_clear_stats

    if new_api_only:
        stats_module.async_get_last_statistics = None  # type: ignore[attr-defined]
        stats_module.async_get_statistics_during_period = None  # type: ignore[attr-defined]
        stats_module.async_delete_statistics = None  # type: ignore[attr-defined]

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
    recorder_module = sys.modules["homeassistant.components.recorder"]

    class _RecorderInstance:
        def __init__(self) -> None:
            async def _call(func, *args, **kwargs):
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

            self.async_add_executor_job = AsyncMock(side_effect=_call)

    recorder_instance = _RecorderInstance()
    recorder_module.get_instance = lambda hass: recorder_instance  # type: ignore[attr-defined]
    sys.modules["homeassistant.components.recorder.statistics"] = stats_module

    entity_registry_module = importlib.import_module(
        "homeassistant.helpers.entity_registry"
    )

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

    monkeypatch.setattr(api_module, "RESTClient", _FakeRESTClient)

    ws_module = importlib.import_module(
        "custom_components.termoweb.backend.ws_client"
    )
    termoweb_ws_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb_ws"
    )

    class _FakeWSClient:
        def __init__(self, hass: Any, dev_id: str, *args: Any, **kwargs: Any) -> None:
            self.hass = hass
            self.dev_id = dev_id

        def start(self) -> asyncio.Task:
            return asyncio.create_task(asyncio.sleep(0))

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(ws_module, "TermoWebWSClient", _FakeWSClient)
    monkeypatch.setattr(termoweb_ws_module, "TermoWebWSClient", _FakeWSClient)
    backend_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb"
    )
    monkeypatch.setattr(
        backend_module, "TermoWebWSClient", _FakeWSClient, raising=False
    )

    energy_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.energy")
    )
    reset_cache = getattr(energy_module, "_reset_integration_dependencies_cache", None)
    if reset_cache is not None:
        reset_cache()
    throttle_module.reset_samples_rate_limit_state()

    if load_coordinator:
        importlib.reload(
            importlib.import_module("custom_components.termoweb.coordinator")
        )
        importlib.reload(importlib.import_module("custom_components.termoweb.sensor"))
    else:
        importlib.reload(
            importlib.import_module("custom_components.termoweb.coordinator")
        )

    const_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.const")
    )
    init_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.__init__")
    )

    compat_last = AsyncMock(return_value={})
    compat_period = AsyncMock(return_value={})
    compat_clear = AsyncMock(return_value="delete")

    if patch_compat:
        monkeypatch.setattr(energy_module, "_get_last_statistics_compat", compat_last)
        monkeypatch.setattr(
            energy_module, "_statistics_during_period_compat", compat_period
        )
        monkeypatch.setattr(energy_module, "_clear_statistics_compat", compat_clear)
    else:
        compat_last = None
        compat_period = None
        compat_clear = None

    ConfigEntry = importlib.import_module("homeassistant.config_entries").ConfigEntry
    HomeAssistant = importlib.import_module("homeassistant.core").HomeAssistant

    return (
        init_module,
        energy_module,
        const_module,
        import_stats,
        update_meta,
        compat_last or last_stats,
        compat_period or get_period,
        compat_clear or delete_stats,
        ConfigEntry,
        HomeAssistant,
        ent_reg,
    )


@pytest.mark.asyncio
async def test_statistics_during_period_compat_sync_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronous helper should receive default types set."""

    energy_module = importlib.import_module("custom_components.termoweb.energy")

    executor = AsyncMock(return_value={})
    sync_helper = Mock(return_value={})

    monkeypatch.setattr(
        energy_module,
        "_resolve_statistics_helpers",
        lambda *args, **kwargs: energy_module._RecorderStatisticsHelpers(
            executor=executor,
            sync_target="hass",
            sync=sync_helper,
            async_fn=None,
        ),
    )

    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=1)
    hass = types.SimpleNamespace()

    await energy_module._statistics_during_period_compat(
        hass, start, end, {"sensor.energy"}
    )

    assert executor.await_count == 1
    exec_args = executor.await_args.args
    assert exec_args[0] is sync_helper
    assert exec_args[1] == "hass"
    assert exec_args[2] == start
    assert exec_args[3] == end
    assert exec_args[4] == {"sensor.energy"}
    assert exec_args[5] == "hour"
    assert exec_args[6] is None
    assert exec_args[7] == {"state", "sum"}


@pytest.mark.asyncio
async def test_statistics_during_period_compat_async_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asynchronous helper should receive default types set."""

    energy_module = importlib.import_module("custom_components.termoweb.energy")

    async_helper = AsyncMock(return_value={})

    monkeypatch.setattr(
        energy_module,
        "_resolve_statistics_helpers",
        lambda *args, **kwargs: energy_module._RecorderStatisticsHelpers(
            executor=None,
            sync_target=None,
            sync=None,
            async_fn=async_helper,
        ),
    )

    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=1)
    hass = types.SimpleNamespace()

    await energy_module._statistics_during_period_compat(
        hass, start, end, {"sensor.energy"}
    )

    async_helper.assert_awaited_once()
    helper_args = async_helper.await_args.args
    helper_kwargs = async_helper.await_args.kwargs
    assert helper_args[0] is hass
    assert helper_args[1] == start
    assert helper_args[2] == end
    assert helper_args[3] == ["sensor.energy"]
    assert helper_kwargs["period"] == "hour"
    assert helper_kwargs["types"] == {"state", "sum"}


def _inventory_for(mod: Any, nodes: dict[str, list[str]] | list[str]) -> list[Any]:
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
            energy_mod,
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

        energy_mod._store_statistics(hass, metadata, stats)

        import_stats.assert_called_once_with(hass, metadata, stats)
        external.assert_not_called()

    asyncio.run(_run())


def test_store_statistics_external_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
            _const,
            import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            _ConfigEntry,
            HomeAssistant,
            ent_reg,
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

        energy_mod._store_statistics(hass, metadata, stats)

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
            energy_mod,
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
        update_entry = Mock()
        hass.config_entries = types.SimpleNamespace(async_update_entry=update_entry)

        entry = ConfigEntry(
            "1",
            options={
                "sentinel": True,
                energy_mod.OPTION_ENERGY_HISTORY_PROGRESS: {"A": 1},
            },
        )
        original_options = copy.deepcopy(entry.options)

        caplog.set_level(logging.DEBUG, logger=energy_mod._LOGGER.name)

        await mod._async_import_energy_history(hass, entry)

        update_entry.assert_not_called()
        assert entry.options == original_options

    asyncio.run(_run())

    assert "no record found for energy import" in caplog.text


def test_async_import_energy_history_skips_without_inventory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
        hass.config_entries = types.SimpleNamespace(async_update_entry=lambda *args, **kwargs: None)

        entry = ConfigEntry("entry-missing")
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": AsyncMock(),
            "dev_id": "dev-missing",
        }

        caplog.set_level(logging.DEBUG, logger=energy_mod._LOGGER.name)

        await mod._async_import_energy_history(hass, entry)

    asyncio.run(_run())

    assert "dev-missing: unable to resolve node inventory" in caplog.text


def test_async_import_energy_history_uses_inventory_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
        hass.config_entries = types.SimpleNamespace(async_update_entry=Mock())

        entry = ConfigEntry("inventory-entry")
        stored_inventory = inventory_module.Inventory(
            "dev",
            {},
            [types.SimpleNamespace(type="htr", addr="A")],
        )

        client = types.SimpleNamespace(
            get_node_samples=AsyncMock(return_value=[]),
        )

        hass.data = {
            const.DOMAIN: {
                entry.entry_id: {
                    "client": client,
                    "dev_id": "dev",
                    "inventory": stored_inventory,
                }
            }
        }

        def _unexpected(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("resolve_record_inventory should not be called")

        monkeypatch.setattr(energy_mod, "resolve_record_inventory", _unexpected)

        await mod._async_import_energy_history(hass, entry)

    asyncio.run(_run())


def test_register_import_service_uses_module_asyncio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            _mod,
            energy_mod,
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

        class Services:
            def __init__(self) -> None:
                self._svcs: dict[str, dict[str, object]] = {}

            def has_service(self, domain: str, service: str) -> bool:
                return service in self._svcs.get(domain, {})

            def async_register(self, domain: str, service: str, func: object) -> None:
                self._svcs.setdefault(domain, {})[service] = func

            def get(self, domain: str, service: str) -> object:
                return self._svcs[domain][service]

        hass.services = Services()

        entries: dict[str, ConfigEntry] = {}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options),
            async_get_entry=lambda entry_id: entries.get(entry_id),
        )

        entry = ConfigEntry("cache-test", options={})
        entries[entry.entry_id] = entry

        raw_nodes = {"nodes": [{"type": "htr", "addr": "A"}]}
        inventory = inventory_module.Inventory(
            "dev",
            raw_nodes,
            inventory_module.build_node_inventory(raw_nodes),
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "inventory": inventory,
            "config_entry": entry,
            "client": AsyncMock(),
            "dev_id": "dev",
        }

        uid = identifiers_module.build_heater_energy_unique_id("dev", "htr", "A")
        ent_reg.add(
            "sensor.dev_A_energy",
            "sensor",
            const.DOMAIN,
            uid,
            "A energy",
            config_entry_id=entry.entry_id,
        )

        gather_calls: list[int] = []

        async def fake_gather(
            *tasks: Any, return_exceptions: bool = False
        ) -> list[Any]:
            gather_calls.append(len(tasks))
            results: list[Any] = []
            for task in tasks:
                try:
                    results.append(await task)
                except Exception as err:  # pragma: no cover - defensive
                    if return_exceptions:
                        results.append(err)
                    else:
                        raise
            return results

        monkeypatch.setattr(energy_mod.asyncio, "gather", fake_gather)

        import_mock = AsyncMock()
        await energy_mod.async_register_import_energy_history_service(hass, import_mock)

        service = hass.services.get(const.DOMAIN, "import_energy_history")
        assert callable(service)

        await service(
            types.SimpleNamespace(data={"entity_id": ["sensor.dev_A_energy"]})
        )

        assert gather_calls == [1]
        import_mock.assert_awaited_once()

    asyncio.run(_run())


def test_service_accepts_single_entity_id_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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

        entries: dict[str, ConfigEntry] = {}
        hass.config_entries = types.SimpleNamespace(
            async_update_entry=lambda entry, *, options: entry.options.update(options),
            async_get_entry=lambda entry_id: entries.get(entry_id),
        )

        entry = ConfigEntry("cache-test", options={})
        entries[entry.entry_id] = entry

        raw_nodes = {"nodes": [{"type": "htr", "addr": "A"}]}
        inventory = inventory_module.Inventory(
            "dev",
            raw_nodes,
            inventory_module.build_node_inventory(raw_nodes),
        )
        hass.data = {const.DOMAIN: {}}
        hass.data[const.DOMAIN][entry.entry_id] = {
            "inventory": inventory,
            "config_entry": entry,
            "client": AsyncMock(),
            "dev_id": "dev",
        }

        uid = identifiers_module.build_heater_energy_unique_id("dev", "htr", "A")
        ent_reg.add(
            "sensor.dev_A_energy",
            "sensor",
            const.DOMAIN,
            uid,
            "A energy",
            config_entry_id=entry.entry_id,
        )

        gather_calls: list[int] = []

        async def fake_gather(*tasks: Any, return_exceptions: bool = False) -> list[Any]:
            gather_calls.append(len(tasks))
            results: list[Any] = []
            for task in tasks:
                try:
                    results.append(await task)
                except Exception as err:  # pragma: no cover - defensive
                    if return_exceptions:
                        results.append(err)
                    else:
                        raise
            return results

        class Services:
            def __init__(self) -> None:
                self._svcs: dict[str, dict[str, object]] = {}

            def has_service(self, domain: str, service: str) -> bool:
                return service in self._svcs.get(domain, {})

            def async_register(self, domain: str, service: str, func: object) -> None:
                self._svcs.setdefault(domain, {})[service] = func

            def get(self, domain: str, service: str) -> object:
                return self._svcs[domain][service]

        hass.services = Services()

        monkeypatch.setattr(energy_mod.asyncio, "gather", fake_gather)

        import_mock = AsyncMock()
        await energy_mod.async_register_import_energy_history_service(hass, import_mock)

        service = hass.services.get(const.DOMAIN, "import_energy_history")
        await service(types.SimpleNamespace(data={"entity_id": "sensor.dev_A_energy"}))

        assert gather_calls == [1]
        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert args[:2] == (hass, entry)
        assert args[2] == [("htr", "A")]
        assert kwargs == {"reset_progress": False, "max_days": None}

    asyncio.run(_run())




























def test_import_energy_history_requested_map_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
                energy_mod.OPTION_ENERGY_HISTORY_PROGRESS: {},
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

        raw_nodes = {
            "nodes": [
                {"type": "htr", "addr": "A"},
                {"type": "acm", "addr": "B"},
                {"type": "pmo", "addr": "ignored"},
            ]
        }
        snapshot = types.SimpleNamespace(
            dev_id="dev",
            raw_nodes=raw_nodes,
            inventory=list(inventory_module.build_node_inventory(raw_nodes)),
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "node_inventory": list(snapshot.inventory),
            "config_entry": entry,
            "snapshot": snapshot,
        }

        uid_a = identifiers_module.build_heater_energy_unique_id("dev", "htr", "A")
        uid_b_legacy = identifiers_module.build_heater_energy_unique_id("dev", "htr", "B")
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

        async def _fake_sleep(_delay: float) -> None:
            return None

        throttle_module.reset_samples_rate_limit_state(
            time_module=types.SimpleNamespace(
                monotonic=lambda: next(monotonic_counter),
                time=lambda: fake_now,
            ),
            sleep=_fake_sleep,
        )
        monkeypatch.setattr(energy_mod, "datetime", FakeDateTime)

        await mod._async_import_energy_history(
            hass,
            entry,
            selection={
                "htr": ["A", "B", "B"],
                "acm": "B",
                "": ["ignored"],
                "thm": ["Z"],
                "pmo": [],
            },
        )

        throttle_module.reset_samples_rate_limit_state(
            time_module=time, sleep=asyncio.sleep
        )

        assert client.get_node_samples.await_count >= 2
        progress = entry.options[energy_mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert set(progress) >= {"htr:A", "acm:B"}
        assert progress

    asyncio.run(_run())






def test_energy_polling_matches_import(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            _mod,
            _energy_mod,
            _const,
            _import_stats,
            _update_meta,
            _last_stats,
            _get_period,
            _delete_stats,
            _ConfigEntry,
            HomeAssistant,
            ent_reg,
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
        raw_nodes = {"nodes": [{"type": "htr", "addr": "A"}]}
        inventory = inventory_module.Inventory(
            "dev",
            raw_nodes,
            inventory_module.build_node_inventory(raw_nodes),
        )
        details = heater_module.HeaterPlatformDetails(
            inventory,
            lambda addr: f"Node {addr}",
        )
        total_entity = sensor_mod.InstallationTotalEnergySensor(
            coordinator,
            "entry",
            "dev",
            "Total",
            "total_uid",
            details,
        )

        assert energy_entity.native_value == pytest.approx(1.3)
        assert total_entity.native_value == pytest.approx(1.3)

        await asyncio.sleep(0)

    asyncio.run(_run())






def test_service_skips_entries_without_inventory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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

        entry = ConfigEntry(
            "entry-skip",
            data={"username": "skip", "password": "pw"},
            options={},
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

        record = hass.data[const.DOMAIN][entry.entry_id]
        record.pop("inventory", None)
        record.pop("node_inventory", None)
        record.pop("snapshot", None)
        record.pop("nodes", None)

        caplog.set_level(logging.DEBUG, logger=energy_mod._LOGGER.name)

        service = hass.services.get(const.DOMAIN, "import_energy_history")
        await service(types.SimpleNamespace(data={}))

        import_mock.assert_not_called()

    asyncio.run(_run())

    assert "skipping energy import for entry entry-skip" in caplog.text


def test_service_filters_invalid_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
            options={energy_mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
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
            identifiers_module.build_heater_energy_unique_id("dev", "htr", "D"),
            "Missing entry",
            config_entry_id="missing",
        )
        ent_reg.add(
            "sensor.no_config",
            "sensor",
            const.DOMAIN,
            identifiers_module.build_heater_energy_unique_id("dev", "htr", "C"),
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


def test_service_uses_snapshot_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
            options={energy_mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
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
        rec.pop("node_inventory", None)
        raw_nodes = {"nodes": [{"type": "htr", "addr": "A"}]}
        rec["inventory"] = inventory_module.Inventory(
            "dev",
            raw_nodes,
            inventory_module.build_node_inventory(raw_nodes),
        )

        hass.data[const.DOMAIN]["other"] = object()

        service = hass.services.get(const.DOMAIN, "import_energy_history")

        await service(types.SimpleNamespace(data={"max_history_retrieval": 3}))

        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert args[0] is hass
        assert args[1] is entry
        assert isinstance(args[2], inventory_module.Inventory)
        assert args[2].heater_sample_targets == [("htr", "A")]
        assert kwargs == {"reset_progress": False, "max_days": 3}

        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(_run())


def test_service_uses_cached_inventory_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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
            "entry-2",
            data={"username": "user", "password": "pass"},
            options={energy_mod.OPTION_ENERGY_HISTORY_IMPORTED: True},
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
        rec.pop("snapshot", None)
        rec["node_inventory"] = _inventory_for(mod, {"htr": ["A"]})

        service = hass.services.get(const.DOMAIN, "import_energy_history")

        await service(types.SimpleNamespace(data={}))

        import_mock.assert_awaited_once()
        args, kwargs = import_mock.await_args
        assert isinstance(args[2], inventory_module.Inventory)
        assert args[2].heater_sample_targets == [("htr", "A")]
        assert kwargs["reset_progress"] is False
        assert kwargs["max_days"] in (None, energy_mod.DEFAULT_MAX_HISTORY_DAYS)

        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(_run())


def test_refresh_fallback_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            energy_mod,
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

        climate_mod = importlib.import_module("custom_components.termoweb.climate")
        coordinator = types.SimpleNamespace(async_request_refresh=AsyncMock())
        heater = climate_mod.HeaterClimateEntity(
            coordinator, "1", "dev", "A", "Heater A"
        )
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
            energy_mod,
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
