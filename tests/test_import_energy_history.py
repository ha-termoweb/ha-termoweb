from __future__ import annotations

import asyncio
import importlib.util
import itertools
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


async def _load_module(monkeypatch: pytest.MonkeyPatch, *, legacy: bool = False):
    package = "custom_components.termoweb"

    # Stub Home Assistant modules
    ha_core = types.ModuleType("homeassistant.core")
    class HomeAssistant:  # pragma: no cover - minimal stub
        pass
    ha_core.HomeAssistant = HomeAssistant
    sys.modules["homeassistant"] = types.ModuleType("homeassistant")
    sys.modules["homeassistant.core"] = ha_core

    ha_cfg = types.ModuleType("homeassistant.config_entries")
    class ConfigEntry:  # pragma: no cover - minimal stub
        def __init__(self, entry_id, data=None, options=None):
            self.entry_id = entry_id
            self.data = data or {}
            self.options = options or {}
    ha_cfg.ConfigEntry = ConfigEntry
    sys.modules["homeassistant.config_entries"] = ha_cfg

    ha_const = types.ModuleType("homeassistant.const")
    ha_const.EVENT_HOMEASSISTANT_STARTED = "homeassistant_started"
    sys.modules["homeassistant.const"] = ha_const

    helpers = types.ModuleType("homeassistant.helpers")
    aiohttp_client = types.ModuleType("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: None
    dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
    dispatcher.async_dispatcher_connect = lambda hass, sig, cb: (lambda: None)
    entity_registry = types.ModuleType("homeassistant.helpers.entity_registry")

    class EntityEntry:
        def __init__(self, entity_id: str, name: str) -> None:
            self.entity_id = entity_id
            self.original_name = name

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
        ) -> None:
            self._by_uid[(domain, platform, unique_id)] = entity_id
            self._by_entity[entity_id] = EntityEntry(entity_id, name)

        def async_get_entity_id(
            self, domain: str, platform: str, unique_id: str
        ) -> str | None:
            return self._by_uid.get((domain, platform, unique_id))

        def async_get(self, entity_id: str) -> EntityEntry | None:
            return self._by_entity.get(entity_id)

    ent_reg = EntityRegistry()

    entity_registry.async_get = lambda hass: ent_reg
    helpers.aiohttp_client = aiohttp_client
    helpers.entity_registry = entity_registry
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.aiohttp_client"] = aiohttp_client
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher
    sys.modules["homeassistant.helpers.entity_registry"] = entity_registry

    loader = types.ModuleType("homeassistant.loader")
    async def async_get_integration(hass, domain):
        return types.SimpleNamespace(version="1.0")
    loader.async_get_integration = async_get_integration
    sys.modules["homeassistant.loader"] = loader

    # Recorder statistics stub
    recorder = types.ModuleType("homeassistant.components.recorder")
    stats = types.ModuleType("homeassistant.components.recorder.statistics")
    sys.modules.setdefault("homeassistant.components", types.ModuleType("homeassistant.components"))
    sys.modules["homeassistant.components.recorder"] = recorder
    sys.modules["homeassistant.components.recorder.statistics"] = stats

    if legacy:
        add_stats = Mock()
        stats.async_add_external_statistics = add_stats
        import_stats = add_stats
        update_meta = None
    else:
        import_stats = Mock()
        update_meta = Mock()
        stats.async_import_statistics = import_stats
        stats.async_update_statistics_metadata = update_meta

    # Stub package modules
    sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
    termoweb_pkg = types.ModuleType(package)
    termoweb_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "custom_components" / "termoweb")]
    sys.modules[package] = termoweb_pkg

    api_stub = types.ModuleType(f"{package}.api")
    class TermoWebClient:  # pragma: no cover - placeholder
        def __init__(self, session, username, password):
            self.session = session
            self.username = username
            self.password = password

        async def list_devices(self):
            return [{"dev_id": "dev"}]

        async def get_nodes(self, dev_id):
            return {"nodes": [{"type": "htr", "addr": "A"}]}

    api_stub.TermoWebClient = TermoWebClient
    sys.modules[f"{package}.api"] = api_stub

    coord_stub = types.ModuleType(f"{package}.coordinator")
    class TermoWebCoordinator:  # pragma: no cover - placeholder
        def __init__(self, hass, client, base_interval, dev_id, dev, nodes):
            self.data = {}

        async def async_config_entry_first_refresh(self):
            return

        def async_add_listener(self, cb):
            return

    coord_stub.TermoWebCoordinator = TermoWebCoordinator
    sys.modules[f"{package}.coordinator"] = coord_stub

    ws_stub = types.ModuleType(f"{package}.ws_client_legacy")
    class TermoWebWSLegacyClient:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return asyncio.create_task(asyncio.sleep(0))

        async def stop(self):
            return

    ws_stub.TermoWebWSLegacyClient = TermoWebWSLegacyClient
    sys.modules[f"{package}.ws_client_legacy"] = ws_stub

    # Load const and __init__ modules
    const_path = Path(__file__).resolve().parents[1] / "custom_components" / "termoweb" / "const.py"
    spec_const = importlib.util.spec_from_file_location(f"{package}.const", const_path)
    const_module = importlib.util.module_from_spec(spec_const)
    sys.modules[f"{package}.const"] = const_module
    spec_const.loader.exec_module(const_module)

    init_path = Path(__file__).resolve().parents[1] / "custom_components" / "termoweb" / "__init__.py"
    spec = importlib.util.spec_from_file_location(f"{package}.__init__", init_path)
    init_module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package}.__init__"] = init_module
    spec.loader.exec_module(init_module)

    return (
        init_module,
        const_module,
        import_stats,
        update_meta,
        ConfigEntry,
        HomeAssistant,
        ent_reg,
    )


def test_import_energy_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
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
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 259_200, "counter": "1.0"}],
                [{"t": 172_800, "counter": "2.0"}],
            ]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "htr_addrs": ["A"],
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

        await mod._async_import_energy_history(hass, entry)

        assert client.get_htr_samples.await_count == 2
        first_call = client.get_htr_samples.await_args_list[0][0]
        second_call = client.get_htr_samples.await_args_list[1][0]
        assert first_call == ("dev", "A", 259_200, 345_600)
        assert second_call == ("dev", "A", 172_800, 259_200)

        update_meta.assert_called_once()
        import_stats.assert_called_once()
        args = import_stats.call_args[0]
        assert args[1]["statistic_id"] == "sensor.dev_A_energy"
        stats_list = args[2]
        assert [s["sum"] for s in stats_list] == [pytest.approx(0.001), pytest.approx(0.002)]
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {"A": 172_800}
        assert entry.options["max_history_retrieved"] == 2

    asyncio.run(_run())


def test_import_energy_history_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            add_stats,
            _,
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
        client.get_htr_samples = AsyncMock(
            return_value=[{"t": 86_400, "counter": "1.0"}]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "htr_addrs": ["A"],
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

        await mod._async_import_energy_history(hass, entry)

        add_stats.assert_called_once()
        _, meta, stats = add_stats.call_args[0]
        assert meta["statistic_id"] == "sensor:dev_A_energy"
        assert meta["source"] == "sensor"
        assert stats[0]["sum"] == pytest.approx(0.001)

    asyncio.run(_run())


def test_import_energy_history_reset_and_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
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
        client.get_htr_samples = AsyncMock(
            return_value=[{"t": 86_400, "counter": "1.0"}]
        )
        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "htr_addrs": ["A", "B"],
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

        await mod._async_import_energy_history(hass, entry, ["A"], reset_progress=True)

        client.get_htr_samples.assert_awaited_once_with("dev", "A", 86_400, 172_800)
        update_meta.assert_called_once()
        import_stats.assert_called_once()
        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"A": 86_400, "B": 0}
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True

    asyncio.run(_run())


def test_setup_defers_import_until_started(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            ConfigEntry,
            HomeAssistant,
            _,
        ) = await _load_module(monkeypatch)

        hass = HomeAssistant()
        hass.data = {const.DOMAIN: {}}
        tasks = []

        def async_create_task(coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        listeners = []

        class Bus:
            def async_listen_once(self, event, cb):
                listeners.append(cb)

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
            "htr_addrs": ["A"],
            "config_entry": entry,
        }

        called = False

        async def fake_import(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(mod, "_async_import_energy_history", fake_import)

        assert await mod.async_setup_entry(hass, entry) is True
        assert not called

        for cb in listeners:
            res = cb(None)
            if asyncio.iscoroutine(res):
                await res
        await asyncio.gather(*tasks)

        assert called

    asyncio.run(_run())
