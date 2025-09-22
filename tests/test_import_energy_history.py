from __future__ import annotations

import asyncio
import copy
from datetime import datetime
import importlib
import importlib.util
import itertools
from pathlib import Path
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest


async def _load_module(monkeypatch: pytest.MonkeyPatch, *, legacy: bool = False):
    package = "custom_components.termoweb"

    # Stub Home Assistant modules
    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistant:  # pragma: no cover - minimal stub
        pass

    def callback(func):  # pragma: no cover - minimal stub
        return func

    class ServiceCall:  # pragma: no cover - minimal stub
        def __init__(self, data=None):
            self.data = data or {}

    ha_core.HomeAssistant = HomeAssistant
    ha_core.callback = callback
    ha_core.ServiceCall = ServiceCall
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

    class UnitOfTemperature:  # pragma: no cover - minimal stub
        CELSIUS = "°C"

    ha_const.UnitOfTemperature = UnitOfTemperature
    ha_const.ATTR_TEMPERATURE = "temperature"
    sys.modules["homeassistant.const"] = ha_const

    ha_exc = types.ModuleType("homeassistant.exceptions")
    class ConfigEntryAuthFailed(Exception):
        pass

    class ConfigEntryNotReady(Exception):
        pass

    ha_exc.ConfigEntryAuthFailed = ConfigEntryAuthFailed
    ha_exc.ConfigEntryNotReady = ConfigEntryNotReady
    sys.modules["homeassistant.exceptions"] = ha_exc

    helpers = types.ModuleType("homeassistant.helpers")
    aiohttp_client = types.ModuleType("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: None
    dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
    dispatcher.async_dispatcher_connect = lambda hass, sig, cb: (lambda: None)
    entity_registry = types.ModuleType("homeassistant.helpers.entity_registry")

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

    entity_registry.async_get = lambda hass: ent_reg
    helpers.aiohttp_client = aiohttp_client
    helpers.entity_registry = entity_registry
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.aiohttp_client"] = aiohttp_client
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher
    sys.modules["homeassistant.helpers.entity_registry"] = entity_registry

    climate = types.ModuleType("homeassistant.components.climate")

    class ClimateEntity:  # pragma: no cover - minimal stub
        async def async_will_remove_from_hass(self) -> None:
            return

    class HVACMode:  # pragma: no cover - minimal stub
        HEAT = "heat"
        OFF = "off"
        AUTO = "auto"

    class HVACAction:  # pragma: no cover - minimal stub
        HEATING = "heating"
        IDLE = "idle"
        OFF = "off"

    class ClimateEntityFeature:  # pragma: no cover - minimal stub
        TARGET_TEMPERATURE = 1
        PRESET_MODE = 2

    climate.ClimateEntity = ClimateEntity
    climate.HVACMode = HVACMode
    climate.HVACAction = HVACAction
    climate.ClimateEntityFeature = ClimateEntityFeature
    sys.modules.setdefault("homeassistant.components", types.ModuleType("homeassistant.components"))
    sys.modules["homeassistant.components.climate"] = climate

    helpers_entity = types.ModuleType("homeassistant.helpers.entity")

    class DeviceInfo(dict):  # pragma: no cover - minimal stub
        pass

    helpers_entity.DeviceInfo = DeviceInfo
    sys.modules["homeassistant.helpers.entity"] = helpers_entity

    entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")
    entity_platform.async_get_current_platform = lambda: types.SimpleNamespace(
        async_register_entity_service=lambda *args, **kwargs: None
    )
    sys.modules["homeassistant.helpers.entity_platform"] = entity_platform

    helpers_update_coordinator = types.ModuleType(
        "homeassistant.helpers.update_coordinator"
    )

    class CoordinatorEntity:  # pragma: no cover - minimal stub
        def __init__(self, coordinator):
            self.coordinator = coordinator

        async def async_will_remove_from_hass(self) -> None:
            return

    helpers_update_coordinator.CoordinatorEntity = CoordinatorEntity
    sys.modules["homeassistant.helpers.update_coordinator"] = (
        helpers_update_coordinator
    )

    ha_util = types.ModuleType("homeassistant.util")
    dt = types.ModuleType("homeassistant.util.dt")
    dt.now = lambda: datetime.utcnow()
    ha_util.dt = dt
    sys.modules["homeassistant.util"] = ha_util
    sys.modules["homeassistant.util.dt"] = dt

    aiohttp_mod = sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

    if not hasattr(aiohttp_mod, "ClientError"):
        class ClientError(Exception):  # pragma: no cover - minimal stub
            pass

        aiohttp_mod.ClientError = ClientError

    sys.modules["voluptuous"] = types.ModuleType("voluptuous")

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

    last_stats = AsyncMock(return_value={})
    stats.async_get_last_statistics = last_stats

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
    class TermoWebAuthError(Exception):  # pragma: no cover - placeholder
        pass

    class TermoWebRateLimitError(Exception):  # pragma: no cover - placeholder
        pass

    api_stub.TermoWebAuthError = TermoWebAuthError
    api_stub.TermoWebRateLimitError = TermoWebRateLimitError
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
        last_stats,
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
            last_stats,
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
        monkeypatch.setattr(
            mod,
            "datetime",
            types.SimpleNamespace(
                now=lambda tz=None: datetime.fromtimestamp(fake_now, tz),
                fromtimestamp=datetime.fromtimestamp,
            ),
        )

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        assert client.get_htr_samples.await_count == 2
        first_call = client.get_htr_samples.await_args_list[0][0]
        second_call = client.get_htr_samples.await_args_list[1][0]
        assert first_call == ("dev", "A", 259_199, 345_599)
        assert second_call == ("dev", "A", 172_799, 259_199)

        last_stats.assert_called_once()
        assert captured["meta"]["statistic_id"] == "sensor.dev_A_energy"
        stats_list = captured["stats"]
        assert [s["sum"] for s in stats_list] == [pytest.approx(0.001), pytest.approx(0.002)]
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {"A": 172_799}
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
            "htr_addrs": ["A"],
            "config_entry": entry,
        }
        uid = f"{const.DOMAIN}:dev:htr:A:energy"
        ent_reg.add("sensor.dev_A_energy", "sensor", const.DOMAIN, uid, "A energy")

        last_stats.return_value = {"sensor.dev_A_energy": [{"sum": 1.0}]}

        fake_now = 4 * 86_400
        monotonic_counter = itertools.count(start=1, step=2)
        monkeypatch.setattr(
            mod,
            "time",
            types.SimpleNamespace(
                time=lambda: fake_now, monotonic=lambda: next(monotonic_counter)
            ),
        )
        monkeypatch.setattr(
            mod,
            "datetime",
            types.SimpleNamespace(
                now=lambda tz=None: datetime.fromtimestamp(fake_now, tz),
                fromtimestamp=datetime.fromtimestamp,
            ),
        )

        captured: dict = {}
        monkeypatch.setattr(
            mod,
            "_store_statistics",
            lambda _h, m, s: captured.update(meta=m, stats=s),
        )

        await mod._async_import_energy_history(hass, entry)

        last_stats.assert_called_once()
        stats_list = captured["stats"]
        assert [s["sum"] for s in stats_list] == [
            pytest.approx(1.001),
            pytest.approx(1.002),
        ]

    asyncio.run(_run())


def test_import_energy_history_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            add_stats,
            _,
            last_stats,
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
            return_value=[
                {"t": 172_800, "counter": "2.0"},
                {"t": 86_400, "counter": "1.0"},
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

        last_stats.assert_called_once()
        meta = captured["meta"]
        stats = captured["stats"]
        assert meta["statistic_id"] == "sensor.dev_A_energy"
        assert meta["source"] == "recorder"
        assert stats[0]["sum"] == pytest.approx(0.001)

    asyncio.run(_run())


def test_import_energy_history_reset_and_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
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
        monkeypatch.setattr(
            mod,
            "datetime",
            types.SimpleNamespace(
                now=lambda tz=None: datetime.fromtimestamp(fake_now, tz),
                fromtimestamp=datetime.fromtimestamp,
            ),
        )

        await mod._async_import_energy_history(hass, entry, ["A"], reset_progress=True)

        client.get_htr_samples.assert_awaited_once_with("dev", "A", 86_399, 172_799)
        last_stats.assert_called_once()
        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"A": 86_399, "B": 0}
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

        call_log: list[tuple[str, str, int, int]] = []

        async def sample_side_effect(dev_id: str, addr: str, start: int, stop: int):
            call_log.append((dev_id, addr, start, stop))
            base = 0.0 if addr == "A" else 100.0
            return [
                {"t": start + 1, "counter": str(base + (start // 3600) + 1)},
                {"t": stop + 1, "counter": str(base + (stop // 3600) + 1.5)},
            ]

        client = types.SimpleNamespace(
            get_htr_samples=AsyncMock(side_effect=sample_side_effect)
        )

        hass.data[const.DOMAIN][entry.entry_id] = {
            "client": client,
            "dev_id": "dev",
            "htr_addrs": ["A", "B"],
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
        monkeypatch.setattr(
            mod,
            "datetime",
            types.SimpleNamespace(
                now=lambda tz=None: datetime.fromtimestamp(fake_now, tz),
                fromtimestamp=datetime.fromtimestamp,
            ),
        )

        await mod._async_import_energy_history(
            hass,
            entry,
            None,
            reset_progress=True,
            max_days=3,
        )

        assert client.get_htr_samples.await_count == 6
        assert call_log == [
            ("dev", "A", 345_599, 431_999),
            ("dev", "A", 259_199, 345_599),
            ("dev", "A", 172_799, 259_199),
            ("dev", "B", 345_599, 431_999),
            ("dev", "B", 259_199, 345_599),
            ("dev", "B", 172_799, 259_199),
        ]
        assert len(updates) == 8
        assert updates[0][mod.OPTION_ENERGY_HISTORY_PROGRESS] == {}
        final_update = updates[-1]
        assert final_update[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {
            "A": 172_799,
            "B": 172_799,
        }
        assert final_update[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True

        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"A": 172_799, "B": 172_799}
        assert entry.options[mod.OPTION_MAX_HISTORY_RETRIEVED] == 1
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert last_stats.await_count == 2

    asyncio.run(_run())


def test_setup_defers_import_until_started(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        (
            mod,
            const,
            import_stats,
            update_meta,
            last_stats,
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
            "htr_addrs": ["A"],
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
        rec["htr_addrs"] = ["A", "B"]

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
        assert set(args[2]) == {"A", "B"}
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
        assert args[2] is None
        assert kwargs == {"reset_progress": False, "max_days": 2}

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
            ConfigEntry,
            HomeAssistant,
            _ent_reg,
        ) = await _load_module(monkeypatch)

        climate_mod = importlib.import_module("custom_components.termoweb.climate")
        coordinator = types.SimpleNamespace(async_request_refresh=AsyncMock())
        heater = climate_mod.TermoWebHeater(coordinator, "1", "dev", "A", "Heater A")
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
