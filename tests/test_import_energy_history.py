from __future__ import annotations

import asyncio
import importlib.util
import itertools
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


async def _load_module(monkeypatch: pytest.MonkeyPatch):
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

    helpers = types.ModuleType("homeassistant.helpers")
    aiohttp_client = types.ModuleType("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: None
    dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
    dispatcher.async_dispatcher_connect = lambda hass, sig, cb: (lambda: None)
    entity_registry = types.ModuleType("homeassistant.helpers.entity_registry")
    entity_registry.async_get = lambda hass: None
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
    add_stats = AsyncMock()
    stats.async_add_external_statistics = add_stats
    sys.modules.setdefault("homeassistant.components", types.ModuleType("homeassistant.components"))
    sys.modules["homeassistant.components.recorder"] = recorder
    sys.modules["homeassistant.components.recorder.statistics"] = stats

    # Stub package modules
    sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
    termoweb_pkg = types.ModuleType(package)
    termoweb_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "custom_components" / "termoweb")]
    sys.modules[package] = termoweb_pkg

    api_stub = types.ModuleType(f"{package}.api")
    class TermoWebClient:  # pragma: no cover - placeholder
        pass
    api_stub.TermoWebClient = TermoWebClient
    sys.modules[f"{package}.api"] = api_stub

    coord_stub = types.ModuleType(f"{package}.coordinator")
    class TermoWebCoordinator:  # pragma: no cover - placeholder
        pass
    coord_stub.TermoWebCoordinator = TermoWebCoordinator
    sys.modules[f"{package}.coordinator"] = coord_stub

    ws_stub = types.ModuleType(f"{package}.ws_client_legacy")
    class TermoWebWSLegacyClient:  # pragma: no cover - placeholder
        pass
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

    return init_module, const_module, add_stats, ConfigEntry, HomeAssistant


def test_import_energy_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        mod, const, add_stats, ConfigEntry, HomeAssistant = await _load_module(monkeypatch)

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

        add_stats.assert_awaited_once()
        args = add_stats.await_args[0]
        assert args[1]["statistic_id"] == f"{const.DOMAIN}:dev:htr:A:energy"
        stats_list = args[2]
        assert [s["sum"] for s in stats_list] == [pytest.approx(0.001), pytest.approx(0.002)]
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True
        assert entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS] == {"A": 172_800}
        assert entry.options["max_history_retrieved"] == 2

    asyncio.run(_run())


def test_import_energy_history_reset_and_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        mod, const, add_stats, ConfigEntry, HomeAssistant = await _load_module(monkeypatch)

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
        add_stats.assert_awaited_once()
        progress = entry.options[mod.OPTION_ENERGY_HISTORY_PROGRESS]
        assert progress == {"A": 86_400, "B": 0}
        assert entry.options[mod.OPTION_ENERGY_HISTORY_IMPORTED] is True

    asyncio.run(_run())
