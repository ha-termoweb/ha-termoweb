from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

WS_CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "ws_client_legacy.py"
)


def _load_ws_client():
    package = "custom_components.termoweb"
    sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
    termoweb_pkg = types.ModuleType(package)
    termoweb_pkg.__path__ = [str(WS_CLIENT_PATH.parent)]
    sys.modules[package] = termoweb_pkg

    ha = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")
    ha_core.HomeAssistant = object  # type: ignore[attr-defined]
    ha_helpers = types.ModuleType("homeassistant.helpers")
    ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")

    def _send(*args, **kwargs):
        return None

    ha_dispatcher.async_dispatcher_send = _send
    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.helpers"] = ha_helpers
    sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

    aiohttp_stub = types.ModuleType("aiohttp")
    class WSMsgType:
        TEXT = 1
        BINARY = 2
        CLOSED = 3
        CLOSE = 4
        ERROR = 5

    aiohttp_stub.WSMsgType = WSMsgType
    aiohttp_stub.ClientSession = object  # pragma: no cover - placeholder
    aiohttp_stub.ClientTimeout = object  # pragma: no cover - placeholder
    aiohttp_stub.WSCloseCode = types.SimpleNamespace(GOING_AWAY=1001)
    sys.modules["aiohttp"] = aiohttp_stub

    spec = importlib.util.spec_from_file_location(
        f"{package}.ws_client_legacy", WS_CLIENT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[f"{package}.ws_client_legacy"] = module
    spec.loader.exec_module(module)
    return module


def test_read_loop_bubbles_exception_on_close():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        api = types.SimpleNamespace(_session=None)
        coordinator = types.SimpleNamespace()
        client = Client(hass, entry_id="e", dev_id="d", api_client=api, coordinator=coordinator)

        aiohttp = sys.modules["aiohttp"]

        class DummyWS:
            def __init__(self):
                self.close_code = 1006

            async def receive(self):
                return types.SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra="bye")

            def exception(self):
                return RuntimeError("boom")

        client._ws = DummyWS()
        with pytest.raises(RuntimeError, match="boom"):
            await client._read_loop()

    asyncio.run(_run())
