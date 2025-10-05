import asyncio
from types import SimpleNamespace

from custom_components.termoweb.api import RESTClient
from custom_components.termoweb.backend.ducaheat import DucaheatBackend, DucaheatRESTClient
from custom_components.termoweb.const import WS_NAMESPACE
from custom_components.termoweb.ws_client import DucaheatWSClient, WebSocketClient


class DummyClient:
    async def list_devices(self) -> list[dict[str, object]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, object]:
        return {"dev_id": dev_id}

    async def get_node_settings(
        self, dev_id: str, node: tuple[str, str | int]
    ) -> dict[str, object]:
        node_type, addr = node
        return {"dev_id": dev_id, "addr": addr, "type": node_type}

    async def set_node_settings(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, object]:
        return {}

    async def get_node_samples(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        start: float,
        stop: float,
    ) -> list[dict[str, object]]:
        return []

    async def _authed_headers(self) -> dict[str, str]:  # pragma: no cover - stub
        return {"Authorization": "Bearer token"}


def test_ducaheat_backend_creates_ws_client() -> None:
    backend = DucaheatBackend(brand="ducaheat", client=DummyClient())
    loop = asyncio.new_event_loop()
    try:
        hass = SimpleNamespace(loop=loop, data={})
        ws_client = backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
        )
    finally:
        loop.close()

    assert isinstance(ws_client, WebSocketClient)
    assert isinstance(ws_client, DucaheatWSClient)
    assert ws_client.dev_id == "dev"
    assert ws_client.entry_id == "entry"
    assert ws_client._protocol_hint is None
    assert ws_client._namespace == WS_NAMESPACE


def test_dummy_client_get_node_settings_accepts_acm() -> None:
    client = DummyClient()

    async def _run() -> dict[str, object]:
        return await client.get_node_settings("dev", ("acm", "3"))

    data = asyncio.run(_run())
    assert data["type"] == "acm"
    assert data["addr"] == "3"


def test_ducaheat_rest_client_fetches_generic_node(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        seen: dict[str, object] = {}

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        async def fake_request(method: str, path: str, **kwargs: object):
            seen["method"] = method
            seen["path"] = path
            seen["kwargs"] = kwargs
            return {"status": {"power": 0}}

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        result = await client.get_node_settings("dev", ("pmo", "9"))
        assert result == {"status": {"power": 0}}
        assert seen["method"] == "GET"
        assert seen["path"] == "/api/v2/devs/dev/pmo/9"
        assert seen["kwargs"] == {"headers": {"Authorization": "Bearer token"}}

    asyncio.run(_run())


def test_ducaheat_rest_client_normalises_acm(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        seen: dict[str, object] = {}

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        async def fake_request(method: str, path: str, **kwargs: object):
            seen["method"] = method
            seen["path"] = path
            seen["kwargs"] = kwargs
            return {"status": {"mode": "AUTO"}}

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        def fake_normalise(self, payload, *, node_type: str = "htr"):
            seen["node_type"] = node_type
            seen["payload"] = payload
            return {"normalized": True}

        monkeypatch.setattr(DucaheatRESTClient, "_normalise_settings", fake_normalise)

        result = await client.get_node_settings("dev", ("acm", "2"))
        assert result == {"normalized": True}
        assert seen["node_type"] == "acm"
        assert seen["payload"] == {"status": {"mode": "AUTO"}}
        assert seen["method"] == "GET"
        assert seen["path"] == "/api/v2/devs/dev/acm/2"
        assert seen["kwargs"] == {"headers": {"Authorization": "Bearer token"}}

    asyncio.run(_run())


def test_ducaheat_rest_set_node_settings_routes_non_special(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        captured: dict[str, object] = {}

        async def fake_super(self, dev_id: str, node: tuple[str, str], **kwargs):
            captured["args"] = (dev_id, node, kwargs)
            return {"ok": True}

        monkeypatch.setattr(RESTClient, "set_node_settings", fake_super)

        result = await client.set_node_settings(
            "dev",
            ("pmo", "4"),
            mode="auto",
            stemp=20.5,
        )

        assert result == {"ok": True}
        assert captured["args"] == (
            "dev",
            ("pmo", "4"),
            {"mode": "auto", "stemp": 20.5, "prog": None, "ptemp": None, "units": "C"},
        )

    asyncio.run(_run())


def test_ducaheat_rest_get_node_samples_forwards_non_htr(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        captured: dict[str, object] = {}

        async def fake_super(
            self, dev_id: str, node: tuple[str, str], start: float, stop: float
        ):
            captured["args"] = (dev_id, node, start, stop)
            return [{"t": 1}]

        monkeypatch.setattr(RESTClient, "get_node_samples", fake_super)

        result = await client.get_node_samples("dev", ("acm", "7"), 1.0, 2.0)
        assert result == [{"t": 1}]
        assert captured["args"] == ("dev", ("acm", "7"), 1.0, 2.0)

    asyncio.run(_run())


def test_ducaheat_rest_normalise_ws_nodes_prog() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    nodes = {
        "acm": {
            "settings": {
                "02": {
                    "prog": {
                        "prog": {
                            str(day): [day % 3] * 48 for day in range(7)
                        }
                    },
                    "mode": "auto",
                }
            },
            "status": {"02": {"temp": 21}},
        }
    }

    result = client.normalise_ws_nodes(nodes)
    settings = result["acm"]["settings"]["02"]
    assert len(settings["prog"]) == 168
    assert settings["prog"][24:48] == [1] * 24
    # Original payload should remain unchanged
    assert len(nodes["acm"]["settings"]["02"]["prog"]["prog"]["1"]) == 48


def test_ducaheat_rest_normalise_ws_nodes_passthrough() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client.normalise_ws_nodes(["bad"]) == ["bad"]

    nodes = {"htr": [1, 2, 3]}
    assert client.normalise_ws_nodes(nodes)["htr"] == [1, 2, 3]

    nodes_with_scalar = {"htr": {"settings": {"01": 5}}}
    normalised = client.normalise_ws_nodes(nodes_with_scalar)
    assert normalised["htr"]["settings"]["01"] == 5
