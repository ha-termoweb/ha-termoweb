import asyncio
from types import SimpleNamespace

from custom_components.termoweb.api import RESTClient
from custom_components.termoweb.backend.ducaheat import DucaheatBackend, DucaheatRESTClient
from custom_components.termoweb.ws_client import DucaheatWSClient


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

    assert isinstance(ws_client, DucaheatWSClient)
    assert ws_client.dev_id == "dev"
    assert ws_client.entry_id == "entry"


def test_dummy_client_get_node_settings_accepts_acm() -> None:
    client = DummyClient()

    async def _run() -> dict[str, object]:
        return await client.get_node_settings("dev", ("acm", "3"))

    data = asyncio.run(_run())
    assert data["type"] == "acm"
    assert data["addr"] == "3"


def test_ducaheat_rest_client_passthrough_for_non_htr(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        captured: dict[str, object] = {}

        async def fake_super(self, dev_id: str, node: tuple[str, str]):
            captured["args"] = (dev_id, node)
            return {"ok": True}

        monkeypatch.setattr(RESTClient, "get_node_settings", fake_super)

        result = await client.get_node_settings("dev", ("pmo", "9"))
        assert result == {"ok": True}
        assert captured["args"] == ("dev", ("pmo", "9"))

    asyncio.run(_run())


def test_ducaheat_rest_client_normalises_acm(monkeypatch) -> None:
    async def _run() -> None:
        session = SimpleNamespace()
        client = DucaheatRESTClient(session, "user", "pass")

        async def fake_super(self, dev_id: str, node: tuple[str, str]):
            return {"status": {"mode": "AUTO"}}

        monkeypatch.setattr(RESTClient, "get_node_settings", fake_super)

        seen: dict[str, object] = {}

        def fake_normalise(self, payload, *, node_type: str = "htr"):
            seen["node_type"] = node_type
            seen["payload"] = payload
            return {"normalized": True}

        monkeypatch.setattr(DucaheatRESTClient, "_normalise_settings", fake_normalise)

        result = await client.get_node_settings("dev", ("acm", "2"))
        assert result == {"normalized": True}
        assert seen["node_type"] == "acm"
        assert seen["payload"] == {"status": {"mode": "AUTO"}}

    asyncio.run(_run())


def test_ducaheat_rest_set_node_settings_routes_non_htr(monkeypatch) -> None:
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
            ("acm", "4"),
            mode="auto",
            stemp=20.5,
        )

        assert result == {"ok": True}
        assert captured["args"] == (
            "dev",
            ("acm", "4"),
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
