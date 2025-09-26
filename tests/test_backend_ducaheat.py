import asyncio
from types import SimpleNamespace

from custom_components.termoweb.backend.ducaheat import DucaheatBackend
from custom_components.termoweb.ws_client_v2 import DucaheatWSClient


class DummyClient:
    async def list_devices(self) -> list[dict[str, object]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, object]:
        return {"dev_id": dev_id}

    async def get_htr_settings(self, dev_id: str, addr: str | int) -> dict[str, object]:
        return {"dev_id": dev_id, "addr": addr}

    async def set_htr_settings(
        self,
        dev_id: str,
        addr: str | int,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, object]:
        return {}

    async def get_htr_samples(
        self,
        dev_id: str,
        addr: str | int,
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
