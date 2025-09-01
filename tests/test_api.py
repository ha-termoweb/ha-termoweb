from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# Provide a minimal aiohttp stub for the module import
aiohttp_stub = types.ModuleType("aiohttp")


class ClientSession:  # pragma: no cover - simple placeholder
    pass


class ClientTimeout:  # pragma: no cover - simple placeholder
    def __init__(self, total: int | None = None) -> None:
        self.total = total


class ClientResponseError(Exception):  # pragma: no cover - simple placeholder
    def __init__(
        self, request_info, history, *, status=None, message=None, headers=None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers
        self.request_info = request_info
        self.history = history


aiohttp_stub.ClientSession = ClientSession
aiohttp_stub.ClientTimeout = ClientTimeout
aiohttp_stub.ClientResponseError = ClientResponseError
aiohttp_stub.ClientError = Exception

sys.modules.setdefault("aiohttp", aiohttp_stub)
aiohttp = aiohttp_stub

API_PATH = (
    Path(__file__).resolve().parents[1] / "custom_components" / "termoweb" / "api.py"
)

package_name = "custom_components.termoweb"
module_name = f"{package_name}.api"

sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(package_name)
termoweb_pkg.__path__ = [str(API_PATH.parent)]
sys.modules[package_name] = termoweb_pkg

spec = importlib.util.spec_from_file_location(module_name, API_PATH)
api = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[module_name] = api
spec.loader.exec_module(api)
TermoWebClient = api.TermoWebClient


class MockResponse:
    def __init__(
        self,
        status: int,
        json_data: Any,
        *,
        headers: dict[str, str] | None = None,
        text_data: str = "",
    ) -> None:
        self.status = status
        self._json = json_data
        self._text = text_data
        self.headers = headers or {}
        self.request_info = None
        self.history = ()

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(
        self, exc_type, exc, tb
    ) -> None:  # pragma: no cover - no special handling
        return None

    async def text(self) -> str:
        return self._text

    async def json(
        self, content_type: str | None = None
    ) -> Any:  # pragma: no cover - simple pass-through
        return self._json


def test_token_refresh(monkeypatch) -> None:
    async def _run() -> None:
        session = MagicMock()
        session.post.side_effect = [
            MockResponse(
                200,
                {"access_token": "t1", "expires_in": 1},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "t2", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        ]

        client = TermoWebClient(session, "user", "pass")

        import custom_components.termoweb.api as api_module

        fake_time = 0.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(api_module.time, "time", _fake_time)
        token1 = await client._ensure_token()
        assert token1 == "t1"

        fake_time = 2.0  # advance beyond expiry
        token2 = await client._ensure_token()
        assert token2 == "t2"
        assert session.post.call_count == 2

    asyncio.run(_run())


def test_get_htr_samples_success() -> None:
    async def _run() -> None:
        session = MagicMock()
        session.post.return_value = MockResponse(
            200,
            {"access_token": "tok", "expires_in": 3600},
            headers={"Content-Type": "application/json"},
        )
        session.request.return_value = MockResponse(
            200,
            {"samples": [{"t": 1000, "counter": "1.5"}]},
            headers={"Content-Type": "application/json"},
        )

        client = TermoWebClient(session, "user", "pass")
        samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == [{"t": 1000, "counter": "1.5"}]
        params = session.request.call_args[1]["params"]
        assert params == {"start": 0, "end": 10}

    asyncio.run(_run())


def test_get_htr_samples_404() -> None:
    async def _run() -> None:
        session = MagicMock()
        session.post.return_value = MockResponse(
            200,
            {"access_token": "tok", "expires_in": 3600},
            headers={"Content-Type": "application/json"},
        )
        session.request.return_value = MockResponse(
            404,
            {},
            headers={"Content-Type": "application/json"},
        )

        client = TermoWebClient(session, "user", "pass")
        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client.get_htr_samples("dev", "A", 0, 10)
        assert err.value.status == 404

    asyncio.run(_run())


def test_request_retries_on_401() -> None:
    async def _run() -> None:
        session = MagicMock()
        session.post.side_effect = [
            MockResponse(
                200,
                {"access_token": "old", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "new", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        ]
        session.request.side_effect = [
            MockResponse(401, {}, headers={"Content-Type": "application/json"}),
            MockResponse(
                200, [{"dev_id": "1"}], headers={"Content-Type": "application/json"}
            ),
        ]

        client = TermoWebClient(session, "user", "pass")
        devices = await client.list_devices()

        assert devices == [{"dev_id": "1"}]
        assert session.request.call_count == 2
        assert session.post.call_count == 2

        # Verify that the Authorization header was updated after the retry
        second_headers = session.request.call_args_list[1][1]["headers"]
        assert second_headers["Authorization"] == "Bearer new"

    asyncio.run(_run())


def test_set_htr_settings_translates_heat(monkeypatch) -> None:
    async def _run() -> None:
        session = MagicMock()
        client = TermoWebClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        monkeypatch.setattr(client, "_authed_headers", fake_headers)

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs) -> Any:
            captured["json"] = kwargs.get("json")
            return {}

        monkeypatch.setattr(client, "_request", fake_request)

        await client.set_htr_settings("dev", 1, mode="heat", stemp=21.0)

        assert captured["json"]["mode"] == "manual"

    asyncio.run(_run())


def test_set_htr_settings_invalid_units(monkeypatch) -> None:
    async def _run() -> None:
        session = MagicMock()
        client = TermoWebClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            raise AssertionError("_authed_headers should not be called")

        monkeypatch.setattr(client, "_authed_headers", fake_headers)

        with pytest.raises(ValueError):
            await client.set_htr_settings("dev", 1, units="K")

    asyncio.run(_run())
