from __future__ import annotations

import asyncio
import copy
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest

# Provide a minimal aiohttp stub for the module import
aiohttp_module = sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))


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


aiohttp_module.ClientSession = ClientSession
aiohttp_module.ClientTimeout = ClientTimeout
aiohttp_module.ClientResponseError = ClientResponseError
aiohttp_module.ClientError = Exception

aiohttp = aiohttp_module

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


class LatchedResponse:
    def __init__(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value


class FakeSession:
    def __init__(self) -> None:
        self._request_queue: list[Any] = []
        self._post_queue: list[Any] = []
        self.request_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def queue_request(self, *responses: Any) -> None:
        self._request_queue.extend(responses)

    def queue_post(self, *responses: Any) -> None:
        self._post_queue.extend(responses)

    def clear_calls(self) -> None:
        self.request_calls.clear()
        self.post_calls.clear()

    def _resolve(self, queue: list[Any], label: str) -> Any:
        if not queue:
            raise AssertionError(f"Unexpected {label} call with no queued response")
        item = queue[0]
        if isinstance(item, LatchedResponse):
            result = item.get()
        else:
            result = queue.pop(0)
        if callable(result):
            result = result()
        return result

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        self.request_calls.append((method, url, copy.deepcopy(kwargs)))
        result = self._resolve(self._request_queue, "request")
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.post_calls.append((url, args, copy.deepcopy(kwargs)))
        result = self._resolve(self._post_queue, "post")
        if isinstance(result, Exception):
            raise result
        return result


def test_token_refresh(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
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
        )

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
        assert len(session.post_calls) == 2

    asyncio.run(_run())


def test_get_htr_samples_success() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1000, "counter": "1.5"}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = TermoWebClient(session, "user", "pass")
        samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == [{"t": 1000, "counter": "1.5"}]
        assert len(session.request_calls) == 1
        params = session.request_calls[0][2]["params"]
        assert params == {"start": 0, "end": 10}

    asyncio.run(_run())


def test_get_htr_samples_404() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                404,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = TermoWebClient(session, "user", "pass")
        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client.get_htr_samples("dev", "A", 0, 10)
        assert err.value.status == 404

    asyncio.run(_run())

def test_request_ignore_status_returns_none() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                404,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = TermoWebClient(session, "user", "pass")
        result = await client._request(
            "GET", "/missing", headers={}, ignore_statuses=(404,)
        )
        assert result is None

    asyncio.run(_run())


def test_request_refreshes_once_then_raises() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "initial", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "refreshed", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        )
        session.queue_request(
            LatchedResponse(
                MockResponse(
                    401,
                    {"error": "invalid_token"},
                    headers={"Content-Type": "application/json"},
                    text_data='{"error":"invalid_token"}',
                )
            )
        )

        client = TermoWebClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(api.TermoWebAuthError):
            await client._request("GET", "/api/test", headers=headers)

        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 1
        first_auth = session.request_calls[0][2]["headers"]["Authorization"]
        second_auth = session.request_calls[1][2]["headers"]["Authorization"]
        assert first_auth == "Bearer initial"
        assert second_auth == "Bearer refreshed"

    asyncio.run(_run())


def test_request_rate_limit_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                429,
                {"error": "rate"},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"rate"}',
            )
        )

        client = TermoWebClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(api.TermoWebRateLimitError):
            await client._request("GET", "/api/rate", headers=headers)

        assert len(session.request_calls) == 1
        assert len(session.post_calls) == 0

    asyncio.run(_run())


def test_request_5xx_surfaces_client_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                503,
                {"error": "down"},
                headers={"Content-Type": "application/json"},
                text_data="Service unavailable",
            )
        )

        client = TermoWebClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client._request("GET", "/api/down", headers=headers)

        assert err.value.status == 503
        assert len(session.request_calls) == 1

    asyncio.run(_run())


def test_request_timeout_propagates() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(asyncio.TimeoutError())

        client = TermoWebClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(asyncio.TimeoutError):
            await client._request("GET", "/api/slow", headers=headers)

        assert len(session.request_calls) == 1

    asyncio.run(_run())


def test_set_htr_settings_invalid_units() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = TermoWebClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid units"):
            await client.set_htr_settings("dev", "1", units="kelvin")

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_htr_settings_invalid_program() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = TermoWebClient(session, "user", "pass")

        with pytest.raises(ValueError, match="prog must be a list of 168"):
            await client.set_htr_settings("dev", "1", prog=[0, 1, 2])

        with pytest.raises(ValueError, match="prog values must be 0, 1, or 2"):
            await client.set_htr_settings("dev", "1", prog=[0] * 167 + [5])

        with pytest.raises(ValueError, match="prog contains non-integer value"):
            await client.set_htr_settings("dev", "1", prog=[0] * 167 + ["bad"])

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_htr_settings_invalid_temperatures() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = TermoWebClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid stemp value"):
            await client.set_htr_settings("dev", "1", stemp="warm")

        with pytest.raises(
            ValueError, match="ptemp must be a list of three numeric values"
        ):
            await client.set_htr_settings("dev", "1", ptemp=[21.0, 19.0])

        with pytest.raises(ValueError, match="ptemp contains non-numeric value"):
            await client.set_htr_settings("dev", "1", ptemp=[21.0, "bad", 23.0])

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_get_htr_samples_empty_payload() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"samples": []},
                headers={"Content-Type": "application/json"},
            )
        )

        client = TermoWebClient(session, "user", "pass")
        samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == []

    asyncio.run(_run())


def test_get_htr_samples_decreasing_counters() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {
                    "samples": [
                        {"t": 1, "counter": "3.0"},
                        {"t": 2, "counter": "2.5"},
                    ]
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = TermoWebClient(session, "user", "pass")
        samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == [
            {"t": 1, "counter": "3.0"},
            {"t": 2, "counter": "2.5"},
        ]

    asyncio.run(_run())


def test_request_recovers_after_token_refresh() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
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
        )
        session.queue_request(
            MockResponse(
                401,
                {"error": "expired"},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"expired"}',
            ),
            MockResponse(
                200,
                [{"dev_id": "1"}],
                headers={"Content-Type": "application/json"},
            ),
        )

        client = TermoWebClient(session, "user", "pass")
        devices = await client.list_devices()

        assert devices == [{"dev_id": "1"}]
        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 2
        refreshed_headers = session.request_calls[1][2]["headers"]
        assert refreshed_headers["Authorization"] == "Bearer new"

    asyncio.run(_run())


def test_set_htr_settings_translates_heat(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = TermoWebClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            captured["json"] = kwargs.get("json")
            return {}

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        await client.set_htr_settings("dev", 1, mode="heat", stemp=21.0)

        assert captured["json"]["mode"] == "manual"
        assert captured["json"]["stemp"] == "21.0"

    asyncio.run(_run())


