from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any, Callable

import aiohttp
import pytest

import custom_components.termoweb.api as api

RESTClient = api.RESTClient


class MockResponse:
    def __init__(
        self,
        status: int,
        json_data: Any,
        *,
        headers: dict[str, str] | None = None,
        text_data: str | Callable[[], str] | None = "",
        text_exc: Exception | None = None,
        json_exc: Exception | None = None,
    ) -> None:
        self.status = status
        self._json = json_data
        self._text = text_data
        self._text_exc = text_exc
        self._json_exc = json_exc
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
        if self._text_exc is not None:
            raise self._text_exc
        value = self._text() if callable(self._text) else self._text
        if value is None:
            return ""
        return value

    async def json(
        self, content_type: str | None = None
    ) -> Any:  # pragma: no cover - simple pass-through
        if self._json_exc is not None:
            raise self._json_exc
        return self._json() if callable(self._json) else self._json


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

        client = RESTClient(session, "user", "pass")

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


def test_ensure_token_401_raises_auth_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                401,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendAuthError):
            await client._ensure_token()

    asyncio.run(_run())


def test_ensure_token_429_raises_rate_limit_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                429,
                {},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"rate"}',
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendRateLimitError):
            await client._ensure_token()

    asyncio.run(_run())


def test_ensure_token_missing_access_token() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"unexpected": True},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendAuthError):
            await client._ensure_token()

    asyncio.run(_run())


def test_ensure_token_non_numeric_expires_in(monkeypatch) -> None:
    fake_time = 1000.0

    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": "soon"},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        token = await client._ensure_token()
        assert token == "tok"
        assert client._token_obtained_at == fake_time
        assert client._token_expiry == fake_time + 3600

    monkeypatch.setattr(api.time, "time", lambda: fake_time)
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

        client = RESTClient(session, "user", "pass")
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

        client = RESTClient(session, "user", "pass")
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

        client = RESTClient(session, "user", "pass")
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

        client = RESTClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(api.BackendAuthError):
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

        client = RESTClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(api.BackendRateLimitError):
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

        client = RESTClient(session, "user", "pass")
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

        client = RESTClient(session, "user", "pass")
        headers = await client._authed_headers()
        session.clear_calls()

        with pytest.raises(asyncio.TimeoutError):
            await client._request("GET", "/api/slow", headers=headers)

        assert len(session.request_calls) == 1

    asyncio.run(_run())


def test_request_preview_logs_json_fallback(caplog) -> None:
    results: list[str] = []

    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"ok": True},
                headers={"Content-Type": "application/json"},
                text_data='{"ok":true}',
                json_exc=ValueError("boom"),
            )
        )

        client = RESTClient(session, "user", "pass")
        result = await client._request("GET", "/api/preview", headers={})
        results.append(result)

    caplog.clear()
    api.API_LOG_PREVIEW = True
    try:
        with caplog.at_level("DEBUG"):
            asyncio.run(_run())
    finally:
        api.API_LOG_PREVIEW = False

    assert results == ['{"ok":true}']
    preview_logs = [rec.message for rec in caplog.records if "body[0:200]" in rec.message]
    assert preview_logs


def test_api_base_property_returns_sanitized() -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pw", api_base="https://api.example.com/")

    assert client.api_base == "https://api.example.com"


def test_request_text_exception_fallback() -> None:
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
                {"ok": True},
                headers={"Content-Type": "application/json"},
                text_exc=RuntimeError("boom"),
            )
        )

        client = RESTClient(session, "user", "pw")
        headers = await client._authed_headers()
        session.clear_calls()

        result = await client._request("GET", "/api/data", headers=headers)
        assert result == {"ok": True}

    asyncio.run(_run())


def test_request_returns_plain_text() -> None:
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
                "ignored",
                headers={"Content-Type": "text/plain"},
                text_data="hello world",
            )
        )

        client = RESTClient(session, "user", "pw")
        headers = await client._authed_headers()
        session.clear_calls()

        result = await client._request("GET", "/api/plain", headers=headers)
        assert result == "hello world"

    asyncio.run(_run())


def test_ensure_token_uses_cache_without_http() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "cached"
        client._token_expiry = time.time() + 1000

        token = await client._ensure_token()
        assert token == "cached"
        assert session.post_calls == []

    asyncio.run(_run())


def test_ensure_token_concurrent_calls_share_refresh() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        tokens = await asyncio.gather(client._ensure_token(), client._ensure_token())

        assert tokens == ["tok", "tok"]
        assert len(session.post_calls) == 1

    asyncio.run(_run())


def test_ensure_token_returns_cached_after_lock_entry() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = None
        client._token_expiry = 0.0

        class FakeLock:
            def __init__(self, owner: RESTClient) -> None:
                self._owner = owner

            async def __aenter__(self) -> "FakeLock":
                self._owner._access_token = "cached"
                self._owner._token_expiry = time.time() + 100
                return self

            async def __aexit__(self, *_exc: Any) -> bool:
                return False

        client._lock = FakeLock(client)  # type: ignore[assignment]

        token = await client._ensure_token()
        assert token == "cached"
        assert session.post_calls == []

    asyncio.run(_run())


def test_token_request_error_raises_client_response_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                500,
                {},
                headers={"Content-Type": "text/plain"},
                text_data="failure",
            )
        )

        client = RESTClient(session, "user", "pw")
        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client._ensure_token()
        assert err.value.status == 500

    asyncio.run(_run())


def test_list_devices_handles_various_shapes() -> None:
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
                {"devs": [{"id": 1}, "bad"]},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"devices": [{"dev_id": "abc"}, 123]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        first = await client.list_devices()
        second = await client.list_devices()

        assert first == [{"id": 1}]
        assert second == [{"dev_id": "abc"}]

    asyncio.run(_run())


def test_device_connected_returns_none() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        assert await client.device_connected("dev") is None

    asyncio.run(_run())


def test_get_nodes_and_settings_use_expected_paths(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        client._token_expiry = time.time() + 1000

        calls: list[tuple[str, str]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((method, path))
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        await client.get_nodes("dev123")
        await client.get_htr_settings("dev123", "5")

        assert calls == [
            ("GET", api.NODES_PATH_FMT.format(dev_id="dev123")),
            ("GET", f"/api/v2/devs/dev123/htr/5/settings"),
        ]

    asyncio.run(_run())


def test_set_htr_settings_includes_prog_and_ptemp(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        client._token_expiry = time.time() + 1000

        received: list[dict[str, Any]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            received.append(kwargs["json"])
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        prog = [0, 1, 2] * 56
        ptemp = [18.0, 19.0, 20.0]
        await client.set_htr_settings("dev123", "7", prog=prog, ptemp=ptemp, units="f")

        assert received == [
            {
                "units": "F",
                "prog": prog,
                "ptemp": ["18.0", "19.0", "20.0"],
            }
        ]

    asyncio.run(_run())


def test_request_cancelled_error_propagates() -> None:
    async def _run() -> None:
        session = FakeSession()

        def raise_cancelled() -> None:
            raise asyncio.CancelledError()

        session.queue_request(raise_cancelled)

        client = RESTClient(session, "user", "pass")

        with pytest.raises(asyncio.CancelledError):
            await client._request("GET", "/api/cancel", headers={})

    asyncio.run(_run())


def test_request_generic_exception_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=api.__name__)

    async def _run() -> None:
        session = FakeSession()
        session.queue_request(RuntimeError("upstream Bearer secret-token failure"))

        client = RESTClient(session, "user", "pass")

        headers = {"Authorization": "Bearer secret-token"}

        with pytest.raises(RuntimeError):
            await client._request("GET", "/api/fail", headers=headers)

    asyncio.run(_run())

    assert "Request GET" in caplog.text
    assert "Bearer ***REDACTED***" in caplog.text


def test_request_final_auth_error_after_retries(monkeypatch) -> None:
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
                {"access_token": "refresh1", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "refresh2", "expires_in": 3600},
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

        client = RESTClient(session, "user", "pass")
        headers = await client._authed_headers()
        monkeypatch.setattr(api, "range", lambda _n: (0, 0), raising=False)

        with pytest.raises(api.BackendAuthError) as err:
            await client._request("GET", "/api/fail", headers=headers)

        assert str(err.value) == "Unauthorized"
        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 3

    asyncio.run(_run())


def test_set_htr_settings_invalid_units() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid units"):
            await client.set_htr_settings("dev", "1", units="kelvin")

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_htr_settings_invalid_program() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

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
        client = RESTClient(session, "user", "pass")

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

        client = RESTClient(session, "user", "pass")
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

        client = RESTClient(session, "user", "pass")
        samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == [
            {"t": 1, "counter": "3.0"},
            {"t": 2, "counter": "2.5"},
        ]

    asyncio.run(_run())


def test_get_htr_samples_malformed_items(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(
            method: str, path: str, **kwargs: Any
        ) -> dict[str, Any]:
            return {
                "samples": [
                    123,
                    {"t": "bad"},
                    {"t": 5, "counter": None},
                ]
            }

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == []

    caplog.clear()
    asyncio.run(_run())
    messages = [rec.message for rec in caplog.records]
    assert any("Unexpected htr sample item" in msg for msg in messages)
    assert any("Unexpected htr sample shape" in msg for msg in messages)


def test_get_htr_samples_unexpected_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            return "garbled"

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_htr_samples("dev", "A", 0, 10)

        assert samples == []

    caplog.clear()
    asyncio.run(_run())
    assert any(
        "Unexpected htr samples payload" in rec.message for rec in caplog.records
    )


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

        client = RESTClient(session, "user", "pass")
        devices = await client.list_devices()

        assert devices == [{"dev_id": "1"}]
        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 2
        refreshed_headers = session.request_calls[1][2]["headers"]
        assert refreshed_headers["Authorization"] == "Bearer new"

    asyncio.run(_run())


def test_list_devices_unexpected_dict_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(
            method: str, path: str, **kwargs: Any
        ) -> dict[str, Any]:
            return {"weird": []}

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            devices = await client.list_devices()

        assert devices == []

    caplog.clear()
    asyncio.run(_run())
    assert any("Unexpected /devs shape" in rec.message for rec in caplog.records)


def test_list_devices_unexpected_string_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> str:
            return "oops"

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            devices = await client.list_devices()

        assert devices == []

    caplog.clear()
    asyncio.run(_run())
    assert any("Unexpected /devs shape" in rec.message for rec in caplog.records)


def test_set_htr_settings_translates_heat(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

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


