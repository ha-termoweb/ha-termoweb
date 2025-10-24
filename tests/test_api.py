from __future__ import annotations

import asyncio
import copy
import logging
import logging
import time
from typing import Any, Callable
from unittest.mock import AsyncMock

import aiohttp
import pytest

import custom_components.termoweb.api as api
from custom_components.termoweb.backend.ducaheat import (
    DucaheatRESTClient,
    DucaheatRequestError,
)
from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    get_brand_requested_with,
    get_brand_user_agent,
)
from custom_components.termoweb.inventory import AccumulatorNode

RESTClient = api.RESTClient


def _patch_api_clock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wall: float | Callable[[], float],
    mono: float | Callable[[], float] | None = None,
) -> None:
    """Patch both wall-clock and monotonic timers used by the API client."""

    if callable(wall):
        wall_func = wall
    else:
        wall_value = float(wall)

        def wall_func() -> float:
            return wall_value

    monkeypatch.setattr(api.time, "time", wall_func)

    if mono is None:
        mono_func = wall_func
    elif callable(mono):
        mono_func = mono
    else:
        mono_value = float(mono)

        def mono_func() -> float:
            return mono_value

    monkeypatch.setattr(api, "time_mod", mono_func)


def _set_token_expiry_seconds(client: RESTClient, seconds: float) -> None:
    """Set token expiry relative to the current time providers."""

    now_wall = api.time.time()
    now_mono = api.time_mod()
    client._token_expiry = now_wall + seconds
    client._token_expiry_monotonic = now_mono + seconds


class MockResponse:
    def __init__(
        self,
        status: int,
        json_data: Any,
        *,
        headers: dict[str, str] | None = None,
        text_data: str | Callable[[], str] | None = "",
        text_exc: Exception | Callable[[], Exception] | None = None,
        json_exc: Exception | Callable[[], Exception] | None = None,
    ) -> None:
        self.status = status
        self._json = json_data
        self._text = text_data
        self._text_exc = text_exc
        self._json_exc = json_exc
        self.headers = headers or {}
        self.request_info = None
        self.history = ()
        self.text_calls = 0
        self.json_calls = 0

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(
        self, exc_type, exc, tb
    ) -> None:  # pragma: no cover - no special handling
        return None

    async def text(self) -> str:
        self.text_calls += 1
        if self._text_exc is not None:
            exc = self._text_exc() if callable(self._text_exc) else self._text_exc
            raise exc
        value = self._text() if callable(self._text) else self._text
        if value is None:
            return ""
        return value

    async def json(
        self, content_type: str | None = None
    ) -> Any:  # pragma: no cover - simple pass-through
        self.json_calls += 1
        if self._json_exc is not None:
            exc = self._json_exc() if callable(self._json_exc) else self._json_exc
            raise exc
        return self._json() if callable(self._json) else self._json


class LatchedResponse:
    def __init__(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value


class _StubRequestInfo:
    def __init__(self, method: str, url: str) -> None:
        self.method = method
        self._url = url

    @property
    def real_url(self) -> str:
        return self._url


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
        if hasattr(result, "request_info") and result.request_info is None:
            result.request_info = _StubRequestInfo(method, url)
        return result

    def post(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.post_calls.append((url, args, copy.deepcopy(kwargs)))
        result = self._resolve(self._post_queue, "post")
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "request_info") and result.request_info is None:
            result.request_info = _StubRequestInfo("POST", url)
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

        fake_time = 0.0

        def _fake_time() -> float:
            return fake_time

        _patch_api_clock(monkeypatch, wall=_fake_time)
        token1 = await client._ensure_token()
        assert token1 == "t1"

        fake_time = 2.0  # advance beyond expiry
        token2 = await client._ensure_token()
        assert token2 == "t2"
        assert len(session.post_calls) == 2

    asyncio.run(_run())


def test_ducaheat_token_headers() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(
            session,
            "user",
            "pass",
            api_base="https://api-tevolve.termoweb.net",
        )

        token = await client._ensure_token()
        assert token == "tok"

        assert session.post_calls
        headers = session.post_calls[0][2]["headers"]
        assert headers["X-SerialId"] == "15"
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_DUCAHEAT)
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_DUCAHEAT)

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


def test_resolve_node_descriptor_validations() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Unsupported node descriptor"):
        client._resolve_node_descriptor("htr")

    with pytest.raises(ValueError, match="Invalid node type"):
        client._resolve_node_descriptor(("  ", "1"))

    with pytest.raises(ValueError, match="Invalid node address"):
        client._resolve_node_descriptor(("htr", ""))


def test_resolve_node_descriptor_normalises_values() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    node = AccumulatorNode(name=" Storage ", addr=" 007 ")
    assert client._resolve_node_descriptor(node) == ("acm", "007")

    assert client._resolve_node_descriptor(("HTR", " 08 ")) == ("htr", "08")


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
        assert client._token_expiry_monotonic == pytest.approx(fake_time + 3600)

    _patch_api_clock(monkeypatch, wall=lambda: fake_time)
    asyncio.run(_run())


def test_get_node_samples_success() -> None:
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
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == [{"t": 1000, "counter": "1.5"}]
        assert len(session.request_calls) == 1
        params = session.request_calls[0][2]["params"]
        assert params == {"start": 0, "end": 10}

    asyncio.run(_run())


def test_get_node_samples_404() -> None:
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
            await client.get_node_samples("dev", ("htr", "A"), 0, 10)
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
        headers = await client.authed_headers()
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


@pytest.mark.asyncio
async def test_authed_headers_builds_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")
    ensure_mock = AsyncMock(return_value="token")
    monkeypatch.setattr(client, "_ensure_token", ensure_mock)

    headers = await client.authed_headers()

    assert ensure_mock.await_count == 1
    expected = {
        "Authorization": "Bearer token",
        "Accept": "application/json",
        "User-Agent": client.user_agent,
        "Accept-Language": api.ACCEPT_LANGUAGE,
    }
    if client.requested_with:
        expected["X-Requested-With"] = client.requested_with
    assert headers == expected


@pytest.mark.asyncio
async def test_refresh_token_public_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")
    client._access_token = "cached"
    client._token_expiry = 123.0
    client._token_expiry_monotonic = 456.0
    client._token_obtained_at = 789.0
    client._token_obtained_monotonic = 987.0
    ensure_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(client, "_ensure_token", ensure_mock)

    await client.refresh_token()

    assert client._access_token is None
    assert client._token_expiry == 0.0
    assert client._token_expiry_monotonic == 0.0
    assert client._token_obtained_at == 0.0
    assert client._token_obtained_monotonic == 0.0
    assert ensure_mock.await_count == 1


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
        headers = await client.authed_headers()
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
        headers = await client.authed_headers()
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
        headers = await client.authed_headers()
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
    preview_logs = [
        rec.message for rec in caplog.records if "body[0:200]" in rec.message
    ]
    assert preview_logs


def test_request_preview_truncates_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[str, tuple[Any, ...]]] = []

        def debug(self, msg: str, *args: Any) -> None:
            self.debug_calls.append((msg, args))

    stub_logger = StubLogger()
    monkeypatch.setattr(api, "_LOGGER", stub_logger)
    monkeypatch.setattr(api, "API_LOG_PREVIEW", True)

    long_body = "Bearer secret-token " + ("0123456789" * 30)

    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"ok": True},
                headers={"Content-Type": "text/plain"},
                text_data=long_body,
            )
        )

        client = RESTClient(session, "user", "pass")
        result = await client._request("GET", "/api/preview", headers={})
        assert result == long_body

    asyncio.run(_run())

    preview_call = next(
        (call for call in stub_logger.debug_calls if "body[0:200]" in call[0]),
        None,
    )
    assert preview_call is not None, "Expected preview log entry"
    snippet = preview_call[1][-1]
    assert isinstance(snippet, str)
    assert len(snippet) == 200
    assert snippet.startswith("Bearer ***")
    assert "secret-token" not in snippet


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
        headers = await client.authed_headers()
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
        headers = await client.authed_headers()
        session.clear_calls()

        result = await client._request("GET", "/api/plain", headers=headers)
        assert result == "hello world"

    asyncio.run(_run())


def test_log_non_htr_payload_truncates_long_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    class StubLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[str, tuple[Any, ...]]] = []

        def debug(self, msg: str, *args: Any) -> None:
            self.debug_calls.append((msg, args))

    stub_logger = StubLogger()
    monkeypatch.setattr(api, "_LOGGER", stub_logger)

    payload = {"data": "x" * 600}
    client._log_non_htr_payload(
        node_type="acm",
        dev_id="device-12345",
        addr="001",
        stage="update",
        payload=payload,
    )

    assert len(stub_logger.debug_calls) == 1
    _, args = stub_logger.debug_calls[0]
    snippet = args[-1]
    assert isinstance(snippet, str)
    assert snippet.endswith("...")
    assert len(snippet) <= 500


def test_ensure_token_uses_cache_without_http() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "cached"
        _set_token_expiry_seconds(client, 1000.0)

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
        client._token_expiry_monotonic = 0.0

        class FakeLock:
            def __init__(self, owner: RESTClient) -> None:
                self._owner = owner

            async def __aenter__(self) -> "FakeLock":
                self._owner._access_token = "cached"
                _set_token_expiry_seconds(self._owner, 100.0)
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


def test_ducaheat_authed_request_headers() -> None:
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
            MockResponse(200, [], headers={"Content-Type": "application/json"})
        )

        client = RESTClient(
            session,
            "user",
            "pw",
            api_base="https://api-tevolve.termoweb.net",
        )

        await client.list_devices()

        assert session.request_calls
        method, url, kwargs = session.request_calls[0]
        assert method == "GET"
        assert url == "https://api-tevolve.termoweb.net/api/v2/devs/"
        headers = kwargs["headers"]
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_DUCAHEAT)
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_DUCAHEAT)
        assert headers["X-SerialId"] == "15"

    asyncio.run(_run())


def test_termoweb_authed_request_headers() -> None:
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
            MockResponse(200, [], headers={"Content-Type": "application/json"})
        )

        client = RESTClient(session, "user", "pw")

        await client.list_devices()

        assert session.request_calls
        method, url, kwargs = session.request_calls[0]
        assert method == "GET"
        assert url == "https://control.termoweb.net/api/v2/devs/"
        headers = kwargs["headers"]
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_TERMOWEB)
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_TERMOWEB)

    asyncio.run(_run())


def test_get_nodes_and_settings_use_expected_paths(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        calls: list[tuple[str, str]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((method, path))
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        await client.get_nodes("dev123")
        await client.get_node_settings("dev123", ("htr", "5"))
        await client.get_node_samples("dev123", ("htr", "5"), 0, 10)

        assert calls == [
            ("GET", api.NODES_PATH_FMT.format(dev_id="dev123")),
            ("GET", f"/api/v2/devs/dev123/htr/5/settings"),
            ("GET", f"/api/v2/devs/dev123/htr/5/samples"),
        ]

    asyncio.run(_run())


def test_get_rtc_time_uses_expected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        calls: list[tuple[str, str]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            calls.append((method, path))
            return {"status": "ok"}

        monkeypatch.setattr(client, "_request", fake_request)

        data = await client.get_rtc_time("dev123")

        assert data == {"status": "ok"}
        assert calls == [("GET", "/api/v2/devs/dev123/mgr/rtc/time")]

    asyncio.run(_run())


def test_get_rtc_time_handles_non_dict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            return ["unexpected"]

        monkeypatch.setattr(client, "_request", fake_request)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        data = await client.get_rtc_time("dev456")

        assert data == {}
        assert any(
            "Unexpected RTC time payload" in record.getMessage()
            for record in caplog.records
        )

    asyncio.run(_run())


def test_get_node_settings_acm_logs(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"status": {"mode": "auto"}, "capabilities": {"boost": {"max": 60}}},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        node = AccumulatorNode(name="Store", addr="7")
        data = await client.get_node_settings("dev", node)

        assert data["status"]["mode"] == "auto"
        expected = (
            f"GET settings node {mask_identifier('dev')}/{mask_identifier('7')}"
            " (acm) payload"
        )
        assert any(expected in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_get_node_settings_pmo_uses_device_endpoint(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            captured["method"] = method
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"status": {"power": 0}}

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "authed_headers", fake_headers)

        payload = await client.get_node_settings("dev", ("pmo", "4"))

        assert payload == {"status": {"power": 0}}
        assert captured == {
            "method": "GET",
            "path": "/api/v2/devs/dev/pmo/4",
            "kwargs": {"headers": {"Authorization": "Bearer tok"}},
        }

    asyncio.run(_run())


def test_get_node_samples_logs_for_unknown_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"unexpected": True},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        samples = await client.get_node_samples("dev", ("pmo", "4"), 0, 5)

        assert samples == []
        expected = (
            f"GET samples node {mask_identifier('dev')}/{mask_identifier('4')}"
            " (pmo) payload"
        )
        assert any(expected in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_set_node_settings_includes_prog_and_ptemp(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        received: list[dict[str, Any]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            received.append(kwargs["json"])
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        prog = [0, 1, 2] * 56
        ptemp = [18.0, 19.0, 20.0]
        await client.set_node_settings(
            "dev123",
            ("htr", "7"),
            prog=prog,
            ptemp=ptemp,
            units="f",
        )

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
    assert "Bearer ***" in caplog.text


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
        headers = await client.authed_headers()
        monkeypatch.setattr(api, "range", lambda _n: (0, 0), raising=False)

        with pytest.raises(api.BackendAuthError) as err:
            await client._request("GET", "/api/fail", headers=headers)

        assert str(err.value) == "Unauthorized"
        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 3

    asyncio.run(_run())


def test_set_node_settings_invalid_units() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid units"):
            await client.set_node_settings("dev", ("htr", "1"), units="kelvin")

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_node_settings_invalid_program() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="prog must be a list of 168"):
            await client.set_node_settings("dev", ("htr", "1"), prog=[0, 1, 2])

        with pytest.raises(ValueError, match="prog values must be 0, 1, or 2"):
            await client.set_node_settings("dev", ("htr", "1"), prog=[0] * 167 + [5])

        with pytest.raises(ValueError, match="prog contains non-integer value"):
            await client.set_node_settings(
                "dev", ("htr", "1"), prog=[0] * 167 + ["bad"]
            )

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_node_settings_invalid_temperatures() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid stemp value"):
            await client.set_node_settings("dev", ("htr", "1"), stemp="warm")

        with pytest.raises(
            ValueError, match="ptemp must be a list of three numeric values"
        ):
            await client.set_node_settings("dev", ("htr", "1"), ptemp=[21.0, 19.0])

        with pytest.raises(ValueError, match="ptemp contains non-numeric value"):
            await client.set_node_settings(
                "dev",
                ("htr", "1"),
                ptemp=[21.0, "bad", 23.0],
            )

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_get_node_samples_empty_payload() -> None:
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
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == []

    asyncio.run(_run())


def test_get_node_samples_decreasing_counters() -> None:
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
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == [
            {"t": 1, "counter": "3.0"},
            {"t": 2, "counter": "2.5"},
        ]

    asyncio.run(_run())


def test_get_node_samples_malformed_items(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "samples": [
                    123,
                    {"t": "bad"},
                    {"t": 5, "counter": None},
                ]
            }

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == []

    caplog.clear()
    asyncio.run(_run())
    messages = [rec.message for rec in caplog.records]
    assert any("Unexpected htr sample item" in msg for msg in messages)
    assert any("Unexpected htr sample shape" in msg for msg in messages)


def test_get_node_samples_unexpected_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            return "garbled"

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

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

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            return {"weird": []}

        monkeypatch.setattr(client, "authed_headers", fake_headers)
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

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            devices = await client.list_devices()

        assert devices == []

    caplog.clear()
    asyncio.run(_run())
    assert any("Unexpected /devs shape" in rec.message for rec in caplog.records)


def test_set_node_settings_translates_heat(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            captured["json"] = kwargs.get("json")
            return {}

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        await client.set_node_settings("dev", ("htr", 1), mode="heat", stemp=21.0)

        assert captured["json"]["mode"] == "manual"
        assert captured["json"]["stemp"] == "21.0"

    asyncio.run(_run())


def test_build_acm_extra_options_payload_with_boost_time_only() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    payload = client._build_acm_extra_options_payload(180, None)

    assert payload == {"extra_options": {"boost_time": 180}}


def test_build_acm_extra_options_payload_with_boost_temp_only() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    payload = client._build_acm_extra_options_payload(None, 23.0)

    assert payload == {"extra_options": {"boost_temp": "23.0"}}


def test_build_acm_extra_options_payload_missing_inputs() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="must be provided"):
        client._build_acm_extra_options_payload(None, None)


def test_build_acm_extra_options_payload_rejects_invalid_boost_temp() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid boost_temp value"):
        client._build_acm_extra_options_payload(None, "invalid")


@pytest.mark.asyncio
async def test_set_acm_extra_options_forwards_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    sentinel_payload: dict[str, Any] = {"extra_options": {"boost_time": 120}}
    request_mock = AsyncMock(return_value={"ok": True})
    headers_mock = AsyncMock(return_value={"Authorization": "Bearer token"})

    monkeypatch.setattr(
        client, "_build_acm_extra_options_payload", lambda *args: sentinel_payload
    )
    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "authed_headers", headers_mock)
    monkeypatch.setattr(client, "_log_non_htr_payload", lambda **_: None)

    response = await client.set_acm_extra_options("dev123", 9, boost_time=120)

    assert response == {"ok": True}
    assert request_mock.await_count == 1
    await_call = request_mock.await_args
    assert await_call.args[0] == "POST"
    assert await_call.args[1] == "/api/v2/devs/dev123/acm/9/setup"
    assert await_call.kwargs["json"] is sentinel_payload
    assert await_call.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert headers_mock.await_count == 1


@pytest.mark.asyncio
async def test_set_acm_boost_state_formats_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    request_mock = AsyncMock(return_value={"ok": True})
    headers_mock = AsyncMock(return_value={"Authorization": "Bearer token"})

    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "authed_headers", headers_mock)
    monkeypatch.setattr(client, "_log_non_htr_payload", lambda **_: None)

    response = await client.set_acm_boost_state(
        "dev123",
        "7",
        boost=True,
        boost_time=120,
        stemp=22.5,
        units=" f ",
    )

    assert response == {"ok": True}
    assert request_mock.await_count == 1
    await_call = request_mock.await_args
    assert await_call.args[0] == "POST"
    assert await_call.args[1] == "/api/v2/devs/dev123/acm/7/boost"
    assert await_call.kwargs["json"] == {
        "boost": True,
        "boost_time": 120,
        "stemp": "22.5",
        "units": "F",
    }
    assert await_call.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert headers_mock.await_count == 1


@pytest.mark.asyncio
async def test_set_acm_boost_state_rejects_invalid_units() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid units"):
        await client.set_acm_boost_state("dev456", "8", boost=True, units="kelvin")


@pytest.mark.asyncio
async def test_set_acm_boost_state_rejects_invalid_stemp() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid stemp value"):
        await client.set_acm_boost_state("dev789", "9", boost=True, stemp="oops")


def test_ducaheat_get_node_settings_normalises_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "mode": "Manual",
                        "state": "heating",
                        "stemp": "21.0",
                        "temp": 20.5,
                        "units": "c",
                        "boost_active": True,
                    },
                    "setup": {
                        "extra_options": {"boost_temp": "23.0", "boost_time": 45}
                    },
                    "prog": {
                        "days": {
                            "mon": {"slots": [0, 1, 2, 0] * 6},
                            "tue": {"slots": [1] * 24},
                            "wed": {"slots": [2] * 24},
                            "thu": {"slots": [0] * 24},
                            "fri": {"slots": [1, 2] * 12},
                            "sat": {"slots": [2, 2, 1, 1] * 6},
                            "sun": {"slots": [0, 0, 1, 2] * 6},
                        }
                    },
                    "prog_temps": {
                        "comfort": "21.0",
                        "eco": "18.0",
                        "antifrost": "7.0",
                    },
                    "addr": "A1",
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        data = await client.get_node_settings("dev", ("htr", "A1"))

        assert data["mode"] == "manual"
        assert data["state"] == "heating"
        assert data["stemp"] == "21.0"
        assert data["mtemp"] == "20.5"
        assert data["units"] == "C"
        assert "boost_active" not in data
        assert "boost_time" not in data
        assert "boost_temp" not in data
        assert len(data["prog"]) == 168
        assert data["ptemp"] == ["7.0", "18.0", "21.0"]
        assert data["raw"]["status"]["mode"] == "Manual"

    asyncio.run(_run())


def test_ducaheat_get_node_settings_acm_merges_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "mode": "auto",
                        "capabilities": {"boost": {"max": 90}},
                    },
                    "setup": {
                        "capabilities": {
                            "boost": {"min": 10},
                            "charge": {"modes": ["eco"]},
                        },
                        "extra_options": {"boost_temp": "25.0", "boost_time": 30},
                    },
                    "capabilities": {"charge": {"levels": [1, 2, 3]}},
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        data = await client.get_node_settings("dev", ("acm", "2"))

        assert data["mode"] == "auto"
        assert data["boost_temp"] == "25.0"
        assert data["boost_time"] == 30
        assert data["capabilities"] == {
            "boost": {"max": 90, "min": 10},
            "charge": {"modes": ["eco"], "levels": [1, 2, 3]},
        }
        assert session.request_calls[0][1] == (
            "https://api.termoweb.fake/api/v2/devs/dev/acm/2"
        )

    asyncio.run(_run())


def test_ducaheat_get_node_settings_acm_handles_half_hour_prog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        half_hour_prog = {
            str(idx): pattern
            for idx, pattern in enumerate(
                [
                    [0, 2] * 24,
                    [0, 0] * 24,
                    [2, 2] * 24,
                    [0, 1] * 24,
                    [1, 2] * 24,
                    [0, 2, 1, 0] * 12,
                    [0] * 48,
                ]
            )
        }

        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "sync_status": "ok",
                        "mode": "off",
                        "heating": False,
                        "units": "C",
                        "stemp": "21.0",
                        "mtemp": 24.4,
                    },
                    "setup": {
                        "sync_status": "ok",
                        "operational_mode": 1,
                        "control_mode": 5,
                        "units": "C",
                        "offset": "0.0",
                        "priority": "medium",
                        "away_offset": "2.0",
                        "window_mode_enabled": False,
                        "prog_resolution": 1,
                        "charging_conf": {
                            "slot_1": {"start": 0, "end": 1430},
                            "slot_2": {"start": 0, "end": 0},
                            "active_days": [1, 1, 1, 1, 1, 1, 1],
                        },
                        "min_stemp": "5.0",
                        "max_stemp": "30.0",
                    },
                    "prog": {"sync_status": "ok", "prog": half_hour_prog},
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        data = await client.get_node_settings("dev", ("acm", "2"))

        assert data["mode"] == "off"
        assert data["stemp"] == "21.0"
        assert data["mtemp"] == "24.4"
        assert data["units"] == "C"
        assert len(data["prog"]) == 168
        assert data["prog"][:24] == [2] * 24
        assert data["prog"][24:48] == [0] * 24
        assert data["prog"][72:96] == [1] * 24
        assert data["prog"][96:120] == [2] * 24
        assert session.request_calls[0][1] == (
            "https://api.termoweb.fake/api/v2/devs/dev/acm/2"
        )

    asyncio.run(_run())


def test_ducaheat_set_node_settings_invalid_stemp(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        selection_calls: list[bool] = []

        async def fake_select_segmented_node(**kwargs: Any) -> None:
            selection_calls.append(kwargs["select"])

        monkeypatch.setattr(
            client, "_select_segmented_node", fake_select_segmented_node
        )

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("htr", "A1"), stemp="bad")

        assert "Invalid stemp value" in str(exc.value)
        assert selection_calls == [True, False]

    asyncio.run(_run())


def test_ducaheat_set_node_settings_invalid_units(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(200, {}, headers={"Content-Type": "application/json"})
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        selection_calls: list[bool] = []

        async def fake_select_segmented_node(**kwargs: Any) -> None:
            selection_calls.append(kwargs["select"])

        monkeypatch.setattr(
            client, "_select_segmented_node", fake_select_segmented_node
        )

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("htr", "A1"), stemp=21.0, units="K")

        assert "Invalid units" in str(exc.value)
        assert selection_calls == [True, False]

    asyncio.run(_run())


def test_ducaheat_set_acm_settings_short_prog(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("acm", "5"), prog=[0] * 24)

        assert "168" in str(exc.value)
        assert not session.request_calls

    asyncio.run(_run())


def test_ducaheat_set_acm_settings_client_error(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                400,
                {},
                headers={"Content-Type": "text/plain"},
                text_data="bad request",
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "5"), mode="boost")

        assert "bad request" in str(exc.value)

    asyncio.run(_run())


def test_rest_client_set_node_settings_rejects_boost_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError):
            await client.set_node_settings("dev", ("htr", "3"), boost_time=30)

    asyncio.run(_run())


def test_ducaheat_get_node_samples_converts_ms(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1_700_000_000_500, "counter": 7.5}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        samples = await client.get_node_samples("dev", ("htr", "A"), 10, 20)
        assert samples == [{"t": 1_700_000_000, "counter": "7.5"}]

        call = session.request_calls[0]
        assert call[1] == "https://api.termoweb.fake/api/v2/devs/dev/htr/A/samples"
        assert call[2]["params"] == {"start": 10, "end": 20}

    asyncio.run(_run())


def test_ducaheat_get_node_samples_keeps_second_payload(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1_700_000_010, "counter": 3}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        samples = await client.get_node_samples("dev", ("htr", "A"), 5, 30)
        assert samples == [{"t": 1_700_000_010, "counter": "3"}]

        call = session.request_calls[0]
        assert call[1] == "https://api.termoweb.fake/api/v2/devs/dev/htr/A/samples"
        assert call[2]["params"] == {"start": 5, "end": 30}

    asyncio.run(_run())


def test_ducaheat_normalise_settings_non_dict() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    assert client._normalise_settings(123) == {}


def test_ducaheat_normalise_settings_fallbacks() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    payload = {
        "status": {
            "set_temp": "20.5",
            "ambient": 19,
            "boost_temp": "  23.0 ",
            "boost_time": 30,
        },
        "prog": None,
        "prog_temps": {"cold": "5", "night": None, "day": " 18 "},
        "name": "Heater",
    }

    result = client._normalise_settings(payload)
    assert result["stemp"] == "20.5"
    assert result["mtemp"] == "19.0"
    assert "boost_temp" not in result
    assert "boost_time" not in result
    assert result["ptemp"] == ["5.0", "", "18.0"]
    assert result["name"] == "Heater"


def test_ducaheat_normalise_settings_acm_status_boost() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    payload = {
        "status": {
            "boost_temp": "24",
            "boost_time": 15,
            "boost_active": True,
        }
    }

    result = client._normalise_settings(payload, node_type="acm")
    assert result["boost_temp"] == "24.0"
    assert result["boost_time"] == 15
    assert result["boost_active"] is True


def test_ducaheat_normalise_settings_acm_boost_metadata() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    payload = {
        "status": {
            "boost": True,
            "boost_end": {"day": 2, "minute": 45},
        },
        "setup": {
            "extra_options": {"boost_end_min": 150},
        },
    }

    result = client._normalise_settings(payload, node_type="acm")
    assert result["boost"] is True
    assert result["boost_end"] == {"day": 2, "minute": 45}
    assert result["boost_end_day"] == 2
    assert result["boost_end_min"] == 45


def test_ducaheat_normalise_settings_acm_boost_metadata_fallback() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    payload = {
        "status": {},
        "setup": {
            "extra_options": {"boost_end_min": 180},
            "boost_end": {"day": 4},
        },
    }

    result = client._normalise_settings(payload, node_type="acm")
    assert result["boost_end_day"] == 4
    assert result["boost_end_min"] == 180


def test_ducaheat_merge_boost_metadata_defensive() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    target: dict[str, Any] = {"boost": True, "boost_end": None}

    client._merge_boost_metadata(target, None)
    client._merge_boost_metadata(target, {"boost_end": None}, prefer_existing=True)

    assert target == {"boost": True, "boost_end": None}


def test_ducaheat_normalise_settings_handles_half_hour_prog() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    payload = {
        "prog": {"prog": {str(day): [day % 3] * 48 for day in range(7)}},
        "status": {"mode": "AUTO"},
    }

    result = client._normalise_settings(payload)
    assert len(result["prog"]) == 168
    assert result["prog"][:24] == [0] * 24
    assert result["prog"][24:48] == [1] * 24


def test_ducaheat_normalise_prog_with_varied_inputs() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )
    data = {
        "mon": {"values": [0, 1, 2]},
        "tue": [1] * 10,
        "wed": {"slots": [2] * 30},
        "fri": {"slots": [0] * 24},
        "sat": {"values": [1] * 24},
    }

    result = client._normalise_prog(data)
    assert result is not None
    assert len(result) == 168
    # Monday should extend with zeros to 24 slots
    assert result[:3] == [0, 1, 2]


def test_ducaheat_normalise_prog_invalid_inputs() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    assert client._normalise_prog("bad") is None
    assert client._normalise_prog({"foo": "bar"}) is None
    assert client._normalise_prog([0] * 168 + ["x"]) is None

    bad_day = {"days": {"mon": ["x"]}}
    assert client._normalise_prog(bad_day) is None
    assert client._normalise_prog({"days": {"mon": {"slots": None}}}) is None
    assert client._normalise_prog({"days": {"mon": {"slots": 123}}}) is None
    assert client._normalise_prog({"days": {"mon": {"values": "abc"}}}) is None


def test_ducaheat_normalise_prog_temps_variations() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    assert client._normalise_prog_temps("bad") is None

    temps = client._normalise_prog_temps(
        {"antifrost": None, "eco": " 18.5 ", "comfort": "abc"}
    )
    assert temps == ["", "18.5", "abc"]

    weird = client._normalise_prog_temps(
        {"antifrost": ["bad"], "eco": None, "comfort": 18}
    )
    assert weird == ["['bad']", "", "18.0"]


def test_ducaheat_serialise_prog_expands_half_hours() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    prog: list[int] = []
    for day in range(7):
        prog.extend([day % 3] * 24)

    serialised = client._serialise_prog(prog)
    assert set(serialised) == {"prog"}
    assert len(serialised["prog"]) == 7
    assert serialised["prog"]["0"] == [0] * 48
    assert serialised["prog"]["1"] == [1] * 48


def test_rest_client_normalise_ws_nodes_passthrough() -> None:
    client = RESTClient(FakeSession(), "user", "pass", api_base="https://api.fake")
    payload = {"htr": {"settings": {"01": {}}}}
    assert client.normalise_ws_nodes(payload) is payload


def test_ducaheat_safe_temperature_handles_strings() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    assert client._safe_temperature(None) is None
    assert client._safe_temperature(" 21.2 ") == "21.2"
    assert client._safe_temperature("   ") is None
    assert client._safe_temperature("abc") == "abc"
    assert client._safe_temperature(["oops"]) is None


def test_extract_samples_handles_list_payload() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    samples = client._extract_samples(
        [
            {"timestamp": 2000.0, "value": 5.5},
            {"t": "bad"},
            {"timestamp": 1000, "energy": 3},
        ]
    )

    assert samples == [{"t": 2000, "counter": "5.5"}, {"t": 1000, "counter": "3"}]


def test_extract_samples_preserves_min_max() -> None:
    client = RESTClient(FakeSession(), "user", "pass", api_base="https://api.fake")

    samples = client._extract_samples(
        [
            {
                "t": 1000,
                "counter": {"value": 3_600_000, "min": 3_500_000, "max": 3_700_000},
            },
            {
                "t": 2000,
                "counter": 7_200_000,
                "counter_min": 7_100_000,
                "counter_max": 7_300_000,
            },
        ]
    )

    assert samples == [
        {
            "t": 1000,
            "counter": "3600000",
            "counter_min": "3500000",
            "counter_max": "3700000",
        },
        {
            "t": 2000,
            "counter": "7200000",
            "counter_min": "7100000",
            "counter_max": "7300000",
        },
    ]


def test_extract_samples_uses_counter_field_when_value_missing() -> None:
    client = RESTClient(FakeSession(), "user", "pass", api_base="https://api.fake")

    samples = client._extract_samples(
        [
            {
                "t": 3000,
                "counter": {
                    "counter": 12_345,
                    "min": 12_000,
                    "max": 13_000,
                },
            }
        ]
    )

    assert samples == [
        {
            "t": 3000,
            "counter": "12345",
            "counter_min": "12000",
            "counter_max": "13000",
        }
    ]


@pytest.mark.asyncio
async def test_rest_client_rejects_cancel_boost_for_non_acm() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="cancel_boost"):
        await client.set_node_settings(
            "dev",
            ("pmo", "1"),
            cancel_boost=True,
        )


def test_rest_client_header_properties_exposed() -> None:
    client = RESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )

    assert client.user_agent == get_brand_user_agent(BRAND_TERMOWEB)
    assert client.requested_with == get_brand_requested_with(BRAND_TERMOWEB)


def test_ducaheat_rest_client_header_properties_exposed() -> None:
    client = DucaheatRESTClient(
        FakeSession(),
        "user",
        "pass",
        api_base=api.DUCAHEAT_API_BASE,
    )

    assert client.user_agent == get_brand_user_agent(BRAND_DUCAHEAT)
    assert client.requested_with == get_brand_requested_with(BRAND_DUCAHEAT)
